"""SFT loss of a model on the held-out split of an SFT mix, measured after training.

The four SFT runs train without validation (F#45: `val_files: null` in
configs/sft/s_dom.yaml), so the val split b2e.data.sft_mix builds is scored
here instead. Each row is rendered by the dataset class the trainer uses
(b2e.train.sft_dataset.SystemPromptSFTDataset, with verl's hf_tokenizer /
hf_processor, as tools/check_sft_render.py), so the tokens and the loss mask
are the ones the student was trained on. vLLM's prompt log-probabilities give
-log p(token | earlier tokens); the loss is their mean over the target tokens
(loss_mask = 1), the token mean verl's SFT loss takes. Run it on M0 for the
value before SFT. Reported per file, per source and per thinking mode, with
every row's -log p sum and token count kept in the output.

Each mix draws its own val split (F#47), so a row in one mix's val can be in
another mix's train: S-fin trained on most of mix_mix's finance rows. Pass the
model's train file(s) as --exclude-train and each file also gets an `unseen`
summary over the rows whose (source, group) is in none of them; run the
model it is compared with on the same flags so both cover the same rows.

    python -P tools/sft_val_loss.py --model /tmp/b2e-models/s_fin \\
        --data $B2E_ROOT/data/sft/mix_fin/val.parquet --out results/s_fin/val_loss.json \\
        --exclude-train $B2E_ROOT/data/sft/mix_fin/train.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

CFG = {"pad_mode": "no_padding", "max_length": 12288, "truncation": "error",
       "messages_key": "messages", "enable_thinking_key": "enable_thinking"}


def masked_nll(prompt_logprobs: list, ids: list[int], mask: list[int]) -> tuple[float, int]:
    """Sum of -log p over the target tokens, and their count. Position 0 has no prediction."""
    nll, n = 0.0, 0
    for pos in range(1, len(ids)):
        if mask[pos]:
            nll -= prompt_logprobs[pos][ids[pos]].logprob
            n += 1
    return nll, n


def summarize(recs: list[dict]) -> dict:
    """Token-mean loss over all rows, per source and per thinking mode."""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for r in recs:
        for key in ("all", f"source:{r['source']}", f"mode:{r['mode']}"):
            acc[key][0] += r["nll"]
            acc[key][1] += r["n"]
    res = {k: {"loss": round(v[0] / v[1], 4), "target_tokens": v[1]} for k, v in sorted(acc.items()) if v[1]}
    if "all" in res:
        res["all"]["rows"] = len(recs)
        res["all"]["ppl"] = round(math.exp(res["all"]["loss"]), 3)
    return res


def train_keys(files: list[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for f in files:
        t = pd.read_parquet(f, columns=["source", "group"])
        keys.update(zip(t["source"], t["group"], strict=True))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="val.parquet files of SFT mixes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.7)
    ap.add_argument("--exclude-train", nargs="*", default=[], help="train.parquet files the model was trained on")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from verl.utils import hf_processor, hf_tokenizer
    from vllm import LLM, SamplingParams

    from b2e.train.sft_dataset import SystemPromptSFTDataset

    tok, proc = hf_tokenizer(args.model), hf_processor(args.model)
    cfg = OmegaConf.create(CFG)
    # prompt log-probs are recomputed for every request, so prefix caching buys nothing here; small
    # prefill chunks bound the log-prob tensor of a 12K-token row (248K vocabulary)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=CFG["max_length"] + 1,
              enable_prefix_caching=False, max_num_batched_tokens=4096)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    seen_keys = train_keys(args.exclude_train)
    result: dict = {"model": args.model, "exclude_train": args.exclude_train, "files": {}}
    for f in args.data:
        df = pd.read_parquet(f)
        ds = SystemPromptSFTDataset(f, tok, cfg, processor=proc)
        rows = [ds[i] for i in range(len(ds))]
        ids = [r["input_ids"].tolist() for r in rows]
        masks = [r["loss_mask"].tolist() for r in rows]
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
        recs = []
        for i, o in enumerate(outs):
            nll, n = masked_nll(o.prompt_logprobs, ids[i], masks[i])
            src, grp = str(df["source"].iat[i]), str(df["group"].iat[i])
            recs.append({"source": src, "group": grp, "mode": "think" if bool(df["enable_thinking"].iat[i]) else "nothink",
                         "nll": nll, "n": n, "seen": (src, grp) in seen_keys})
        res = summarize(recs)
        res["all"]["seconds"] = round(time.time() - t0, 1)
        if args.exclude_train:
            res["unseen"] = summarize([r for r in recs if not r["seen"]])
            res["excluded_rows"] = sum(r["seen"] for r in recs)
        res["per_row"] = recs
        result["files"][f] = res
        print(f, json.dumps(res["all"]), "excluded_rows", res.get("excluded_rows"), flush=True)
        for k, v in res.items():
            if k.startswith(("source:", "mode:")):
                print("   ", k, v, flush=True)
        if args.exclude_train:
            print("    unseen", json.dumps(res["unseen"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
