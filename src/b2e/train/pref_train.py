"""Offline preference optimization on the P5 pairs with TRL 0.27: DPO, SimPO, KTO, ORPO.

A run is configs/po/<run>.yaml: the method, the starting model, the rendered pairs
(b2e.data.pref_render: strings, so no trainer renders them again) and the fields of
the method's TRL config. One GPU, full fine-tuning:
  - the policy is loaded in fp32 and trained under bf16 autocast (bf16: true), so
    AdamW updates fp32 weights. At DPO's learning rates (~5e-7) a bf16 weight cannot
    hold the update: with 8 bits of mantissa, a relative change below ~0.4 % rounds
    away.
  - Qwen3_5ForConditionalGeneration's vision tower is frozen but kept, so the export
    has S-fin's architecture (cluster/common.sh check_arch).
  - DPO and KTO compare against S-fin itself, loaded again in bf16 and frozen; SimPO
    (CPO's simpo loss, cpo_alpha 0) and ORPO need no reference.
Before training, every row the trainer built is checked against the sequence the
model generated, prompt + completion + eos, which b2e.data.pref_render matched to
the conversation through <|im_end|>: each trainer tokenizes with its own code (DPO
tokenize_row; CPO and ORPO the older build_tokenized_answer path; KTO _tokenize),
so the check is on what each one built, and a mismatch stops the run.
CPO, KTO and ORPO come from trl.experimental: the trl.trainer names only forward
there and warn (removed in TRL 0.29).

    python -P -m b2e.train.pref_train --config configs/po/dpo_fin.yaml --model $MODELS/s_fin \\
        --data $B/data/processed/pref_fin --out $CKPT/dpo_fin [--max-steps 3 --rows 16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

METHODS = ("dpo", "simpo", "kto", "orpo")
LABEL_PAD = -100


def load_pairs(data: Path, split: str, rows: int | None = None) -> list[dict]:
    """The rendered pairs of a split (train or test) as dicts of strings."""
    df = pd.read_parquet(data / f"{split}_rendered.parquet")
    if rows:
        df = df.head(rows)
    return df[["prompt", "chosen", "rejected"]].to_dict("records")


def unpaired(pairs: list[dict]) -> list[dict]:
    """KTO's format: each pair as a desirable and an undesirable completion."""
    return [row for p in pairs for row in ({"prompt": p["prompt"], "completion": p["chosen"], "label": True},
                                           {"prompt": p["prompt"], "completion": p["rejected"], "label": False})]


def expected(tok, prompt: str, completion: str) -> list[int]:
    """The sequence the model generated: prompt, completion, one eos (<|im_end|>)."""
    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]
    return ids(prompt) + ids(completion) + [tok.eos_token_id]


def check_rows(method: str, built: list[dict], source: list[dict], tok) -> list[str]:
    """Mismatches between the trainer's rows and the generated sequences, one line each."""
    bad = []
    for i, (b, s) in enumerate(zip(built, source, strict=True)):
        if method == "dpo":
            sides = {k: b["prompt_input_ids"] + b[f"{k}_input_ids"] for k in ("chosen", "rejected")}
            labels = {}
        elif method == "kto":
            sides = {"completion": b["completion_input_ids"]}
            labels = {"completion": (b["completion_labels"], len(b["prompt_input_ids"]))}
        else:  # simpo (CPO), orpo
            sides = {k: b[f"{k}_input_ids"] for k in ("chosen", "rejected")}
            labels = {k: (b[f"{k}_labels"], len(b["prompt_input_ids"])) for k in ("chosen", "rejected")}
        for k, got in sides.items():
            if got != expected(tok, s["prompt"], s[k]):
                bad.append(f"row {i} {k}: tokens differ from the generated sequence")
            if k in labels:
                lab, n = labels[k]
                if lab[:n] != [LABEL_PAD] * n or lab[n:] != got[n:]:
                    bad.append(f"row {i} {k}: labels do not mask exactly the prompt")
    return bad


def padding_check(model, tok, pairs: list[dict]) -> dict:
    """|log-prob difference| on the completion tokens of the shortest and the longest of the
    pairs' chosen sequences, each run alone and both in one batch, the shorter right-padded
    as TRL's flush_left leaves it. Padding after the real tokens must not change them;
    Qwen3.5's linear-attention layers carry a state, so this is measured, not assumed."""
    import torch

    def ids_of(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    # (tokens, index of the first completion token), shortest and longest
    seqs = sorted(((expected(tok, p["prompt"], p["chosen"]), len(ids_of(p["prompt"]))) for p in pairs),
                  key=lambda s: len(s[0]))
    seqs = [seqs[0], seqs[-1]]
    dev = next(model.parameters()).device

    def logps(ids, mask):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        return torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)

    alone = [logps(torch.tensor([s], device=dev), torch.ones(1, len(s), dtype=torch.long, device=dev))[0]
             for s, _ in seqs]
    n = len(seqs[1][0])
    ids = torch.full((2, n), tok.eos_token_id, device=dev)
    mask = torch.zeros((2, n), dtype=torch.long, device=dev)
    for i, (s, _) in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, device=dev)
        mask[i, :len(s)] = 1
    both = logps(ids, mask)
    out = {}
    for i, (name, (s, start)) in enumerate(zip(("padded", "longest"), seqs, strict=True)):
        k, st = len(s), start - 1  # the log-prob of token t sits at position t - 1
        d = (alone[i][st:k - 1] - both[i][st:k - 1]).abs()
        out[name] = {"tokens": k, "max": round(d.max().item(), 5), "mean": round(d.mean().item(), 5)}
    return out


def trainer_for(method: str, cfg: dict, model, ref, tok, train: list[dict], test: list[dict], out: Path):
    from datasets import Dataset

    if method == "dpo":
        from trl import DPOConfig as Config
        from trl import DPOTrainer as Trainer
        extra = {"ref_model": ref}
    elif method == "kto":
        from trl.experimental.kto import KTOConfig as Config
        from trl.experimental.kto import KTOTrainer as Trainer
        extra = {"ref_model": ref}
        train, test = unpaired(train), unpaired(test)
    elif method == "simpo":
        from trl.experimental.cpo import CPOConfig as Config
        from trl.experimental.cpo import CPOTrainer as Trainer
        extra = {}
        cfg = {**cfg, "loss_type": "simpo", "cpo_alpha": 0.0}
    else:
        from trl.experimental.orpo import ORPOConfig as Config
        from trl.experimental.orpo import ORPOTrainer as Trainer
        extra = {}
    args = Config(output_dir=str(out), **cfg)
    return Trainer(model=model, args=args, train_dataset=Dataset.from_list(train),
                   eval_dataset=Dataset.from_list(test), processing_class=tok, **extra), train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="the starting model dir (S-fin)")
    ap.add_argument("--data", type=Path, required=True, help="dir with train/test_rendered.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke: stop after this many steps")
    ap.add_argument("--rows", type=int, default=None, help="smoke: only the first rows of each split")
    ap.add_argument("--padding-check", action="store_true",
                    help="smoke: before training, compare log-probs alone and right-padded in a batch")
    return ap.parse_args(argv)


def main() -> None:
    a = parse_args()
    spec = yaml.safe_load(a.config.read_text())
    method, cfg = spec["method"], dict(spec["trl"])
    if method not in METHODS:
        raise SystemExit(f"method {method!r} is not one of {METHODS}")
    if a.max_steps:
        cfg.update(max_steps=a.max_steps, save_strategy="no", eval_strategy="no")

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float32)
    frozen = [n for n, p in model.named_parameters() if "visual" in n]
    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad_(False)
    ref = None
    if method in ("dpo", "kto"):
        ref = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).eval()
        ref.requires_grad_(False)

    train, test = load_pairs(a.data, "train", a.rows), load_pairs(a.data, "test", a.rows)
    if a.padding_check:
        model.cuda()
        print(json.dumps({"padding_check": padding_check(model, tok, train)}), flush=True)
    trainer, train_rows = trainer_for(method, cfg, model, ref, tok, train, test, a.out)
    bad = check_rows(method, list(trainer.train_dataset), train_rows, tok)
    print(json.dumps({"method": method, "train_rows": len(train_rows), "frozen_params": len(frozen),
                      "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "row_check_failures": len(bad)}), flush=True)
    if bad:
        raise SystemExit("rows differ from the generated sequences:\n" + "\n".join(bad[:10]))
    result = trainer.train()
    print(json.dumps({"train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
                      "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
    if a.max_steps:
        return
    # weights only, in bf16 as every other export; the job copies the starting model's own
    # tokenizer and processor files next to them (cluster/common.sh keep_tokenizer), since
    # transformers re-serializes them differently
    final = a.out / "final"
    model.to(torch.bfloat16).save_pretrained(final, safe_serialization=True)
    print(json.dumps({"saved": str(final)}), flush=True)


if __name__ == "__main__":
    main()
