"""Offline preference optimization on the P5 pairs with TRL 0.27: DPO, SimPO, KTO, ORPO.

A run is configs/po/<run>.yaml: the method, the starting model, the rendered pairs
(b2e.data.pref_render: strings, so no trainer renders them again) and the fields of
the method's TRL config. One GPU, full fine-tuning:
  - the policy is loaded in fp32 and trained under bf16 autocast (bf16: true), so
    AdamW updates fp32 weights. At DPO's learning rates (~5e-7) a bf16 weight cannot
    hold the update: bf16 keeps 8 significant bits, so round-to-nearest drops a change
    below half the spacing, ~0.2-0.4 % of the weight; Adam's step at 5e-7 is ~1e-5 of it.
  - Qwen3_5ForConditionalGeneration's vision tower is frozen but kept, so the export
    has S-fin's architecture (cluster/common.sh check_arch).
  - DPO and KTO compare against S-fin itself, loaded again in bf16 and frozen; SimPO
    (CPO's simpo loss, cpo_alpha 0) and ORPO need no reference.
Before training, every row the trainer built is checked against the sequence the
model generated, prompt + completion + eos, which b2e.data.pref_render matched to
the conversation through <|im_end|>: each trainer tokenizes with its own code (DPO
tokenize_row; CPO and ORPO the older build_tokenized_answer path; KTO _tokenize),
so the check is on what each one built, and a mismatch stops the run. KTO's KL rows, which
TRL pairs up by dataset order, are checked too: each must pair a prompt with another prompt's answer.
CPO, KTO and ORPO come from trl.experimental: the trl.trainer names only forward
there and warn (removed in TRL 0.29). TRL 0.27 next to transformers 5.9 needs one fix before
any trainer is imported (fix_trl_availability).

    python -P -m b2e.train.pref_train --config configs/po/dpo_fin.yaml --model $MODELS/s_fin \\
        --data $B/data/processed/pref_fin --out $CKPT/dpo_fin [--max-steps 3 --rows 16]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    """KTO's format: each pair's chosen answer a desirable completion and its rejected answer an
    undesirable one, in rows of two with different prompts. TRL builds KTO's KL rows (its
    reference point) at load time by moving each batch's completions one row down, batches of
    per_device_train_batch_size rows in dataset order (kto_trainer._get_kl_dataset), so a pair's
    two rows side by side would give each row the other answer to its own prompt: a matched
    pair, which the KL term must not see. Row 2k is pair k's chosen and row 2k + 1 the rejected
    of pair k + s, the pairs sorted by prompt and s the most pairs any prompt has."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i]["prompt"])
    s = max(Counter(p["prompt"] for p in pairs).values(), default=0)
    if 2 * s > len(pairs):  # then some row of that prompt can only be put next to the same prompt
        raise ValueError(f"{s} of the {len(pairs)} pairs share one prompt, more than half")
    rows = []
    for k, i in enumerate(order):
        j = order[(k + s) % len(order)]
        rows += [{"prompt": pairs[i]["prompt"], "completion": pairs[i]["chosen"], "label": True},
                 {"prompt": pairs[j]["prompt"], "completion": pairs[j]["rejected"], "label": False}]
    return rows


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


def kl_partner(i: int, n: int, group: int) -> int:
    """The row whose completion TRL gives row i as its KL completion (kto_trainer._get_kl_dataset:
    batches of `group` rows, the completions moved one row down, the last to the top)."""
    start = i - i % group
    return start + (i - start - 1) % min(group, n - start)


def check_kl_rows(built: list[dict], source: list[dict], tok, group: int) -> list[str]:
    """KTO's KL rows as the trainer built them: each row's prompt with another prompt's completion."""
    bad = []
    for i, b in enumerate(built):
        j = kl_partner(i, len(built), group)
        if source[j]["prompt"] == source[i]["prompt"]:
            bad.append(f"row {i}: its KL completion (row {j}'s) answers the same prompt")
        got, n = b["KL_completion_input_ids"], len(b["KL_prompt_input_ids"])
        if got != expected(tok, source[i]["prompt"], source[j]["completion"]):
            bad.append(f"row {i}: KL tokens are not its prompt with row {j}'s completion")
        if b["KL_completion_labels"][:n] != [LABEL_PAD] * n or b["KL_completion_labels"][n:] != got[n:]:
            bad.append(f"row {i}: KL labels do not mask exactly the prompt")
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


def bool_found(found):
    """transformers' _is_package_available, but a bool again when return_version is False."""
    def available(name: str, return_version: bool = False):
        out = found(name, return_version)
        return out if return_version else out[0]
    return available


def fix_trl_availability() -> bool:
    """TRL 0.27's is_<package>_available() helpers (weave, mergekit, unsloth and others) return
    transformers' _is_package_available(name), which transformers 5.x makes a tuple, (found,
    version), even without return_version; a tuple is always true, so TRL imports packages that
    are not installed. Importing DPO's trainer runs trl.trainer.callbacks' `if
    is_weave_available(): import weave`, which failed in the train env (11559903). TRL's copy
    of the helper is pointed at bool_found before any trainer is imported; returns whether that
    was needed (on a transformers that returns a bool it changes nothing)."""
    import trl.import_utils as tiu

    if not isinstance(tiu._is_package_available("b2e_no_such_package"), tuple):
        return False
    tiu._is_package_available = bool_found(tiu._is_package_available)
    return True


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
    fixed = fix_trl_availability()
    trainer, train_rows = trainer_for(method, cfg, model, ref, tok, train, test, a.out)
    built = list(trainer.train_dataset)
    bad = check_rows(method, built, train_rows, tok)
    if method == "kto":
        bad += check_kl_rows(built, train_rows, tok, trainer.args.per_device_train_batch_size)
    print(json.dumps({"method": method, "trl_availability_fixed": fixed,
                      "train_rows": len(train_rows), "frozen_params": len(frozen),
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
