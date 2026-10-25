"""Instruction-following prompts in verl's RL format, checked by b2e.reward.ifcheck.

  if_multi   allenai/IF_multi_constraints_upto5: 95,373 prompts with 1-5
             constraints each (IFEval's 25 types and IFBench-Train's 29); the
             SFT / RL pool. Its ground_truth column is a Python literal
             (single quotes, None), not JSON.
  ifeval     google/IFEval, 541 prompts: the held-out test.

Every row is validated before it is written: each instruction id must be in
the vendored registry and each checker must build from the row's kwargs, so
a reward never fails in the middle of an RL step. Training prompts whose
request (constraint sentences removed, see user_query) covers >= 60% of an
eval item (b2e.data.decontam.ItemNgramIndex, the same rule as the corpus
cleaning) are dropped.

The validation set is pinned by id (configs/data/if_multi_val_ids.txt, the
1,000 drawn on 10-24). It used to be the first 1,000 of a shuffle of the pool,
so dropping rows from the pool (F#33) reshuffled it, and prompts already
synthesized as training data could have moved into validation.

    python -P -m b2e.data.if_data --hf $B2E_ROOT/data/hf --index $B2E_ROOT/data/processed/eval_items \\
        --val-ids configs/data/if_multi_val_ids.txt --out $B2E_ROOT/data/processed/if
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def row(source: str, idx: str, prompt: str, constraints: list[dict], split: str, **extra) -> dict:
    gt = {"kind": "ifeval", "prompt": prompt, "constraints": constraints}
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "instruction_following",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
        "extra_info": {"split": split, "id": idx, "n_constraints": sum(len(c["instruction_id"]) for c in constraints),
                       **extra},
    }


def valid(constraints: list[dict], prompt: str = "", fixed_args: bool = True) -> str | None:
    """None if every checker builds, else the reason.

    fixed_args: also reject a constraint whose kwargs are missing or out of
    range, which build_description replaces with a random draw (626 if_multi
    rows ask for paragraph 8 of 7, 20 copy_span_idx rows have a span that does
    not fit). The model would be scored on a constraint other than the one its
    prompt states. Detected by building with two seeds and comparing the args."""
    import random

    from b2e.reward.ifcheck import build
    from b2e.reward.ifevalg import instructions_registry as reg

    for c in constraints:
        if len(c["instruction_id"]) != len(c["kwargs"]):
            return "kwargs_length"
        for iid, kw in zip(c["instruction_id"], c["kwargs"], strict=True):
            if iid not in reg.INSTRUCTION_DICT:
                return f"unknown:{iid}"
            try:
                drawn = []
                for seed in (1, 2):
                    random.seed(seed)
                    drawn.append(json.dumps(build(iid, kw, prompt).get_instruction_args(), sort_keys=True, default=str))
            except Exception as e:  # noqa: BLE001 - any failure makes the row unusable
                return f"build:{iid}:{type(e).__name__}"
            if fixed_args and drawn[0] != drawn[1]:
                return f"randomized:{iid}"
    return None


def user_query(prompt: str, constraint: str) -> str:
    """The prompt without its constraint sentences (the dataset's `constraint` column, tab-separated).

    Decontamination looks at this part only. IFEval items are short and consist
    largely of the same constraint templates ("First repeat the request word for
    word without change, ..."), so with the template sentences left in, one
    template covered >= 60% of an IFEval item and 1,903 prompts were dropped
    whose actual requests had nothing to do with IFEval's (11352234)."""
    for c in (constraint or "").split("\t"):
        if c.strip():
            prompt = prompt.replace(c.strip(), " ")
    return prompt


def unique_ids(keys) -> list[str]:
    """The dataset's keys, with a suffix on repeats: 4 keys ("hard_coded_1", ...) name several different
    prompts, and every later step (synthesis pairs, pass rates, --only-ids, the pinned validation set)
    identifies a prompt by its id. The first occurrence keeps its key, so existing ids do not change."""
    seen: Counter = Counter()
    out = []
    for k in keys:
        out.append(k if not seen[k] else f"{k}#{seen[k]}")
        seen[k] += 1
    return out


def if_multi(hf: Path, index, drops: Counter, max_chars: int) -> list[dict]:
    df = pd.read_parquet(hf / "allenai/IF_multi_constraints_upto5/data/train-00000-of-00001.parquet")
    df["key"] = unique_ids(df["key"])
    rows = []
    for r in df.itertuples(index=False):
        msgs = list(r.messages)
        if len(msgs) != 1 or msgs[0]["role"] != "user":
            drops["not_single_user_turn"] += 1
            continue
        constraints = ast.literal_eval(r.ground_truth)
        prompt = msgs[0]["content"]
        reason = valid(constraints, prompt)
        if reason:
            drops[reason] += 1
            continue
        if len(prompt) > max_chars:
            drops["too_long"] += 1
            continue
        if index is not None and index.contaminated(user_query(prompt, r.constraint)):
            drops["eval_overlap"] += 1
            continue
        rows.append(row("if_multi", r.key, prompt, constraints, "train", constraint_type=r.constraint_type))
    return rows


def ifeval(hf: Path, drops: Counter) -> list[dict]:
    d = hf / "google/IFEval"
    files = sorted(d.rglob("*.parquet")) or sorted(d.rglob("*.jsonl"))
    df = pd.concat([pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_json(f, lines=True) for f in files])
    rows = []
    for r in df.itertuples(index=False):
        kwargs = [dict(k) if k is not None else None for k in r.kwargs]
        constraints = [{"instruction_id": list(r.instruction_id_list), "kwargs": kwargs}]
        # the test set stays as published: its two letter_frequency items with a non-letter ("!", "#")
        # get a drawn letter, as in the official evaluation (deterministic here, see ifcheck)
        reason = valid(constraints, r.prompt, fixed_args=False)
        if reason:
            drops["ifeval_" + reason] += 1
            continue
        rows.append(row("ifeval", str(r.key), r.prompt, constraints, "test"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", default="", help="ItemNgramIndex prefix for decontamination")
    ap.add_argument("--val", type=int, default=1000, help="if_multi prompts held out as a validation set")
    ap.add_argument("--val-ids", default="", help="file of if_multi ids that form the validation set (overrides --val)")
    ap.add_argument("--max-prompt-chars", type=int, default=12000,
                    help="longer prompts are dropped: RL's prompt budget is 4,096 tokens (configs/rl/base.yaml), "
                         "and one 16 K-token prompt stopped an eval job (F#31)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    index = None
    if args.index:
        from b2e.data.decontam import ItemNgramIndex

        index = ItemNgramIndex(args.index)
    drops: Counter = Counter()
    pool = pd.DataFrame(if_multi(args.hf, index, drops, args.max_prompt_chars)).sample(frac=1.0, random_state=args.seed)
    test = pd.DataFrame(ifeval(args.hf, drops))
    for name, d in (("if_multi", pool), ("ifeval", test)):
        ids = d["extra_info"].apply(lambda e: e["id"])
        if ids.duplicated().any():
            raise SystemExit(f"{name}: duplicate ids {sorted(set(ids[ids.duplicated()]))[:5]}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.val_ids:
        pinned = set(Path(args.val_ids).read_text().split("\n")) - {""}
        is_val = pool["extra_info"].apply(lambda e: e["id"] in pinned)
        val, train = pool[is_val].copy(), pool[~is_val]
        drops["val_ids_not_in_pool"] = len(pinned) - len(val)  # dropped by the checks above
    else:
        val, train = pool.iloc[: args.val].copy(), pool.iloc[args.val:]
    val["extra_info"] = [{**e, "split": "val"} for e in val["extra_info"]]
    train.to_parquet(args.out / "if_multi_train.parquet", index=False)
    val.to_parquet(args.out / "if_multi_val.parquet", index=False)
    test.to_parquet(args.out / "ifeval_test.parquet", index=False)
    n = Counter(e["n_constraints"] for e in pool["extra_info"])
    print(json.dumps({"train": len(train), "val": len(val), "ifeval_test": len(test), "dropped": dict(drops),
                      "constraints_per_prompt": dict(sorted(n.items()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
