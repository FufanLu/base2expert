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

    python -P -m b2e.data.if_data --hf $B2E_ROOT/data/hf --index $B2E_ROOT/data/processed/eval_items \\
        --out $B2E_ROOT/data/processed/if
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


def valid(constraints: list[dict]) -> str | None:
    """None if every checker builds, else the reason."""
    from b2e.reward.ifevalg import instructions_registry as reg

    for c in constraints:
        if len(c["instruction_id"]) != len(c["kwargs"]):
            return "kwargs_length"
        for iid, kw in zip(c["instruction_id"], c["kwargs"], strict=True):
            if iid not in reg.INSTRUCTION_DICT:
                return f"unknown:{iid}"
            try:
                reg.INSTRUCTION_DICT[iid](iid).build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
            except Exception as e:  # noqa: BLE001 - any failure makes the row unusable
                return f"build:{iid}:{type(e).__name__}"
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


def if_multi(hf: Path, index, drops: Counter) -> list[dict]:
    df = pd.read_parquet(hf / "allenai/IF_multi_constraints_upto5/data/train-00000-of-00001.parquet")
    rows = []
    for r in df.itertuples(index=False):
        msgs = list(r.messages)
        if len(msgs) != 1 or msgs[0]["role"] != "user":
            drops["not_single_user_turn"] += 1
            continue
        constraints = ast.literal_eval(r.ground_truth)
        reason = valid(constraints)
        if reason:
            drops[reason] += 1
            continue
        prompt = msgs[0]["content"]
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
        reason = valid(constraints)
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    index = None
    if args.index:
        from b2e.data.decontam import ItemNgramIndex

        index = ItemNgramIndex(args.index)
    drops: Counter = Counter()
    pool = pd.DataFrame(if_multi(args.hf, index, drops)).sample(frac=1.0, random_state=args.seed)
    test = pd.DataFrame(ifeval(args.hf, drops))
    args.out.mkdir(parents=True, exist_ok=True)
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
