"""RL candidates: the train prompts a domain's SFT did not use.

Each domain's SFT data came from b2e.data.synth sampling part of the train prompts; its
*_passrate.parquet lists every prompt it sampled, kept for SFT or not. RL should train on
prompts the model was not fine-tuned on, so the P7 pools are drawn from the rest, sampled by
the starting policy (b2e.data.synth, the RL rollout settings) and cut by b2e.data.rl_pool.
F#58: the finance pool was too small for 150 steps, so these start from many more prompts.

    python -P -m b2e.data.rl_candidates --exclude $B2E_ROOT/data/sft/law_passrate.parquet \\
        --out $B2E_ROOT/data/processed/law_pool --train $B2E_ROOT/data/processed/law_med/cail_train.parquet

Prompts are matched by (data_source, id): ids repeat across sources (MedQA and CMExam both
number theirs from 0). Both file lists are named options (F#48).

--val-per-source N sets N of each source's unused prompts aside as the RL val set
(<stem>_val.parquet) and leaves them out of <stem>_unused.parquet, so the pool sampled from
the rest never holds a val prompt. The val prompts are not filtered by pass rate, as
P6's dev sets were not, and no eval uses a train split (P6's val shared TAT-QA dev with
the eval, F#66).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def unused(pool: pd.DataFrame, used: pd.DataFrame) -> pd.DataFrame:
    """Rows of pool whose (data_source, id) is not in used (columns data_source, id)."""
    keys = set(zip(used["data_source"], used["id"], strict=True))
    mask = [(ds, e["id"]) not in keys for ds, e in zip(pool["data_source"], pool["extra_info"], strict=True)]
    return pool[mask]


def split_val(left: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rest, val): per data_source, a seeded sample of n rows (all of them if fewer) is the val set."""
    picked = [i for _, g in left.groupby("data_source") for i in g.sample(n=min(n, len(g)), random_state=seed).index]
    return left.drop(index=picked), left.loc[sorted(picked)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="+", required=True,
                    help="b2e.data.synth *_passrate.parquet files of the SFT sampling (columns data_source, id)")
    ap.add_argument("--train", nargs="+", required=True, help="processed train parquet files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-per-source", type=int, default=0,
                    help="unused prompts per source set aside as the RL val set (<stem>_val.parquet)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    used = pd.concat([pd.read_parquet(f, columns=["data_source", "id"]) for f in args.exclude], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"exclude": args.exclude, "val_per_source": args.val_per_source, "seed": args.seed, "files": {}}
    for f in args.train:
        pool = pd.read_parquet(f)
        left = unused(pool, used)
        row: dict = {"rows": len(pool)}
        if args.val_per_source:
            left, val = split_val(left, args.val_per_source, args.seed)
            val.to_parquet(args.out / (Path(f).stem + "_val.parquet"), index=False)
            row["val_by_source"] = {k: int(v) for k, v in val["data_source"].value_counts().items()}
        left.to_parquet(args.out / (Path(f).stem + "_unused.parquet"), index=False)
        summary["files"][Path(f).stem] = {
            **row, "unused": len(left),
            "unused_by_source": {k: int(v) for k, v in left["data_source"].value_counts().items()}}
    (args.out / "candidates.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
