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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="+", required=True,
                    help="b2e.data.synth *_passrate.parquet files of the SFT sampling (columns data_source, id)")
    ap.add_argument("--train", nargs="+", required=True, help="processed train parquet files")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    used = pd.concat([pd.read_parquet(f, columns=["data_source", "id"]) for f in args.exclude], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"exclude": args.exclude, "files": {}}
    for f in args.train:
        pool = pd.read_parquet(f)
        left = unused(pool, used)
        left.to_parquet(args.out / (Path(f).stem + "_unused.parquet"), index=False)
        summary["files"][Path(f).stem] = {
            "rows": len(pool), "unused": len(left),
            "unused_by_source": {k: int(v) for k, v in left["data_source"].value_counts().items()}}
    (args.out / "candidates.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
