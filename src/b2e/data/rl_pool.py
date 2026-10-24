"""RL prompt pool: keep the prompts the starting policy solves sometimes, not always or never.

GRPO-style advantages compare the samples of one prompt with each other. A
prompt whose samples are all right or all wrong gives every sample the same
reward, so its advantages are zero and it only costs rollout time. DAPO drops
such groups online (dynamic sampling); in verl 0.9.1 that exists only in the
V1 trainer (docs/runs.md), so this is the offline version: sample the starting
policy k times per prompt beforehand (b2e.data.synth with --temperature 1.0
--top-p 1.0, the RL rollout settings) and keep prompts whose pass rate lies in
[--low, --high].

    python -P -m b2e.data.rl_pool --passrate s_fin_k8_passrate.parquet --low 0.1 --high 0.9 \\
        --out $B2E_ROOT/data/processed/finance_pool --val finqa_dev.parquet tatqa_dev.parquet \\
        finqa_train.parquet tatqa_train.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def select(pool: pd.DataFrame, rates: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    """Rows of pool whose (data_source, id) has a pass rate in [low, high]."""
    keep = rates[(rates["pass_rate"] >= low) & (rates["pass_rate"] <= high)]
    keys = set(zip(keep["data_source"], keep["id"], strict=True))
    mask = [(ds, e["id"]) in keys for ds, e in zip(pool["data_source"], pool["extra_info"], strict=True)]
    return pool[mask]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passrate", required=True, help="b2e.data.synth *_passrate.parquet of the starting policy")
    ap.add_argument("--low", type=float, default=0.1)
    ap.add_argument("--high", type=float, default=0.9)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", nargs="+", required=True, help="validation parquet files, listed as they are")
    ap.add_argument("train", nargs="+")
    args = ap.parse_args()

    rates = pd.read_parquet(args.passrate)
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"passrate": args.passrate, "low": args.low, "high": args.high, "sources": {}}
    listed = []
    for f in args.train:
        pool = pd.read_parquet(f)
        kept = select(pool, rates, args.low, args.high)
        dst = args.out / (Path(f).stem + "_pool.parquet")
        kept.to_parquet(dst, index=False)
        listed.append(str(dst))
        scored = rates[rates["data_source"].isin(pool["data_source"].unique())]
        summary["sources"][Path(f).stem] = {
            "rows": len(pool), "scored": len(scored), "kept": len(kept),
            "all_wrong": int((scored["pass_rate"] == 0).sum()), "all_right": int((scored["pass_rate"] == 1).sum())}
    (args.out / "rl_train.txt").write_text("\n".join(listed) + "\n")
    (args.out / "rl_val.txt").write_text("\n".join(str(Path(v).resolve()) for v in args.val) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
