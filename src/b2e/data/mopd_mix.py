"""The prompt mix for multi-teacher on-policy distillation (P8): rows from each domain's RL pool with a
`teacher` column naming the domain teacher verl routes them to (distillation.teacher_key=teacher).
verl routes on one field, and a domain has several data sources (FinQA and TAT-QA both go to R-fin),
so the route is its own column and data_source stays as it is for the per-source val metrics.
Each domain is drawn to the same number of rows (seeded) unless it has fewer.

    python -P -m b2e.data.mopd_mix --out $B2E_ROOT/data/processed/mopd_smoke --n-train 64 --n-val 32 \\
        --train fin=finqa_train_pool.parquet,tatqa_train_pool.parquet law=cail_pool.parquet med=med_pool.parquet \\
        --val fin=rl_val512.parquet law=cail_train_val.parquet med=med_val.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_domains(items: list[str]) -> dict[str, list[str]]:
    out = {}
    for it in items:
        name, _, files = it.partition("=")
        if not name or not files:
            raise SystemExit(f"expected DOMAIN=FILE[,FILE...], got {it!r}")
        out[name] = files.split(",")
    return out


def draw(frames, n: int, seed: int, teacher: str):
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    if 0 < n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.assign(teacher=teacher)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, metavar="DOMAIN=FILES")
    ap.add_argument("--val", nargs="+", required=True, metavar="DOMAIN=FILES")
    ap.add_argument("--n-train", type=int, default=0, help="rows per domain (0: all)")
    ap.add_argument("--n-val", type=int, default=0, help="rows per domain (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import pandas as pd

    train, val = parse_domains(args.train), parse_domains(args.val)
    if set(train) != set(val):
        raise SystemExit(f"train domains {sorted(train)} and val domains {sorted(val)} differ")
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"teacher_key": "teacher", "train": {}, "val": {}}
    for split, spec, n, name in (("train", train, args.n_train, "train"), ("val", val, args.n_val, "val")):
        parts = []
        for dom, files in spec.items():
            d = draw([pd.read_parquet(f) for f in files], n, args.seed, dom)
            parts.append(d)
            summary[split][dom] = {"rows": len(d), "sources": d["data_source"].value_counts().to_dict()}
        pd.concat(parts, ignore_index=True).to_parquet(args.out / f"{name}.parquet", index=False)
    (args.out / "rl_train.txt").write_text(str((args.out / "train.parquet").resolve()) + "\n")
    (args.out / "rl_val.txt").write_text(str((args.out / "val.parquet").resolve()) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
