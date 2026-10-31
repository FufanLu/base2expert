"""Writes the validation rows that verl's RL trainer scores, to evaluate exported models on the same rows.

verl 0.9.1 (RLHFDataset._read_files_and_tokenize) concatenates data.val_files in the order given and,
when val_max_samples is below the total and data.shuffle is on (ours is), keeps the rows
np.random.default_rng(data.seed).choice(total, size=val_max_samples, replace=False); it then drops
prompts longer than max_prompt_length (none of ours: the trainer logs 279 FinQA + 233 TAT-QA = 512).

    python -P tools/rl_val_rows.py --n 512 --seed 42 --out $B2E_ROOT/data/processed/finance_pool/rl_val512.parquet \\
        $B2E_ROOT/data/processed/finance/finqa_dev.parquet $B2E_ROOT/data/processed/finance/tatqa_dev.parquet

The files must be listed in rl_val.txt's order. It reads them with `datasets`, as verl does.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np


def val_indices(total: int, n: int, seed: int | None) -> list[int]:
    """Row indices verl keeps out of `total` for val_max_samples `n` (shuffle on)."""
    if n <= 0 or n >= total:
        return list(range(total))
    rng = np.random.default_rng(*((seed,) if seed is not None else ()))
    return rng.choice(total, size=n, replace=False).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="data.val_files, in order")
    ap.add_argument("--n", type=int, required=True, help="data.val_max_samples")
    ap.add_argument("--seed", type=int, default=42, help="data.seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import datasets

    ds = datasets.concatenate_datasets([datasets.load_dataset("parquet", data_files=f)["train"] for f in args.files])
    sub = ds.select(val_indices(len(ds), args.n, args.seed))
    sub.to_parquet(args.out)
    print(f"{len(ds)} rows -> {len(sub)}:", dict(Counter(sub["data_source"])))


if __name__ == "__main__":
    main()
