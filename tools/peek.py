"""Print the columns and first rows of parquet / json / jsonl files.

    python -P tools/peek.py [--rows 2] file [file ...]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd


def load(path: str, rows: int) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path).head(rows)
    if path.endswith(".jsonl"):
        with open(path) as f:
            return pd.DataFrame([json.loads(next(f)) for _ in range(rows)])
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data[:rows] if isinstance(data, list) else [data])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for f in a.files:
        df = load(f, a.rows)
        print(f"== {f}\ncolumns: {list(df.columns)}")
        for _, r in df.iterrows():
            print(json.dumps({k: (str(v)[:300]) for k, v in r.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
