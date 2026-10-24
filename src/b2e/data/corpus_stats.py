"""Size of corpus shards in documents, characters and tokens.

Reads parquet shards, finds the text column, and counts characters for every
row and Qwen3.5 tokens on a random sample of rows (tokenizing whole shards is
slow and not needed for budgeting). Prints one JSON line per shard and the
schema of the first one, so the pipeline knows the column names.

    python -P -m b2e.data.corpus_stats --tokenizer <model dir> shard.parquet [...]
"""

from __future__ import annotations

import argparse
import json
import random

import pyarrow.parquet as pq

TEXT_COLUMNS = ("text", "content", "raw_content", "document")


def text_column(names: list[str]) -> str:
    for c in TEXT_COLUMNS:
        if c in names:
            return c
    raise SystemExit(f"no text column among {names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(0)
    for i, path in enumerate(args.files):
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if i == 0:
            print(json.dumps({"schema": str(pf.schema_arrow)}, ensure_ascii=False))
        col = text_column(names)
        texts = pf.read(columns=[col]).column(col).to_pylist()
        chars = sum(len(t or "") for t in texts)
        sample = rng.sample(texts, min(args.sample, len(texts)))
        s_chars = sum(len(t or "") for t in sample)
        s_tokens = sum(len(tok(t or "", add_special_tokens=False)["input_ids"]) for t in sample)
        chars_per_token = s_chars / max(s_tokens, 1)
        print(json.dumps({
            "file": path,
            "docs": len(texts),
            "chars": chars,
            "chars_per_token": round(chars_per_token, 3),
            "est_tokens": int(chars / chars_per_token),
            "median_doc_chars": sorted(len(t or "") for t in sample)[len(sample) // 2],
            "columns": names,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
