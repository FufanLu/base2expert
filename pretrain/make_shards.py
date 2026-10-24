"""Bilingual pre-training shards in nanochat's format.

nanochat reads every `*.parquet` in its data dir in sorted order, one row
group at a time, from a `text` column, and uses the last file as validation
(nanochat/dataset.py). This writes such a directory from several sources
mixed by UTF-8 bytes (bits per byte is the metric, so bytes are the budget):

    python -P pretrain/make_shards.py --out DIR --shard-bytes 250e6 --val-bytes 50e6 \\
        --source en=0.65:/path/climbmix/*.parquet \\
        --source zh=0.30:/path/fineweb-edu-zh/*.parquet \\
        --source domain=0.05:/path/industrycorpus2/*/high/*.parquet

Documents are interleaved by a seeded weighted draw, so every shard has the
same mix and the validation shard matches training. A manifest.json records
sources, weights, documents and bytes per source.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP = 1024


def docs(files: list[str], column: str = "text") -> Iterator[str]:
    for f in files:
        pf = pq.ParquetFile(f)
        col = column if column in pf.schema_arrow.names else pf.schema_arrow.names[0]
        for i in range(pf.num_row_groups):
            for t in pf.read_row_group(i, columns=[col]).column(col).to_pylist():
                if t:
                    yield t


def parse_source(s: str) -> tuple[str, float, list[str]]:
    name, rest = s.split("=", 1)
    weight, pattern = rest.split(":", 1)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    return name, float(weight), files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True, help="name=weight:glob")
    ap.add_argument("--shard-bytes", type=float, default=250e6)
    ap.add_argument("--val-bytes", type=float, default=50e6)
    ap.add_argument("--total-bytes", type=float, default=0, help="stop after this many training bytes (0: until a source runs out)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sources = [parse_source(s) for s in args.source]
    names = [n for n, _, _ in sources]
    weights = [w for _, w, _ in sources]
    iters = {n: docs(f) for n, _, f in sources}
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {n: {"docs": 0, "bytes": 0} for n in names}
    # Byte-weighted draw: pick the source whose share is furthest below target.
    shards: list[dict] = []

    def fill(limit: float) -> tuple[list[str], dict]:
        buf, local = [], {n: 0 for n in names}
        total = 0
        while total < limit:
            deficits = [w * (total + 1) - local[n] for n, w in zip(names, weights, strict=True)]
            n = names[max(range(len(names)), key=lambda i: (deficits[i], rng.random()))]
            try:
                t = next(iters[n])
            except StopIteration:
                return buf, {"exhausted": n, **local}
            b = len(t.encode())
            buf.append(t)
            local[n] += b
            total += b
            stats[n]["docs"] += 1
            stats[n]["bytes"] += b
        return buf, local

    def write(name: str, texts: list[str]) -> None:
        rng.shuffle(texts)
        table = pa.table({"text": texts})
        pq.write_table(table, out / name, row_group_size=ROW_GROUP)

    # Validation first, so it never shares documents with training; it is
    # written last in sort order (nanochat takes the last file as val).
    val, _ = fill(args.val_bytes)
    i, trained = 0, 0.0
    while not args.total_bytes or trained < args.total_bytes:
        texts, local = fill(args.shard_bytes)
        if not texts:
            break
        write(f"shard_{i:05d}.parquet", texts)
        b = sum(v for k, v in local.items() if k != "exhausted")
        shards.append({"file": f"shard_{i:05d}.parquet", "docs": len(texts), "bytes": b})
        trained += b
        i += 1
        if "exhausted" in local:
            break
    write("shard_99999.parquet", val)
    (out / "manifest.json").write_text(json.dumps({
        "sources": {n: {"weight": w, "files": f} for n, w, f in sources},
        "seed": args.seed, "train_shards": shards, "val_file": "shard_99999.parquet",
        "per_source": stats,
    }, indent=2, ensure_ascii=False))
    print(json.dumps({"train_shards": len(shards), "train_bytes": trained, "per_source": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
