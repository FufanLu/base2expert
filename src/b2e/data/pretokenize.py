"""Tokenize text corpora for mid-training (continued pre-training).

Each document is tokenized, ends with the end-of-text token so the model
learns where documents stop, and is cut into chunks of at most --max-len
tokens (no overlap). Chunks shorter than --min-len are dropped. The output is
parquet with `input_ids` (list<int32>), `source` and `doc_index`, read by
b2e.train.cpt_dataset during training.

    python -P -m b2e.data.pretokenize --tokenizer <model dir> --max-len 8192 \\
        --source fin_zh --out fin_zh.parquet input.parquet [...]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator


def chunk(ids: list[int], max_len: int, min_len: int) -> Iterator[list[int]]:
    for start in range(0, len(ids), max_len):
        piece = ids[start:start + max_len]
        if len(piece) >= min_len:
            yield piece


def tokenize_docs(texts: Iterable[str], tok, eos_id: int, max_len: int, min_len: int,
                  batch: int = 256) -> Iterator[tuple[int, list[int]]]:
    """Yields (document index, chunk) pairs."""
    buf: list[str] = []
    base = 0

    def flush() -> Iterator[tuple[int, list[int]]]:
        enc = tok(buf, add_special_tokens=False)["input_ids"]
        for j, ids in enumerate(enc):
            for piece in chunk(ids + [eos_id], max_len, min_len):
                yield base + j, piece

    for t in texts:
        buf.append(t)
        if len(buf) == batch:
            yield from flush()
            base += len(buf)
            buf = []
    if buf:
        yield from flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--min-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=float, default=0, help="stop after this many tokens (0: all)")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.convert_tokens_to_ids("<|endoftext|>")

    def texts() -> Iterator[str]:
        for f in args.files:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                for t in pf.read_row_group(i, columns=[args.text_column]).column(args.text_column).to_pylist():
                    if t:
                        yield t

    rows: dict[str, list] = {"input_ids": [], "source": [], "doc_index": []}
    n_tok = 0
    for d, piece in tokenize_docs(texts(), tok, eos_id, args.max_len, args.min_len):
        rows["input_ids"].append(piece)
        rows["source"].append(args.source)
        rows["doc_index"].append(d)
        n_tok += len(piece)
        if args.max_tokens and n_tok >= args.max_tokens:
            break
    table = pa.table({"input_ids": pa.array(rows["input_ids"], type=pa.list_(pa.int32())),
                      "source": rows["source"], "doc_index": rows["doc_index"]})
    pq.write_table(table, args.out, row_group_size=1024)
    print(json.dumps({"source": args.source, "chunks": len(rows["input_ids"]), "tokens": n_tok,
                      "docs": (rows["doc_index"][-1] + 1) if rows["doc_index"] else 0, "out": args.out}))


if __name__ == "__main__":
    main()
