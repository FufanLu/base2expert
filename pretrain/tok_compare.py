"""Compression of several tokenizers on the same texts: UTF-8 bytes per token.

More bytes per token means shorter sequences for the same text, so more text
fits in a context and a training token budget covers more data. Compared on
held-out text per source:
  nanochat:DIR   a nanochat tokenizer.pkl (pickled tiktoken Encoding)
  tiktoken:NAME  an OpenAI encoding (gpt2, cl100k_base)
  hf:DIR         a Hugging Face tokenizer (Qwen3.5)
Texts come from the end of each parquet file (--docs per source): the
bilingual tokenizer was trained on shards that took documents from the start
of the same domain files (pretrain/make_shards.py), and the end was never read.

    python -P pretrain/tok_compare.py --docs 200 \\
        --tok bilingual=nanochat:$ROOT/pretrain/base_bilingual/tokenizer --tok gpt4=tiktoken:cl100k_base \\
        --text fin_zh=$DATA/clean/fin_zh/000_00000.parquet --out tok_compare.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Callable
from pathlib import Path


def load(spec: str) -> tuple[Callable[[str], int], int]:
    """(text -> number of tokens, vocabulary size)."""
    kind, _, arg = spec.partition(":")
    if kind == "nanochat":
        with open(Path(arg) / "tokenizer.pkl", "rb") as f:
            enc = pickle.load(f)
        return (lambda t: len(enc.encode_ordinary(t))), enc.n_vocab
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(arg)
        return (lambda t: len(enc.encode_ordinary(t))), enc.n_vocab
    if kind == "hf":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(arg)
        return (lambda t: len(tok(t, add_special_tokens=False)["input_ids"])), len(tok)
    raise ValueError(f"unknown tokenizer kind in {spec!r}")


def tail_docs(path: str, n: int) -> list[str]:
    import pyarrow.parquet as pq

    col = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return [t for t in col if t][-n:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", action="append", required=True, help="name=kind:arg")
    ap.add_argument("--text", action="append", required=True, help="name=parquet file")
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    toks = {name: load(spec) for name, _, spec in (t.partition("=") for t in args.tok)}
    texts = {name: tail_docs(path, args.docs) for name, _, path in (t.partition("=") for t in args.text)}
    table: dict = {"vocab": {k: v[1] for k, v in toks.items()}, "bytes_per_token": {}}
    print(f"{'text':<12}" + "".join(f"{k:>12}" for k in toks))
    for tname, docs in texts.items():
        nbytes = sum(len(d.encode()) for d in docs)
        row = {k: round(nbytes / sum(count(d) for d in docs), 3) for k, (count, _) in toks.items()}
        table["bytes_per_token"][tname] = row
        print(f"{tname:<12}" + "".join(f"{row[k]:>12}" for k in toks), flush=True)
    Path(args.out).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
