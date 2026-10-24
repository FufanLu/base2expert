"""Build the M0 mid-training mix from a recipe: one pre-tokenized file per source.

The recipe (configs/data/m0.yaml) lists each source with its share of the
total token budget, its files, and its kind:
  text  a `text` column (cleaned corpora, general replay)
  chat  a `conversations` column of {from, value} turns (OpenThoughts3),
        rendered with the chat template so long chain-of-thought arrives in
        the same <think> format the model will be fine-tuned on
Every source is cut at its budget (chunks of at most max_len tokens, ending
documents with the end-of-text token) and written to <out>/<name>.parquet,
with a manifest of the tokens each source actually contributed.
The first --heldout-docs documents of each text source are not trained on:
they go to <out>/heldout/<name>.parquet as plain text, where b2e.eval.bpb
measures bits per byte before and after mid-training (domain text should
get cheaper, general text should not get dearer).

    python -P -m b2e.data.m0_mix --recipe configs/data/m0.yaml --total-tokens 5e8 \\
        --data-root $B2E_ROOT/data --tokenizer <model dir> --out $B2E_ROOT/data/cpt/m0
"""

from __future__ import annotations

import argparse
import ast
import glob
import itertools
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

from b2e.data.pretokenize import tokenize_docs

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


def read_column(files: list[str], column: str) -> Iterator:
    import pyarrow.parquet as pq

    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            yield from pf.read_row_group(i, columns=[column]).column(column).to_pylist()


def split_heldout(docs: Iterable[str], n: int) -> tuple[list[str], Iterator[str]]:
    """The first n documents, and an iterator over the rest."""
    it = iter(docs)
    return list(itertools.islice(it, n)), it


def render_chat(conv, tok) -> str:
    turns = ast.literal_eval(conv) if isinstance(conv, str) else conv
    msgs = [{"role": ROLE[t["from"]], "content": t["value"]} for t in turns if t.get("from") in ROLE]
    return tok.apply_chat_template(msgs, tokenize=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--total-tokens", type=float, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--data-root", default="", help="prefix for relative file patterns in the recipe")
    ap.add_argument("--decontam-index", default="",
                    help="ItemNgramIndex prefix; documents covering >= 60%% of an eval item are skipped. "
                         "Needed for sources that bypass clean_pipeline (the chat-format long-CoT data)")
    ap.add_argument("--heldout-docs", type=int, default=500, help="per text source, kept out of training for bpb")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    shares = sum(s["share"] for s in recipe["sources"])
    assert abs(shares - 1.0) < 1e-6, f"shares sum to {shares}"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    index = None
    if args.decontam_index:
        from b2e.data.decontam import ItemNgramIndex

        index = ItemNgramIndex(args.decontam_index)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"total_tokens_target": args.total_tokens, "sources": {}}
    for src in recipe["sources"]:
        patterns = [p if Path(p).is_absolute() else str(Path(args.data_root) / p) for p in src["files"]]
        files = sorted(f for pattern in patterns for f in glob.glob(pattern))
        if not files:
            raise SystemExit(f"{src['name']}: no files match {src['files']}")
        budget = src["share"] * args.total_tokens
        if src["kind"] == "chat":
            texts: Iterator[str] = (render_chat(c, tok) for c in read_column(files, "conversations"))
        else:
            texts = (t for t in read_column(files, "text") if t)
            if args.heldout_docs:
                held, texts = split_heldout(texts, args.heldout_docs)
                (out / "heldout").mkdir(exist_ok=True)
                pq.write_table(pa.table({"text": held}), out / "heldout" / f"{src['name']}.parquet")
        skipped = {"n": 0}
        if index is not None and src.get("decontam", src["kind"] == "chat"):
            def keep(it: Iterator[str], skipped=skipped) -> Iterator[str]:
                for t in it:
                    if index.contaminated(t):
                        skipped["n"] += 1
                        continue
                    yield t
            texts = keep(texts)
        chunks, n_tok, docs = [], 0, 0
        for d, piece in tokenize_docs(texts, tok, eos, args.max_len, src.get("min_len", 64)):
            chunks.append(piece)
            n_tok += len(piece)
            docs = d + 1
            if n_tok >= budget:
                break
        pq.write_table(pa.table({"input_ids": pa.array(chunks, type=pa.list_(pa.int32()))}),
                       out / f"{src['name']}.parquet", row_group_size=1024)
        manifest["sources"][src["name"]] = {"share": src["share"], "budget": int(budget), "tokens": n_tok,
                                            "chunks": len(chunks), "docs": docs, "files": files,
                                            "exhausted": n_tok < budget, "decontam_skipped": skipped["n"],
                                            "heldout_docs": args.heldout_docs if src["kind"] != "chat" else 0}
        print(src["name"], manifest["sources"][src["name"]]["tokens"], "tokens", "(short of budget)" if n_tok < budget else "", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
