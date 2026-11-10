"""Bits per byte of a nanochat base checkpoint on held-out text, per source.

The protocol of b2e.eval.bpb (the Qwen side), so the two compare directly:
each document is scored on its own, cut to --max-tokens tokens, and its first
token is not scored; bpb = summed nats / ln 2 / UTF-8 bytes of the scored
tokens (nanochat's token_bytes, special tokens count 0). The one difference is
nanochat's <|bos|> in front of every document, as in its training; it is
context only. Input: a directory of parquet files with a `text` column, one
per source (data/cpt/m0/heldout/<source>.parquet).

    NANOCHAT_BASE_DIR=$ROOT/pretrain/bl2_d5 python -P pretrain/eval_bpb.py --model-tag d12 \\
        --data $ROOT/data/cpt/m0/heldout --out $ROOT/results/nanochat/bl2_d5_d12_heldout.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def doc_rows(ids: list[list[int]], bos: int, max_tokens: int) -> list[tuple[list[int], list[int]]]:
    """One (inputs, targets) row per document, cut to its first max_tokens
    tokens: <|bos|> + all but the last in, the tokens themselves as targets,
    the first one's -1 (not scored), so max_tokens - 1 are scored, as on the
    Qwen side. Documents under 2 tokens have nothing to score and are dropped."""
    rows = []
    for x in ids:
        x = x[:max_tokens]
        if len(x) < 2:
            continue
        inp = [bos] + x[:-1]
        tgt = [-1] + x[1:]
        rows.append((inp, tgt))
    return rows


def batches(rows, batch_size: int, device):
    import torch

    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        T = max(len(r[0]) for r in chunk)
        x = torch.zeros((len(chunk), T), dtype=torch.long)
        y = torch.full((len(chunk), T), -1, dtype=torch.long)
        for j, (inp, tgt) in enumerate(chunk):
            x[j, : len(inp)] = torch.tensor(inp)
            y[j, : len(tgt)] = torch.tensor(tgt)
        yield x.to(device), y.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048, help="document tokens kept (= input length)")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import torch
    from nanochat.checkpoint_manager import load_model
    from nanochat.common import compute_init
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.tokenizer import get_token_bytes

    *_, device = compute_init("cuda")
    model, tok, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    token_bytes = get_token_bytes(device=device)
    bos = tok.get_bos_token_id()
    result: dict = {"model_tag": args.model_tag, "step": meta.get("step"), "max_tokens": args.max_tokens,
                    "sources": {}}
    for f in sorted(Path(args.data).glob("*.parquet")):
        texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        rows = doc_rows(tok.encode(texts), bos, args.max_tokens)
        t0 = time.time()
        steps = (len(rows) + args.batch_size - 1) // args.batch_size
        bpb = evaluate_bpb(model, batches(rows, args.batch_size, device), steps, token_bytes)
        n_tok = sum(len(r[1]) - 1 for r in rows)
        result["sources"][f.stem] = {"docs": len(rows), "tokens": n_tok, "bpb": round(bpb, 4),
                                     "seconds": round(time.time() - t0, 1)}
        print(f.stem, result["sources"][f.stem], flush=True)
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
