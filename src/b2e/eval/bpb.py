"""Bits per byte on held-out text, from vLLM's prompt log-probabilities.

    bpb = sum over scored tokens of -log2 p(token | earlier tokens)
          / UTF-8 bytes of the text those tokens cover

Measured per byte rather than per token, so it does not depend on the
tokenizer: a tokenizer that cuts text into more, easier pieces gets no
credit, and models with different vocabularies (the nanochat side track vs
Qwen3.5-0.8B-Base) compare directly. Each document is cut to --max-tokens
tokens; the first token has no prediction, so its bytes are not counted.
Input: a directory of parquet files with a `text` column, one per source
(b2e.data.m0_mix writes <out>/heldout/<source>.parquet).

    python -P -m b2e.eval.bpb --model /tmp/b2e-models/Qwen3.5-4B-Base-chat \\
        --data $B2E_ROOT/data/cpt/m0/heldout --out results/m0/bpb_base.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def scored_bytes(text: str, offsets: list[tuple[int, int]]) -> int:
    """UTF-8 bytes from the second token's start to the last token's end."""
    return len(text[offsets[1][0]: offsets[-1][1]].encode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="documents per source (0 = all)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    # prompt log-probs are recomputed for every request, so prefix caching buys nothing here
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_tokens + 1,
              enable_prefix_caching=False)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    result: dict = {"model": args.model, "max_tokens": args.max_tokens, "sources": {}}
    for f in sorted(Path(args.data).glob("*.parquet")):
        texts = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        if args.limit:
            texts = texts[: args.limit]
        ids, nbytes = [], []
        for t in texts:
            enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
            x, off = enc["input_ids"][: args.max_tokens], enc["offset_mapping"][: args.max_tokens]
            if len(x) < 2:
                continue
            ids.append(x)
            nbytes.append(scored_bytes(t, off))
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
        nll, n_tok = 0.0, 0
        for o, x in zip(outs, ids, strict=True):
            for pos, lp in enumerate(o.prompt_logprobs[1:], start=1):
                nll -= lp[x[pos]].logprob
                n_tok += 1
        bits = nll / math.log(2)
        result["sources"][f.stem] = {"docs": len(ids), "tokens": n_tok, "bytes": sum(nbytes),
                                     "bpb": round(bits / sum(nbytes), 4), "nats_per_token": round(nll / n_tok, 4),
                                     "seconds": round(time.time() - t0, 1)}
        print(f.stem, result["sources"][f.stem], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
