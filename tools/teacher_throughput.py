"""Generation throughput of the teacher model in vLLM (offline batch mode).

P1 synthesizes SFT data with Qwen3.6-35B-A3B; this measures how many output
tokens per second one H200 gives, with thinking on, on GSM8K questions as
stand-in prompts. The number sets how many jobs the synthesis needs.

    python -P tools/teacher_throughput.py --model /tmp/b2e-models/Qwen3.6-35B-A3B \\
        --prompts /scratch/$USER/b2e/data/processed/gsm8k/train.parquet --n 512
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", default="true")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.prompts).head(args.n)
    msgs = [list(p) for p in df["prompt"]]
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": args.thinking == "true"})
    dt = time.time() - t0
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tok = sum(len(o.prompt_token_ids) for o in outs)
    truncated = sum(o.outputs[0].finish_reason == "length" for o in outs)
    print(json.dumps({
        "model": args.model, "prompts": len(outs), "thinking": args.thinking,
        "seconds": round(dt, 1), "output_tokens": out_tok, "input_tokens": in_tok,
        "output_tokens_per_s": round(out_tok / dt, 1),
        "mean_output_tokens": round(out_tok / len(outs), 1),
        "truncated_at_max_tokens": truncated,
    }))
    print("sample:", outs[0].outputs[0].text[:1500])


if __name__ == "__main__":
    main()
