"""Batch evaluation with vLLM, scored by the same reward code as training.

Reads verl-format parquet files (prompt messages, data_source,
reward_model.ground_truth), generates with vLLM, scores every answer with
b2e.reward.score.compute_score, and writes per-sample records plus a summary.
--repeats runs the same generation several times (seeds 0..k-1) to measure
the run-to-run spread, which on 128 GSM8K questions was already 8 points
between two greedy runs of the same model (docs/runs.md, 11343461/62).

    python -P -m b2e.eval.run_eval --model /tmp/b2e-models/Qwen3.5-4B-Base-chat \\
        --data finqa_test.parquet tatqa_dev.parquet --out results/base --thinking false --repeats 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd

from b2e.prompts import fits
from b2e.reward.score import compute_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", default="false")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True)
    summary = {"model": args.model, "thinking": args.thinking, "temperature": args.temperature, "sets": {}}
    for path in args.data:
        df = pd.read_parquet(path)
        if args.limit:
            df = df.head(args.limit)
        name = Path(path).stem
        tok = llm.get_tokenizer()
        ok = [fits(tok, list(p), args.thinking == "true", args.max_tokens, args.max_model_len) for p in df["prompt"]]
        skipped = len(ok) - sum(ok)
        df = df[ok]
        msgs = [list(p) for p in df["prompt"]]
        accs, runs = [], []
        for r in range(args.repeats):
            sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=r)
            t0 = time.time()
            res = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": args.thinking == "true"})
            dt = time.time() - t0
            recs = []
            for row, o in zip(df.itertuples(index=False), res, strict=True):
                text = o.outputs[0].text
                score = compute_score(row.data_source, text, row.reward_model["ground_truth"],
                                      thinking=args.thinking == "true")
                recs.append({"data_source": row.data_source, "id": row.extra_info.get("id"),
                             "score": score, "boxed": "\\boxed{" in text, "tokens": len(o.outputs[0].token_ids),
                             "finish": o.outputs[0].finish_reason, "response": text})
            with (out / f"{name}.run{r}.jsonl").open("w") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            acc = sum(x["score"] for x in recs) / len(recs)
            accs.append(acc)
            # strict: full marks only (all constraints of an instruction-following prompt, IFEval's
            # prompt-level number; the same as accuracy for 0/1 verifiers)
            strict = sum(x["score"] >= 1.0 - 1e-9 for x in recs) / len(recs)
            runs.append({"accuracy": round(acc, 4), "strict_accuracy": round(strict, 4),
                         "boxed_rate": round(sum(x["boxed"] for x in recs) / len(recs), 4),
                         "mean_tokens": round(statistics.mean(x["tokens"] for x in recs), 1),
                         # answers that open a <think> block themselves: with thinking off, the template
                         # already closed an empty one, so this is the mode switch being ignored (M0 did
                         # it on every finance answer after mid-training on thinking-only chats)
                         "reopened_think": round(sum("<think>" in x["response"] for x in recs) / len(recs), 4),
                         "truncated": sum(x["finish"] == "length" for x in recs), "seconds": round(dt, 1)})
            print(name, f"run {r}", runs[-1], flush=True)
        summary["sets"][name] = {"n": len(df), "skipped_too_long": skipped, "runs": runs,
                                 "accuracy_mean": round(statistics.mean(accs), 4),
                                 "accuracy_spread": round(max(accs) - min(accs), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
