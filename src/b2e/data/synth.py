"""SFT data by rejection sampling from the teacher (Qwen3.6-35B-A3B).

For each training prompt (verl RL format) the teacher samples k answers in
thinking mode. Each answer is scored by the same verifier the RL reward
uses (b2e.reward.score), and only verified answers are kept, the shortest
one per prompt (long-winded reasoning is not what the student should copy).
Every kept example is written twice:

  thinking:     assistant = "<think>\\n{reasoning}\\n</think>\\n\\n{answer}", enable_thinking=True
  non-thinking: assistant = "{answer}", enable_thinking=False

so the student learns both modes from the same problems. The per-prompt pass
rate (correct / k) is saved as well: it measures difficulty for the teacher
and is used later to filter the RL prompt pool.

    python -P -m b2e.data.synth --teacher /tmp/b2e-models/Qwen3.6-35B-A3B --k 4 \\
        --per-source 2000 --out sft_fin.parquet finqa_train.parquet tatqa_train.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from b2e.reward.score import compute_score


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, answer). The teacher's text starts inside the <think> block
    that the chat template opened, so the reasoning is everything before </think>."""
    if "</think>" not in text:
        return "", text.strip()
    reasoning, answer = text.split("</think>", 1)
    return reasoning.replace("<think>", "").strip(), answer.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--per-source", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--min-score", type=float, default=1.0, help="graded rewards (CAIL term) count from here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    frames = []
    for path in args.inputs:
        df = pd.read_parquet(path)
        for _, g in df.groupby("data_source"):
            frames.append(g.sample(n=min(args.per_source, len(g)), random_state=args.seed))
    df = pd.concat(frames, ignore_index=True)
    msgs = [list(p) for p in df["prompt"]]
    llm = LLM(model=args.teacher, gpu_memory_utilization=0.9, max_model_len=16384, enable_prefix_caching=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens,
                        seed=args.seed)
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})
    dt = time.time() - t0

    sft, stats = [], []
    gen_tokens = 0
    for row, o in zip(df.itertuples(index=False), outs, strict=True):
        gt = row.reward_model["ground_truth"]
        scored = []
        for c in o.outputs:
            gen_tokens += len(c.token_ids)
            s = compute_score(row.data_source, c.text, gt)
            scored.append((s, len(c.token_ids), c.text, c.finish_reason))
        good = [x for x in scored if x[0] >= args.min_score and x[3] == "stop"]
        stats.append({"data_source": row.data_source, "id": row.extra_info["id"],
                      "pass_rate": sum(x[0] >= args.min_score for x in scored) / len(scored),
                      "mean_score": sum(x[0] for x in scored) / len(scored)})
        if not good:
            continue
        _, _, text, _ = min(good, key=lambda x: x[1])
        reasoning, answer = split_thinking(text)
        user = list(row.prompt)
        base = {"data_source": row.data_source, "id": row.extra_info["id"], "ground_truth": gt}
        sft.append({**base, "enable_thinking": True,
                    "messages": user + [{"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"}]})
        sft.append({**base, "enable_thinking": False,
                    "messages": user + [{"role": "assistant", "content": answer}]})
    out = Path(args.out)
    pd.DataFrame(sft).to_parquet(out, index=False)
    st = pd.DataFrame(stats)
    st.to_parquet(out.with_name(out.stem + "_passrate.parquet"), index=False)
    summary = {
        "prompts": len(df), "k": args.k, "kept_prompts": len(sft) // 2, "seconds": round(dt, 1),
        "generated_tokens": gen_tokens, "tokens_per_s": round(gen_tokens / dt, 1),
        "pass_rate_by_source": st.groupby("data_source")["pass_rate"].mean().round(3).to_dict(),
        "kept_by_source": pd.Series([r["data_source"] for r in sft[::2]]).value_counts().to_dict(),
    }
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
