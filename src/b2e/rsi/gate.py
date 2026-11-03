"""Validity gate for self-written problems (P7.5, the 11-03 redesign).

The pass-rate filter cannot tell a hard problem from a flawed one. In the
pilots' hand audits the solver often agreed with a wrong gold, so a majority
vote kept the problem (docs/runs.md, 11-03; CoVerRL calls this the consensus
trap, and R-Quest finds invalid self-made questions growing over rounds).
Before that filter an external model, Qwen3.6-35B-A3B, reads each problem. It
either rejects the question as not answerable with one number, or answers it;
a problem stays only if it is answered and the answer matches the program's
under the finance verifier, so the gold has a second, independent derivation.

  valid      answered, and the answer matches the gold          kept
  disagree   answered, and it does not                          dropped
  invalid    \\boxed{INVALID}: missing, contradictory or ambiguous
  no_box     closed its thinking but gave no box
  unclosed   cut inside its thinking

    python -P -m b2e.rsi.gate --judge /tmp/b2e-models/Qwen3.6-35B-A3B \\
        --problems $B2E_ROOT/data/rsi/round1/problems.parquet --out $B2E_ROOT/data/rsi/round1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from b2e.reward.numeric import extract_boxed
from b2e.reward.score import compute_score
from b2e.rsi.docs import split_prompt

GATE_INSTRUCTION = (
    "Before answering, check the question: it must have exactly one correct numeric answer, with every quantity "
    "it needs given in the excerpt or the question and wording that allows only one reading. If it does not, "
    "write \\boxed{INVALID} and one sentence saying why. Otherwise answer it step by step and give the final "
    "answer as a single number in \\boxed{} (write percentages with %).")


def gate_messages(row: dict) -> list[dict]:
    excerpt, question = split_prompt(row["prompt"][0]["content"])
    return [{"role": "user", "content": f"{excerpt}\n\nQuestion: {question}\n\n{GATE_INSTRUCTION}"}]


def verdict(text: str, ground_truth: str) -> str:
    """The judge's verdict on one problem; text starts inside the <think> block the template opened."""
    if "</think>" not in text:
        return "unclosed"
    box = extract_boxed(text)
    if box is None:
        return "no_box"
    if box.strip().strip(".").upper() == "INVALID":
        return "invalid"
    return "valid" if compute_score("rsi_fin", text, ground_truth, thinking=True) >= 1.0 else "disagree"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--problems", type=Path, required=True, help="b2e.rsi.propose problems.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import pandas as pd
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.problems)
    rows = df.to_dict("records")
    llm = LLM(model=args.judge, gpu_memory_utilization=0.9, max_model_len=16384, enable_prefix_caching=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.chat([gate_messages(r) for r in rows], sp, chat_template_kwargs={"enable_thinking": True})
    texts = [o.outputs[0].text for o in outs]
    verdicts = [verdict(t, r["reward_model"]["ground_truth"]) for t, r in zip(texts, rows, strict=True)]
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": [r["extra_info"]["id"] for r in rows], "verdict": verdicts, "box": [extract_boxed(t) for t in texts],
        "tokens": [len(o.outputs[0].token_ids) for o in outs], "finish": [o.outputs[0].finish_reason for o in outs],
        "text": texts,
    }).to_parquet(args.out / "gate.parquet", index=False)
    df[[v == "valid" for v in verdicts]].to_parquet(args.out / "problems_gated.parquet", index=False)
    summary = {"problems": len(rows), **dict(Counter(verdicts))}
    (args.out / "summary_gate.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
