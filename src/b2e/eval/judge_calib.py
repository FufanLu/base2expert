"""How well the rubric judge agrees with physicians, on HealthBench's meta-eval (P5, 11-03).

Each meta-eval row is one rubric item on one completion, labelled met / not met by several
physicians. The judge (b2e.reward.rubric's grader, Qwen3.6-35B-A3B in vLLM) labels a seeded
sample of rows; its labels are compared with every physician's (class-balanced F1, the
simple-evals metric `pairwise_model_f1_balanced`), next to the physicians' agreement with one
another on the same rows. The judge is used as a reward only if it is near that bar.

    python -P -m b2e.eval.judge_calib --judge /tmp/b2e-models/Qwen3.6-35B-A3B --n 3000 \\
        --meta $B2E_ROOT/data/raw/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl \\
        --out $B2E_ROOT/results/judge_calib
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from b2e.reward.rubric import f1_balanced, grader_messages, judge_pairs, load_jsonl, parse_verdict, physician_pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--meta", required=True, help="HealthBench meta-eval jsonl")
    ap.add_argument("--n", type=int, default=3000, help="rows to grade (seeded sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import pandas as pd
    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.meta)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    msgs = [grader_messages(list(r["prompt"]) + [{"role": "assistant", "content": r["completion"]}], r["rubric"])
            for r in sample]
    llm = LLM(model=args.judge, gpu_memory_utilization=0.9, max_model_len=16384, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": args.thinking})
    texts = [o.outputs[0].text for o in outs]
    verdicts = [parse_verdict(t) for t in texts]
    judge = f1_balanced(judge_pairs(sample, verdicts))
    doctors = f1_balanced(physician_pairs(sample))
    by_cat: dict[str, dict] = {}
    for cat in sorted({str(r.get("category", "")) for r in sample}):
        idx = [i for i, r in enumerate(sample) if str(r.get("category", "")) == cat]
        sub = [sample[i] for i in idx]
        by_cat[cat] = {"rows": len(idx), "judge": f1_balanced(judge_pairs(sub, [verdicts[i] for i in idx]))["f1_balanced"],
                       "physicians": f1_balanced(physician_pairs(sub))["f1_balanced"]}
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rubric": [r["rubric"] for r in sample], "category": [str(r.get("category", "")) for r in sample],
                  "labels": [list(r["binary_labels"]) for r in sample], "verdict": verdicts, "text": texts,
                  "tokens": [len(o.outputs[0].token_ids) for o in outs]}).to_parquet(args.out / "verdicts.parquet")
    summary = {"rows_total": len(rows), "graded": len(sample), "thinking": args.thinking,
               "no_verdict": sum(v is None for v in verdicts), "verdicts": dict(Counter(map(str, verdicts))),
               "judge_vs_physicians": judge, "physicians_vs_each_other": doctors, "by_category": by_cat}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("graded", "no_verdict", "judge_vs_physicians", "physicians_vs_each_other")}))


if __name__ == "__main__":
    main()
