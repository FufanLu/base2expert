"""Score saved b2e.eval.run_eval answers again with the current reward code.

run_eval writes every answer (<set>.run<r>.jsonl). After a verifier fix the
numbers are recomputed from those files on a CPU; nothing is generated again.
Writes <results>/rescore.json with the same metrics as summary.json and the
number of answers whose score changed; the original files stay as they are.

    python -P tools/rescore_eval.py --results $B2E_ROOT/results/baseline/Qwen3.5-4B-nothink-if \\
        --data $B2E_ROOT/data/processed/if/ifeval_test.parquet $B2E_ROOT/data/processed/if/if_multi_val.parquet --thinking false
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd

from b2e.reward.score import compute_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--thinking", choices=["true", "false"], required=True, help="the run_eval --thinking of these answers")
    args = ap.parse_args()
    logging.disable(logging.ERROR)  # langdetect logs every answer it cannot classify

    gt = {}
    for path in args.data:
        for r in pd.read_parquet(path).itertuples(index=False):
            gt[(r.data_source, r.extra_info["id"])] = r.reward_model["ground_truth"]
    commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out = {"results": str(args.results), "code": commit, "sets": {}}
    for f in sorted(args.results.glob("*.run*.jsonl")):
        name, run = f.name.rsplit(".run", 1)
        recs = [json.loads(line) for line in f.open()]
        missing = sum((r["data_source"], r["id"]) not in gt for r in recs)  # no longer in the set (e.g. F#33 drops)
        recs = [r for r in recs if (r["data_source"], r["id"]) in gt]
        new = [compute_score(r["data_source"], r["response"], gt[(r["data_source"], r["id"])], thinking=args.thinking == "true")
               for r in recs]
        out["sets"].setdefault(name, {})[f"run{run.split('.')[0]}"] = {
            "n": len(recs),
            "not_in_data": missing,
            "accuracy": round(sum(new) / len(new), 4),
            "strict_accuracy": round(sum(s >= 1.0 - 1e-9 for s in new) / len(new), 4),
            "accuracy_before": round(sum(r["score"] for r in recs) / len(recs), 4),
            "strict_before": round(sum(r["score"] >= 1.0 - 1e-9 for r in recs) / len(recs), 4),
            "changed": sum(abs(a - r["score"]) > 1e-9 for a, r in zip(new, recs, strict=True)),
        }
    (args.results / "rescore.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
