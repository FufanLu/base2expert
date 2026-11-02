"""Pass rates of a b2e.data.synth run, re-scored with the current verifier.

synth scores every sample when it generates it. When the verifier changes afterwards (F#86: a
CAIL charge label with a bracketed group, such as [走私、贩卖、运输、制造]毒品, could never score),
the stored scores are stale, and the pool selection (b2e.data.rl_pool) would keep the wrong
prompts. The texts are in <out>_parts/samples_*.parquet, so the scores are recomputed without
generating again, the way synth.score_prompt computed them.

    python -P -m b2e.data.rescore --parts rl_pool/p7/s_law_k8_parts \\
        --out rl_pool/p7/s_law_k8_rescored_passrate.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from b2e.data.pref_pairs import load_samples
from b2e.reward.score import compute_score, extract_boxed

COLUMNS = ["data_source", "id", "pass_rate", "mean_score", "ground_truth", "answers", "scores", "tokens",
           "finish", "seconds"]  # synth's *_passrate.parquet


def unequal(scores) -> bool:
    """rl_pool --unequal's rule: the k scores are not all the same (a nonzero group advantage)."""
    return len(set(round(float(s), 9) for s in scores)) > 1


def rescore(df: pd.DataFrame, min_score: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """load_samples' rows with scores, answers, pass_rate and mean_score recomputed (thinking on, as
    synth samples), in synth's passrate columns; and per source, the samples whose score changed and
    the pass rate and the prompts with unequal scores, before and after."""
    out: list[dict] = []
    summary: dict[str, dict] = {}
    for r in df.to_dict("records"):
        old = [float(s) for s in r["scores"]]
        new = [compute_score(r["data_source"], t, r["ground_truth"], thinking=True) for t in r["texts"]]
        s = summary.setdefault(r["data_source"], {"prompts": 0, "samples_changed": 0, "pass_rate_before": 0.0,
                                                   "pass_rate_after": 0.0, "unequal_before": 0, "unequal_after": 0})
        s["prompts"] += 1
        s["samples_changed"] += sum(abs(a - b) > 1e-9 for a, b in zip(old, new, strict=True))
        s["pass_rate_before"] += float(r["pass_rate"])
        s["pass_rate_after"] += sum(x >= min_score for x in new) / len(new)
        s["unequal_before"] += unequal(old)
        s["unequal_after"] += unequal(new)
        out.append({**r, "scores": new, "answers": [extract_boxed(t) for t in r["texts"]],
                    "pass_rate": sum(x >= min_score for x in new) / len(new), "mean_score": sum(new) / len(new)})
    for s in summary.values():
        for k in ("pass_rate_before", "pass_rate_after"):
            s[k] = round(s[k] / s["prompts"], 4)
    return pd.DataFrame(out)[COLUMNS], summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, required=True, help="a synth run's <out>_parts directory")
    ap.add_argument("--out", type=Path, required=True, help="the re-scored *_passrate.parquet")
    ap.add_argument("--min-score", type=float, default=1.0, help="as synth's: graded rewards count from here")
    a = ap.parse_args()
    rates, summary = rescore(load_samples(a.parts), a.min_score)
    rates.to_parquet(a.out, index=False)
    a.out.with_name(a.out.stem + "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
