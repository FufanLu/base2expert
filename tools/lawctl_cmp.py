"""Articles-prompt control (F#92): each example against `orig` on the same rows.

    ENV_KIND=tools sbatch cluster/cpu_run.sbatch python -P tools/lawctl_cmp.py \\
        $B2E_ROOT/results/s_law/s_law-think-lawctl $B2E_ROOT/data/processed/law_med/control

RESULTS_DIR holds cail_test_articles_{orig,ex1,ex2}.run0.jsonl (cluster/submit_eval_lawctl.sh),
CONTROL_DIR the parquets and cail_test_report.json (b2e.data.law_control). Per example: accuracy,
the rows right with only one of the two prompts (exact McNemar test), the same without the rows
whose gold holds an article of the example (a model copying it would score there), copies of the
example, and answers with general-part articles (1-101, which no CAIL label holds). Writes
RESULTS_DIR/lawctl_cmp.json.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

VARIANTS = ("orig", "ex1", "ex2")
SPECIFIC_PART = 102


def nums(text: str) -> set[int]:
    """The article numbers in the last \\boxed{}."""
    b = text.split("\\boxed{")[-1].split("}")[0] if "\\boxed{" in text else ""
    return {int(n) for n in re.findall(r"\d+", b)}


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar test on b and c discordant pairs."""
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


def discordant(new: dict, old: dict, ids) -> tuple[int, int]:
    """(right only with the new prompt, right only with the old one)."""
    return (sum(old[i]["score"] < 1 <= new[i]["score"] for i in ids),
            sum(new[i]["score"] < 1 <= old[i]["score"] for i in ids))


def compare(runs: dict[str, dict], gold: dict, examples: dict[str, set[int]]) -> dict:
    out = {}
    o = runs["orig"]
    for v, rs in runs.items():
        ans = {i: nums(r["response"]) for i, r in rs.items()}
        row = {"n": len(rs), "acc": round(sum(r["score"] for r in rs.values()) / len(rs), 4),
               "with_67": sum(67 in a for a in ans.values()),
               "with_general": sum(any(x < SPECIFIC_PART for x in a) for a in ans.values()),
               "mean_tokens": round(sum(r["tokens"] for r in rs.values()) / len(rs), 1)}
        if v != "orig":
            ex = examples[v]
            keep = [i for i in rs if not gold[i] & ex]
            b, c = discordant(rs, o, rs)
            bk, ck = discordant(rs, o, keep)
            row.update({
                "example": sorted(ex), "gold_has_example_article": len(rs) - len(keep),
                "copied_example_where_gold_differs": sum(ans[i] == ex and gold[i] != ex for i in rs),
                "discordant_new_right_old_wrong": b, "discordant_old_right_new_wrong": c,
                "mcnemar_p": mcnemar(b, c),
                "without_example_rows": {"n": len(keep),
                                         "acc_new": round(sum(rs[i]["score"] for i in keep) / len(keep), 4),
                                         "acc_orig": round(sum(o[i]["score"] for i in keep) / len(keep), 4),
                                         "discordant": [bk, ck], "mcnemar_p": mcnemar(bk, ck)},
                # of the rows the new example fixed, those whose old answer had a general-part article
                "fixed_rows_old_had_general": sum(o[i]["score"] < 1 <= rs[i]["score"] and
                                                  any(x < SPECIFIC_PART for x in nums(o[i]["response"]))
                                                  for i in rs),
            })
        out[v] = row
    return out


def main(argv: list[str]) -> dict:
    import pandas as pd

    res, ctl = Path(argv[0]), Path(argv[1])
    report = json.loads((ctl / "cail_test_report.json").read_text())
    gold: dict = {}
    for v in VARIANTS:
        for r in pd.read_parquet(ctl / f"cail_test_articles_{v}.parquet").itertuples(index=False):
            gold.setdefault(r.extra_info["id"], set(json.loads(r.reward_model["ground_truth"])["value"]))
    runs = {}
    for v in VARIANTS:
        with open(res / f"cail_test_articles_{v}.run0.jsonl") as f:
            runs[v] = {r["id"]: r for r in map(json.loads, f)}
    if not all(set(r) == set(gold) and len(r) == report["data_rows"] for r in runs.values()):
        raise ValueError("the runs and the control set hold different rows")
    examples = {v: set(map(int, re.findall(r"\d+", report[v]["example"]))) for v in VARIANTS if v != "orig"}
    out = compare(runs, gold, examples)
    (res / "lawctl_cmp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
