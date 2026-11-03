"""The low-effort prompt pool for R-fin-low (P7, Kimi K3's effort control; docs/plan.md, 11-03).

Every finance pool prompt is copied with a system message asking for low reasoning effort, a
data_source ending in "_low" (P8 routes these to the R-fin-low teacher), and its token budget:
S-fin's median length over its finished samples on that prompt in the pool sampling. A prompt
whose samples were all cut has no length to go by and is left out. The val rows get the system
message and the source suffix but no budget (S-fin was not sampled on them), so their length is
measured, not rewarded.

    python -P -m b2e.data.effort_pool --passrate $B2E_ROOT/data/rl_pool/s_fin_k8_passrate.parquet \\
        --train $B2E_ROOT/data/processed/finance_pool/finqa_train_pool.parquet \\
                $B2E_ROOT/data/processed/finance_pool/tatqa_train_pool.parquet \\
        --val $B2E_ROOT/data/processed/finance_pool/rl_val512.parquet \\
        --out $B2E_ROOT/data/processed/finance_low
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

LOW_EFFORT = "Reasoning effort: low. Think briefly, then answer."


def budget(tokens: Sequence[int], finish: Sequence[str]) -> float | None:
    done = [t for t, f in zip(tokens, finish, strict=True) if f == "stop"]
    return float(statistics.median(done)) if done else None


def low_row(row: dict, budget_tokens: float | None) -> dict:
    extra = dict(row["extra_info"])
    if budget_tokens is not None:
        extra["budget"] = budget_tokens
    return {**row, "data_source": row["data_source"] + "_low",
            "prompt": [{"role": "system", "content": LOW_EFFORT}, *list(row["prompt"])], "extra_info": extra}


def build(rows: Sequence[dict], budgets: dict[tuple[str, str], float | None]) -> tuple[list[dict], int]:
    """Low-effort copies of the rows that have a budget, and the number left out."""
    out, skipped = [], 0
    for r in rows:
        b = budgets.get((r["data_source"], str(r["extra_info"]["id"])))
        if b is None:
            skipped += 1
            continue
        out.append(low_row(r, b))
    return out, skipped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passrate", required=True, help="b2e.data.synth *_passrate.parquet of the cold-start model")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import pandas as pd

    rates = pd.read_parquet(args.passrate)
    budgets = {(ds, str(i)): budget(list(t), list(f))
               for ds, i, t, f in zip(rates["data_source"], rates["id"], rates["tokens"], rates["finish"], strict=True)}
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"passrate": args.passrate, "system": LOW_EFFORT, "train": {}, "val": {}}
    listed = []
    for f in args.train:
        rows, skipped = build(pd.read_parquet(f).to_dict("records"), budgets)
        dst = args.out / (Path(f).stem + "_low.parquet")
        pd.DataFrame(rows).to_parquet(dst, index=False)
        listed.append(str(dst))
        b = [r["extra_info"]["budget"] for r in rows]
        summary["train"][Path(f).stem] = {"rows": len(rows), "no_finished_sample": skipped,
                                          "budget_median": statistics.median(b) if b else None}
    vals = []
    for f in args.val:
        rows = [low_row(r, None) for r in pd.read_parquet(f).to_dict("records")]
        dst = args.out / (Path(f).stem + "_low.parquet")
        pd.DataFrame(rows).to_parquet(dst, index=False)
        vals.append(str(dst))
        summary["val"][Path(f).stem] = len(rows)
    (args.out / "rl_train.txt").write_text("\n".join(listed) + "\n")
    (args.out / "rl_val.txt").write_text("\n".join(vals) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
