"""RL prompt pool: keep the prompts the starting policy solves sometimes, not always or never.

GRPO-style advantages compare the samples of one prompt with each other. A
prompt whose samples are all right or all wrong gives every sample the same
reward, so its advantages are zero and it only costs rollout time. DAPO drops
such groups online (dynamic sampling); in verl 0.9.1 that exists only in the
V1 trainer (docs/runs.md), so this is the offline version: sample the starting
policy k times per prompt beforehand (b2e.data.synth with --temperature 1.0
--top-p 1.0, the RL rollout settings) and keep prompts whose pass rate lies in
[--low, --high].

    python -P -m b2e.data.rl_pool --passrate s_fin_k8_passrate.parquet --low 0.1 --high 0.9 \\
        --out $B2E_ROOT/data/processed/finance_pool --train finqa_train.parquet tatqa_train.parquet \\
        --val finqa_dev.parquet tatqa_dev.parquet

The pass rate counts samples at score 1, so a graded reward (CAIL term gives
0.2-0.8 for a near miss) can leave a group with no sample at 1 and still unequal
scores. --unequal keeps the prompts whose k scores are not all equal instead,
which is the condition for a nonzero advantage and, for a 0/1 reward, the same
prompts as the pass-rate range (P7 pools; docs/runs.md, 11-01).

The train files are a named option: as a positional list after `--val`, they
were taken as more val files and the run stopped with "the following arguments
are required: train" (F#48).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from b2e.reward.numeric import numeric_match, parse_number


def wrong_majority(answers: Sequence[str | None], scores: Sequence[float], finish: Sequence[str]) -> bool:
    """True if one wrong numeric answer is given by more finished samples than score 1.

    For problems a model wrote itself (P7.5), whose gold comes from its own program: in
    the pilot (11511420) 20 of the 23 such problems had a wrong or ambiguous gold, or
    one that S-fin reports at another precision. Two answers count as the same one when
    the finance verifier would accept either for the other (numeric_match).
    """
    right = sum(s >= 1 - 1e-9 for s in scores)
    reps: list[str] = []
    counts: list[int] = []
    for a, s, f in zip(answers, scores, finish, strict=True):
        if s >= 1 - 1e-9 or f != "stop" or not a or parse_number(a) is None:
            continue
        for i, r in enumerate(reps):
            if numeric_match(a, r):
                counts[i] += 1
                break
        else:
            reps.append(a)
            counts.append(1)
    return max(counts, default=0) > right


def select(pool: pd.DataFrame, rates: pd.DataFrame, low: float, high: float, unequal: bool = False,
           majority: bool = False) -> pd.DataFrame:
    """Rows of pool whose (data_source, id) has a pass rate in [low, high].

    With unequal, the rows whose k scores are not all equal instead: the condition for a
    nonzero group advantage. For a 0/1 reward these are the prompts with a pass rate in
    [1/k, 1 - 1/k], [0.1, 0.9] at k = 8; a graded reward keeps more (CAIL term in the P7
    pilot, 11527721: 232 of 300 prompts against 147 by the pass rate at score 1). With
    majority, prompts where a wrong answer outvotes the gold (wrong_majority) are dropped.
    """
    if unequal:
        keep = rates[rates["scores"].map(lambda s: max(s) > min(s))]
    else:
        keep = rates[(rates["pass_rate"] >= low) & (rates["pass_rate"] <= high)]
    if majority:
        out = [wrong_majority(a, s, f) for a, s, f in zip(keep["answers"], keep["scores"], keep["finish"], strict=True)]
        keep = keep[[not x for x in out]]
    keys = set(zip(keep["data_source"], keep["id"], strict=True))
    mask = [(ds, e["id"]) in keys for ds, e in zip(pool["data_source"], pool["extra_info"], strict=True)]
    return pool[mask]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passrate", required=True, help="b2e.data.synth *_passrate.parquet of the starting policy")
    ap.add_argument("--low", type=float, default=0.1)
    ap.add_argument("--high", type=float, default=0.9)
    ap.add_argument("--unequal", action="store_true",
                    help="keep prompts whose k scores are not all equal instead of --low/--high (P7: graded rewards)")
    ap.add_argument("--majority", action="store_true",
                    help="drop prompts where one wrong numeric answer outvotes the gold (P7.5: self-made problems)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", nargs="+", required=True, help="validation parquet files, listed as they are")
    ap.add_argument("--train", nargs="+", required=True, help="train parquet files to select from")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    rates = pd.read_parquet(args.passrate)
    args.out.mkdir(parents=True, exist_ok=True)
    rule = {"unequal": True} if args.unequal else {"low": args.low, "high": args.high}
    summary: dict = {"passrate": args.passrate, **rule, "majority": args.majority, "sources": {}}
    listed = []
    for f in args.train:
        pool = pd.read_parquet(f)
        kept = select(pool, rates, args.low, args.high, args.unequal, args.majority)
        dst = args.out / (Path(f).stem + "_pool.parquet")
        kept.to_parquet(dst, index=False)
        listed.append(str(dst))
        scored = rates[rates["data_source"].isin(pool["data_source"].unique())]
        summary["sources"][Path(f).stem] = {
            "rows": len(pool), "scored": len(scored), "kept": len(kept),
            "all_wrong": int((scored["pass_rate"] == 0).sum()), "all_right": int((scored["pass_rate"] == 1).sum())}
    (args.out / "rl_train.txt").write_text("\n".join(listed) + "\n")
    (args.out / "rl_val.txt").write_text("\n".join(str(Path(v).resolve()) for v in args.val) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
