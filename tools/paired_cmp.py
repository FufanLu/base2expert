"""Paired per-item comparison of two models' answers to the same eval sets.

Reads the <set>.run<r>.jsonl files b2e.eval.run_eval wrote into two result
dirs (one model each) and pairs the answers by (data_source, id). Per set it
prints the mean score of A and of B, the mean difference B - A, its standard
error over the items, and the number of items each scored higher on. With two
or more sets a pooled row takes all their items together: the P7 recipe rule
(docs/runs.md, 10-31) ranks the P6 arms by the pooled FinQA test + TAT-QA dev
score with thinking.

    python3 -I tools/paired_cmp.py $B2E_ROOT/results/s_fin/s_fin-think \\
        $B2E_ROOT/results/grpo_fin_step90/grpo_fin_step90-think

An answer that opened a thinking block (always, in a dir run with thinking, as
its summary.json says) and never closed it scores 0, as compute_score does
since F#53, so dirs scored before that rule compare the same way. Items only
one dir has are left out and counted in "unpaired". --exclude leaves out the
items of other answer files too, e.g. the 512 RL val rows, 233 of which are
TAT-QA dev items (F#66):

    python3 -I tools/paired_cmp.py A B --exclude $B2E_ROOT/results/grpo_fin_step90/valchk/rl_val512.run0.jsonl

Runs on the login node's system Python (3.9, standard library only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def thinking_of(d: Path) -> bool:
    return json.loads((d / "summary.json").read_text())["thinking"] == "true"


def load(d: Path, name: str, run: int, thinking: bool) -> dict[tuple[str, str], float] | None:
    """{(data_source, id): score} of one set's answers in a result dir; None if the dir lacks it."""
    p = d / f"{name}.run{run}.jsonl"
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            t = r["response"]
            unclosed = (thinking or "<think>" in t) and "</think>" not in t
            out[(r["data_source"], r["id"])] = 0.0 if unclosed else float(r["score"])
    return out


def keys_of(paths: list[Path]) -> set[tuple[str, str]]:
    """(data_source, id) of every answer in these run_eval answer files."""
    out = set()
    for p in paths:
        with open(p) as f:
            out |= {(r["data_source"], r["id"]) for r in map(json.loads, f)}
    return out


def compare(a: dict, b: dict) -> dict:
    """Means over the items both have, the mean of B - A and its standard error."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        raise ValueError("no item in both")
    d = [b[i] - a[i] for i in ids]
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    return {"n": n, "unpaired": len(set(a) ^ set(b)),
            "a": sum(a[i] for i in ids) / n, "b": sum(b[i] for i in ids) / n, "diff": mean, "se": se,
            "a_higher": sum(a[i] > b[i] for i in ids), "b_higher": sum(b[i] > a[i] for i in ids)}


def row(name: str, c: dict) -> str:
    return (f"{name:14} {c['n']:5} {c['unpaired']:8} {c['a']:7.4f} {c['b']:7.4f} {c['diff']:+7.4f} {c['se']:6.4f}"
            f" {c['a_higher']:5} {c['b_higher']:5}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", type=Path, help="result dir of model A (a run_eval --out)")
    ap.add_argument("b", type=Path, help="result dir of model B")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--sets", nargs="+", help="set names (default: every set of A's run that B also has)")
    ap.add_argument("--exclude", nargs="+", type=Path, default=[],
                    help="run_eval answer files (.jsonl): their (data_source, id) items are left out")
    args = ap.parse_args(argv)
    think = {d: thinking_of(d) for d in (args.a, args.b)}
    for label, d in (("A", args.a), ("B", args.b)):
        print(f"{label}: {d} (thinking {str(think[d]).lower()}, run {args.run})")
    drop = keys_of(args.exclude)
    if drop:
        print(f"excluding the {len(drop)} items of {', '.join(str(p) for p in args.exclude)}")
    suffix = f".run{args.run}.jsonl"
    names = args.sets or sorted(p.name[: -len(suffix)] for p in args.a.glob("*" + suffix))
    print(f"{'set':14} {'n':>5} {'unpaired':>8} {'A':>7} {'B':>7} {'B-A':>7} {'se':>6} {'A>B':>5} {'B>A':>5}")
    pooled_a: dict = {}
    pooled_b: dict = {}
    for name in names:
        a, b = load(args.a, name, args.run, think[args.a]), load(args.b, name, args.run, think[args.b])
        if a is None or b is None:
            if args.sets:
                sys.exit(f"no {name}{suffix} in {args.a if a is None else args.b}")
            continue
        a = {k: v for k, v in a.items() if k not in drop}
        b = {k: v for k, v in b.items() if k not in drop}
        print(row(name, compare(a, b)))
        pooled_a.update({(name, *k): v for k, v in a.items()})
        pooled_b.update({(name, *k): v for k, v in b.items()})
    if not pooled_a:
        sys.exit("no set in both dirs")
    if len({k[0] for k in pooled_a}) > 1:
        print(row("pooled", compare(pooled_a, pooled_b)))


if __name__ == "__main__":
    main()
