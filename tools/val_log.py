"""Validation readings from a verl RL log ($B2E_ROOT/logs/rl_<run>.log).

verl prints one metrics line per step ("step:N - key:value - ..."), and the
steps it validated carry val-core/<source>/acc/mean@1. Each leg re-validates
the step it resumes from, so a leg boundary has two readings of the same
weights; the P6 step-90 rule (docs/runs.md, 10-30) counts the first one, and
so do --mean and the "first" column here.

    python3 -I tools/val_log.py $B2E_ROOT/logs/rl_grpo_fin.log --rows finqa=279,tatqa=233
    python3 -I tools/val_log.py $B2E_ROOT/logs/rl_grpo_fin.log --rows finqa=279,tatqa=233 \\
        --mean 40,50,60 --mean 70,80,90

--rows gives each source's number of val rows, for the accuracy over all of
them; an accuracy times its row count must be a whole number of rows, so a
wrong count is caught rather than averaged in. Runs on the login node's system
Python (3.9, standard library only).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STEP = re.compile(r"\bstep:(\d+) - ")
ACC = re.compile(r"val-core/([\w.-]+)/acc/mean@\d+:(?:np\.float\d*\()?([-+0-9.eE]+)")


def parse(lines) -> list[tuple[int, dict[str, float]]]:
    """(step, {source: accuracy}) for each metrics line with val, in log order."""
    out = []
    for line in lines:
        m = STEP.search(line)
        if not m:
            continue
        acc = {src: float(v) for src, v in ACC.findall(line)}
        if acc:
            out.append((int(m.group(1)), acc))
    return out


def first_readings(readings: list[tuple[int, dict[str, float]]]) -> dict[int, dict[str, float]]:
    first: dict[int, dict[str, float]] = {}
    for step, acc in readings:
        first.setdefault(step, acc)
    return first


def rows_right(acc: dict[str, float], rows: dict[str, int]) -> tuple[int, int]:
    """(rows answered right, rows) over all sources; each product must be whole."""
    if set(acc) != set(rows):
        raise ValueError(f"val sources {sorted(acc)} but --rows names {sorted(rows)}")
    right = 0
    for src, a in acc.items():
        k = a * rows[src]
        if abs(k - round(k)) > 1e-6:
            raise ValueError(f"{src}: accuracy {a} is not a whole number of {rows[src]} rows")
        right += round(k)
    return right, sum(rows.values())


def parse_rows(spec: str) -> dict[str, int]:
    return {k: int(v) for k, v in (kv.split("=") for kv in spec.split(","))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--rows", type=parse_rows, help="source=rows,... for the accuracy over all rows")
    ap.add_argument("--mean", action="append", default=[], metavar="STEPS",
                    help="comma-separated steps: mean of their first readings (twice, with --rows: also the difference)")
    args = ap.parse_args(argv)
    with open(args.log, errors="replace") as f:
        readings = parse(f)
    if not readings:
        sys.exit(f"no val readings in {args.log}")
    first = first_readings(readings)
    sources = sorted({s for _, acc in readings for s in acc})
    print("step  first  " + "  ".join(f"{s:>7}" for s in sources) + ("      all" if args.rows else ""))
    seen: set[int] = set()
    for step, acc in readings:
        cells = "  ".join(f"{acc[s]:7.4f}" if s in acc else "      -" for s in sources)
        total = ""
        if args.rows:
            k, n = rows_right(acc, args.rows)
            total = f"  {k / n:.4f} ({k}/{n})"
        print(f"{step:>4}  {'yes' if step not in seen else 'no':>5}  {cells}{total}")
        seen.add(step)
    means = []
    for spec in args.mean:
        steps = [int(s) for s in spec.split(",")]
        missing = [s for s in steps if s not in first]
        if missing:
            sys.exit(f"no val reading at steps {missing} (have {sorted(first)})")
        per = {s: sum(first[t].get(s, float("nan")) for t in steps) / len(steps) for s in sources}
        line = "  ".join(f"{s} {v:.4f}" for s, v in per.items())
        if args.rows:
            k = sum(rows_right(first[t], args.rows)[0] for t in steps) / len(steps)
            n = sum(args.rows.values())
            means.append(k / n)
            line += f"  all {k / n:.4f} ({k:.2f}/{n} rows)"
        print(f"mean of the first readings at {spec}: {line}")
    if len(means) == 2:
        print(f"difference (second - first): {means[1] - means[0]:+.4f}")


if __name__ == "__main__":
    main()
