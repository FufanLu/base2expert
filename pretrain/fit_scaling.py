"""Scaling-law fit over a nanochat series, and its prediction for held-out sizes.

Each run of cluster/nanochat_series.sbatch leaves <base>_d<depth>_train.log
(nanochat prints the training FLOPs, the parameter counts, the val bpb at the
end and CORE) and <base>_d<depth>_heldout.json (pretrain/eval_bpb.py). At a
fixed data:param ratio the runs lie on one compute frontier, so bpb is fit as
a power law in training compute,

    log bpb = a + b log C,

by least squares on the --fit depths, and evaluated at the --predict depths,
whose measured bpb is the check. Done for the run's own val shard and for each
held-out source. Standard library only (runs on the login node):

    python3 -I pretrain/fit_scaling.py --results $ROOT/results/nanochat --base bl2_d5 \\
        --fit 6 8 10 12 --predict 14 --out $ROOT/results/nanochat/scaling_bl2_d5.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

PATTERNS = {
    "flops": r"Total training FLOPs estimate: ([\d.e+]+)",
    "tokens": r"Total number of training tokens: ([\d,]+)",
    "params_total": r"^total\s*: ([\d,]+)",
    "val_bpb": r"Validation bpb: ([\d.]+)",
    "core": r"CORE metric: ([\d.]+)",
}


def parse_log(text: str) -> dict[str, float]:
    out = {}
    for k, p in PATTERNS.items():
        m = re.findall(p, text, flags=re.M)
        if m:
            out[k] = float(m[-1].replace(",", ""))
    return out


def fit_power(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares of log y = a + b log x; returns (a, b)."""
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((u - mx) ** 2 for u in lx)
    b = sum((u - mx) * (v - my) for u, v in zip(lx, ly, strict=True)) / sxx
    return my - b * mx, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--fit", type=int, nargs="+", required=True)
    ap.add_argument("--predict", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    R = Path(args.results)
    runs = {}
    for d in args.fit + args.predict:
        row = parse_log((R / f"{args.base}_d{d}_train.log").read_text())
        held = json.loads((R / f"{args.base}_d{d}_heldout.json").read_text())
        row.update({f"heldout_{s}": v["bpb"] for s, v in held["sources"].items()})
        runs[d] = row
    metrics = ["val_bpb"] + sorted(k for k in runs[args.fit[0]] if k.startswith("heldout_"))
    fits = {}
    for m in metrics:
        a, b = fit_power([runs[d]["flops"] for d in args.fit], [runs[d][m] for d in args.fit])
        resid = [math.log(runs[d][m]) - (a + b * math.log(runs[d]["flops"])) for d in args.fit]
        pred = {d: {"predicted": round(math.exp(a + b * math.log(runs[d]["flops"])), 4),
                    "measured": runs[d][m]} for d in args.predict}
        for p in pred.values():
            p["error"] = round(p["measured"] - p["predicted"], 4)
            p["error_rel"] = round(p["measured"] / p["predicted"] - 1, 4)
        fits[m] = {"a": a, "b": round(b, 5), "max_abs_log_resid": round(max(abs(r) for r in resid), 5),
                   "predict": pred}
        line = f"{m:24} b {b:+.4f}  fit resid {max(abs(r) for r in resid):.4f}"
        for d, p in pred.items():
            line += f"  d{d}: pred {p['predicted']:.4f} meas {p['measured']:.4f} ({p['error_rel']:+.2%})"
        print(line)
    Path(args.out).write_text(json.dumps({"base": args.base, "fit": args.fit, "predict": args.predict,
                                          "runs": runs, "fits": fits}, indent=2))


if __name__ == "__main__":
    main()
