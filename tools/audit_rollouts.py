"""Audit RL rollouts for reward hacking and lost learning signal (P6: every
algorithm's trajectories are read by hand before its numbers are trusted).

verl writes each step's rollouts to <ckpt>/rollouts/<step>.jsonl (input,
output, gts, score, step, uid; the uid's first field is the prompt, shared by
its n samples). Four reports:

  scan   per step: the score recomputed with the trainer's compute_score and
         compared with the logged one (any difference is a reward bug or a
         shaped reward such as DAPO's overlong penalty); answers still inside
         </think>; missing or repeated \\boxed{}; length of the thinking and of
         the answer after it (an RL policy that drops the step-by-step answer
         the prompt asks for, because only the box is scored, shows here);
         abstentions (a box with no number on a numeric question); outputs
         with Chinese the prompt does not have; repetition; and why wrong
         numeric answers were wrong (sign, scale, percent, rounding);
  groups per step: prompts whose samples all scored the same. GRPO-style
         advantages are zero for such a group, so it costs rollouts and
         teaches nothing (DAPO's dynamic sampling drops them);
  show   N random rollouts of one step in a compact form, to read;
  signs  sign-flipped answers grouped by prompt: a model that flips every
         sample disagrees with the gold's convention (it gets no gradient
         there), one that flips some can still learn it.

    python -P tools/audit_rollouts.py scan   $B2E_ROOT/ckpt/grpo_fin 1,30,60
    python -P tools/audit_rollouts.py groups $B2E_ROOT/ckpt/grpo_fin
    python -P tools/audit_rollouts.py show   $B2E_ROOT/ckpt/grpo_fin 60 --n 50 --seed 0
    python -P tools/audit_rollouts.py signs  $B2E_ROOT/ckpt/grpo_fin 60

It also runs on the login node's system Python (3.9, no b2e installed, no
torch): python3 -I tools/audit_rollouts.py ... imports b2e from ../src.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
import zlib
from pathlib import Path

try:
    import b2e  # noqa: F401
except ImportError:  # the login node: the repo's src, not an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from b2e.reward.numeric import extract_boxed, parse_number  # noqa: E402
from b2e.reward.score import SCALE, compute_score  # noqa: E402

CJK = re.compile(r"[一-鿿]")
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def steps_of(ckpt: Path, spec: str | None) -> list[int]:
    have = sorted(int(p.stem) for p in (ckpt / "rollouts").glob("*.jsonl"))
    if not spec:
        return have
    want = [int(s) for s in spec.split(",")]
    missing = [s for s in want if s not in have]
    if missing:
        raise SystemExit(f"no rollouts for steps {missing} in {ckpt}/rollouts (have {have[0]}..{have[-1]})")
    return want


def gold(row: dict) -> dict:
    g = row["gts"]
    return json.loads(g) if isinstance(g, str) else g


def split_think(out: str) -> tuple[str, str]:
    """(thinking, answer after </think>); an unclosed answer is all thinking."""
    if "</think>" in out:
        t, a = out.split("</think>", 1)
        return t, a
    return out, ""


def golds(gt: dict) -> list[float]:
    g = float(gt["value"])
    s = SCALE.get(gt.get("scale", ""), 1.0)
    return [g] if s == 1.0 else [g, g * s]


def near_miss(ans: str, gt: dict) -> str:
    """Why a boxed answer to a numeric question scored 0: the first reason that fits."""
    p = parse_number(ans)
    nums = NUM.findall(ans.replace(",", ""))
    if p is None:
        return "several numbers" if len(nums) > 1 else "no number"
    for x in golds(gt):
        if math.isclose(p.value, -x, rel_tol=1e-2):
            return "sign"
        if nums and math.isclose(float(nums[0]), x, rel_tol=1e-2):
            return "scale word"  # the digits are right; a unit word in the box moved them
        for f in (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
            if math.isclose(p.value, x * f, rel_tol=1e-2):
                return f"scale x1e{round(math.log10(f)):+d}"
        if math.isclose(p.value, x * 100, rel_tol=1e-2) or math.isclose(p.value * 100, x, rel_tol=1e-2):
            return "percent x100"
        if x != 0 and abs(p.value - x) / abs(x) <= 0.05:
            return "within 5%"
    return "wrong"


def max_line_repeats(text: str) -> int:
    c = collections.Counter(s.strip() for s in text.splitlines() if len(s.strip()) >= 10)
    return max(c.values()) if c else 0


def quantile(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def scan_step(rows: list[dict], thinking: bool = True) -> dict:
    st: collections.Counter = collections.Counter()
    miss: collections.Counter = collections.Counter()
    think_len, ans_len, zr = [], [], []
    for r in rows:
        gt, out, s = gold(r), r["output"], float(r["score"])
        t, a = split_think(out)
        closed = "</think>" in out
        st["closed"] += closed
        think_len.append(len(t))
        if closed:
            ans_len.append(len(a))
            nb = a.count("\\boxed{")
            st["no box"] += nb == 0
            st["box only in thinking"] += nb == 0 and "\\boxed{" in t
            st["several boxes"] += nb > 1
        st["line repeated 8+"] += max_line_repeats(out) >= 8
        zr.append(len(zlib.compress(out.encode())) / max(1, len(out.encode())))
        st["chinese only in output"] += bool(CJK.search(out)) and not CJK.search(r["input"])
        if compute_score(r.get("data_source", ""), out, gt, thinking=thinking) != s:
            st["score differs from compute_score"] += 1
        st["score sum"] += s
        if gt["kind"] != "number" or not closed:
            continue
        st["numeric"] += 1
        box = extract_boxed(out)
        if box is not None and not re.search(r"\d", box):
            st["abstain"] += 1
        if s <= 0 and box is not None:
            miss[near_miss(box, gt)] += 1
    n = len(rows)
    return {
        "n": n,
        "score": st["score sum"] / n,
        "closed": st["closed"] / n,
        "counts": {k: v for k, v in st.items() if k not in ("closed", "score sum")},
        "think_chars_median": statistics.median(think_len) if think_len else 0,
        "answer_chars": {"p10": quantile(ans_len, 0.1), "median": quantile(ans_len, 0.5), "p90": quantile(ans_len, 0.9)},
        "zlib_ratio_p5": round(sorted(zr)[int(0.05 * len(zr))], 3) if zr else 0,
        "wrong_numeric": dict(miss.most_common()),
    }


def group_signal(rows: list[dict]) -> dict:
    """Prompts whose samples all scored the same, by the uid's prompt field."""
    by: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        by[r["uid"].split("_")[0]].append(float(r["score"]))
    same = [v for v in by.values() if min(v) == max(v)]
    n = len(by)
    return {
        "prompts": n,
        "samples_per_prompt": dict(collections.Counter(len(v) for v in by.values())),
        "all_right": sum(1 for v in same if v[0] >= 1) / n,
        "all_wrong": sum(1 for v in same if v[0] <= 0) / n,
        "mixed": (n - len(same)) / n,
    }


def cmd_scan(args: argparse.Namespace) -> None:
    for s in steps_of(args.ckpt, args.steps):
        print(s, json.dumps(scan_step(load(args.ckpt / "rollouts" / f"{s}.jsonl"), thinking=not args.no_thinking)))


def cmd_groups(args: argparse.Namespace) -> None:
    print("step  prompts  all-right  all-wrong  mixed")
    for s in steps_of(args.ckpt, args.steps):
        g = group_signal(load(args.ckpt / "rollouts" / f"{s}.jsonl"))
        print(f"{s:>4}  {g['prompts']:>7}  {g['all_right']:>9.1%}  {g['all_wrong']:>9.1%}  {g['mixed']:>5.1%}")


def question(inp: str) -> str:
    i = inp.rfind("Question:")
    q = inp[i:] if i >= 0 else inp[-400:]
    return " ".join(q.split("Answer the question", 1)[0].split())[:300]


def cmd_show(args: argparse.Namespace) -> None:
    rows = load(args.ckpt / "rollouts" / f"{args.step}.jsonl")
    random.Random(args.seed).shuffle(rows)
    for k, r in enumerate(rows[: args.n]):
        gt = gold(r)
        t, a = split_think(r["output"])
        box = extract_boxed(r["output"]) if "</think>" in r["output"] else None
        g = {key: v for key, v in gt.items() if key != "kind"}
        print(f"\n### {k + 1} score {r['score']} kind {gt['kind']} gold {g} boxed {box!r} think {len(t)} answer {len(a)}")
        print("Q:", question(r["input"]))
        print("A:", a.strip()[:700].replace("\n", " | "))
        if float(r["score"]) <= 0:
            print("T-tail:", " ".join(t[-400:].split()))


def sign_flipped(r: dict, gt: dict) -> bool:
    if gt["kind"] != "number" or float(r["score"]) >= 1 or "</think>" not in r["output"]:
        return False
    box = extract_boxed(r["output"])
    p = parse_number(box) if box else None
    return p is not None and any(math.isclose(p.value, -x, rel_tol=1e-2) for x in golds(gt))


def cmd_signs(args: argparse.Namespace) -> None:
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in load(args.ckpt / "rollouts" / f"{args.step}.jsonl"):
        by[r["uid"].split("_")[0]].append(r)
    found = []
    for v in by.values():
        k = sum(sign_flipped(r, gold(v[0])) for r in v)
        if k:
            found.append((k, int(sum(float(r["score"]) >= 1 for r in v)), v))
    print(f"{sum(f[0] for f in found)} sign-flipped samples in {len(found)} of {len(by)} prompts; "
          f"{sum(1 for f in found if f[1] == 0)} of these prompts have no right sample")
    for k, right, v in sorted(found, key=lambda f: (-f[0], f[1]))[: args.n]:
        gt = gold(v[0])
        boxes = [extract_boxed(x["output"]) if "</think>" in x["output"] else None for x in v]
        print(f"--- flipped {k}/{len(v)}, right {right}, gold {gt['value']} {gt.get('scale', '')} | {question(v[0]['input'])}")
        print("    boxed:", boxes)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "groups"):
        p = sub.add_parser(name)
        p.add_argument("ckpt", type=Path)
        p.add_argument("steps", nargs="?", help="comma-separated; default every step")
        if name == "scan":
            p.add_argument("--no-thinking", action="store_true", help="the run's prompts did not open a thinking block")
    for name in ("show", "signs"):
        p = sub.add_parser(name)
        p.add_argument("ckpt", type=Path)
        p.add_argument("step", type=int)
        p.add_argument("--n", type=int, default=50 if name == "show" else 12)
        if name == "show":
            p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"scan": cmd_scan, "groups": cmd_groups, "show": cmd_show, "signs": cmd_signs}[args.cmd](args)


if __name__ == "__main__":
    main()
