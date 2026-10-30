import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from audit_rollouts import group_signal, near_miss, scan_step, sign_flipped  # noqa: E402


def num(value, scale=""):
    return {"kind": "number", "value": value, "scale": scale}


def row(output, gts, score, uid="p_0_0", inp="Question: q"):
    return {"input": inp, "output": output, "gts": json.dumps(gts), "score": score, "step": 1, "uid": uid}


def test_near_miss_reasons():
    # the cases read in the P6 audit (results/p6_audit)
    assert near_miss("-394", num(394.0)) == "sign"  # "from december 2012 to december 2011"
    assert near_miss("1171400", num(1171.4)) == "scale x1e+3"  # converted to thousands; the gold is in millions
    assert near_miss("3,931 million", num(3931.0)) == "scale word"  # right digits, a unit word in the box
    assert near_miss("5000", num(50.0)) == "percent x100"  # not a percent on either side, so not accepted
    assert near_miss("103", num(100.0)) == "within 5%"
    assert near_miss("8,708 - 12,639 = -3,931", num(-3931.0)) == "several numbers"
    assert near_miss("Not available", num(0.87)) == "no number"
    assert near_miss("338.7", num(366.9)) == "wrong"


def test_near_miss_uses_the_gold_scale():
    # TAT-QA gold 1.9 with scale million: 1,900,000 matches the gold, -1,900,000 is a sign miss
    assert near_miss("-1,900,000", num(1.9, "million")) == "sign"


def test_group_signal_counts_prompts_not_samples():
    rows = [row("", num(1.0), s, uid=f"{p}_0_{i}") for p, scores in
            (("a", [1, 1, 1]), ("b", [0, 0, 0]), ("c", [1, 0, 0])) for i, s in enumerate(scores)]
    g = group_signal(rows)
    assert g["prompts"] == 3 and g["samples_per_prompt"] == {3: 3}
    assert (g["all_right"], g["all_wrong"], g["mixed"]) == (1 / 3, 1 / 3, 1 / 3)


def test_scan_rescores_and_flags():
    right = row("think</think>Step 1: 2 + 2 = 4\n\\boxed{4}", num(4.0), 1.0)
    cut = row("still thinking \\boxed{4} and", num(4.0), 0.0)  # no </think>: cut off at the limit (F#53)
    abstain = row("t</think>The excerpt has no 2016 data. \\boxed{Not available}", num(0.87), 0.0)
    shaped = row("t</think>\\boxed{4}", num(4.0), 0.5)  # a logged score that is not compute_score's (DAPO penalty)
    s = scan_step([right, cut, abstain, shaped])
    assert s["n"] == 4 and s["closed"] == 0.75
    assert s["counts"]["abstain"] == 1
    assert s["counts"]["score differs from compute_score"] == 1
    assert s["wrong_numeric"] == {"no number": 1}
    assert s["counts"].get("box only in thinking", 0) == 0  # the cut answer is not "closed" at all


def test_sign_flipped_needs_a_closed_wrong_answer():
    gt = num(394.0)
    assert sign_flipped(row("t</think>\\boxed{-394}", gt, 0.0), gt)
    assert not sign_flipped(row("t</think>\\boxed{394}", gt, 1.0), gt)
    assert not sign_flipped(row("t \\boxed{-394}", gt, 0.0), gt)  # unclosed: no final answer
