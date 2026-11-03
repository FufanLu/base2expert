"""Rubric grading by a generative judge (P5, the 11-03 plan: Kimi K3's protocol, calibrated on
HealthBench before it rewards medicine's open prompts).

One call grades one rubric item on the last assistant reply of a conversation and returns
criteria_met. HealthBench's rubrics mix items a good reply meets (positive points) with mistakes
(negative points, e.g. "Fails to advise ..."): a mistake item is "met" when the reply makes the
mistake, so its points count against the reply. A reply's score is the points it achieves over
the positive points available, clipped to [0, 1] (HealthBench's own aggregation).

The judge's wording is ours; the protocol (per-item yes/no in JSON, points-weighted score)
follows HealthBench's grader in OpenAI simple-evals.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

GRADER = """You are grading one rubric item against the last assistant reply in a conversation.

<conversation>
{conversation}
</conversation>

<rubric_item>
{item}
</rubric_item>

Decide whether the last assistant reply meets the rubric item. Some items describe what a good reply does; others describe a mistake (for example "Fails to ..." or "Recommends ... without ..."), and such an item is met when the reply makes that mistake. An item with several requirements is met only if the reply meets all of them. Judge only the last assistant reply, using the rest of the conversation as context.

Answer with JSON only: {{"explanation": "<one or two sentences>", "criteria_met": true or false}}"""

_MET = re.compile(r'"criteria_met"\s*:\s*(true|false)', re.I)


def render_conversation(messages: Sequence[dict]) -> str:
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def grader_messages(conversation: Sequence[dict], item: str) -> list[dict]:
    return [{"role": "user", "content": GRADER.format(conversation=render_conversation(conversation), item=item)}]


def parse_verdict(text: str) -> bool | None:
    """criteria_met from the judge's reply (after its thinking), or None if it gave none."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    found = _MET.findall(text)
    if not found:
        return None
    return found[-1].lower() == "true"


def rubric_score(items: Sequence[tuple[float, bool]]) -> float | None:
    """(points, met) per item -> achieved / possible positive points, clipped to [0, 1]."""
    possible = sum(p for p, _ in items if p > 0)
    if possible <= 0:
        return None
    achieved = sum(p for p, met in items if met)
    return min(1.0, max(0.0, achieved / possible))


def f1_balanced(pairs: Sequence[tuple[bool, bool]]) -> dict:
    """Agreement of predicted labels with reference labels, (pred, ref) per pair: the F1 of
    the "met" class, the F1 of the "not met" class, and their mean (class-balanced)."""

    def f1(cls: bool) -> float:
        tp = sum(p == cls and r == cls for p, r in pairs)
        fp = sum(p == cls and r != cls for p, r in pairs)
        fn = sum(p != cls and r == cls for p, r in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    pos, neg = f1(True), f1(False)
    return {"f1_met": pos, "f1_not_met": neg, "f1_balanced": (pos + neg) / 2, "pairs": len(pairs)}


def physician_pairs(rows: Sequence[dict]) -> list[tuple[bool, bool]]:
    """Each physician's label against each other physician's label on the same item: the
    agreement physicians have with one another, the bar a judge is compared with."""
    out = []
    for r in rows:
        labels = list(r["binary_labels"])
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i != j:
                    out.append((bool(a), bool(b)))
    return out


def judge_pairs(rows: Sequence[dict], verdicts: Sequence[bool | None]) -> list[tuple[bool, bool]]:
    """The judge's label against every physician's label on the items it gave a verdict on."""
    return [(v, bool(lab)) for r, v in zip(rows, verdicts, strict=True) if v is not None
            for lab in r["binary_labels"]]


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
