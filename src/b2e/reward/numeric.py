"""Numeric answer checking for finance questions (FinQA, TAT-QA, ConvFinQA).

The model is trained to end its answer with ``\\boxed{...}``. Finance answers
arrive in many spellings of the same number, and a reward that rejects a
right answer for its spelling teaches the model formatting, not finance:

    "12.4%"  "12.4 %"  "0.124"            (gold may be stored as 0.124 or 12.4)
    "1,234.5"  "$1,234.5"  "(1,234.5)"   (parentheses = negative, accounting style)
    "1.2 billion"  "12亿"  "3.5万元"
    "-3.1"  "−3.1" (unicode minus)

``numeric_match`` accepts a prediction when, after parsing, it equals the gold
value within a relative tolerance, also allowing the percent-vs-fraction
mismatch (12.4 vs 0.124) that FinQA's gold programs produce.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Word and character multipliers. Chinese units can stack ("百万", "千亿").
_EN_SCALE = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12,
}
_ZH_SCALE = {"十": 1e1, "百": 1e2, "千": 1e3, "万": 1e4, "亿": 1e8}

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Parsed:
    value: float
    percent: bool  # the text carried a percent sign or word


def extract_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (braces may nest), after any thinking block."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return None  # unbalanced


def parse_number(s: str) -> Parsed | None:
    """Parse one number with its sign, percent and scale words; None if there is
    not exactly one number in the string (ambiguous answers get no credit)."""
    t = s.strip()
    t = t.replace("\u2212", "-").replace("\u2013", "-").replace("，", ",")
    t = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", t)  # \text{million}
    t = t.replace("\\%", "%").replace("\\$", "$").replace("\\,", "")
    negative = False
    if re.fullmatch(r"\(.*\)", t):  # (1,234) accounting negative
        negative, t = True, t[1:-1]
    percent = "%" in t or "percent" in t.lower() or "百分" in t
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # thousands separators
    nums = _NUM_RE.findall(t)
    if len(nums) != 1:
        return None
    value = float(nums[0])
    rest = t[t.find(nums[0]) + len(nums[0]):].lower()
    word = re.match(r"\s*([a-z]+)", rest)
    if word and word.group(1) in _EN_SCALE and not percent:
        # bare "m"/"b"/"k" count only when directly attached ("1.2b"), not "1.2 bags"
        w = word.group(1)
        if len(w) > 2 or rest[: len(w)] == w:
            value *= _EN_SCALE[w]
    zh = re.match(r"\s*([十百千万亿]+)", rest)
    if zh:
        for ch in zh.group(1):
            value *= _ZH_SCALE[ch]
    if negative:
        value = -value
    return Parsed(value, percent)


def numeric_match(pred: str, gold: float | str, rel_tol: float = 1e-2, abs_tol: float = 1e-6,
                  allow_percent_scale: bool = True) -> bool:
    """True if ``pred`` (an answer string) equals ``gold`` within tolerance.

    rel_tol 1e-2 accepts the rounding finance answers are written with
    ("12.4%" for 0.12384...). With allow_percent_scale, a prediction may differ
    from gold by exactly a factor of 100 when one side is a percent.
    """
    p = parse_number(pred)
    if p is None:
        return False
    if isinstance(gold, str):
        g = parse_number(gold)
        if g is None:
            return False
        g_val, g_pct = g.value, g.percent
    else:
        g_val, g_pct = float(gold), False
    if not math.isfinite(p.value) or not math.isfinite(g_val):
        return False

    def close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    if close(p.value, g_val):
        return True
    if allow_percent_scale and (p.percent or g_pct or abs(g_val) <= 1.0 or abs(p.value) <= 1.0):
        return close(p.value, g_val * 100) or close(p.value * 100, g_val)
    return False
