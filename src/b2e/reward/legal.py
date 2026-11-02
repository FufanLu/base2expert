"""Verifiers for CAIL2018 judgment prediction: charges, articles, prison term.

Charges score 1 when the answer splits into exactly the gold charge labels, a selective charge
in any of its forms (charge_forms); articles are compared as sets (exact match scores 1, else 0).
The prison term uses CAIL2018's official scoring: with v = |ln(pred + 1) -
ln(gold + 1)| on months, the score is 1.0 if v <= 0.2, 0.8 if <= 0.4,
0.6 if <= 0.6, 0.4 if <= 0.8, 0.2 if <= 1.0, else 0. Death penalty and life
imprisonment score 1 only when predicted exactly. A graded reward tells
"off by one month" from "off by five years", which a 0/1 reward cannot.
"""

from __future__ import annotations

import itertools
import math
import re

_SPLIT = re.compile(r"[、，,；;\s/和及]+")
_SEPS = "、，,；;/和及"
_GROUP = re.compile(r"(\[[^\]]*\])")
_ZH_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def norm_charges(s: str) -> set[str]:
    out = set()
    for part in _SPLIT.split(s.replace("罪名", "").replace("：", ":").split(":")[-1]):
        p = part.strip().strip("《》“”\"'")
        if p.endswith("罪"):
            p = p[:-1]
        if p:
            out.add(p)
    return out


def charge_forms(label: str) -> list[str]:
    """Every way to write a CAIL2018 charge label. CAIL merges a selective charge (选择性罪名)
    into one label and brackets its acts: [走私、贩卖、运输、制造]毒品 is charged as 贩卖毒品,
    贩卖、运输毒品 or all four, so a bracketed group stands for any non-empty subset of its items,
    in the statute's order. 28 % of the CAIL test's charges rows have such a label."""
    options = []
    for part in _GROUP.split(label):
        if part.startswith("[") and part.endswith("]"):
            items = [x for x in part[1:-1].split("、") if x]
            options.append(["、".join(c) for k in range(1, len(items) + 1) for c in itertools.combinations(items, k)])
        elif part:
            options.append([part])
    return ["".join(p) for p in itertools.product(*options)]


def charges_match(pred: str, gold: list[str]) -> bool:
    """True if the answer names exactly the gold charges: it splits into forms of the gold labels
    (each may end in 罪), separated by 、，; and the like, and covers every label. Splitting at 、
    first (norm_charges) cannot work: 、 also separates the acts inside a selective charge."""
    text = re.sub(r"[\s\[\]《》“”\"'。.]", "", pred.replace("罪名", "").replace("：", ":").split(":")[-1])
    forms = sorted({(f, g) for g in gold for f in charge_forms(g)}, key=lambda x: -len(x[0]))

    def split(i: int, found: frozenset[str]) -> frozenset[str] | None:
        while i < len(text) and text[i] in _SEPS:
            i += 1
        if i == len(text):
            return found
        for form, label in forms:
            if text.startswith(form, i):
                j = i + len(form) + (text.startswith("罪", i + len(form)))
                if j == len(text) or text[j] in _SEPS:
                    done = split(j, found | {label})
                    if done is not None:
                        return done
        return None

    return split(0, frozenset()) == set(gold)


def norm_articles(s: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", s)}


def _zh_number(s: str) -> float | None:
    """Small Chinese numerals as used in sentences: 三, 十, 十二, 二十, 两."""
    if not s:
        return None
    if s.isdigit():
        return float(s)
    if "十" in s:
        a, _, b = s.partition("十")
        tens = _ZH_DIGITS.get(a, 1) if a else 1
        ones = _ZH_DIGITS.get(b, 0) if b else 0
        return float(tens * 10 + ones)
    return float(_ZH_DIGITS[s]) if s in _ZH_DIGITS else None


def parse_term(s: str) -> tuple[str, float]:
    """('death'|'life'|'months', value). Accepts '6个月', '一年六个月', '18', '3年'."""
    t = s.replace(" ", "")
    if "死刑" in t:
        return "death", 0.0
    if "无期" in t:
        return "life", 0.0
    months = 0.0
    found = False
    for num, unit in re.findall(r"([\d.]+|[一二两三四五六七八九十]+)(年|个?月)", t):
        v = float(num) if re.fullmatch(r"[\d.]+", num) else _zh_number(num)
        if v is None:
            continue
        months += v * 12 if unit == "年" else v
        found = True
    if not found:
        m = re.search(r"[\d.]+", t)
        if m:
            months, found = float(m.group(0)), True
    return ("months", months) if found else ("none", 0.0)


def term_score(pred: str, gold_months: float, death: bool = False, life: bool = False) -> float:
    kind, v = parse_term(pred)
    if death or life:
        return 1.0 if kind == ("death" if death else "life") else 0.0
    if kind != "months":
        return 0.0
    d = abs(math.log(v + 1) - math.log(gold_months + 1))
    for limit, score in ((0.2, 1.0), (0.4, 0.8), (0.6, 0.6), (0.8, 0.4), (1.0, 0.2)):
        if d <= limit:
            return score
    return 0.0
