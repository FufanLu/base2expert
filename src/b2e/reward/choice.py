"""Multiple-choice verifier: the boxed answer's option letters vs the gold set."""

from __future__ import annotations

import re


def letters(s: str, allowed: str = "ABCDEFGHIJ") -> set[str]:  # MMLU-Pro has up to 10 options
    # "\boxed{B}", "B", "(B)", "B. Cross-linking", "答案：AB"
    t = s.strip()
    m = re.match(rf"^[\(\[]?([{allowed}](?:\s*[,，、和]?\s*[{allowed}])*)[\)\]]?(?:[.．、:：\s]|$)", t.upper())
    if m:
        return set(re.findall(rf"[{allowed}]", m.group(1)))
    return set()


def choice_match(answer: str, gold: str) -> bool:
    got = letters(answer)
    return bool(got) and got == set(gold.upper())
