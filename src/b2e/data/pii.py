"""Scrub personal information from Chinese and English corpus text.

Replaces e-mail addresses, IPv4 addresses, mainland-China mobile numbers,
18-digit resident ID numbers and bank-card numbers with placeholders. Long
digit runs are common in finance text (amounts, account codes, report
numbers), so the two numeric kinds are replaced only when their own check
digit is valid: the ISO 7064 MOD 11-2 checksum for ID numbers and the Luhn
checksum for card numbers. A random 18-digit amount passes the ID check
about 1 time in 11 and a random card-length number the Luhn check 1 time in
10, so some false positives remain; they cost a number, not a document.
"""

from __future__ import annotations

import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
CN_MOBILE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
CN_ID = re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")

_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_ID_CHECK = "10X98765432"


def valid_cn_id(s: str) -> bool:
    if len(s) != 18 or not s[:17].isdigit():
        return False
    total = sum(int(c) * w for c, w in zip(s[:17], _ID_WEIGHTS, strict=True))
    return _ID_CHECK[total % 11] == s[17].upper()


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def scrub(text: str) -> tuple[str, dict[str, int]]:
    counts = {"email": 0, "ip": 0, "mobile": 0, "id": 0, "card": 0}

    def sub(pattern: re.Pattern, kind: str, repl: str, check=None):
        nonlocal text

        def f(m: re.Match) -> str:
            raw = m.group(0)
            if check is not None and not check(raw):
                return raw
            counts[kind] += 1
            return repl

        text = pattern.sub(f, text)

    sub(EMAIL, "email", "<EMAIL>")
    sub(IPV4, "ip", "<IP>")
    sub(CN_ID, "id", "<ID_NUMBER>", valid_cn_id)
    sub(CARD, "card", "<CARD_NUMBER>", lambda s: luhn_ok(re.sub(r"[ -]", "", s)))
    sub(CN_MOBILE, "mobile", "<PHONE>")
    return text, counts
