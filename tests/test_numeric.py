import pytest

from b2e.reward.numeric import extract_boxed, numeric_match, parse_number


@pytest.mark.parametrize(
    "text, expected",
    [
        ("so the answer is \\boxed{12.4\\%}", "12.4\\%"),
        ("\\boxed{1} then \\boxed{2}", "2"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("<think>maybe \\boxed{3}</think>final \\boxed{4}", "4"),
        ("<think>\\boxed{3}</think>no box here", None),
        ("no box", None),
        ("\\boxed{unbalanced", None),
    ],
)
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    "s, value, percent",
    [
        ("1,234.5", 1234.5, False),
        ("$1,234.5", 1234.5, False),
        ("(1,234.5)", -1234.5, False),
        ("12.4%", 12.4, True),
        ("12.4 percent", 12.4, True),
        ("\u22123.1", -3.1, False),
        ("1.2 billion", 1.2e9, False),
        ("1.2b", 1.2e9, False),
        ("12亿", 12e8, False),
        ("3.5万元", 3.5e4, False),
        ("2.5百万", 2.5e6, False),
        ("\\text{1.5 million}", 1.5e6, False),
    ],
)
def test_parse_number(s, value, percent):
    p = parse_number(s)
    assert p is not None
    assert p.value == pytest.approx(value)
    assert p.percent is percent


@pytest.mark.parametrize("s", ["", "about 3 or 4", "n/a", "between 1 and 2"])
def test_parse_number_rejects_ambiguous(s):
    assert parse_number(s) is None


def test_word_after_number_is_not_a_scale():
    # "3 bags" must not become 3e9
    assert parse_number("3 bags").value == 3


@pytest.mark.parametrize(
    "pred, gold, ok",
    [
        ("12.4%", 0.124, True),          # FinQA gold programs give fractions
        ("0.124", "12.4%", True),
        ("12.38%", 0.1238, True),
        ("12.4%", 0.1238, True),         # written rounding within 1%
        ("13%", 0.124, False),
        ("1,234", 1234, True),
        ("(5)", -5, True),
        ("5", -5, False),
        ("1.2 billion", 1.2e9, True),
        ("1200 million", 1.2e9, True),
        ("12亿", 1.2e9, True),
        ("about 3 or 4", 3, False),
        ("120", 1.2, False),             # x100 only allowed when a side is a percent or a fraction
        ("120%", 1.2, True),
    ],
)
def test_numeric_match(pred, gold, ok):
    assert numeric_match(pred, gold) is ok
