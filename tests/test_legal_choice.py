import pytest

from b2e.reward.choice import choice_match, letters
from b2e.reward.legal import charge_forms, charges_match, norm_articles, norm_charges, parse_term, term_score

DRUGS = "[走私、贩卖、运输、制造]毒品"  # the most frequent CAIL label with a bracketed group


def test_charges_normalized():
    assert norm_charges("盗窃罪、诈骗罪") == {"盗窃", "诈骗"}
    assert norm_charges("罪名：故意伤害") == {"故意伤害"}


def test_charge_forms_expand_a_selective_charge_in_the_statutes_order():
    assert charge_forms("盗窃") == ["盗窃"]
    forms = charge_forms(DRUGS)
    assert len(forms) == 15 and {"贩卖毒品", "贩卖、运输毒品", "走私、贩卖、运输、制造毒品"} <= set(forms)
    assert "运输、贩卖毒品" not in forms
    two = charge_forms("[掩饰、隐瞒][犯罪所得、犯罪所得收益]")
    assert len(two) == 9 and {"掩饰犯罪所得", "掩饰、隐瞒犯罪所得、犯罪所得收益"} <= set(two)


@pytest.mark.parametrize("pred, gold, ok", [
    ("盗窃罪、诈骗罪", ["盗窃", "诈骗"], True),
    ("诈骗，盗窃", ["盗窃", "诈骗"], True),
    ("盗窃", ["盗窃", "诈骗"], False),
    ("盗窃、抢劫", ["盗窃"], False),
    ("贩卖毒品", [DRUGS], True),
    ("贩卖、运输毒品罪", [DRUGS], True),
    (DRUGS, [DRUGS], True),  # the label itself, brackets and all
    ("走私、贩卖、运输、制造毒品罪", [DRUGS], True),
    ("毒品", [DRUGS], False),
    ("贩卖毒品罪、盗窃罪", [DRUGS, "盗窃"], True),
    ("贩卖毒品", [DRUGS, "盗窃"], False),
    ("包庇", ["[窝藏、包庇]"], True),
    ("罪名：容留卖淫", ["[引诱、容留、介绍]卖淫"], True),
    ("被告人犯盗窃罪", ["盗窃"], False),
    ("", ["盗窃"], False),
])
def test_charges_match(pred, gold, ok):
    assert charges_match(pred, gold) is ok
    # the old comparison, norm_charges against the gold names, could never pay a bracketed label
    if any("[" in g for g in gold) and ok:
        assert norm_charges(pred) != set(gold)


def test_articles():
    assert norm_articles("第264条、第67条") == {264, 67}


def test_parse_term():
    assert parse_term("有期徒刑一年六个月") == ("months", 18.0)
    assert parse_term("6个月") == ("months", 6.0)
    assert parse_term("3年") == ("months", 36.0)
    assert parse_term("十二个月") == ("months", 12.0)
    assert parse_term("无期徒刑")[0] == "life"
    assert parse_term("18") == ("months", 18.0)


def test_term_score_is_graded():
    assert term_score("18个月", 18) == 1.0
    assert term_score("一年", 18) == 0.8      # |ln 13 - ln 19| = 0.38
    assert term_score("6个月", 36) == 0.0     # |ln 7 - ln 37| = 1.67
    assert term_score("死刑", 0, death=True) == 1.0
    assert term_score("10年", 0, life=True) == 0.0


def test_choice():
    assert letters("B") == {"B"}
    assert letters("(D) Cross-linking of DNA") == {"D"}
    assert letters("AB") == {"A", "B"}
    assert choice_match("C", "C")
    assert not choice_match("C", "D")
    assert not choice_match("Cross-linking", "D")


def test_ten_option_letters():
    # MMLU-Pro options run A-J
    assert choice_match("J", "J")
    assert letters("I. none of the above") == {"I"}
