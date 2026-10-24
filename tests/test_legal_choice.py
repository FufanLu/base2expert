from b2e.reward.choice import choice_match, letters
from b2e.reward.legal import norm_articles, norm_charges, parse_term, term_score


def test_charges_normalized():
    assert norm_charges("盗窃罪、诈骗罪") == {"盗窃", "诈骗"}
    assert norm_charges("罪名：故意伤害") == {"故意伤害"}


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
