from b2e.reward.rubric import (
    f1_balanced,
    grader_messages,
    judge_pairs,
    parse_verdict,
    physician_pairs,
    rubric_score,
)


def test_verdict_is_read_after_the_thinking():
    assert parse_verdict('<think>maybe "criteria_met": false</think>{"explanation": "ok", "criteria_met": true}')
    assert parse_verdict('{"explanation": "no", "criteria_met": False}') is False
    assert parse_verdict("thinking that never closed") is None
    assert parse_verdict('</think>I think it is met.') is None


def test_score_counts_mistakes_against_the_reply():
    # +5 met, +5 not met, -3 mistake made: (5 - 3) / 10
    assert rubric_score([(5, True), (5, False), (-3, True)]) == 0.2
    assert rubric_score([(5, False), (-3, True)]) == 0.0  # clipped at 0
    assert rubric_score([(-3, True)]) is None  # nothing to earn


def test_balanced_f1_weighs_both_classes():
    pairs = [(True, True)] * 8 + [(False, True)] * 2 + [(False, False)] * 1 + [(True, False)] * 1
    m = f1_balanced(pairs)
    assert round(m["f1_met"], 4) == round(2 * (8 / 9) * (8 / 10) / (8 / 9 + 8 / 10), 4)
    assert round(m["f1_not_met"], 4) == round(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2), 4)
    assert m["f1_balanced"] == (m["f1_met"] + m["f1_not_met"]) / 2 and m["pairs"] == 12


def test_pairs_with_physicians():
    rows = [{"binary_labels": [True, False, True]}, {"binary_labels": [False, False]}]
    assert len(physician_pairs(rows)) == 3 * 2 + 2 * 1  # ordered pairs of different physicians
    assert judge_pairs(rows, [True, None]) == [(True, True), (True, False), (True, True)]


def test_the_grader_sees_the_conversation_and_the_item():
    content = grader_messages([{"role": "user", "content": "I have a headache"},
                               {"role": "assistant", "content": "Drink water."}], "Asks about red-flag symptoms")[0]["content"]
    assert "user: I have a headache" in content and "assistant: Drink water." in content
    assert "<rubric_item>\nAsks about red-flag symptoms\n</rubric_item>" in content
    assert '"criteria_met": true or false' in content
