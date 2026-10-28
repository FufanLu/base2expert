import json

from b2e.data.finance import md_table
from b2e.reward.score import compute_score


def gt(**kw):
    return json.dumps(kw)


def test_finqa_fraction_gold_accepts_percent_answer():
    assert compute_score("finqa", "... so \\boxed{14.46\\%}", gt(kind="number", value=0.14464, scale="")) == 1.0
    assert compute_score("finqa", "\\boxed{0.1446}", gt(kind="number", value=0.14464, scale="")) == 1.0
    assert compute_score("finqa", "\\boxed{15.5\\%}", gt(kind="number", value=0.14464, scale="")) == 0.0


def test_tatqa_scale_word():
    g = gt(kind="number", value=1.9, scale="million")
    assert compute_score("tatqa", "\\boxed{1.9}", g) == 1.0
    assert compute_score("tatqa", "\\boxed{1.9 million}", g) == 1.0
    assert compute_score("tatqa", "\\boxed{1,900,000}", g) == 1.0
    assert compute_score("tatqa", "\\boxed{19 million}", g) == 0.0


def test_no_box_scores_zero():
    assert compute_score("finqa", "The answer is 94.", gt(kind="number", value=94.0, scale="")) == 0.0


def test_yes_no():
    assert compute_score("finqa", "\\boxed{Yes}", gt(kind="text", value="yes")) == 1.0
    assert compute_score("finqa", "\\boxed{no}", gt(kind="text", value="yes")) == 0.0


def test_md_table():
    assert md_table([["", "2019"], ["revenue", "$ 5"]]) == "|  | 2019 |\n|---|---|\n| revenue | $ 5 |"


def test_law_and_choice_dispatch():
    assert compute_score("cail2018_charges", "\\boxed{盗窃罪}", gt(kind="charges", value=["盗窃"])) == 1.0
    assert compute_score("cail2018_articles", "\\boxed{264}", gt(kind="articles", value=[264])) == 1.0
    assert compute_score("cail2018_term", "\\boxed{一年}", gt(kind="term", months=18.0)) == 0.8
    assert compute_score("medqa", "\\boxed{B}", gt(kind="choice", value="B")) == 1.0
    assert compute_score("cmexam", "\\boxed{AC}", gt(kind="choice", value="AC")) == 1.0


def test_answer_cut_off_inside_thinking_scores_zero():
    # F#53: a \boxed{} written while reasoning is not a final answer
    g = gt(kind="number", value=0.0523, scale="")
    cut = "the ratio is 1555/29734, so \\boxed{5.23\\%}. Let me double-check the table"
    assert compute_score("finqa", cut, g, thinking=True) == 0.0
    assert compute_score("finqa", cut + "\n</think>\n\n\\boxed{5.23\\%}", g, thinking=True) == 1.0
    # thinking off: no block, the whole text is the answer; one it opens itself must be closed
    assert compute_score("finqa", "\\boxed{5.23\\%}", g) == 1.0
    assert compute_score("finqa", "<think>\n" + cut, g) == 0.0
    g_if = json.dumps({"kind": "ifeval", "prompt": "p",
                       "constraints": [{"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}]})
    assert compute_score("if_multi", "no commas here", g_if) == 1.0
    assert compute_score("if_multi", "no commas here", g_if, thinking=True) == 0.0
    assert compute_score("if_multi", "x\n</think>\n\nno commas here", g_if, thinking=True) == 1.0
