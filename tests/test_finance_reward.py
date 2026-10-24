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
