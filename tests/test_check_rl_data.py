import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from check_rl_data import gold_text  # noqa: E402

from b2e.reward.score import compute_score  # noqa: E402

GOLDS = [
    {"kind": "number", "value": 1e-05},
    {"kind": "choice", "value": "B"},
    {"kind": "charges", "value": ["抢劫", "盗窃"]},
    {"kind": "charges", "value": ["[走私、贩卖、运输、制造]毒品", "盗窃"]},
    {"kind": "articles", "value": [25, 264]},
    {"kind": "term", "months": 18.0, "death": False, "life": False},
    {"kind": "term", "months": 0.0, "death": True, "life": False},
    {"kind": "term", "months": 0.0, "death": False, "life": True},
]


@pytest.mark.parametrize("gt", GOLDS, ids=lambda g: g["kind"])
def test_gold_text_is_an_answer_the_verifier_pays(gt):
    # the check writes the gold answer after the thinking, as a model does (11564356: CAIL's kinds
    # had no "value" or came out as "['盗窃']")
    answer = f"<think>\nok\n</think>\n\n\\boxed{{{gold_text(gt)}}}"
    assert compute_score("x", answer, json.dumps(gt, ensure_ascii=False), thinking=True) == 1.0


def test_gold_text_writes_cail_answers_plainly():
    assert gold_text({"kind": "charges", "value": ["抢劫", "盗窃"]}) == "抢劫、盗窃"
    assert gold_text({"kind": "term", "months": 18.0}) == "18个月"
