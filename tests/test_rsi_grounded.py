import json

import pandas as pd

from b2e.data.finance import finqa_rows
from b2e.data.rl_pool import mixed_finished, select
from b2e.rsi.docs import build, split_prompt
from b2e.rsi.gate import gate_messages, verdict
from b2e.rsi.propose import filter_grounded, grounded_messages, prior_problems, same_value

EXCERPT_TEXT = "Net revenues were $4,981 million in 2018 and $5,260 million in 2019."
TABLE = [["", "2019", "2018"], ["net revenues", "5,260", "4,981"], ["cost of revenues", "3,031", "2,940"]]


def finqa_example(i: int, question: str, ans) -> dict:
    return {"id": f"ABC/2019/page_7.pdf-{i}", "pre_text": [EXCERPT_TEXT], "table": TABLE, "post_text": [],
            "qa": {"question": question, "exe_ans": ans, "program": ""}}


def docs_from(tmp_path, examples: list[dict]) -> list[dict]:
    path = tmp_path / "train.json"
    path.write_text(json.dumps(examples))
    return build(finqa_rows(path, "train"))


def proposal(question: str, program: str) -> str:
    return f"<question>\n{question}\n</question>\n<program>\n{program}\n</program>"


def test_split_prompt_gives_back_the_excerpt_and_the_question(tmp_path):
    (doc,) = docs_from(tmp_path, [finqa_example(1, "What was the change in net revenues?", 279.0)])
    assert doc["excerpt"].startswith(EXCERPT_TEXT) and "| net revenues | 5,260 | 4,981 |" in doc["excerpt"]
    assert doc["questions"] == ["What was the change in net revenues?"]
    assert doc["source"] == "finqa" and doc["answers"] == [279.0]


def test_questions_on_one_excerpt_make_one_doc_and_only_numbers_are_answers(tmp_path):
    docs = docs_from(tmp_path, [finqa_example(1, "What was the change in net revenues?", 279.0),
                                finqa_example(2, "Did net revenues rise?", "yes")])
    assert len(docs) == 1
    assert docs[0]["questions"] == ["What was the change in net revenues?", "Did net revenues rise?"]
    assert docs[0]["answers"] == [279.0] and docs[0]["row_ids"] == ["ABC/2019/page_7.pdf-1", "ABC/2019/page_7.pdf-2"]


def test_the_proposer_sees_the_excerpt_and_the_questions_already_asked(tmp_path):
    (doc,) = docs_from(tmp_path, [finqa_example(1, "What was the change in net revenues?", 279.0)])
    content = grounded_messages(doc, "a margin")[0]["content"]
    assert doc["excerpt"] in content and "- What was the change in net revenues?" in content
    assert "preferably a margin if the excerpt supports one" in content


def test_grounded_filter_keeps_a_new_question_and_drops_repeats(tmp_path):
    (doc,) = docs_from(tmp_path, [finqa_example(1, "What was the change in net revenues?", 279.0)])
    margin = "By how many percentage points did the gross margin change from 2018 to 2019?"
    margin_prog = "m19 = (5260 - 3031) / 5260 * 100\nm18 = (4981 - 2940) / 4981 * 100\nprint(m19 - m18)"
    texts = [
        proposal(margin, margin_prog),                                        # kept
        proposal("What was the increase in net revenues from 2018 to 2019?", "print(5260 - 4981)"),  # the dataset's
        proposal("What was the gross margin change in points?", margin_prog),  # same answer as the kept one
        proposal("What would revenues be at 6,000?", "print(6100 - 4981)"),    # 6100 is not in the excerpt
        "no tags",
    ]
    outcomes: list[str] = []
    rows, stats = filter_grounded(texts, [doc] * 5, round_idx=1, outcomes=outcomes)
    assert outcomes == ["kept", "asked", "duplicate", "ungrounded", "parse"]
    (row,) = rows
    assert row["extra_info"]["doc_id"] == doc["doc_id"] and row["extra_info"]["doc_source"] == "finqa"
    excerpt, question = split_prompt(row["prompt"][0]["content"])
    assert excerpt == doc["excerpt"] and question == margin
    # the unrounded result, 1.40072..., not 1.4
    assert json.loads(row["reward_model"]["ground_truth"])["value"] == (5260 - 3031) / 5260 * 100 - (4981 - 2940) / 4981 * 100


def test_an_answer_in_percent_is_the_same_answer():
    assert same_value(0.25, 25.0) and same_value(25.0, 0.25) and same_value(-1.21, -1.21)
    assert not same_value(0.25, 0.26) and not same_value(25.0, -25.0)


def test_earlier_rounds_count_and_other_excerpts_do_not(tmp_path):
    (doc,) = docs_from(tmp_path, [finqa_example(1, "What was the change in net revenues?", 279.0)])
    other = {**doc, "doc_id": "another-excerpt", "answers": []}
    cost = proposal("What was the percentage change in cost of revenues from 2018 to 2019?",
                    "print((3031 - 2940) / 2940 * 100)")
    first, _ = filter_grounded([cost], [doc], round_idx=1)
    prior = prior_problems(first)
    _, stats = filter_grounded([cost, cost], [doc, other], round_idx=2, prior=prior)
    assert stats == {"duplicate": 1, "kept": 1}  # same excerpt as round 1: repeat; another excerpt: new


def test_cut_proposals_are_unclosed_in_thinking_mode(tmp_path):
    (doc,) = docs_from(tmp_path, [finqa_example(1, "What was the change in net revenues?", 279.0)])
    _, stats = filter_grounded(["<think>still planning"], [doc], round_idx=1, thinking=True)
    assert stats == {"unclosed": 1}


def test_gate_verdicts():
    gt = json.dumps({"kind": "number", "value": 1.40072, "scale": ""})
    assert verdict("plan</think>The margin rose by \\boxed{1.40} points.", gt) == "valid"
    assert verdict("plan</think>\\boxed{1.40%}", gt) == "valid"
    assert verdict("plan</think>\\boxed{2.1}", gt) == "disagree"
    assert verdict("plan</think>\\boxed{INVALID} The 2017 figure is not given.", gt) == "invalid"
    assert verdict("plan</think>I cannot tell.", gt) == "no_box"
    assert verdict("plan \\boxed{1.40} still checking", gt) == "unclosed"  # a box inside thinking is a draft


def test_gate_prompt_keeps_the_problem_and_adds_the_check(tmp_path):
    (doc,) = docs_from(tmp_path, [finqa_example(1, "What was the change in net revenues?", 279.0)])
    margin_prog = "m19 = (5260 - 3031) / 5260 * 100\nm18 = (4981 - 2940) / 4981 * 100\nprint(m19 - m18)"
    rows, _ = filter_grounded([proposal("By how many points did the gross margin change?", margin_prog)], [doc], 1)
    content = gate_messages(rows[0])[0]["content"]
    assert content.startswith(doc["excerpt"] + "\n\nQuestion: By how many points did the gross margin change?")
    assert "\\boxed{INVALID}" in content


def test_finished_samples_decide_mixed():
    assert mixed_finished([1, 0, 0], ["stop", "stop", "length"])
    assert not mixed_finished([1, 1, 0], ["stop", "stop", "length"])  # right whenever it finished: F#89
    assert not mixed_finished([0, 0, 0], ["stop", "length", "length"])
    pool = pd.DataFrame({"data_source": ["rsi_fin"] * 2, "extra_info": [{"id": "0"}, {"id": "1"}]})
    rates = pd.DataFrame({"data_source": ["rsi_fin"] * 2, "id": ["0", "1"], "pass_rate": [0.5, 0.5],
                          "scores": [[1, 1, 0, 0], [1, 0, 1, 0]],
                          "finish": [["stop", "stop", "length", "length"], ["stop"] * 4]})
    assert [e["id"] for e in select(pool, rates, 0.1, 0.9)["extra_info"]] == ["0", "1"]
    assert [e["id"] for e in select(pool, rates, 0.1, 0.9, finished=True)["extra_info"]] == ["1"]
