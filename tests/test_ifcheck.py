import json

from b2e.reward.ifcheck import check, if_score
from b2e.reward.score import compute_score


def gt(ids, kwargs, prompt="p"):
    return {"kind": "ifeval", "prompt": prompt, "constraints": [{"instruction_id": ids, "kwargs": kwargs}]}


def test_constraints_checked_one_by_one():
    g = gt(["punctuation:no_comma", "keywords:existence", "last_word:last_word_answer"],
           [None, {"keywords": ["ledger"]}, {"last_word": "brief"}])
    assert check("The ledger is balanced and brief", g) == [True, True, True]
    assert check("The ledger, balanced. Done", g) == [False, True, False]
    assert if_score("The ledger, balanced. Done", g) == 1 / 3


def test_only_the_answer_counts():
    g = gt(["punctuation:no_comma"], [None])
    assert if_score("<think>a, b, c</think>\n\nNo commas here", g) == 1.0
    assert if_score("<think>fine</think>\n\nOne, two", g) == 0.0


def test_empty_answer_fails_and_unset_kwargs_are_dropped():
    g = gt(["change_case:english_lowercase"], [{"unused": None}])
    assert if_score("   ", g) == 0.0
    assert if_score("all lower case", g) == 1.0


def test_routed_from_compute_score_without_boxed():
    g = gt(["punctuation:no_comma"], [None])
    assert compute_score("if_multi", "no commas", json.dumps(g)) == 1.0


def test_if_rows_are_validated():
    from b2e.data.if_data import row, valid

    assert valid([{"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}]) is None
    assert valid([{"instruction_id": ["no:such_check"], "kwargs": [None]}]) == "unknown:no:such_check"
    r = row("if_multi", "k1", "Write a poem.", [{"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}], "train")
    assert r["extra_info"]["n_constraints"] == 1
    assert compute_score("if_multi", "a poem without commas", r["reward_model"]["ground_truth"]) == 1.0


def test_constraint_sentences_are_left_out_of_decontamination():
    from b2e.data.if_data import user_query

    c = "First repeat the request word for word without change.\tAnswer in lowercase."
    q = user_query("Explain ballistics. First repeat the request word for word without change. Answer in lowercase.", c)
    assert "repeat" not in q and "lowercase" not in q and "ballistics" in q
