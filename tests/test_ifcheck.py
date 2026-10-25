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


def test_blank_line_after_think_is_not_a_paragraph():
    # F#32: "</think>\n\n" left the answer starting with an empty paragraph, so paragraph 1 was ""
    g = gt(["length_constraints:nth_paragraph_first_word"], [{"num_paragraphs": 2, "nth_paragraph": 1,
                                                              "first_word": "ledger"}])
    answer = "Ledger entries come first.\n\nThen the totals."
    assert if_score(answer, g) == 1.0
    assert if_score("<think>plan</think>\n\n" + answer, g) == 1.0


def test_checks_are_deterministic_and_leave_random_alone():
    import random

    # a letter_frequency item with a non-letter is given a drawn letter (as two IFEval items are)
    g = gt(["keywords:letter_frequency", "change_case:english_capital"],
           [{"let_relation": "at least", "letter": "!", "let_frequency": 3}, None])
    text = "WOW! A NEW ICE CREAM! TRY IT TODAY, FRIENDS! IT IS GENTLE ON THE STOMACH."
    random.seed(123)
    before = random.random()
    random.seed(123)
    results = [tuple(check(text, g)) for _ in range(20)]
    assert len(set(results)) == 1
    assert random.random() == before


def test_rows_with_randomized_kwargs_are_rejected():
    from b2e.data.if_data import valid

    para = [{"instruction_id": ["length_constraints:nth_paragraph_first_word"],
             "kwargs": [{"num_paragraphs": 7, "nth_paragraph": 8, "first_word": "complaint"}]}]
    assert valid(para) == "randomized:length_constraints:nth_paragraph_first_word"
    assert valid(para, fixed_args=False) is None
    ok = [{"instruction_id": ["length_constraints:nth_paragraph_first_word"],
           "kwargs": [{"num_paragraphs": 7, "nth_paragraph": 3, "first_word": "complaint"}]}]
    assert valid(ok) is None


def test_repeated_dataset_keys_get_unique_ids():
    from b2e.data.if_data import unique_ids

    assert unique_ids(["a", "hard_coded_1", "b", "hard_coded_1", "hard_coded_1"]) == [
        "a", "hard_coded_1", "b", "hard_coded_1#1", "hard_coded_1#2"]
