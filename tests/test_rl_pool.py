import pandas as pd

from b2e.data.rl_pool import parse_args, select, wrong_majority


def test_keeps_only_prompts_the_policy_sometimes_solves():
    pool = pd.DataFrame({"data_source": ["finqa"] * 4, "extra_info": [{"id": str(i)} for i in range(4)]})
    rates = pd.DataFrame({"data_source": ["finqa"] * 4, "id": ["0", "1", "2", "3"], "pass_rate": [0.0, 0.125, 0.875, 1.0]})
    kept = select(pool, rates, 0.1, 0.9)
    assert [e["id"] for e in kept["extra_info"]] == ["1", "2"]


def test_train_and_val_lists_in_either_order():
    # F#48: with the train files positional, `--val a b c d` swallowed them all
    for argv in (["--val", "a", "b", "--train", "c", "d"], ["--train", "c", "d", "--val", "a", "b"]):
        args = parse_args(["--passrate", "p.parquet", "--out", "o", *argv])
        assert args.val == ["a", "b"] and args.train == ["c", "d"]


def test_unequal_keeps_graded_groups_with_any_spread():
    # CAIL term's reward is graded: a group can have no sample at 1.0 and still unequal scores
    pool = pd.DataFrame({"data_source": ["cail2018_term"] * 4, "extra_info": [{"id": str(i)} for i in range(4)]})
    scores = [[0.0] * 8, [0.8] * 8, [0.8] * 7 + [0.6], [1.0] * 8]
    rates = pd.DataFrame({"data_source": ["cail2018_term"] * 4, "id": ["0", "1", "2", "3"],
                          "pass_rate": [0.0, 0.0, 0.0, 1.0], "scores": scores})
    assert [e["id"] for e in select(pool, rates, 0.1, 0.9, unequal=True)["extra_info"]] == ["2"]
    assert select(pool, rates, 0.1, 0.9).empty


def test_unequal_is_the_pass_rate_range_for_a_0_1_reward():
    k = 8
    ids = [str(r) for r in range(k + 1)]
    pool = pd.DataFrame({"data_source": ["finqa"] * (k + 1), "extra_info": [{"id": i} for i in ids]})
    rates = pd.DataFrame({"data_source": ["finqa"] * (k + 1), "id": ids, "pass_rate": [r / k for r in range(k + 1)],
                          "scores": [[1.0] * r + [0.0] * (k - r) for r in range(k + 1)]})
    by_rate = [e["id"] for e in select(pool, rates, 0.1, 0.9)["extra_info"]]
    by_spread = [e["id"] for e in select(pool, rates, 0.1, 0.9, unequal=True)["extra_info"]]
    assert by_rate == by_spread == ids[1:k]


def test_unequal_flag():
    args = parse_args(["--passrate", "p.parquet", "--out", "o", "--val", "a", "--train", "c", "--unequal"])
    assert args.unequal and not parse_args(["--passrate", "p", "--out", "o", "--val", "a", "--train", "c"]).unequal


def test_wrong_majority_counts_finished_answers_the_verifier_calls_the_same():
    # pilot r0-276: gold 0.0636, S-fin wrote $0.06 five times (two spellings) and the gold once
    answers = ["\\$0.06", None, None, "0.0636", "\\$0.06", "0.06", "0.06", "0.06"]
    scores = [0, 0, 0, 1, 0, 0, 0, 0]
    finish = ["stop", "length", "length", "stop", "stop", "stop", "stop", "stop"]
    assert wrong_majority(answers, scores, finish)
    # pilot r0-91: gold -1.21 once, -1.19 twice, four cut: 2 > 1
    assert wrong_majority(["", "", "-1.21\\%", "-1.19\\%", "-1.19\\%", "", "", ""], [0, 0, 1, 0, 0, 0, 0, 0],
                          ["length", "length", "stop", "stop", "stop", "length", "length", "length"])


def test_wrong_majority_needs_more_votes_than_the_gold():
    stop = ["stop"] * 8
    # 3 right, 3 of one wrong answer, 2 of another: a tie is kept
    assert not wrong_majority(["5"] * 3 + ["7"] * 3 + ["9"] * 2, [1] * 3 + [0] * 5, stop)
    # spread-out wrong answers never outvote 2 right ones
    assert not wrong_majority(["5", "5", "1", "2", "3", "4", "6", "8"], [1, 1, 0, 0, 0, 0, 0, 0], stop)
    # cut samples do not vote, even with a number boxed in their thinking
    assert not wrong_majority(["5", "7", "7", "7", "", "", "", ""], [1, 0, 0, 0, 0, 0, 0, 0],
                              ["stop", "length", "length", "length", "length", "length", "length", "length"])


def test_majority_drops_outvoted_prompts_after_the_range():
    pool = pd.DataFrame({"data_source": ["rsi_fin"] * 3, "extra_info": [{"id": str(i)} for i in range(3)]})
    stop = ["stop"] * 8
    rates = pd.DataFrame({"data_source": ["rsi_fin"] * 3, "id": ["0", "1", "2"], "pass_rate": [0.25, 0.75, 1.0],
                          "scores": [[1, 1, 0, 0, 0, 0, 0, 0], [1] * 6 + [0, 0], [1] * 8],
                          "answers": [["24", "24"] + ["48"] * 6, ["3"] * 6 + ["4", "5"], ["1"] * 8],
                          "finish": [stop, stop, stop]})
    assert [e["id"] for e in select(pool, rates, 0.1, 0.9)["extra_info"]] == ["0", "1"]
    assert [e["id"] for e in select(pool, rates, 0.1, 0.9, majority=True)["extra_info"]] == ["1"]
