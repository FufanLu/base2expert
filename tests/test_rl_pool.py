import pandas as pd

from b2e.data.rl_pool import parse_args, select


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
