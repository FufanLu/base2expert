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
