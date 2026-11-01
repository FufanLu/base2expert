import pandas as pd

from b2e.data.rl_candidates import parse_args, split_val, unused


def test_drops_the_prompts_sft_sampled_matching_source_and_id():
    # MedQA and CMExam both number from 0: CMExam's "0" stays when only MedQA's "0" was used
    pool = pd.DataFrame({"data_source": ["medqa", "medqa", "cmexam", "cmexam"],
                         "extra_info": [{"id": "0"}, {"id": "1"}, {"id": "0"}, {"id": "1"}]})
    used = pd.DataFrame({"data_source": ["medqa", "cmexam"], "id": ["0", "1"]})
    left = unused(pool, used)
    assert list(zip(left["data_source"], [e["id"] for e in left["extra_info"]], strict=True)) == [
        ("medqa", "1"), ("cmexam", "0")]


def test_exclude_and_train_lists_in_either_order():
    for argv in (["--exclude", "a", "b", "--train", "c", "d"], ["--train", "c", "d", "--exclude", "a", "b"]):
        args = parse_args(["--out", "o", *argv])
        assert args.exclude == ["a", "b"] and args.train == ["c", "d"]


def test_val_rows_are_left_out_of_the_unused_rows():
    left = pd.DataFrame({"data_source": ["medqa"] * 5 + ["cmexam"] * 2,
                         "extra_info": [{"id": str(i)} for i in range(5)] + [{"id": "0"}, {"id": "1"}]})

    def keys(d):
        return set(zip(d["data_source"], [e["id"] for e in d["extra_info"]], strict=True))

    rest, val = split_val(left, 3, 0)
    assert val["data_source"].value_counts().to_dict() == {"medqa": 3, "cmexam": 2}
    assert not keys(rest) & keys(val) and keys(rest) | keys(val) == keys(left)
    assert split_val(left, 3, 0)[1].equals(val)  # seeded: a rerun sets the same prompts aside
    assert parse_args(["--exclude", "a", "--train", "c", "--out", "o"]).val_per_source == 0
