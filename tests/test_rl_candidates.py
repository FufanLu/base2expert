import pandas as pd

from b2e.data.rl_candidates import parse_args, unused


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
