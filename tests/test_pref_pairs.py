import itertools
import json
import random

import pandas as pd
import pytest

from b2e.data.pref_pairs import build, held_out, load_samples, match_by_length, pair_prompt
from b2e.reward.score import compute_score


def test_length_matching_is_the_least_total_gap():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 4)
        short = [rng.randint(500, 4000) for _ in range(n)]
        long = [rng.randint(500, 4000) for _ in range(n + rng.randint(0, 3))]
        pairs = match_by_length(short, long)
        assert sorted(i for i, _ in pairs) == list(range(n))
        assert len({j for _, j in pairs}) == n
        best = min((sum(abs(short[i] - long[j]) for i, j in enumerate(p)) for p in itertools.permutations(range(len(long)), n)),
                   default=0)
        assert sum(abs(short[i] - long[j]) for i, j in pairs) == best


def test_match_by_length_wants_the_shorter_list_first():
    with pytest.raises(ValueError):
        match_by_length([1, 2], [3])


def test_pairs_use_finished_answers_once_each_matched_by_length():
    #            right right wrong wrong wrong  cut   stopped-in-thinking  graded
    scores = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    finish = ["stop", "stop", "stop", "stop", "stop", "length", "stop", "stop"]
    tokens = [1000, 3000, 2900, 1100, 5000, 4096, 1000, 2000]
    texts = ["r</think>a"] * 6 + ["no end of thinking", "g</think>a"]
    assert pair_prompt(scores, finish, tokens, texts) == [(0, 3), (1, 2)]


def test_more_right_than_wrong_pairs_the_wrong_one_with_the_nearest_right():
    texts = ["x</think>y"] * 5
    pairs = pair_prompt([1, 1, 1, 1, 0], ["stop"] * 5, [800, 2600, 2000, 3900, 2500], texts)
    assert pairs == [(1, 4)]


def test_no_pairs_without_a_finished_right_and_a_finished_wrong():
    texts = ["x</think>y"] * 4
    assert pair_prompt([1, 1, 1, 1], ["stop"] * 4, [1, 2, 3, 4], texts) == []
    assert pair_prompt([0, 0, 0, 0], ["stop"] * 4, [1, 2, 3, 4], texts) == []
    # the only wrong answer was cut: nothing to reject
    assert pair_prompt([1, 1, 0], ["stop", "stop", "length"], [1, 2, 3], texts[:3]) == []


def test_held_out_prompts_per_source_seeded():
    keys = pd.DataFrame({"data_source": ["finqa"] * 10 + ["tatqa"] * 3, "id": [str(i) for i in range(13)]})
    a, b = held_out(keys, 5, 0), held_out(keys, 5, 0)
    assert a == b and len(a) == 8
    assert sum(ds == "finqa" for ds, _ in a) == 5 and sum(ds == "tatqa" for ds, _ in a) == 3
    assert held_out(keys, 0, 0) == set()


def gt(value):
    return json.dumps({"kind": "number", "value": value, "scale": ""})


def answer(x):
    return f"Some reasoning.\n</think>\n\nSo the answer is \\boxed{{{x}}}."


@pytest.fixture
def parts(tmp_path):
    rows = [  # (source, id, gold, answers, stored scores, finish, tokens)
        ("finqa", "a", 12.0, [12, 12, 13, 14], [1, 1, 0, 0], ["stop"] * 4, [900, 2100, 2000, 1000]),
        # sample 1 was stored as wrong; the verifier now accepts 7.0 for 7: re-scored, it is right
        ("finqa", "b", 7.0, [7, "7.0", 8, 8], [1, 0, 0, 0], ["stop"] * 4, [1500, 1600, 1550, 1700]),
        ("tatqa", "c", 3.0, [3, 4, 4, 4], [1, 0, 0, 0], ["stop", "stop", "length", "stop"], [800, 900, 4096, 820]),
        ("tatqa", "d", 5.0, [5, 5, 5, 5], [1, 1, 1, 1], ["stop"] * 4, [700, 710, 720, 730]),
    ]
    stats = pd.DataFrame([{"data_source": s, "id": i, "ground_truth": gt(g), "scores": sc, "finish": f, "tokens": t}
                          for s, i, g, _, sc, f, t in rows])
    samples = pd.DataFrame([{"data_source": s, "id": i,
                             "texts": [answer(x) if fin == "stop" else "cut mid-thought" for x, fin in zip(a, f, strict=True)]}
                            for s, i, _, a, _, f, _ in rows])
    stats.to_parquet(tmp_path / "stats_0000.parquet", index=False)
    samples.to_parquet(tmp_path / "samples_0000.parquet", index=False)
    prompts = pd.DataFrame({"data_source": [r[0] for r in rows], "extra_info": [{"id": r[1]} for r in rows],
                            "prompt": [[{"role": "user", "content": f"question {r[1]}"}] for r in rows]})
    return tmp_path, prompts


def test_build_pairs_rescored_held_out_and_in_trl_format(parts):
    path, prompts = parts
    train, test, summary = build(load_samples(path), prompts, per_source=1, seed=0)
    both = pd.concat([train, test], ignore_index=True)
    # a: 2 right, 2 wrong -> 2 pairs by length; b: 2 right after re-scoring, 2 wrong -> 2; c: the cut one left out -> 1
    assert both.groupby("id").size().to_dict() == {"a": 2, "b": 2, "c": 1}
    assert summary["scores_changed_by_rescoring"] == 1
    assert set(train["id"]).isdisjoint(test["id"])
    assert summary["sources"]["finqa"]["test_prompts"] == 1 and summary["sources"]["tatqa"]["test_prompts"] == 1
    a = both[both["id"] == "a"].sort_values("chosen_index")
    assert list(zip(a["chosen_index"], a["rejected_index"], strict=True)) == [(0, 3), (1, 2)]
    for r in both.itertuples():
        assert r.prompt == [{"role": "user", "content": f"question {r.id}"}]
        for msgs, want in ((r.chosen, 1.0), (r.rejected, 0.0)):
            assert len(msgs) == 1 and msgs[0]["role"] == "assistant"
            assert msgs[0]["content"].startswith("<think>\n") and "\n</think>\n\n" in msgs[0]["content"]
            assert compute_score(r.data_source, msgs[0]["content"], r.ground_truth, thinking=True) == want
    assert summary["tokens"]["matched_abs_gap_mean"] <= summary["tokens"]["random_abs_gap_mean"]


def test_load_samples_needs_texts_for_every_prompt(parts):
    path, _ = parts
    pd.DataFrame({"data_source": ["finqa"], "id": ["a"], "texts": [[answer(12)] * 4]}).to_parquet(
        path / "samples_0000.parquet", index=False)
    with pytest.raises(SystemExit):
        load_samples(path)


def test_build_stops_when_a_sampled_prompt_is_not_in_the_prompt_files(parts):
    path, prompts = parts
    with pytest.raises(SystemExit):
        build(load_samples(path), prompts[prompts["data_source"] == "finqa"], per_source=1, seed=0)
