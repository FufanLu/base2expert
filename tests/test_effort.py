import json

import pytest

import b2e.reward.budget as budget_mod
from b2e.data.effort_pool import LOW_EFFORT, budget, build, low_row

ROW = {"data_source": "finqa", "prompt": [{"role": "user", "content": "excerpt\n\nQuestion: q"}],
       "reward_model": {"style": "rule", "ground_truth": json.dumps({"kind": "number", "value": 2.0, "scale": ""})},
       "extra_info": {"id": "a-1", "split": "train"}}


def test_budget_is_the_median_of_finished_samples():
    assert budget([100, 300, 4096, 200], ["stop", "stop", "length", "stop"]) == 200.0
    assert budget([4096, 4096], ["length", "length"]) is None  # all cut: no length to go by


def test_low_rows_carry_the_flag_the_source_suffix_and_the_budget():
    r = low_row(ROW, 250.0)
    assert r["data_source"] == "finqa_low"
    assert r["prompt"][0] == {"role": "system", "content": LOW_EFFORT} and r["prompt"][1] == ROW["prompt"][0]
    assert r["extra_info"]["budget"] == 250.0 and "budget" not in ROW["extra_info"]  # the source row is untouched
    assert "budget" not in low_row(ROW, None)["extra_info"]


def test_rows_without_a_budget_are_left_out():
    rows, skipped = build([ROW, {**ROW, "extra_info": {"id": "a-2"}}], {("finqa", "a-1"): 250.0, ("finqa", "a-2"): None})
    assert [r["extra_info"]["id"] for r in rows] == ["a-1"] and skipped == 1


@pytest.fixture
def lengths(monkeypatch):
    def set_length(n: int) -> None:
        monkeypatch.setattr(budget_mod, "n_tokens", lambda text, model_path=None: n)
    return set_length


def test_over_budget_scores_minus_one_even_when_right(lengths, monkeypatch):
    monkeypatch.setenv("B2E_BUDGET_MULT", "0.5")
    gt = ROW["reward_model"]["ground_truth"]
    lengths(120)  # budget 200 x 0.5 = 100
    out = budget_mod.compute_score("finqa_low", "</think>\\boxed{2}", gt, {"budget": 200.0})
    assert out == {"score": -1.0, "acc": 1.0, "over_budget": 1.0, "length": 120.0}
    lengths(90)
    assert budget_mod.compute_score("finqa_low", "</think>\\boxed{2}", gt, {"budget": 200.0})["score"] == 1.0
    assert budget_mod.compute_score("finqa_low", "</think>\\boxed{3}", gt, {"budget": 200.0})["score"] == 0.0


def test_no_budget_means_the_usual_score(lengths):
    lengths(10_000)
    out = budget_mod.compute_score("finqa_low", "</think>\\boxed{2}", ROW["reward_model"]["ground_truth"], {})
    assert out["score"] == 1.0 and out["over_budget"] == 0.0


def test_the_multiplier_and_thinking_come_from_reward_kwargs(lengths, monkeypatch):
    monkeypatch.delenv("B2E_BUDGET_MULT", raising=False)
    gt = ROW["reward_model"]["ground_truth"]
    lengths(150)
    kw = {"budget_mult": 0.5, "model_path": "unused", "thinking": True}
    assert budget_mod.compute_score("finqa_low", "</think>\\boxed{2}", gt, {"budget": 400.0}, **kw)["score"] == 1.0
    assert budget_mod.compute_score("finqa_low", "</think>\\boxed{2}", gt, {"budget": 200.0}, **kw)["score"] == -1.0
    # thinking=True reaches the verifier: an answer that never closed its thinking scores 0
    assert budget_mod.compute_score("finqa_low", "\\boxed{2} still thinking", gt, {"budget": 400.0}, **kw)["acc"] == 0.0
