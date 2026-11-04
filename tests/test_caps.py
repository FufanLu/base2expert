import asyncio
import importlib
import sys
import types

import numpy as np
import pytest


class FakeDAPO:
    """The attributes CappedDAPO reads from verl's DAPORewardManager.__init__."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        kw = config.reward.get("reward_kwargs", {})
        self.tokenizer, self.compute_score, self.is_async_reward_score = tokenizer, compute_score, False
        self.overlong_buffer_cfg, self.max_resp_len = kw.get("overlong_buffer_cfg"), kw.get("max_resp_len")
        self.reward_router_address, self.reward_model_tokenizer = reward_router_address, reward_model_tokenizer


@pytest.fixture
def caps(monkeypatch):
    names = ["verl", "verl.experimental", "verl.experimental.reward_loop",
             "verl.experimental.reward_loop.reward_manager", "verl.experimental.reward_loop.reward_manager.dapo"]
    for n in names:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    sys.modules[names[-1]].DAPORewardManager = FakeDAPO
    monkeypatch.delitem(sys.modules, "b2e.reward.caps", raising=False)
    yield importlib.import_module("b2e.reward.caps")
    sys.modules.pop("b2e.reward.caps", None)


class Tok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


def config(caps, enable=False, max_resp_len=8):
    kw = {"max_resp_len": max_resp_len, "caps": caps, "overlong_buffer_cfg": types.SimpleNamespace(enable=enable)}
    return types.SimpleNamespace(reward={"reward_kwargs": kw})


def item(domain, valid, width=8):
    mask = np.zeros(4 + width, dtype=int)
    mask[:4 + valid] = 1  # 4 prompt tokens, then the response
    return types.SimpleNamespace(
        batch={"responses": np.arange(width), "attention_mask": mask},
        non_tensor_batch={"teacher": domain, "data_source": "x", "reward_model": {"ground_truth": "g"}})


def test_the_penalty_is_dapos_over_the_domains_zone(caps):
    assert caps.overlong_penalty(3000, 4096, 1024) == 0.0
    assert caps.overlong_penalty(3584, 4096, 1024) == -0.5
    assert caps.overlong_penalty(4096, 4096, 1024) == -1.0
    assert caps.overlong_penalty(6000, 4096, 1024) == -1.0  # past the cap: cut there
    assert caps.overlong_penalty(7168, 8192, 2048) == -0.5


def test_a_zone_must_fit_its_cap(caps):
    assert caps.parse_caps({"fin": [4096, 1024]}) == {"fin": (4096, 1024)}
    for bad in ([4096, 0], [1024, 4096]):
        with pytest.raises(ValueError, match="zone"):
            caps.parse_caps({"fin": bad})


def test_a_response_is_scored_as_cut_at_its_domains_cap(caps):
    score = lambda solution_str, **kw: 1.0 if "t5" in solution_str else 0.0  # noqa: E731
    m = caps.CappedDAPO(config({"fin": [4, 2], "med": [8, 2]}), Tok(), score)

    async def run(it):
        m.loop = asyncio.get_running_loop()
        return await m.run_single([it])

    fin, med = asyncio.run(run(item("fin", 6))), asyncio.run(run(item("med", 6)))
    # finance: the answer (token 5) is past its 4-token cap, so it is cut, with the full penalty
    assert fin["reward_score"] == -1.0 and fin["reward_extra_info"]["over_cap"] == 1.0
    # medicine: the same 6 tokens end where its zone starts (8 - 2): right, no penalty
    assert med["reward_score"] == 1.0 and med["reward_extra_info"]["acc"] == 1.0
    with pytest.raises(KeyError, match="law"):
        asyncio.run(run(item("law", 6)))


def test_the_run_wide_penalty_must_be_off_and_every_cap_within_the_run(caps):
    with pytest.raises(ValueError, match="disable"):
        caps.CappedDAPO(config({"fin": [4, 2]}, enable=True), Tok(), None)
    with pytest.raises(ValueError, match="max_resp_len"):
        caps.CappedDAPO(config({"med": [16, 4]}), Tok(), None)
