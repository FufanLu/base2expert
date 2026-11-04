"""Each domain's response cap in one RL run over several domains (P8 baseline (a), mixed RL).

Each expert was trained with its own cap and DAPO overlong zone (configs/rl/dapo_{fin,law,med}.yaml:
4,096 / 1,024, 4,096 / 1,024, 8,192 / 2,048). Mixed RL from S-mix trains every domain in one run,
and verl has one max_response_length, so the run's cap is the largest and this reward manager gives
each sample the reward its expert's run would have given: the response is cut at its domain's cap
before it is scored, and the overlong penalty is DAPO's over that domain's zone. The domain is the
row's routing column (`teacher`, b2e.data.mopd_mix), so mixed RL and MOPD read the same prompt mix.

What stays different from the experts' runs: a finance or law rollout can still run past 4,096
tokens (the sampler stops at the run's cap), and those tokens are in the loss with the cut answer's
reward. They cost rollout time, not reward.

    reward.reward_manager.source=importlib reward.reward_manager.module.path=pkg://b2e.reward.caps
    reward.reward_manager.name=CappedDAPO
    reward.reward_kwargs: max_resp_len, caps {domain: [cap, zone]}, overlong_buffer_cfg.enable false
"""

from __future__ import annotations

from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager

KEY = "teacher"


def parse_caps(raw) -> dict[str, tuple[int, int]]:
    caps = {str(k): (int(v[0]), int(v[1])) for k, v in dict(raw).items()}
    for k, (cap, zone) in caps.items():
        if not 0 < zone <= cap:
            raise ValueError(f"{k}: the overlong zone {zone} must be in (0, cap {cap}]")
    return caps


def overlong_penalty(length: int, cap: int, zone: int, factor: float = 1.0) -> float:
    """DAPO's linear penalty over the last `zone` tokens before `cap`, -factor at the cap; a longer
    response counts as cut at the cap."""
    return min(-(min(length, cap) - (cap - zone)) / zone * factor, 0.0)


class CappedDAPO(DAPORewardManager):
    """DAPORewardManager with a cap and an overlong zone per domain."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)
        kw = config.reward.get("reward_kwargs", {})
        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            raise ValueError("the run-wide overlong penalty would come on top of each domain's: disable it")
        self.caps = parse_caps(kw["caps"])
        self.factor = float(kw.get("penalty_factor", 1.0))
        over = {k: cap for k, (cap, _) in self.caps.items() if cap > self.max_resp_len}
        if over:
            raise ValueError(f"caps over the run's max_resp_len {self.max_resp_len}: {over}")

    async def run_single(self, data) -> dict:
        data_item = data[-1:][0]
        domain = str(data_item.non_tensor_batch[KEY])
        if domain not in self.caps:
            raise KeyError(f"{KEY}={domain!r} has no cap in {sorted(self.caps)}")
        cap, zone = self.caps[domain]
        response_ids = data_item.batch["responses"]
        length = int(data_item.batch["attention_mask"][-response_ids.shape[-1]:].sum())
        ids = response_ids[:min(length, cap)]  # what the expert's run would have sampled
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(ids, skip_special_tokens=True))
        extra = {"reward_router_address": self.reward_router_address,
                 "reward_model_tokenizer": self.reward_model_tokenizer} if self.reward_router_address else {}
        kwargs = dict(data_source=data_item.non_tensor_batch["data_source"], solution_str=response_str,
                      ground_truth=data_item.non_tensor_batch["reward_model"]["ground_truth"],
                      extra_info=data_item.non_tensor_batch.get("extra_info", {}), **extra)
        if self.is_async_reward_score:
            result = await self.compute_score(**kwargs)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))
        info = dict(result) if isinstance(result, dict) else {"acc": result}
        score = float(result["score"] if isinstance(result, dict) else result)
        penalty = overlong_penalty(length, cap, zone, self.factor)
        info.update(overlong_reward=penalty, over_cap=float(length > cap))
        return {"reward_score": score + penalty, "reward_extra_info": info}
