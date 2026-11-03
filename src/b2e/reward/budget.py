"""Per-prompt token budgets for effort control (P7: the low-effort finance teacher R-fin-low).

Kimi K3's recipe (docs/plan.md, 11-03): each prompt gets a budget estimated from the
cold-start model, and a rollout over a scaled budget gets reward -1 even when its answer is
right. Here the budget is S-fin's median length, in tokens, over its finished samples on that
prompt in the pool sampling (b2e.data.synth's passrate records), stored as extra_info["budget"]
by b2e.data.effort_pool. The scale is B2E_BUDGET_MULT, set per leg and lowered leg by leg, so
one config serves the whole schedule.

verl's reward managers pass the decoded response text, not its length, so the length is
counted again with the policy's tokenizer (B2E_MODEL). A prompt without a budget (the val
rows: S-fin was not sampled on them) is scored as usual and only its length is logged.

    reward.custom_reward_function.path=$B2E_CODE/src/b2e/reward/budget.py
    reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

import functools
import os

from b2e.reward.score import compute_score as verifier_score


@functools.lru_cache(maxsize=2)
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def n_tokens(text: str, model_path: str | None = None) -> int:
    tok = _tokenizer(model_path or os.environ["B2E_MODEL"])
    return len(tok(text, add_special_tokens=False)["input_ids"])


def over_budget(length: int, budget: float | None, mult: float) -> bool:
    return budget is not None and budget > 0 and length > budget * mult


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    """The verifier's score, or -1 over the budget; acc and over_budget are logged per source."""
    acc = verifier_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    budget = (extra_info or {}).get("budget")
    length = n_tokens(solution_str)
    over = over_budget(length, budget, float(os.environ.get("B2E_BUDGET_MULT", "1.0")))
    return {"score": -1.0 if over else acc, "acc": acc, "over_budget": float(over), "length": float(length)}
