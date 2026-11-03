"""Per-prompt token budgets for effort control (P7: the low-effort finance teacher R-fin-low).

Kimi K3's recipe (docs/plan.md, 11-03): each prompt gets a budget estimated from the
cold-start model, and a rollout over a scaled budget gets reward -1 even when its answer is
right. Here the budget is S-fin's median length, in tokens, over its finished samples on that
prompt in the pool sampling (b2e.data.synth's passrate records), stored as extra_info["budget"]
by b2e.data.effort_pool. The scale (budget_mult) is set per leg and lowered leg by leg, so one
config serves the whole schedule (configs/rl/dapo_fin_low.yaml passes it, and the tokenizer
path, as reward_kwargs from B2E_BUDGET_MULT and B2E_MODEL, resolved in the driver).

verl's reward managers pass the decoded response text, not its length, so the length is
counted again with the policy's tokenizer. A prompt without a budget (the val rows: S-fin was
not sampled on them) is scored as usual and only its length is logged.
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


def compute_score(data_source, solution_str, ground_truth, extra_info=None, budget_mult=None, model_path=None,
                  **kwargs) -> dict:
    """The verifier's score (kwargs such as thinking go to it), or -1 over the budget; acc,
    over_budget and length are logged per source."""
    acc = verifier_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    budget = (extra_info or {}).get("budget")
    length = n_tokens(solution_str, model_path)
    mult = float(budget_mult if budget_mult is not None else os.environ.get("B2E_BUDGET_MULT", "1.0"))
    over = over_budget(length, budget, mult)
    return {"score": -1.0 if over else acc, "acc": acc, "over_budget": float(over), "length": float(length)}
