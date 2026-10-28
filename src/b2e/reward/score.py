"""verl reward entry point: route each sample to its verifier by data_source.

verl calls compute_score(data_source, solution_str, ground_truth, extra_info)
for every rollout. Scores are 1.0 (verified correct) or 0.0, except the
CAIL prison term, which is graded (b2e.reward.legal.term_score), and
instruction following, the fraction of constraints met (b2e.reward.ifcheck). A missing
\\boxed{} answer scores 0, the same as a wrong one, so the model is never
paid for the format alone.

An answer still inside its thinking block also scores 0: it was cut off at the
token limit and gave no final answer, even when its reasoning wrote a \\boxed{}
on the way (F#53: 57 of the 3,072 answers in the GRPO smoke 11423710 were paid
for that). With thinking=True the chat template opened the block in the prompt,
so the answer must contain </think>; an answer that opens <think> itself must
close it either way. verl passes thinking from
reward.custom_reward_function.reward_kwargs.
"""

from __future__ import annotations

import json

from b2e.reward.choice import choice_match
from b2e.reward.ifcheck import if_score
from b2e.reward.legal import norm_articles, norm_charges, term_score
from b2e.reward.numeric import extract_boxed, numeric_match

SCALE = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "percent": 1.0}


def check_number(answer: str, gt: dict) -> bool:
    value = float(gt["value"])
    scale = gt.get("scale", "")
    if numeric_match(answer, value):
        return True
    # "1.9 million" for a TAT-QA gold of 1.9 with scale "million"
    return scale in SCALE and SCALE[scale] != 1.0 and numeric_match(answer, value * SCALE[scale])


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None, thinking: bool = False) -> float:
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(gt, dict) or "kind" not in gt:
        raise ValueError(f"{data_source}: ground truth {ground_truth!r} is not a b2e ground-truth object "
                         "(JSON with a 'kind'); build the set with a b2e.data module")
    if (thinking or "<think>" in solution_str) and "</think>" not in solution_str:
        return 0.0
    if gt["kind"] == "ifeval":  # free-form answers with constraints, no \boxed{}
        return if_score(solution_str, gt)
    answer = extract_boxed(solution_str)
    if answer is None:
        return 0.0
    if gt["kind"] == "number":
        return 1.0 if check_number(answer, gt) else 0.0
    if gt["kind"] == "text":
        return 1.0 if answer.strip().lower().rstrip(".") == gt["value"] else 0.0
    if gt["kind"] == "choice":
        return 1.0 if choice_match(answer, gt["value"]) else 0.0
    if gt["kind"] == "charges":
        return 1.0 if norm_charges(answer) == set(gt["value"]) else 0.0
    if gt["kind"] == "articles":
        return 1.0 if norm_articles(answer) == set(gt["value"]) else 0.0
    if gt["kind"] == "term":
        return term_score(answer, float(gt["months"]), gt.get("death", False), gt.get("life", False))
    raise ValueError(f"unknown ground truth kind {gt['kind']!r} for {data_source}")
