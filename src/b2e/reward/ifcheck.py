"""Instruction-following reward: every constraint of a prompt checked by the IFEvalG verifiers.

Ground truth in verl format (b2e.data.if_data writes it):

    {"kind": "ifeval", "prompt": <user text>,
     "constraints": [{"instruction_id": ["punctuation:no_comma", ...], "kwargs": [null, {...}, ...]}]}

the layout of allenai/IF_multi_constraints_upto5 and google/IFEval. Each
checker is built exactly as IFEval's own evaluation does it: build the
description from the kwargs (unset ones dropped), rebuild it with the prompt
when the checker takes one, and count an empty response as a failure.

Only the answer is checked: everything up to </think> is dropped, so a
reasoning block that mentions a forbidden word or breaks the format does not
count against the answer. The answer is stripped, as open-instruct's
remove_thinking_section does: the blank line a model writes after </think>
otherwise reads as an empty first paragraph, and nth_paragraph_first_word and
sentence_hyphens failed on every thinking-mode answer (F#32).

Checking is deterministic. langdetect (response_language, english_capital,
english_lowercase) samples at random unless seeded, and a checker whose
kwargs are missing or out of range draws a replacement from `random`
(b2e.data.if_data drops such training rows; two IFEval items keep theirs, as
the official evaluation does). Unseeded, the same answer scored differently
from call to call, which in RL is reward noise.

The training reward is the fraction of constraints met. With up to five
constraints per prompt, all-or-nothing scoring leaves most early rollouts at
0, and a GRPO group whose answers all score 0 has zero advantage and teaches
nothing; partial credit keeps a signal. Evaluation reports the strict rate
(all constraints met) as well, the usual IFEval "prompt-level" number.
"""

from __future__ import annotations

import random

SEED = 0


def answer_part(text: str) -> str:
    return (text.rsplit("</think>", 1)[1] if "</think>" in text else text).strip()


def build(iid: str, kwargs: dict | None, prompt: str):
    """The checker for one constraint, built the way IFEval's evaluation builds it."""
    from b2e.reward.ifevalg import instructions_registry as reg

    checker = reg.INSTRUCTION_DICT[iid](iid)
    checker.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
    args = checker.get_instruction_args()
    if args and "prompt" in args:
        checker.build_description(prompt=prompt)
    return checker


def check(response: str, gt: dict) -> list[bool]:
    """One result per constraint, in order."""
    import langdetect

    langdetect.DetectorFactory.seed = SEED
    response = answer_part(response)
    state = random.getstate()  # langdetect and the checkers use the global generator; leave it as found
    try:
        random.seed(SEED)
        out = []
        for c in gt["constraints"]:
            for iid, kwargs in zip(c["instruction_id"], c["kwargs"], strict=True):
                checker = build(iid, kwargs, gt.get("prompt", ""))
                try:
                    ok = bool(response) and bool(checker.check_following(response))
                except Exception:  # a checker that cannot parse the response counts it as not following
                    ok = False
                out.append(ok)
    finally:
        random.setstate(state)
    return out


def if_score(response: str, gt: dict) -> float:
    results = check(response, gt)
    return sum(results) / len(results) if results else 0.0
