"""Prompt length checks before handing a batch to vLLM.

vLLM rejects a request whose prompt plus max_tokens exceeds max_model_len, and
`LLM.chat` raises for the whole batch on the first such request (docs/failures.md
F#31: one 16 K-token instruction-following prompt ended an eval job). Callers
drop those prompts up front and report how many they dropped.
"""

from __future__ import annotations


def prompt_tokens(tok, messages: list[dict], thinking: bool) -> int:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def fits(tok, messages: list[dict], thinking: bool, max_tokens: int, max_model_len: int) -> bool:
    return prompt_tokens(tok, messages, thinking) + max_tokens <= max_model_len
