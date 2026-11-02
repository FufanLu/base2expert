import re

import pandas as pd
import pytest

from b2e.data.pref_render import matches, render, render_rows

SPECIAL = {"<|im_start|>": 0, "<|im_end|>": 1, "<think>": 2, "</think>": 3}


class QwenLikeTok:
    """Qwen-style turns; thinking mode opens <think> in the generation prompt. Special
    tokens are single ids, every other character its own id."""

    eos_token = "<|im_end|>"
    eos_token_id = 1

    def __init__(self, thinking_prompt: str = "<think>\n"):
        self.thinking_prompt = thinking_prompt

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n" + (self.thinking_prompt if enable_thinking else "")
        return s

    def __call__(self, text, add_special_tokens=False):
        parts = re.split("(" + "|".join(map(re.escape, SPECIAL)) + ")", text)
        return {"input_ids": [i for p in parts if p for i in ([SPECIAL[p]] if p in SPECIAL else [ord(c) + 10 for c in p])]}


Q = [{"role": "user", "content": "What is 2 + 3?"}]
A = [{"role": "assistant", "content": "<think>\n2 plus 3.\n</think>\n\nThe answer is \\boxed{5}."}]


def test_render_splits_after_the_generation_prompt_and_before_the_turn_end():
    tok = QwenLikeTok()
    prompt, completion = render(tok, Q, A)
    assert prompt == "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert completion == "2 plus 3.\n</think>\n\nThe answer is \\boxed{5}."
    # TRL appends eos (<|im_end|>) to the completion: one turn end, as generated
    assert matches(tok, prompt, completion, tok.apply_chat_template(Q + A))


def test_render_refuses_a_generation_prompt_that_is_not_a_prefix():
    # a template that closes an empty thinking block in the prompt would move the boundary
    with pytest.raises(ValueError):
        render(QwenLikeTok(thinking_prompt="<think>\n\n</think>\n\n"), Q, A)


def test_render_rows_keeps_metadata_and_checks_every_side():
    df = pd.DataFrame([{"data_source": "finqa", "id": "a", "prompt": Q, "chosen": A,
                        "rejected": [{"role": "assistant", "content": "<think>\nhm\n</think>\n\n\\boxed{6}"}],
                        "chosen_tokens": 30, "rejected_tokens": 20}])
    out = render_rows(QwenLikeTok(), df)
    r = out.iloc[0]
    assert r["prompt"].endswith("<think>\n") and r["chosen"].endswith("\\boxed{5}.") and r["rejected"] == "hm\n</think>\n\n\\boxed{6}"
    assert (r["data_source"], r["id"], r["chosen_tokens"], r["rejected_tokens"]) == ("finqa", "a", 30, 20)

    class WrongEos(QwenLikeTok):
        eos_token_id = 3
    with pytest.raises(ValueError):
        render_rows(WrongEos(), df)
