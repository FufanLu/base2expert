from b2e.prompts import fits, prompt_tokens


class FakeTok:
    """One token per character of the rendered chat."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "".join(m["content"] for m in messages) + ("<a>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


def test_prompt_plus_output_budget_must_fit():
    tok = FakeTok()
    msgs = [{"role": "user", "content": "x" * 97}]
    assert prompt_tokens(tok, msgs, False) == 100
    assert fits(tok, msgs, False, max_tokens=28, max_model_len=128)
    assert not fits(tok, msgs, False, max_tokens=29, max_model_len=128)
