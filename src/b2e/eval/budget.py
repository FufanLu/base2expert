"""Thinking-budget forcing for evaluation (P7: accuracy against the thinking budget).

The model thinks for at most `budget` tokens. Phase 1 generates from the thinking prompt and
stops at </think> or at the budget; a thinking that the budget cut (or that ended without
</think>) is closed by us with the chat template's own separator ("\n</think>\n\n"), and phase 2
generates the answer from the prompt plus that thinking, with `answer_tokens` more. This is s1's
budget forcing (the end-of-thinking delimiter appended when the budget runs out) without its
"Wait" extension, which makes a model think longer than it chose to. </think> is an ordinary
added token in the Qwen3.5 tokenizer (special: false, id 248069), so it shows in the text.
"""

from __future__ import annotations

THINK_END = "</think>"


def continuation(text: str) -> str:
    """Phase 1's text as the assistant text phase 2 continues from: the thinking, closed."""
    body = text[: text.index(THINK_END)] if THINK_END in text else text
    return body + ("" if body.endswith("\n") else "\n") + THINK_END + "\n\n"


def generate(llm, messages: list[list[dict]], budget: int, answer_tokens: int, temperature: float,
             seed: int, sampling_params=None) -> list[dict]:
    """One record per prompt: the full response (thinking, </think>, answer) and its token counts.
    sampling_params defaults to vLLM's SamplingParams (a stand-in in the tests)."""
    if sampling_params is None:
        from vllm import SamplingParams

        sampling_params = SamplingParams

    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids(THINK_END)
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
               for m in messages]
    first = llm.generate(prompts, sampling_params(temperature=temperature, max_tokens=budget, seed=seed,
                                                  stop_token_ids=[end_id]))
    heads, closed = [], []
    for o in first:
        c = o.outputs[0]
        closed.append(c.stop_reason == end_id or THINK_END in c.text)
        heads.append(continuation(c.text))
    second = llm.generate([p + h for p, h in zip(prompts, heads, strict=True)],
                          sampling_params(temperature=temperature, max_tokens=answer_tokens, seed=seed))
    out = []
    for o1, head, done, o2 in zip(first, heads, closed, second, strict=True):
        c1, c2 = o1.outputs[0], o2.outputs[0]
        out.append({"text": head + c2.text, "thinking_tokens": len(c1.token_ids), "answer_tokens": len(c2.token_ids),
                    "forced": not done, "cut_by_budget": c1.finish_reason == "length", "finish": c2.finish_reason})
    return out
