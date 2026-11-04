import types

from b2e.eval.budget import THINK_END, continuation, generate


def test_continuation_closes_the_thinking_once():
    assert continuation("step 1\nstep 2") == "step 1\nstep 2\n</think>\n\n"  # cut by the budget
    assert continuation("step 1\n") == "step 1\n</think>\n\n"
    assert continuation("step 1\n</think>") == "step 1\n</think>\n\n"  # closed by the model


class FakeLLM:
    """Phase 1 answers per prompt from `thinking`; phase 2 always answers \\boxed{2}."""

    def __init__(self, thinking):
        self.thinking = thinking
        self.calls = []

    def get_tokenizer(self):
        return types.SimpleNamespace(
            convert_tokens_to_ids=lambda t: 248069,
            apply_chat_template=lambda m, **kw: f"<|im_start|>user\n{m[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n")

    def generate(self, prompts, sp):
        self.calls.append((prompts, sp))
        outs = []
        for p in prompts:
            if "stop_token_ids" in sp:  # phase 1
                text, finish, stop = self.thinking[p.split("user\n")[1].split("<|im_end|>")[0]]
            else:
                text, finish, stop = "\\boxed{2}", "stop", None
            outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(
                text=text, token_ids=[0] * len(text.split()), finish_reason=finish, stop_reason=stop)]))
        return outs


def test_generate_forces_an_answer_after_the_budget():
    llm = FakeLLM({"q1": ("a b c d", "length", None),  # the budget ran out
                   "q2": ("a b </think>", "stop", 248069)})  # closed in time
    out = generate(llm, [[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]], budget=4,
                   answer_tokens=16, temperature=0.0, seed=0, sampling_params=dict)
    assert out[0]["text"] == "a b c d\n</think>\n\n\\boxed{2}" and out[0]["forced"] and out[0]["cut_by_budget"]
    assert out[1]["text"] == "a b \n</think>\n\n\\boxed{2}" and not out[1]["forced"]
    (p1, sp1), (p2, sp2) = llm.calls
    assert sp1["max_tokens"] == 4 and sp1["stop_token_ids"] == [248069] and sp2["max_tokens"] == 16
    assert p2[0] == p1[0] + "a b c d\n</think>\n\n"  # phase 2 continues the same prompt
    assert all(o["text"].count(THINK_END) == 1 for o in out)
