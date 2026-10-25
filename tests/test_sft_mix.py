import random

import pandas as pd

from b2e.data.sft_mix import first_exchange, load_synth


def test_first_exchange_keeps_system_prompt_and_one_turn():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<think>\nx\n</think>\n\nhello"},
            {"role": "user", "content": "more"}, {"role": "assistant", "content": "<think>\ny\n</think>\n\nok"}]
    out, think = first_exchange(msgs, {"enable_thinking": True, "custom_instructions": "You are a vampire."})
    assert think is True
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[2]["content"].endswith("hello")


def test_first_exchange_rejects_mode_mismatch():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert first_exchange(msgs, {"enable_thinking": True}) is None  # thinking example without reasoning
    assert first_exchange(msgs, {"enable_thinking": False})[1] is False


GT = '{"kind": "number", "value": 3.0, "scale": ""}'


def synth_rows(n, answer=lambda i, think: "\\boxed{3}"):
    return [{"data_source": "finqa", "id": str(i), "enable_thinking": think, "ground_truth": GT,
             "messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": answer(i, think)}]}
            for i in range(n) for think in (True, False)]


def test_synth_pairs_are_sampled_together(tmp_path):
    f = tmp_path / "s.parquet"
    pd.DataFrame(synth_rows(5)).to_parquet(f)
    out, failed = load_synth([str(f)], 2, random.Random(0))
    assert len(out) == 4 and failed == 0
    groups = {r["group"] for r in out}
    assert len(groups) == 2 and all(sum(r["group"] == g for r in out) == 2 for g in groups)


def test_synth_pairs_failing_the_current_verifier_are_dropped(tmp_path):
    # prompt 0's non-thinking answer no longer passes: both of its rows go, before sampling
    f = tmp_path / "s.parquet"
    pd.DataFrame(synth_rows(3, lambda i, think: "\\boxed{4}" if (i == 0 and not think) else "\\boxed{3}")).to_parquet(f)
    out, failed = load_synth([str(f)], "all", random.Random(0))
    assert failed == 1 and {r["group"] for r in out} == {"finqa/1", "finqa/2"}
