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


def test_synth_pairs_are_sampled_together(tmp_path):
    rows = []
    for i in range(5):
        for think in (True, False):
            rows.append({"data_source": "finqa", "id": str(i), "enable_thinking": think, "ground_truth": "{}",
                         "messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]})
    f = tmp_path / "s.parquet"
    pd.DataFrame(rows).to_parquet(f)
    out = load_synth([str(f)], 2, random.Random(0))
    assert len(out) == 4
    groups = {r["group"] for r in out}
    assert len(groups) == 2 and all(sum(r["group"] == g for r in out) == 2 for g in groups)
