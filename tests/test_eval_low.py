import pytest

from b2e.data.effort_pool import LOW_EFFORT
from b2e.eval.run_eval import with_system


def test_low_effort_message_goes_first_and_is_the_trained_text():
    p = [{"role": "user", "content": "What was the change?"}]
    out = with_system(p, LOW_EFFORT)
    assert out == [{"role": "system", "content": "Reasoning effort: low. Think briefly, then answer."}, *p]
    assert p == [{"role": "user", "content": "What was the change?"}]  # the row's own prompt is left as it was


def test_a_prompt_with_a_system_message_is_refused():
    with pytest.raises(ValueError, match="system message"):
        with_system([{"role": "system", "content": LOW_EFFORT}, {"role": "user", "content": "q"}], LOW_EFFORT)
