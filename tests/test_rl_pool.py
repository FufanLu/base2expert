import pandas as pd

from b2e.data.rl_pool import select


def test_keeps_only_prompts_the_policy_sometimes_solves():
    pool = pd.DataFrame({"data_source": ["finqa"] * 4, "extra_info": [{"id": str(i)} for i in range(4)]})
    rates = pd.DataFrame({"data_source": ["finqa"] * 4, "id": ["0", "1", "2", "3"], "pass_rate": [0.0, 0.125, 0.875, 1.0]})
    kept = select(pool, rates, 0.1, 0.9)
    assert [e["id"] for e in kept["extra_info"]] == ["1", "2"]
