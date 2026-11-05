import json

import pandas as pd
import pytest

from b2e.data.finance import INSTRUCTION
from b2e.rsi.round import dedup, halves, join, main


def row(pid: str, doc: str, question: str, value: float) -> dict:
    return {"data_source": "rsi_fin",
            "prompt": [{"role": "user", "content": f"Excerpt {doc}.\n\nQuestion: {question}\n\n{INSTRUCTION}"}],
            "ability": "finance",
            "reward_model": {"style": "rule", "ground_truth": json.dumps({"kind": "number", "value": value, "scale": ""})},
            "extra_info": {"split": "train", "id": pid, "doc_id": doc}}


def rates(ids: list[str], scores: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame({"data_source": ["rsi_fin"] * len(ids), "id": ids, "scores": scores,
                         "finish": [["stop"] * len(s) for s in scores],
                         "pass_rate": [sum(s) / len(s) for s in scores]})


def test_halves_alternate_and_join_refuses_an_id_twice():
    assert halves(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]
    a, b = pd.DataFrame({"id": ["x", "y"]}), pd.DataFrame({"id": ["z"]})
    assert list(join([a, b])["id"]) == ["x", "y", "z"]
    with pytest.raises(ValueError, match="more than one shard"):
        join([a, pd.DataFrame({"id": ["y"]})])


def test_a_problem_repeating_one_on_its_excerpt_is_dropped():
    rows = [row("0", "d1", "What is the growth rate of revenue from 2018 to 2019?", 5.6),
            row("1", "d1", "By what percent did cost of revenues change?", 0.056),  # 5.6 % again, as a fraction
            row("2", "d1", "What is the growth rate of revenue from 2018 to 2019 ?", 7.0),  # the same question
            row("3", "d1", "What was the gross margin in 2019?", 42.4),
            row("4", "d2", "What is the growth rate of revenue from 2018 to 2019?", 5.6)]  # another excerpt
    kept, dropped = dedup(rows)
    assert [r["extra_info"]["id"] for r in kept] == ["0", "3", "4"] and dropped == 2


def test_pool_keeps_the_usable_of_each_source_once(tmp_path):
    # the pilot's gate saw every problem, so its valid ones still need the pass rates; a round's saw the mixed only
    pilot = [row("r0-0", "d1", "What was the gross margin in 2019?", 42.4),
             row("r0-1", "d2", "What is the growth rate of revenue?", 5.6),
             row("r0-2", "d3", "What was the operating margin in 2018?", 12.0)]
    pd.DataFrame(pilot).to_parquet(tmp_path / "pilot_gated.parquet")
    rates(["r0-0", "r0-1", "r0-2"], [[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 0, 0]]).to_parquet(tmp_path / "pilot_rates.parquet")
    round1 = [row("r1-0", "d1", "What was the gross margin in 2019?", 42.4),  # the pilot's r0-0 again
              row("r1-1", "d4", "What share of revenue was R&D in 2019?", 9.1)]
    pd.DataFrame(round1).to_parquet(tmp_path / "r1_gated.parquet")
    val = tmp_path / "val.parquet"
    pd.DataFrame(pilot[:1]).to_parquet(val)
    out = tmp_path / "rsi_r1"
    main(["pool", "--out", str(out), "--val", str(val), "--sources",
          f"p={tmp_path / 'pilot_gated.parquet'},{tmp_path / 'pilot_rates.parquet'}", f"r1={tmp_path / 'r1_gated.parquet'}"])
    train = pd.read_parquet(out / "rsi_train.parquet")
    assert [e["id"] for e in train["extra_info"]] == ["p.r0-0", "p.r0-2", "r1.r1-1"]  # r0-1 always right
    assert (out / "rl_train.txt").read_text().strip() == str((out / "rsi_train.parquet").resolve())
    assert (out / "rl_val.txt").read_text().strip() == str(val.resolve())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["sources"]["p"]["usable"] == 2 and summary["duplicates_dropped"] == 1


def test_mixed_keeps_the_problems_right_and_wrong_among_finished(tmp_path):
    pd.DataFrame([row("a", "d1", "q one?", 1.0), row("b", "d2", "q two?", 2.0)]).to_parquet(tmp_path / "p.parquet")
    rates(["a", "b"], [[1, 0, 1, 0], [1, 1, 1, 1]]).to_parquet(tmp_path / "r.parquet")
    main(["mixed", "--problems", str(tmp_path / "p.parquet"), "--passrate", str(tmp_path / "r.parquet"),
          "--out", str(tmp_path / "new" / "m.parquet")])  # a directory not made yet (the dry check, 11630054)
    assert [e["id"] for e in pd.read_parquet(tmp_path / "new" / "m.parquet")["extra_info"]] == ["a"]
