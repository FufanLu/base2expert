import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from paired_cmp import compare, load, main  # noqa: E402


def results(d: Path, thinking: bool, sets: dict[str, list[tuple[str, float, str]]]) -> Path:
    """A run_eval result dir: summary.json and one <set>.run0.jsonl per set of (id, score, response)."""
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"thinking": "true" if thinking else "false"}))
    for name, rows in sets.items():
        with open(d / f"{name}.run0.jsonl", "w") as f:
            for i, score, response in rows:
                f.write(json.dumps({"data_source": name.split("_")[0], "id": i, "score": score,
                                    "response": response}) + "\n")
    return d


def answers(scores: list[float]) -> list[tuple[str, float, str]]:
    return [(str(i), s, "</think> \\boxed{1}") for i, s in enumerate(scores)]


def test_per_set_and_pooled_differences_with_their_standard_errors(tmp_path, capsys):
    a = results(tmp_path / "a", True, {"finqa_test": answers([1, 0, 1, 0]), "tatqa_dev": answers([1, 1])})
    b = results(tmp_path / "b", True, {"finqa_test": answers([1, 1, 1, 0]), "tatqa_dev": answers([0, 1])})
    main([str(a), str(b)])
    out = {line.split()[0]: line.split()[1:] for line in capsys.readouterr().out.splitlines()[3:]}
    # finqa: differences 0, 1, 0, 0: mean 0.25, sd sqrt(0.75 / 3), se sd / 2
    assert out["finqa_test"] == ["4", "0", "0.5000", "0.7500", "+0.2500", "0.2500", "0", "1"]
    assert out["tatqa_dev"] == ["2", "0", "1.0000", "0.5000", "-0.5000", "0.5000", "1", "0"]
    # the six items together, ids "0" and "1" of both sets kept apart
    assert out["pooled"][:5] == ["6", "0", "0.6667", "0.6667", "+0.0000"]
    assert float(out["pooled"][5]) == pytest.approx(math.sqrt(2 / 5 / 6), abs=1e-4)


def test_an_unclosed_thinking_block_scores_zero(tmp_path):
    rows = [("0", 1.0, "so the answer is \\boxed{1}"), ("1", 1.0, "done </think> \\boxed{1}"),
            ("2", 1.0, "<think> no end \\boxed{1}")]
    think = results(tmp_path / "think", True, {"finqa_test": rows})
    nothink = results(tmp_path / "nothink", False, {"finqa_test": rows})
    assert load(think, "finqa_test", 0, True) == {("finqa", "0"): 0.0, ("finqa", "1"): 1.0, ("finqa", "2"): 0.0}
    assert load(nothink, "finqa_test", 0, False) == {("finqa", "0"): 1.0, ("finqa", "1"): 1.0, ("finqa", "2"): 0.0}
    assert load(think, "tatqa_dev", 0, True) is None


def test_items_in_one_dir_only_are_left_out_and_counted():
    a = {("finqa", "0"): 1.0, ("finqa", "1"): 0.0, ("finqa", "2"): 1.0}
    b = {("finqa", "1"): 1.0, ("finqa", "2"): 1.0, ("finqa", "3"): 0.0}
    c = compare(a, b)
    assert (c["n"], c["unpaired"], c["a"], c["b"], c["diff"]) == (2, 2, 0.5, 1.0, 0.5)
    with pytest.raises(ValueError, match="no item"):
        compare({("finqa", "0"): 1.0}, {("tatqa", "0"): 1.0})


def test_exclude_leaves_out_the_items_of_other_answer_files(tmp_path, capsys):
    a = results(tmp_path / "a", True, {"finqa_test": answers([1, 0, 1, 0]), "tatqa_dev": answers([1, 1])})
    b = results(tmp_path / "b", True, {"finqa_test": answers([1, 1, 1, 0]), "tatqa_dev": answers([0, 1])})
    val = tmp_path / "rl_val512.run0.jsonl"
    val.write_text(json.dumps({"data_source": "tatqa", "id": "0", "score": 1.0, "response": "x"}) + "\n")
    main([str(a), str(b), "--exclude", str(val)])
    out = {w[0]: w[1:] for w in map(str.split, capsys.readouterr().out.splitlines())
           if w[0] in ("finqa_test", "tatqa_dev", "pooled")}
    assert out["finqa_test"][:5] == ["4", "0", "0.5000", "0.7500", "+0.2500"]  # FinQA's id "0" is another item
    assert out["tatqa_dev"][:5] == ["1", "0", "1.0000", "1.0000", "+0.0000"]
    assert out["pooled"][0] == "5"
