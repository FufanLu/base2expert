import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lawctl_cmp import compare, mcnemar, nums  # noqa: E402


def test_nums_reads_the_last_box():
    assert nums("先写 \\boxed{1} 再改：\\boxed{264、67}") == {264, 67}
    assert nums("没有框") == set()


def test_mcnemar_is_the_exact_binomial():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(5, 0) == pytest.approx(2 / 32)
    assert mcnemar(3, 3) == 1.0


def rec(score, box):
    return {"score": score, "response": f"\\boxed{{{box}}}", "tokens": 10}


def test_compare_pairs_rows_and_drops_those_whose_gold_holds_the_example():
    gold = {"a": {264}, "b": {133}, "c": {303}}
    runs = {"orig": {"a": rec(0, "264、67"), "b": rec(0, "133、67"), "c": rec(1, "303")},
            "ex1": {"a": rec(1, "264"), "b": rec(1, "133"), "c": rec(1, "303")},
            "ex2": {"a": rec(0, "133"), "b": rec(1, "133"), "c": rec(1, "303")}}
    out = compare(runs, gold, {"ex1": {133}, "ex2": {133}})
    assert out["orig"]["with_67"] == 2 and out["orig"]["with_general"] == 2
    e1 = out["ex1"]
    assert (e1["discordant_new_right_old_wrong"], e1["discordant_old_right_new_wrong"]) == (2, 0)
    assert e1["gold_has_example_article"] == 1 and e1["without_example_rows"]["discordant"] == [1, 0]
    assert e1["fixed_rows_old_had_general"] == 2
    assert out["ex2"]["copied_example_where_gold_differs"] == 1  # row a: answered 133, gold 264
