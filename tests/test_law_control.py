import json

import pandas as pd
import pytest

from b2e.data.law_control import OLD, box, check_example, labels, main, pick_examples, swap
from b2e.data.law_med import LAW_TASKS


def row(source, gold, fact="某甲盗窃财物。"):
    task = source.removeprefix("cail2018_")
    gt = {"kind": task, "value": gold} if task != "term" else {"kind": "term", "months": 6.0}
    return {"data_source": source,
            "prompt": [{"role": "user", "content": f"{LAW_TASKS[task]}\n\n案件事实：{fact}"}],
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)}}


def frame():
    return pd.DataFrame([row("cail2018_articles", [264]), row("cail2018_articles", [264]),
                         row("cail2018_articles", [266]), row("cail2018_articles", [347, 357]),
                         row("cail2018_articles", [25, 234]), row("cail2018_articles", [25, 234]),
                         row("cail2018_charges", ["盗窃"]), row("cail2018_term", None)])


def test_the_examples_come_from_the_specific_part_of_the_labels():
    golds = labels(frame())
    assert len(golds) == 6
    # (25, 234) is the most common pair but holds a general-part article
    assert pick_examples(golds) == {1: (264,), 2: (347, 357)}
    check_example((264,), golds)
    with pytest.raises(ValueError, match="general part"):
        check_example((264, 67), golds)
    with pytest.raises(ValueError, match="no label"):
        check_example((999,), golds)


def test_swap_rewrites_only_the_example_of_the_articles_rows():
    out = swap(frame(), (347, 357))
    assert len(out) == 6 and set(out["data_source"]) == {"cail2018_articles"}
    text = out["prompt"][0][0]["content"]
    assert OLD not in text and box((347, 357)) == "\\boxed{347、357}" and "\\boxed{347、357}" in text
    assert text.replace("\\boxed{347、357}", OLD) == frame()["prompt"][0][0]["content"]
    bad = frame()
    bad.at[0, "prompt"] = [{"role": "user", "content": "别的提示"}]
    with pytest.raises(ValueError, match="expected instruction"):
        swap(bad, (264,))


def test_main_writes_the_original_rows_and_both_controls(tmp_path):
    frame().to_parquet(tmp_path / "pool.parquet")
    frame().to_parquet(tmp_path / "cail_test.parquet")
    main(["--pool", str(tmp_path / "pool.parquet"), "--data", str(tmp_path / "cail_test.parquet"),
          "--out", str(tmp_path / "control")])
    d = tmp_path / "control"
    assert [len(pd.read_parquet(d / f"cail_test_articles_{k}.parquet")) for k in ("orig", "ex1", "ex2")] == [6, 6, 6]
    rep = json.loads((d / "cail_test_report.json").read_text())
    assert rep["ex1"]["example"] == "\\boxed{264}" and rep["general_part_in_labels"] == 2
