import json

import pandas as pd
import pytest

from b2e.data.finance import main, tatqa_rows


def tatqa_file(path, questions):
    path.write_text(json.dumps([{
        "table": {"table": [["", "2019"], ["Revenue", "1.9"]]},
        "paragraphs": [{"order": 2, "text": "Second."}, {"order": 1, "text": "First."}],
        "questions": questions}]))
    return path


def q(uid, answer_type, answer, scale="million"):
    return {"uid": uid, "answer_type": answer_type, "answer": answer, "scale": scale, "question": f"Q {uid}?"}


def test_tatqa_keeps_arithmetic_and_count_questions_with_a_number(tmp_path):
    f = tatqa_file(tmp_path / "gold.json", [q("a", "arithmetic", 1.9), q("b", "span", ["Revenue"]),
                                            q("c", "count", "2", ""), q("d", "arithmetic", "n/a")])
    rows = tatqa_rows(f, "test")
    assert [r["extra_info"]["id"] for r in rows] == ["a", "c"]
    assert rows[0]["extra_info"]["split"] == "test"
    assert json.loads(rows[0]["reward_model"]["ground_truth"]) == {"kind": "number", "value": 1.9, "scale": "million"}
    assert "First.\nSecond." in rows[0]["prompt"][0]["content"]  # paragraphs in their order


def test_only_writes_the_named_sets(tmp_path):
    tatqa = tmp_path / "tatqa"
    tatqa.mkdir()
    tatqa_file(tatqa / "tatqa_dataset_test_gold.json", [q("a", "arithmetic", 1.9)])
    out = tmp_path / "out"
    out.mkdir()
    (out / "tatqa_dev.parquet").write_text("kept")
    # FinQA's files and the other TAT-QA splits are not there: --only must not read them
    main(["--finqa", str(tmp_path / "none"), "--tatqa", str(tatqa), "--out", str(out), "--only", "tatqa_test"])
    assert sorted(p.name for p in out.iterdir()) == ["tatqa_dev.parquet", "tatqa_test.parquet"]
    assert (out / "tatqa_dev.parquet").read_text() == "kept"
    assert pd.read_parquet(out / "tatqa_test.parquet")["extra_info"][0]["split"] == "test"
    with pytest.raises(SystemExit, match="no set named"):
        main(["--finqa", "x", "--tatqa", str(tatqa), "--out", str(out), "--only", "tatqa_tset"])
