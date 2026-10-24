import pandas as pd

from b2e.data.general_eval import mmlu_pro
from b2e.reward.score import compute_score


def test_mmlu_pro_stratified_sample_keeps_category(tmp_path):
    d = tmp_path / "TIGER-Lab/MMLU-Pro/data"
    d.mkdir(parents=True)
    rows = [{"question_id": i, "question": f"q{i}", "options": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
             "answer": "J", "answer_index": 9, "category": "law" if i < 5 else "math"} for i in range(8)]
    pd.DataFrame(rows).to_parquet(d / "test-00000-of-00001.parquet")
    out = mmlu_pro(tmp_path, per_category=2, seed=0)
    assert sorted(r["extra_info"]["category"] for r in out) == ["law", "law", "math", "math"]
    assert "J. j" in out[0]["prompt"][0]["content"]
    assert compute_score("mmlu_pro", "\\boxed{J}", out[0]["reward_model"]["ground_truth"]) == 1.0
