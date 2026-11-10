import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_bpb", Path("pretrain/eval_bpb.py"))
eval_bpb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_bpb)


def test_rows_score_all_but_the_first_token_and_cut_to_the_input_length():
    rows = eval_bpb.doc_rows([[5, 6, 7, 8, 9], [3], [4, 4]], bos=0, max_tokens=4)
    # the first 4 tokens kept, 3 of them scored; [3] has nothing to score
    assert rows == [([0, 5, 6, 7], [-1, 6, 7, 8]), ([0, 4], [-1, 4])]
    assert all(len(x) <= 4 for x, _ in rows)
