import json

from b2e.reward.score import compute_score
from b2e.rsi.propose import EXAMPLE, filter_problems, grounded, parse


def test_the_prompt_example_passes_every_check():
    rows, stats = filter_problems(["<think>plan</think>\n" + EXAMPLE], round_idx=1)
    assert stats["kept"] == 1
    gt = json.loads(rows[0]["reward_model"]["ground_truth"])
    # (5260-3031)/5260*100 - (4981-2940)/4981*100 = 42.376... - 40.976... = 1.40
    assert gt["value"] == 1.4
    assert compute_score("rsi_fin", "\\boxed{1.40}", rows[0]["reward_model"]["ground_truth"]) == 1.0


def test_program_must_use_numbers_from_the_excerpt():
    ctx = "Revenue was $1,200 million in 2019 and $1,000 million in 2018."
    assert grounded("print((1200 - 1000) / 1000 * 100)", ctx)
    assert not grounded("print((1300 - 1000) / 1000 * 100)", ctx)  # 1300 is not in the excerpt


def test_failures_are_counted_by_kind():
    bad_parse = "no tags here"
    bad_prog = EXAMPLE.replace("print(round(margin_2019 - margin_2018, 2))", "import os\nprint(1)")
    ungrounded = EXAMPLE.replace("(5260 - 3031)", "(5261 - 3031)")
    rows, stats = filter_problems([bad_parse, bad_prog, ungrounded, EXAMPLE, EXAMPLE], round_idx=1)
    assert len(rows) == 1
    assert stats == {"parse": 1, "program:rejected": 1, "ungrounded": 1, "kept": 1, "duplicate": 1}


def test_parse_strips_code_fences():
    p = parse(EXAMPLE.replace("<program>\n", "<program>\n```python\n").replace("\n</program>", "\n```\n</program>"))
    assert p is not None and not p["program"].startswith("```")


def test_a_proposal_cut_off_inside_its_thinking_is_not_kept():
    cut = "<think>a draft:\n" + EXAMPLE + "\nbut let me check the margin again"  # no </think>: hit the limit
    rows, stats = filter_problems([cut, "<think>plan</think>\n" + EXAMPLE], round_idx=1, thinking=True)
    assert stats == {"unclosed": 1, "kept": 1} and len(rows) == 1
