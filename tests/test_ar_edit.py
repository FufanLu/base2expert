import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pretrain" / "autoresearch"))
from edit import EditError, apply, parse  # noqa: E402

SRC = "DEPTH = 8\nMATRIX_LR = 0.04  # muon\nx = 1\nx = 1\n"


def test_parse_and_apply():
    ans = "<think>try a higher lr</think>\nDESCRIPTION: lr 0.05\n<<<<<<< SEARCH\nMATRIX_LR = 0.04  # muon\n=======\nMATRIX_LR = 0.05  # muon\n>>>>>>> REPLACE\n"
    desc, blocks = parse(ans)
    assert desc == "lr 0.05"
    assert apply(SRC, blocks) == SRC.replace("0.04", "0.05")


def test_ambiguous_or_missing_search_is_rejected():
    with pytest.raises(EditError, match="2 times"):
        apply(SRC, [("x = 1", "x = 2")])
    with pytest.raises(EditError, match="0 times"):
        apply(SRC, [("DEPTH = 9", "DEPTH = 10")])
    with pytest.raises(EditError, match="no SEARCH"):
        parse("DESCRIPTION: nothing")


def test_random_proposal_is_a_valid_edit_of_one_constant():
    import random

    from agent import random_propose, summary

    src = ("MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)\nDEPTH = 8               # layers\n"
           "TOTAL_BATCH_SIZE = 2**19 # tokens\nWARMDOWN_RATIO = 0.5\nEMBEDDING_LR = 0.6\nUNEMBEDDING_LR = 0.004\n"
           "SCALAR_LR = 0.5\nWEIGHT_DECAY = 0.2\nWARMUP_RATIO = 0.0\nASPECT_RATIO = 64\n")
    rng = random.Random(0)
    for _ in range(20):
        desc, blocks = parse(random_propose(rng, src))
        out = apply(src, blocks)
        changed = [a for a, b in zip(src.splitlines(), out.splitlines(), strict=True) if a != b]
        assert len(changed) <= 1 and desc.startswith("(random)")
    log = "---\nval_bpb:          0.997900\npeak_vram_mb:     45060.2\nnum_steps:        953\n"
    assert summary(log) == {"val_bpb": 0.9979, "peak_vram_mb": 45060.2, "num_steps": 953.0}
