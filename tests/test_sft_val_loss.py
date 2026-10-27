import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from sft_val_loss import masked_nll  # noqa: E402


def lp(token: int, p: float) -> dict:
    """vLLM's prompt_logprobs entry for one position: {token id: Logprob}."""
    return {token: SimpleNamespace(logprob=math.log(p))}


def test_only_target_positions_count():
    ids = [7, 8, 9, 10]
    mask = [0, 0, 1, 1]  # prompt 7 8, target 9 10
    plp = [None, lp(8, 0.5), lp(9, 0.25), lp(10, 0.5)]
    nll, n = masked_nll(plp, ids, mask)
    assert n == 2
    assert math.isclose(nll, -math.log(0.25) - math.log(0.5))


def test_first_position_has_no_prediction():
    # a mask that starts at position 0 cannot score it: vLLM gives None there
    nll, n = masked_nll([None, lp(5, 0.5)], [4, 5], [1, 1])
    assert n == 1
    assert math.isclose(nll, math.log(2))
