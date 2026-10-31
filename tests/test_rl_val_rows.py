import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from rl_val_rows import val_indices  # noqa: E402


def test_draw_is_fixed_by_the_seed_and_has_no_repeats():
    a = val_indices(1000, 512, 42)
    assert a == val_indices(1000, 512, 42)
    assert len(set(a)) == 512 and all(0 <= i < 1000 for i in a)
    assert a != val_indices(1000, 512, 43)


def test_all_rows_in_order_when_the_limit_does_not_bind():
    assert val_indices(5, 5, 42) == [0, 1, 2, 3, 4]
    assert val_indices(5, 8, 42) == [0, 1, 2, 3, 4]
    assert val_indices(5, -1, 42) == [0, 1, 2, 3, 4]
