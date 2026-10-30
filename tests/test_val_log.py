import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from val_log import first_readings, main, parse, rows_right  # noqa: E402

P = "\x1b[36m(TaskRunnerV1 pid=1)\x1b[0m "  # how Ray prefixes the trainer's lines
LOG = [
    P + "step:0 - val-aux/finqa/reward/mean@1:np.float64(0.75) - val-core/finqa/acc/mean@1:np.float64(0.75)"
    " - val-core/tatqa/acc/mean@1:np.float64(0.5)",
    P + "step:1 - actor/entropy:0.16 - critic/score/mean:0.7",  # a training step: no val
    P + " \"np.float64(0.75), 'val-core/finqa/acc/mean@1': \"",  # the end-of-leg metrics dict: no step
    P + "step:10 - actor/entropy:0.15 - val-core/finqa/acc/mean@1:np.float64(1.0) - val-core/tatqa/acc/mean@1:np.float64(0.5)",
    P + "step:10 - val-core/finqa/acc/mean@1:np.float64(0.5) - val-core/tatqa/acc/mean@1:np.float64(0.75)",  # re-read
    P + "step:20 - val-core/finqa/acc/mean@1:0.75 - val-core/tatqa/acc/mean@1:0.75",  # a plain float
]
ROWS = {"finqa": 4, "tatqa": 4}


def test_parse_keeps_val_lines_in_order():
    r = parse(LOG)
    assert [s for s, _ in r] == [0, 10, 10, 20]
    assert r[0][1] == {"finqa": 0.75, "tatqa": 0.5}


def test_a_boundary_counts_its_first_reading():
    assert first_readings(parse(LOG))[10] == {"finqa": 1.0, "tatqa": 0.5}  # not the next leg's re-read


def test_rows_right_checks_the_counts():
    assert rows_right({"finqa": 0.75, "tatqa": 0.5}, ROWS) == (5, 8)
    with pytest.raises(ValueError, match="whole number"):
        rows_right({"finqa": 0.75, "tatqa": 0.5}, {"finqa": 5, "tatqa": 4})
    with pytest.raises(ValueError, match="sources"):
        rows_right({"finqa": 0.75}, ROWS)


def test_means_and_difference(tmp_path, capsys):
    log = tmp_path / "rl.log"
    log.write_text("\n".join(LOG) + "\n")
    main([str(log), "--rows", "finqa=4,tatqa=4", "--mean", "0,10", "--mean", "20"])
    out = capsys.readouterr().out
    assert "all 0.6875 (5.50/8 rows)" in out  # steps 0 and 10 (first reading): 5 and 6 of 8
    assert "difference (second - first): +0.0625" in out  # step 20: 6 of 8
    with pytest.raises(SystemExit):
        main([str(log), "--mean", "30"])
