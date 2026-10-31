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
    tried: set[str] = set()
    for _ in range(20):
        desc, blocks = parse(random_propose(rng, src, tried))
        out = apply(src, blocks)
        changed = [a for a, b in zip(src.splitlines(), out.splitlines(), strict=True) if a != b]
        assert len(changed) == 1 and desc.startswith("(random)")
        assert desc not in tried  # F#36: never the same change twice
        tried.add(desc)
    log = "---\nval_bpb:          0.997900\npeak_vram_mb:     45060.2\nnum_steps:        953\n"
    assert summary(log) == {"val_bpb": 0.9979, "peak_vram_mb": 45060.2, "num_steps": 953.0}


def test_random_search_stops_when_every_change_was_tried():
    import random

    from agent import RANDOM_SPACE, random_propose

    src = "DEPTH = 8\n"
    tried = {f"(random) DEPTH 8 -> {v}" for v in RANDOM_SPACE["DEPTH"]}
    assert random_propose(random.Random(0), src, tried) is None


def test_confirm_rule_pools_repeats_and_ignores_decision_rows():
    from agent import keep_margin, mean, pooled_sd, runs_by_commit

    tsv = ("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
           "base\t0.996\t44.0\tkeep\tbaseline\n"
           "base\t0.998\t44.0\tnoise\tbaseline repeat 1\n"
           "base\t1.000\t44.0\tnoise\tbaseline repeat 2\n"
           "c1\t1.010\t44.0\tdiscard\tworse\n"
           "c2\t0.000\t0.0\tcrash\toom\n"
           "c3\t0.990\t44.0\tscreen\tbetter?\n"
           "base\t0.997\t44.0\trecheck\tbest again, alongside: better?\n"
           "c3\t0.992\t44.0\tconfirm\tbetter?\n"
           "c3\t0.991\t44.0\trejected\tbetter? | mean of 2 runs\n")  # a decision row: a mean, not a run
    g = runs_by_commit(tsv)
    assert g == {"base": [0.996, 0.998, 1.0, 0.997], "c1": [1.01], "c3": [0.99, 0.992]}
    assert mean(g["c3"]) == pytest.approx(0.991)
    # pooled over base (4 runs) and c3 (2 runs): (sum of squares 8.75e-6 + 2e-6) / (3 + 1) degrees of freedom
    assert pooled_sd(g) == pytest.approx((10.75e-6 / 4) ** 0.5)
    assert pooled_sd({"a": [1.0]}) is None and mean([]) == float("inf")
    # two standard errors of a 3-run mean against a 2-run mean, or min_gain when that is larger
    assert keep_margin(0.002, 3, 2, 0.001) == pytest.approx(2 * 0.002 * (1 / 3 + 1 / 2) ** 0.5)
    assert keep_margin(0.0001, 3, 2, 0.001) == 0.001 and keep_margin(None, 3, 2, 0.001) == 0.001


def test_loop_with_confirm_keeps_a_real_gain_only_after_reruns(tmp_path, monkeypatch):
    import json
    import subprocess

    import agent

    # A stand-in train.py: val_bpb is 1.0, 0.01 lower when MATRIX_LR > 0.05, plus the next value of a
    # fixed noise sequence (so every run of the same code differs a little, as on the real GPU).
    repo, noise = tmp_path / "repo", tmp_path / "noise.json"
    repo.mkdir()
    noise.write_text(json.dumps([0.0, 0.001, -0.001] * 20))
    (repo / "train.py").write_text(
        "MATRIX_LR = 0.04  # learning rate\n"
        "import json, os, pathlib\n"
        "p = pathlib.Path(os.environ['AR_TEST_NOISE'])\n"
        "seq = json.loads(p.read_text())\n"
        "x = seq.pop(0)\n"
        "p.write_text(json.dumps(seq))\n"
        "print('---')\n"
        "print(f'val_bpb: {1.0 - (0.01 if MATRIX_LR > 0.05 else 0.0) + x:.6f}')\n"
        "print('peak_vram_mb: 1000.0')\n")
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                ["add", "train.py"], ["commit", "-qm", "base"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    monkeypatch.setenv("AR_TEST_NOISE", str(noise))
    monkeypatch.setattr(sys, "argv", ["agent.py", "--repo", str(repo), "--python", sys.executable,
                                      "--proposer", "random", "--rounds", "7", "--noise-runs", "2",
                                      "--confirm", "2", "--min-gain", "0.001"])
    agent.main()

    rows = [r.split("\t") for r in (repo / "results.tsv").read_text().splitlines()[1:]]
    status = [r[3] for r in rows]
    assert status[:3] == ["keep", "noise", "noise"]
    # the first change that raises MATRIX_LR above 0.05 is the one kept, after two reruns of each side
    k = status.index("kept")
    assert status[k - 5:k] == ["screen", "recheck", "confirm", "recheck", "confirm"]
    assert status.count("kept") == 1
    assert rows[k - 4][0] == rows[0][0]  # the rechecks train the best commit, the baseline here
    text = (repo / "train.py").read_text()
    assert float(text.split("=")[1].split("#")[0]) > 0.05  # HEAD and the file are the kept change
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                          capture_output=True, text=True).stdout == ""
    runs = (repo / "runs.jsonl").read_text().splitlines()
    assert len(runs) == len(rows) and json.loads(runs[0])["val_bpb"] == 1.0
