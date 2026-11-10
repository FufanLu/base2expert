import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("fit_scaling", Path("pretrain/fit_scaling.py"))
fit_scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit_scaling)

LOG = """Parameter counts:
wte                     : 33,554,432
total                   : {total:,}
Total number of training tokens: {tokens:,}
Total training FLOPs estimate: {flops:e}
Step 00250 | Validation bpb: 9.000000
Step {last:05d} | Validation bpb: {bpb:.6f}
Step {last:05d} | CORE metric: 0.1234
"""


def test_parse_takes_the_last_val_bpb():
    row = fit_scaling.parse_log(LOG.format(total=1_000_000, tokens=12_000_000, flops=3.2e17, last=900, bpb=1.05))
    assert row == {"flops": 3.2e17, "tokens": 12_000_000, "params_total": 1_000_000, "val_bpb": 1.05, "core": 0.1234}


def test_power_law_is_recovered_and_predicts_the_held_out_size(tmp_path: Path):
    flops = {6: 1e16, 8: 3e16, 10: 1e17, 12: 3e17, 14: 1e18}
    for d, c in flops.items():
        bpb = 5.0 * c ** -0.05
        (tmp_path / f"m_d{d}_train.log").write_text(LOG.format(total=d, tokens=d, flops=c, last=d, bpb=bpb))
        (tmp_path / f"m_d{d}_heldout.json").write_text(json.dumps({"sources": {"fin_zh": {"bpb": round(2 * bpb, 4)}}}))
    a, b = fit_scaling.fit_power(list(flops.values())[:4], [5.0 * c ** -0.05 for c in list(flops.values())[:4]])
    assert math.isclose(b, -0.05, rel_tol=1e-9) and math.isclose(math.exp(a), 5.0, rel_tol=1e-9)
    out = tmp_path / "fit.json"
    subprocess.run([sys.executable, "pretrain/fit_scaling.py", "--results", str(tmp_path), "--base", "m",
                    "--fit", "6", "8", "10", "12", "--predict", "14", "--out", str(out)], check=True, capture_output=True)
    f = json.loads(out.read_text())["fits"]
    assert abs(f["val_bpb"]["predict"]["14"]["error_rel"]) < 1e-3
    assert abs(f["heldout_fin_zh"]["predict"]["14"]["error_rel"]) < 1e-3
