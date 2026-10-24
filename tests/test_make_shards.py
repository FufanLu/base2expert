import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def test_mix_by_bytes(tmp_path: Path):
    (tmp_path / "en").mkdir()
    (tmp_path / "zh").mkdir()
    pq.write_table(pa.table({"text": ["english text " * 20] * 2000}), tmp_path / "en" / "a.parquet")
    pq.write_table(pa.table({"text": ["中文文本" * 20] * 2000}), tmp_path / "zh" / "a.parquet")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "pretrain/make_shards.py", "--out", str(out),
                        "--source", f"en=0.7:{tmp_path}/en/*.parquet", "--source", f"zh=0.3:{tmp_path}/zh/*.parquet",
                        "--shard-bytes", "100000", "--val-bytes", "20000", "--total-bytes", "300000"],
                       capture_output=True, text=True, check=True)
    m = json.loads((out / "manifest.json").read_text())
    files = sorted(p.name for p in out.glob("*.parquet"))
    assert files[-1] == "shard_99999.parquet"  # validation sorts last
    assert len(m["train_shards"]) == 3
    en, zh = m["per_source"]["en"]["bytes"], m["per_source"]["zh"]["bytes"]
    assert abs(en / (en + zh) - 0.7) < 0.02
    t = pq.read_table(out / "shard_00000.parquet")
    assert t.column_names == ["text"]
    assert r.returncode == 0
