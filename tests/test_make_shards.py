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


def test_exclude_drops_held_out_documents(tmp_path: Path):
    (tmp_path / "src").mkdir()
    texts = [f"document number {i} " * 10 for i in range(300)]
    pq.write_table(pa.table({"text": texts}), tmp_path / "src" / "a.parquet")
    pq.write_table(pa.table({"text": texts[:40]}), tmp_path / "heldout.parquet")
    out = tmp_path / "out"
    subprocess.run([sys.executable, "pretrain/make_shards.py", "--out", str(out),
                    "--source", f"all=1.0:{tmp_path}/src/*.parquet", "--exclude", str(tmp_path / "heldout.parquet"),
                    "--shard-bytes", "20000", "--val-bytes", "5000"],
                   capture_output=True, text=True, check=True)
    m = json.loads((out / "manifest.json").read_text())
    assert m["exclude"]["documents"] == 40
    assert m["exclude"]["dropped"] == {"all": 40}
    written = [t for p in out.glob("*.parquet") for t in pq.read_table(p).column("text").to_pylist()]
    assert set(written) == set(texts[40:])
