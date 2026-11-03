import sys
import textwrap
import types

import pytest

from b2e.distill import share_gpu

FAKE = "b2e_fake_teacher_model"


def fake_module(tmp_path, monkeypatch):
    """A stand-in for verl's teacher module: managers that record the pool they were given."""
    (tmp_path / f"{FAKE}.py").write_text(textwrap.dedent("""
        class TeacherModelManager:
            def __init__(self, distillation_config, teacher_model_config, resource_pool):
                self.pool = resource_pool
                self.server_addresses = [teacher_model_config.key]
                self.server_handles = []
                self.load_balancer_handle = None

        class MultiTeacherModelManager:
            def __init__(self, distillation_config, resource_pool):
                self.distillation_config = distillation_config
                self.resource_pool = resource_pool
                self.teacher_model_managers, self.server_addresses = {}, {}
                self.server_handles, self.load_balancer_handle = {}, {}
                self._initialize_teacher_model_managers()

            def _initialize_teacher_model_managers(self):
                raise RuntimeError("unpatched: splits the pool")
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, FAKE, raising=False)


def cfg(n_teachers, world_size=1):
    teachers = {k: types.SimpleNamespace(key=k, world_size=world_size) for k in ["fin", "law", "med"][:n_teachers]}
    return types.SimpleNamespace(teacher_models=teachers)


def test_the_hook_patches_the_module_when_it_is_imported(tmp_path, monkeypatch):
    fake_module(tmp_path, monkeypatch)
    hooks = list(sys.meta_path)
    try:
        share_gpu.install(FAKE)
        import importlib

        mod = importlib.import_module(FAKE)
        pool = types.SimpleNamespace(world_size=1, max_colocate_count=3)
        m = mod.MultiTeacherModelManager(cfg(3), pool)
        assert mod.B2E_TEACHERS_SHARE_GPU
        assert set(m.teacher_model_managers) == {"fin", "law", "med"}
        assert all(t.pool is pool for t in m.teacher_model_managers.values())  # the whole pool, not slices
        assert m.server_addresses == {"fin": ["fin"], "law": ["law"], "med": ["med"]}
    finally:
        sys.meta_path[:] = hooks


def test_sharing_refuses_what_does_not_fit(tmp_path, monkeypatch):
    fake_module(tmp_path, monkeypatch)
    import importlib

    mod = importlib.import_module(FAKE)
    share_gpu.patch(mod)
    with pytest.raises(ValueError, match="holds 2 worker groups"):
        mod.MultiTeacherModelManager(cfg(3), types.SimpleNamespace(world_size=1, max_colocate_count=2))
    with pytest.raises(ValueError, match="needs 2 GPUs"):
        mod.MultiTeacherModelManager(cfg(2, world_size=2), types.SimpleNamespace(world_size=1, max_colocate_count=3))


def test_the_mopd_mix_routes_each_domain_to_its_teacher(tmp_path):
    import pandas as pd

    from b2e.data.mopd_mix import main

    for name, src, n in (("fq", "finqa", 5), ("tq", "tatqa", 5), ("law", "cail2018_term", 3)):
        pd.DataFrame({"data_source": [src] * n, "x": range(n)}).to_parquet(tmp_path / f"{name}.parquet")
    out = tmp_path / "mix"
    main(["--out", str(out), "--n-train", "4", "--train", f"fin={tmp_path}/fq.parquet,{tmp_path}/tq.parquet",
          f"law={tmp_path}/law.parquet", "--val", f"fin={tmp_path}/fq.parquet", f"law={tmp_path}/law.parquet"])
    tr = pd.read_parquet(out / "train.parquet")
    assert tr.groupby("teacher").size().to_dict() == {"fin": 4, "law": 3}  # law has only 3
    assert set(tr[tr.teacher == "fin"].data_source) <= {"finqa", "tatqa"}  # sources kept for the val metrics
    assert pd.read_parquet(out / "val.parquet").groupby("teacher").size().to_dict() == {"fin": 5, "law": 3}
    assert (out / "rl_train.txt").read_text().strip().endswith("train.parquet")
