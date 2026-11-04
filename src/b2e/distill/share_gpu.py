"""Several teachers on one GPU for multi-teacher on-policy distillation (P8, P0's option (c)).

verl 0.9.1 gives each teacher a disjoint slice of the teacher resource pool:
MultiTeacherModelManager._initialize_teacher_model_managers splits the pool by each teacher's
world size, so three single-GPU teachers need three GPUs, and with the student that is four in
one Ray cluster, more than any one job gets under the QoS (gpu: 1 GPU a job; gpu-short: 2;
docs/runs.md, 11-03). The pool itself is built with max_colocate_count=3, and every worker group
on it asks for 1 / max_colocate_count of a GPU (RayWorkerGroup), so up to three single-GPU
teachers fit on one GPU if each gets the whole pool instead of a slice. That is the one change
made here. Each teacher's vLLM must then be given about a third of the GPU
(inference.gpu_memory_utilization).

It has to happen in the process that builds the teachers, the trainer's TaskRunner (a Ray
actor), so it is installed at interpreter start-up: cluster/mopd_site/sitecustomize.py calls
install() when B2E_TEACHERS_SHARE_GPU=1, and install() patches verl's teacher module as it is
imported.
"""

from __future__ import annotations

import sys
from types import ModuleType

from b2e.distill.hook import on_import

TARGET = "verl.experimental.teacher_loop.teacher_model"


def shared_initialize(self) -> None:
    """MultiTeacherModelManager._initialize_teacher_model_managers, with the whole pool for every teacher."""
    tm = sys.modules[type(self).__module__]
    teachers = self.distillation_config.teacher_models
    pool = self.resource_pool
    for key, cfg in teachers.items():
        if cfg.world_size != pool.world_size:
            raise ValueError(f"teacher {key!r} needs {cfg.world_size} GPUs; sharing gives every teacher the whole "
                             f"pool of {pool.world_size}")
    if len(teachers) > pool.max_colocate_count:
        raise ValueError(f"{len(teachers)} teachers on one pool, but it holds {pool.max_colocate_count} worker groups")
    for key, cfg in teachers.items():
        manager = tm.TeacherModelManager(distillation_config=self.distillation_config, teacher_model_config=cfg,
                                         resource_pool=pool)
        self.teacher_model_managers[key] = manager
        self.server_addresses[key] = manager.server_addresses
        self.server_handles[key] = manager.server_handles
        self.load_balancer_handle[key] = manager.load_balancer_handle


def patch(module: ModuleType) -> None:
    module.MultiTeacherModelManager._initialize_teacher_model_managers = shared_initialize  # type: ignore[attr-defined]
    module.B2E_TEACHERS_SHARE_GPU = True  # type: ignore[attr-defined]  # visible in logs and tests
    print(f"b2e: {module.__name__} patched, teachers share the teacher pool's GPUs", file=sys.stderr, flush=True)


def install(target: str = TARGET) -> None:
    """Patch the module now if it is imported, else when it is."""
    on_import(target, patch)
