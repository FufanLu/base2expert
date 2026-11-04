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

verl also checks the teachers against the pool where it builds its DistillationConfig, in the
TaskRunner and in the actor worker: their world sizes must sum to the pool's. Sharing needs each
teacher's to equal it instead, so DistillationConfig.__post_init__ is patched too (the MOPD smoke
11598610 failed on that check in the actor worker, 11-04).

Both happen in Ray processes, out of reach of the job's main process, so they are installed at
interpreter start-up: cluster/mopd_site/sitecustomize.py calls install() when
B2E_TEACHERS_SHARE_GPU=1, and install() patches each verl module as it is imported.
"""

from __future__ import annotations

import sys
from types import ModuleType

from b2e.distill.hook import on_import

TARGET = "verl.experimental.teacher_loop.teacher_model"
CONFIG = "verl.workers.config.distillation"


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


def shared_post_init(self) -> None:
    """DistillationConfig.__post_init__ with every teacher on the whole pool: each teacher's world size
    must be the pool's, where verl wants their sum to be."""
    if not self.enabled:
        return
    self.teacher_models = self._resolve_teacher_models()
    pool = self.n_gpus_per_node * self.nnodes
    loss = self.distillation_loss
    for key, t in self.teacher_models.items():
        t.validate_and_prepare_for_distillation(use_topk=loss.loss_settings.use_topk, topk=loss.topk)
        if t.world_size != pool:
            raise ValueError(f"teacher {key!r} needs {t.world_size} GPUs; sharing gives every teacher the whole "
                             f"pool of {pool}")


def patch_config(module: ModuleType) -> None:
    module.DistillationConfig.__post_init__ = shared_post_init  # type: ignore[attr-defined]
    print(f"b2e: {module.__name__} patched, each teacher's world size checked against the whole pool",
          file=sys.stderr, flush=True)


def patch(module: ModuleType) -> None:
    module.MultiTeacherModelManager._initialize_teacher_model_managers = shared_initialize  # type: ignore[attr-defined]
    module.B2E_TEACHERS_SHARE_GPU = True  # type: ignore[attr-defined]  # visible in logs and tests
    print(f"b2e: {module.__name__} patched, teachers share the teacher pool's GPUs", file=sys.stderr, flush=True)


def install(target: str = TARGET, config: str = CONFIG) -> None:
    """Patch each module now if it is imported, else when it is."""
    on_import(target, patch)
    on_import(config, patch_config)
