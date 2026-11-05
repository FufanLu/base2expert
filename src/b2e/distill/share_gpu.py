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

The patched check also refuses a teacher whose vLLM would load dummy weights. A teacher added with
`+distillation.teacher_models.<key>.*` does not inherit the `teacher_model` template of verl's
distillation.yaml (load_format auto, enforce_eager true, ...): it gets RolloutConfig's defaults,
load_format "dummy" among them, which suits the student's rollout engine (the trainer syncs its
weights) but not a teacher, whose weights nothing ever loads. The MOPD smoke 11626637 ran its four
steps that way: every teacher near uniform, the k3 loss at its clamp of 10 on almost every token
(F#104).

Both happen in Ray processes, out of reach of the job's main process, so they are installed at
interpreter start-up: cluster/mopd_site/sitecustomize.py calls install() when
B2E_TEACHERS_SHARE_GPU=1, and install() patches each verl module as it is imported.

Three is verl's limit, not the GPU's: ResourcePoolManager builds every pool with
max_colocate_count 3 (a dataclass field nothing sets). P8 has four teachers (R-fin, R-fin-low,
R-law, R-med), so the teacher pool, and only it, is built to hold B2E_TEACHERS_PER_GPU worker
groups when that is more than 3 (cluster/mopd.sbatch sets it to the number of teachers).
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

from b2e.distill.hook import on_import

TARGET = "verl.experimental.teacher_loop.teacher_model"
CONFIG = "verl.workers.config.distillation"
POOLS = "verl.single_controller.ray.base"
TEACHER_POOL = "teacher_pool"  # the name verl's trainer gives the teacher pool in resource_pool_spec


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
        if t.inference.load_format == "dummy":
            raise ValueError(f"teacher {key!r} would serve dummy weights (RolloutConfig's default load_format): "
                             "set its inference.load_format to auto")
        t.validate_and_prepare_for_distillation(use_topk=loss.loss_settings.use_topk, topk=loss.topk)
        if t.world_size != pool:
            raise ValueError(f"teacher {key!r} needs {t.world_size} GPUs; sharing gives every teacher the whole "
                             f"pool of {pool}")


def shared_create_resource_pool(self) -> None:
    """ResourcePoolManager.create_resource_pool, with the teacher pool holding B2E_TEACHERS_PER_GPU
    worker groups when that is more than the default; every other pool as verl builds it."""
    ray_base = sys.modules[type(self).__module__]
    per_gpu = int(os.environ.get("B2E_TEACHERS_PER_GPU", "0"))
    for name, process_on_nodes in self.resource_pool_spec.items():
        count = max(self.max_colocate_count, per_gpu) if name == TEACHER_POOL else self.max_colocate_count
        self.resource_pool_dict[name] = ray_base.RayResourcePool(
            process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=count, name_prefix=name)
    self._check_resource_available()


def patch_pools(module: ModuleType) -> None:
    module.ResourcePoolManager.create_resource_pool = shared_create_resource_pool  # type: ignore[attr-defined]
    print(f"b2e: {module.__name__} patched, the teacher pool holds "
          f"{os.environ.get('B2E_TEACHERS_PER_GPU', 'the default number of')} worker groups", file=sys.stderr, flush=True)


def patch_config(module: ModuleType) -> None:
    module.DistillationConfig.__post_init__ = shared_post_init  # type: ignore[attr-defined]
    print(f"b2e: {module.__name__} patched, each teacher's world size checked against the whole pool",
          file=sys.stderr, flush=True)


def patch(module: ModuleType) -> None:
    module.MultiTeacherModelManager._initialize_teacher_model_managers = shared_initialize  # type: ignore[attr-defined]
    module.B2E_TEACHERS_SHARE_GPU = True  # type: ignore[attr-defined]  # visible in logs and tests
    print(f"b2e: {module.__name__} patched, teachers share the teacher pool's GPUs", file=sys.stderr, flush=True)


def install(target: str = TARGET, config: str = CONFIG, pools: str = POOLS) -> None:
    """Patch each module now if it is imported, else when it is."""
    on_import(target, patch)
    on_import(config, patch_config)
    on_import(pools, patch_pools)
