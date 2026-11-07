"""Open-MOPD's per-domain loss weights for multi-teacher on-policy distillation (P8, 11-03).

Plain MOPD averages the per-token distillation loss over every response token of a batch, so a
domain's share of the update is its share of the tokens, and a domain with long answers
(medicine, cap 8,192) outweighs one with short ones. Open-MOPD (arXiv 2608.19098) calls this a
misallocated token budget and reweights each domain's per-token loss:

- token-share balancing (its Eqs. 6, 8): s_d = n_d L_d / sum_j n_j L_j is domain d's share of the
  batch's response tokens, and w_d = g_d / s_d with a uniform target share g, so that after
  weighting every domain has the same effective share (sum_d w_d s_d = 1);
- gap-aware allocation (Eq. 9): w_d is multiplied by clamp((m_d / m_ref) ** alpha, 0.05, 20) and
  renormalised to sum_d w_d s_d = 1, where m_d is a running mean of domain d's per-token reward
  magnitude (|distillation loss|, a proxy for how far the student still is from that teacher)
  and m_ref the mean of m_d over the batch's domains. alpha = 1, the paper's (alpha < 0 collapsed
  its run).

A domain here is a teacher routing key (the `teacher` column of b2e.data.mopd_mix). The paper
does not give the running mean's decay; ours is an EMA with decay 0.9, updated every step from
that step's losses at rollout time. Its third fix, refreshing the reward per mini-batch, is not
done (P8 compares one change at a time).

How it is wired into verl 0.9.1, installed by cluster/mopd_site/sitecustomize.py when
B2E_MOPD_BALANCE is "share" or "gap":
- in the TaskRunner, the trainer's _update_actor is wrapped: before the update it gives every
  sample one weight, from the routing key and response length (and for "gap" the rollout-time
  per-token losses, verl's estimator between old_log_probs and the teacher's log-probs). verl
  0.9.1 runs its V1 trainer by default (trainer.use_v1, TaskRunnerV1): PPOTrainer in
  verl.trainer.ppo.v1.trainer_base, whose batch is a KVBatchMeta, keys into TransferQueue, so the
  weights are put there as a field the actor worker then reads. The legacy RayPPOTrainer is
  patched too, putting them into the DataProto. Patching only the legacy one left the 11-07 gap
  smoke unweighted with no error, since its check sat in a wrapper that never ran (F#113);
- in the actor worker, every registered distillation loss is wrapped to multiply its per-token
  losses by the sample's weight, and to stop when a sample has none: with the wrapper installed
  every training loss must be weighted, whichever trainer ran. verl clamps per-token losses at
  loss_max_clamp after the loss function returns, which would cut the weighted values, so a
  weighted run sets loss_max_clamp to null and the wrapper clamps the unweighted losses at
  RAW_CLAMP (verl's default) itself.
"""

from __future__ import annotations

import functools
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from types import ModuleType

from b2e.distill.hook import on_import

MODES = ("share", "gap")
ALPHA = 1.0
GAP_CLAMP = (0.05, 20.0)
DECAY = 0.9
RAW_CLAMP = 10.0
WEIGHT = "b2e_domain_weight"  # per-sample weight, a key of the batch
APPLIED = "b2e/weighted"  # set by the loss wrapper; the trainer checks it came back
TRAINER = "verl.trainer.ppo.ray_trainer"  # the legacy trainer (trainer.use_v1=false)
TRAINER_V1 = "verl.trainer.ppo.v1.trainer_base"
LOSSES = "verl.trainer.distillation.losses"


def token_shares(tokens: Mapping[str, float]) -> dict[str, float]:
    total = sum(tokens.values())
    return {d: t / total for d, t in tokens.items()}


def share_weights(tokens: Mapping[str, float]) -> dict[str, float]:
    """w_d = g_d / s_d with g uniform over the batch's domains (Eq. 8)."""
    s = token_shares(tokens)
    return {d: 1 / (len(s) * s[d]) for d in s}


def gap_weights(tokens: Mapping[str, float], gaps: Mapping[str, float], alpha: float = ALPHA) -> dict[str, float]:
    """The share weights scaled by each domain's relative gap, renormalised (Eq. 9)."""
    s = token_shares(tokens)
    w = share_weights(tokens)
    ref = sum(gaps[d] for d in s) / len(s)
    if ref <= 0:  # no gap measured anywhere: the share weights
        return w
    lo, hi = GAP_CLAMP
    raw = {d: w[d] * min(max((gaps[d] / ref) ** alpha, lo), hi) for d in s}
    z = sum(raw[d] * s[d] for d in s)
    return {d: raw[d] / z for d in s}


class RunningGap:
    """m_d: an EMA of each domain's mean per-token |loss|, the first value taken as it is."""

    def __init__(self, decay: float = DECAY) -> None:
        self.decay = decay
        self.m: dict[str, float] = {}

    def update(self, means: Mapping[str, float]) -> dict[str, float]:
        for d, x in means.items():
            self.m[d] = x if d not in self.m else self.decay * self.m[d] + (1 - self.decay) * x
        return dict(self.m)


def sample_weights(keys: Sequence[str], lengths: Sequence[float], mode: str, running: RunningGap | None = None,
                   loss_sums: Sequence[float] | None = None) -> tuple[list[float], dict[str, float]]:
    """One weight per sample, and the per-domain shares, weights (and gaps) as metrics."""
    tokens: dict[str, float] = defaultdict(float)
    for k, n in zip(keys, lengths, strict=True):
        tokens[k] += n
    tokens = {k: n for k, n in tokens.items() if n > 0}
    metrics = {f"b2e/share/{d}": s for d, s in token_shares(tokens).items()}
    if mode == "share":
        w = share_weights(tokens)
    elif mode == "gap":
        if running is None or loss_sums is None:
            raise ValueError("gap weights need the running gap and the per-sample loss sums")
        sums: dict[str, float] = defaultdict(float)
        for k, x in zip(keys, loss_sums, strict=True):
            sums[k] += x
        m = running.update({d: sums[d] / tokens[d] for d in tokens})
        w = gap_weights(tokens, m)
        metrics.update({f"b2e/gap/{d}": m[d] for d in tokens})
    else:
        raise ValueError(f"B2E_MOPD_BALANCE must be one of {MODES}, got {mode!r}")
    metrics.update({f"b2e/weight/{d}": x for d, x in w.items()})
    return [w.get(k, 0.0) for k in keys], metrics  # a key absent from w has no response tokens


def rollout_token_losses(trainer, batch):
    """The per-token distillation loss at rollout time, (bsz, response_len), clamped as the loss is."""
    from verl.trainer.ppo.core_algos import kl_penalty

    cfg = trainer.distillation_config.distillation_loss
    if cfg.loss_settings.use_topk:
        raise NotImplementedError("gap weights need the teacher's log-prob of the sampled token; a top-k loss is "
                                  "computed in the actor from the student's logits, so use B2E_MOPD_BALANCE=share")
    p, r = batch.batch["prompts"].shape[1], batch.batch["responses"].shape[1]
    # teacher_logprobs run over prompt + response, (bsz, p + r, 1); the response's are the ones
    # verl's loss takes (no_padding_2_padding shifts model outputs by one token)
    teacher = batch.batch["teacher_logprobs"][:, p - 1 : p - 1 + r].squeeze(-1)
    losses = kl_penalty(logprob=batch.batch["old_log_probs"], ref_logprob=teacher, kl_penalty=cfg.loss_mode)
    return losses.clamp(-RAW_CLAMP, RAW_CLAMP)


def patch_trainer(module: ModuleType) -> None:
    cls = module.RayPPOTrainer
    update = cls._update_actor
    running = RunningGap()

    @functools.wraps(update)
    def _update_actor(self, batch):
        import torch

        mode = os.environ["B2E_MOPD_BALANCE"]
        keys = [str(k) for k in batch.non_tensor_batch[self.distillation_config.teacher_key]]
        mask = batch.batch["response_mask"].float()
        sums = None
        if mode == "gap":
            sums = (rollout_token_losses(self, batch).abs() * mask).sum(-1).tolist()
        w, metrics = sample_weights(keys, mask.sum(-1).tolist(), mode, running, sums)
        batch.batch[WEIGHT] = torch.tensor(w, dtype=torch.float32)
        out = update(self, batch)
        got = out.meta_info["metrics"]
        if not any(k.endswith(APPLIED) for k in got):  # the actor never saw the weights: stop, not train unweighted
            raise RuntimeError("b2e: the distillation loss was not weighted; is the loss patch installed in the actor?")
        got.update(metrics)
        return out

    cls._update_actor = _update_actor
    print(f"b2e: {module.__name__} patched, MOPD loss weights ({os.environ.get('B2E_MOPD_BALANCE')})",
          file=sys.stderr, flush=True)


def padded(t, value=0):
    """A nested (jagged) tensor as a padded one; a padded tensor as it is."""
    return t.to_padded_tensor(value) if getattr(t, "is_nested", False) else t


def as_list(v) -> list:
    """A TransferQueue non-tensor field (NonTensorStack) or a list, as a list."""
    return v.tolist() if hasattr(v, "tolist") else list(v)


def v1_inputs(trainer, data, mode: str) -> tuple[list[float], list[float] | None]:
    """Each sample's response length and, for "gap", its summed rollout-time |loss|, from the fields
    the V1 trainer reads out of TransferQueue (nested tensors there, one row per sample)."""
    mask = padded(data["response_mask"]).float()
    if mode != "gap":
        return mask.sum(-1).tolist(), None
    from verl.trainer.ppo.core_algos import kl_penalty
    from verl.workers.utils.padding import no_padding_2_padding

    cfg = trainer.distillation_config.distillation_loss
    if cfg.loss_settings.use_topk:
        raise NotImplementedError("gap weights need the teacher's log-prob of the sampled token; a top-k loss is "
                                  "computed in the actor from the student's logits, so use B2E_MOPD_BALANCE=share")
    # teacher_logprobs cover prompt + response; no_padding_2_padding takes the response's, shifted by one
    # token as the actor's loss does, padded to the batch's longest response like old_log_probs here
    teacher = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    losses = kl_penalty(logprob=padded(data["old_log_probs"], 0.0), ref_logprob=teacher, kl_penalty=cfg.loss_mode)
    return mask.sum(-1).tolist(), (losses.clamp(-RAW_CLAMP, RAW_CLAMP).abs() * mask).sum(-1).tolist()


def weight_field(w: list[float]):
    import torch
    from tensordict import TensorDict

    return TensorDict({WEIGHT: torch.tensor(w, dtype=torch.float32)}, batch_size=[len(w)])


def patch_trainer_v1(module: ModuleType) -> None:
    cls = module.PPOTrainer
    update = cls._update_actor
    running = RunningGap()

    @functools.wraps(update)
    def _update_actor(self, batch, metrics):
        mode = os.environ["B2E_MOPD_BALANCE"]
        key = self.distillation_config.teacher_key
        fields = [key, "prompts", "responses", "response_mask"]
        if mode == "gap":
            fields += ["old_log_probs", "teacher_logprobs"]
        data = module.tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        lengths, sums = v1_inputs(self, data, mode)
        # samples copied in to make the batch divisible (tag is_padding) count in no domain's share
        real = [not t.get("is_padding", False) for t in batch.tags]
        lengths = [n if r else 0.0 for n, r in zip(lengths, real, strict=True)]
        if sums is not None:
            sums = [x if r else 0.0 for x, r in zip(sums, real, strict=True)]
        w, m = sample_weights([str(k) for k in as_list(data[key])], lengths, mode, running, sums)
        module.tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=weight_field(w))
        if batch.fields is not None:  # the actor reads only the fields listed, when there is a list
            listed = [batch.fields] if isinstance(batch.fields, str) else list(batch.fields)
            batch.fields = listed + ([WEIGHT] if WEIGHT not in listed else [])
        out = update(self, batch, metrics)
        if not any(k.endswith(APPLIED) for k in metrics):  # the actor never saw the weights: stop
            raise RuntimeError("b2e: the distillation loss was not weighted; is the loss patch installed in the actor?")
        metrics.update(m)
        return out

    cls._update_actor = _update_actor
    print(f"b2e: {module.__name__} patched, MOPD loss weights ({os.environ.get('B2E_MOPD_BALANCE')})",
          file=sys.stderr, flush=True)


def applied_metric():
    """The value the loss wrapper logs as APPLIED: verl's Metric where verl has one (its V1 actor
    aggregates Metric objects), else a number."""
    try:
        from verl.utils.metric import AggregationType, Metric
    except ImportError:
        return 1.0
    return Metric(AggregationType.MEAN, 1.0)


def weighted(fn):
    """A registered distillation loss whose per-token losses are scaled by the sample's weight."""

    @functools.wraps(fn)
    def loss_fn(config, distillation_config, model_output, data):
        losses, metrics = fn(config=config, distillation_config=distillation_config, model_output=model_output,
                             data=data)
        w = data.get(WEIGHT, None)
        if w is None:  # installed means weighted: a loss without weights is a trainer that was not patched
            raise RuntimeError(f"b2e: B2E_MOPD_BALANCE is set but the batch has no {WEIGHT}; the trainer that ran "
                               "was not patched (F#113)")
        if distillation_config.distillation_loss.loss_max_clamp is not None:
            raise ValueError("weighted distillation losses need distillation.distillation_loss.loss_max_clamp=null "
                             f"(the weights are applied after a clamp at {RAW_CLAMP})")
        losses = losses.clamp(-RAW_CLAMP, RAW_CLAMP) * w.to(device=losses.device, dtype=losses.dtype).unsqueeze(-1)
        metrics[APPLIED] = applied_metric()
        return losses, metrics

    return loss_fn


def patch_losses(module: ModuleType) -> None:
    registry = module.DISTILLATION_LOSS_REGISTRY
    for name, fn in list(registry.items()):
        registry[name] = weighted(fn)
    print(f"b2e: {module.__name__} patched, {len(registry)} losses take per-sample weights", file=sys.stderr,
          flush=True)


def install() -> None:
    mode = os.environ.get("B2E_MOPD_BALANCE")
    if mode not in MODES:
        raise ValueError(f"B2E_MOPD_BALANCE must be one of {MODES}, got {mode!r}")
    on_import(TRAINER, patch_trainer)
    on_import(TRAINER_V1, patch_trainer_v1)
    on_import(LOSSES, patch_losses)
