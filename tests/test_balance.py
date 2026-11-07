import sys
import types

import pytest

from b2e.distill import balance
from b2e.distill.balance import (
    APPLIED,
    WEIGHT,
    RunningGap,
    gap_weights,
    sample_weights,
    share_weights,
    token_shares,
    weighted,
)


def effective(tokens, w):
    s = token_shares(tokens)
    return {d: w[d] * s[d] for d in s}


def test_share_weights_give_every_domain_the_same_effective_share():
    tokens = {"fin": 600, "law": 300, "med": 100}
    w = share_weights(tokens)
    assert w == pytest.approx({"fin": 1 / 1.8, "law": 1 / 0.9, "med": 1 / 0.3})
    assert effective(tokens, w) == pytest.approx({"fin": 1 / 3, "law": 1 / 3, "med": 1 / 3})


def test_gap_weights_move_the_budget_towards_the_larger_gap_and_keep_the_scale():
    tokens = {"a": 100, "b": 100, "c": 100}
    assert gap_weights(tokens, {"a": 1, "b": 1, "c": 1}) == pytest.approx(share_weights(tokens))
    w = gap_weights(tokens, {"a": 3, "b": 1, "c": 2})  # relative gaps 1.5, 0.5, 1
    assert w == pytest.approx({"a": 1.5, "b": 0.5, "c": 1.0})
    assert sum(effective(tokens, w).values()) == pytest.approx(1.0)
    # a relative gap below 0.05 counts as 0.05
    w = gap_weights(tokens, {"a": 1000, "b": 1, "c": 1})
    ref = 1002 / 3
    raw = {"a": 1000 / ref, "b": 0.05, "c": 0.05}
    z = sum(raw.values()) / 3
    assert w == pytest.approx({d: x / z for d, x in raw.items()})
    assert gap_weights(tokens, {"a": 0, "b": 0, "c": 0}) == pytest.approx(share_weights(tokens))


def test_the_running_gap_is_an_ema_that_starts_at_the_first_value():
    g = RunningGap(decay=0.9)
    assert g.update({"fin": 0.4}) == {"fin": 0.4}
    assert g.update({"fin": 0.2, "law": 1.0}) == pytest.approx({"fin": 0.38, "law": 1.0})


def test_sample_weights_follow_the_routing_key():
    keys = ["fin", "fin", "med", "law"]
    w, m = sample_weights(keys, [300, 300, 300, 100], "share")
    assert w == pytest.approx([1 / 1.8, 1 / 1.8, 1 / 0.9, 1 / 0.3])
    assert m["b2e/share/fin"] == pytest.approx(0.6) and m["b2e/weight/law"] == pytest.approx(1 / 0.3)
    # a domain without response tokens is left out of the shares; its samples weigh nothing
    w, m = sample_weights(["fin", "law"], [100, 0], "share")
    assert w == [1.0, 0.0] and "b2e/share/law" not in m


def test_gap_mode_reads_the_mean_per_token_loss_of_each_domain():
    g = RunningGap()
    w, m = sample_weights(["a", "b"], [100, 100], "gap", g, loss_sums=[30.0, 10.0])  # means 0.3, 0.1
    assert m["b2e/gap/a"] == pytest.approx(0.3) and m["b2e/gap/b"] == pytest.approx(0.1)
    assert w == pytest.approx([1.5, 0.5])
    with pytest.raises(ValueError, match="loss sums"):
        sample_weights(["a"], [1], "gap")
    with pytest.raises(ValueError, match="one of"):
        sample_weights(["a"], [1], "tokens")


def test_the_loss_wrapper_stops_on_an_unweighted_batch_and_refuses_a_clamp():
    def loss(config, distillation_config, model_output, data):
        return "losses", {"x": 1}

    fn = weighted(loss)
    cfg = types.SimpleNamespace(distillation_loss=types.SimpleNamespace(loss_max_clamp=10.0))
    # installed means weighted: no weights means the trainer that ran was not patched (F#113)
    with pytest.raises(RuntimeError, match="was not patched"):
        fn(config=None, distillation_config=cfg, model_output={}, data={})
    with pytest.raises(ValueError, match="loss_max_clamp=null"):
        fn(config=None, distillation_config=cfg, model_output={}, data={WEIGHT: object()})
    assert APPLIED.startswith("b2e/")  # logged by the actor as actor/b2e/weighted, which the trainer checks


class FakeTQ:
    def __init__(self, rows):
        self.rows, self.asked, self.put = rows, None, None

    def kv_batch_get(self, keys, partition_id, select_fields):
        self.asked = (list(keys), partition_id, list(select_fields))
        return {f: [self.rows[k][f] for k in keys] for f in select_fields if f in self.rows[keys[0]]}

    def kv_batch_put(self, keys, partition_id, fields):
        self.put = (list(keys), partition_id, fields)


def fake_v1(tq, applied=True):
    """verl's V1 PPOTrainer as far as the patch needs it: _update_actor(batch, metrics) runs the actor,
    whose metrics come back with actor/b2e/weighted when the loss wrapper saw the weights."""

    class PPOTrainer:
        distillation_config = types.SimpleNamespace(teacher_key="teacher")

        def _update_actor(self, batch, metrics):
            self.seen_fields = batch.fields
            metrics["actor/pg_loss"] = 0.5
            if applied:
                metrics["actor/" + APPLIED] = 1.0
            return batch

    mod = types.ModuleType("fake_trainer_base")
    mod.PPOTrainer, mod.tq = PPOTrainer, tq
    return mod


def test_the_v1_trainer_puts_weights_into_transfer_queue(monkeypatch):
    rows = {"a": {"teacher": "fin"}, "b": {"teacher": "fin"}, "c": {"teacher": "med"}, "d": {"teacher": "med"}}
    tq = FakeTQ(rows)
    mod = fake_v1(tq)
    balance.patch_trainer_v1(mod)
    monkeypatch.setenv("B2E_MOPD_BALANCE", "share")
    # response lengths 300, 300, 100, and a padding copy (tag is_padding) of 500 tokens
    monkeypatch.setattr(balance, "v1_inputs", lambda trainer, data, mode: ([300.0, 300.0, 100.0, 500.0], None))
    monkeypatch.setattr(balance, "weight_field", lambda w: {WEIGHT: w})
    batch = types.SimpleNamespace(keys=["a", "b", "c", "d"], partition_id="train",
                                  tags=[{}, {}, {}, {"is_padding": True}], fields=["input_ids", "old_log_probs"])
    metrics: dict = {}
    trainer = mod.PPOTrainer()
    assert trainer._update_actor(batch, metrics) is batch
    assert tq.asked == (["a", "b", "c", "d"], "train", ["teacher", "prompts", "responses", "response_mask"])
    keys, pid, fields = tq.put
    assert keys == ["a", "b", "c", "d"] and pid == "train"
    # the padding copy counts in no share: fin 600 of 700 tokens, med 100
    assert metrics["b2e/share/fin"] == pytest.approx(6 / 7) and metrics["b2e/share/med"] == pytest.approx(1 / 7)
    assert fields[WEIGHT] == pytest.approx([7 / 12, 7 / 12, 3.5, 3.5])
    assert trainer.seen_fields == ["input_ids", "old_log_probs", WEIGHT]  # the actor is told to read it
    assert metrics["actor/pg_loss"] == 0.5


def test_the_v1_trainer_asks_for_the_rollout_losses_and_stops_when_the_actor_ignored_the_weights(monkeypatch):
    rows = {k: {"teacher": t} for k, t in (("a", "fin"), ("b", "law"))}
    monkeypatch.setenv("B2E_MOPD_BALANCE", "gap")
    monkeypatch.setattr(balance, "v1_inputs", lambda trainer, data, mode: ([100.0, 100.0], [30.0, 10.0]))
    monkeypatch.setattr(balance, "weight_field", lambda w: {WEIGHT: w})
    tq = FakeTQ(rows)
    mod = fake_v1(tq)
    balance.patch_trainer_v1(mod)
    batch = types.SimpleNamespace(keys=["a", "b"], partition_id="train", tags=[{}, {}], fields=None)
    metrics: dict = {}
    mod.PPOTrainer()._update_actor(batch, metrics)
    assert tq.asked[2][-2:] == ["old_log_probs", "teacher_logprobs"]
    assert tq.put[2][WEIGHT] == pytest.approx([1.5, 0.5]) and metrics["b2e/gap/fin"] == pytest.approx(0.3)
    assert batch.fields is None  # no list: the actor reads every field, the weights among them

    mod = fake_v1(FakeTQ(rows), applied=False)
    balance.patch_trainer_v1(mod)
    with pytest.raises(RuntimeError, match="not weighted"):
        mod.PPOTrainer()._update_actor(types.SimpleNamespace(keys=["a", "b"], partition_id="train",
                                                             tags=[{}, {}], fields=None), {})


def test_patch_losses_wraps_every_registered_loss():
    def k3(**kw):
        return "k3", {}

    mod = types.ModuleType("fake_losses")
    mod.DISTILLATION_LOSS_REGISTRY = {"k3": k3, "forward_kl_topk": k3}
    balance.patch_losses(mod)
    assert all(f is not k3 and f.__wrapped__ is k3 for f in mod.DISTILLATION_LOSS_REGISTRY.values())


def test_install_needs_a_known_mode(monkeypatch):
    hooks = list(sys.meta_path)
    try:
        monkeypatch.setenv("B2E_MOPD_BALANCE", "tokens")
        with pytest.raises(ValueError, match="one of"):
            balance.install()
        monkeypatch.setenv("B2E_MOPD_BALANCE", "gap")
        balance.install()
        targets = {getattr(f, "target", None) for f in sys.meta_path}
        assert {balance.TRAINER, balance.TRAINER_V1, balance.LOSSES} <= targets
    finally:
        sys.meta_path[:] = hooks
