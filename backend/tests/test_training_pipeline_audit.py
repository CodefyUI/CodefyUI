"""Findings of the 2026-10-13 end-to-end training-pipeline audit.

Each test pins one behaviour that was wrong (or failed with a message naming
nothing useful) when TrainingLoop and its collaborators were checked against
a hand-written torch loop on the same model, data and seed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from app.nodes.training.loss_node import LossNode
from app.nodes.training.training_loop_node import TrainingLoopNode


class _RecordingContext:
    deterministic = False
    current_node_id = "train"
    seed = None
    device = "cpu"

    def __init__(self):
        self.metrics: list[tuple[str, float, int]] = []

    def should_stop(self):
        return False

    def log_metric(self, name, value, step, node_id=None):
        self.metrics.append((name, value, step))

    def can_record_artifacts(self):
        return False


def _features(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4, generator=g)


def _train(model, dataset, loss_fn, *, lr=0.1, val=None, context=None,
           batch_size=8, **params):
    inputs = {
        "model": model,
        "dataloader": DataLoader(dataset, batch_size=batch_size),
        "optimizer": torch.optim.SGD(model.parameters(), lr=lr),
        "loss_fn": loss_fn,
    }
    if val is not None:
        inputs["val_dataloader"] = DataLoader(val, batch_size=batch_size)
    return TrainingLoopNode().execute(
        inputs, {"device": "cpu", "epochs": 1, **params}, context=context)


# ── integer targets on an element-wise loss ─────────────────────────────


def test_integer_targets_train_a_bce_model_as_their_float_values():
    """0/1 labels as int64 used to fail inside torch with "result type Float
    can't be cast to the desired output type Long"."""
    x = _features()
    labels = (x[:, 0] > 0).long().unsqueeze(1)
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    reference = copy.deepcopy(model)

    out = _train(model, TensorDataset(x, labels), nn.BCEWithLogitsLoss(), lr=0.0)

    expected = F.binary_cross_entropy_with_logits(
        reference(x), labels.float()).item()
    assert out["losses"][0].item() == pytest.approx(expected, abs=1e-6)


def test_integer_targets_still_meet_the_shape_guard():
    x = _features()
    labels = (x[:, 0] > 0).long()  # [N] against a [B, 1] output
    with pytest.raises(ValueError, match="element by element"):
        _train(nn.Linear(4, 1), TensorDataset(x, labels), nn.MSELoss())


# ── reduction = none ────────────────────────────────────────────────────


def test_reduction_none_is_refused_with_the_param_named():
    """It used to fail inside autograd: "grad can be implicitly created only
    for scalar outputs"."""
    x = _features()
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(1))
    loss_fn = LossNode().execute({}, {"type": "CrossEntropyLoss",
                                      "reduction": "none"})["loss_fn"]
    with pytest.raises(ValueError, match="reduction = none"):
        _train(nn.Linear(4, 3), TensorDataset(x, y), loss_fn)


def test_reduction_sum_still_trains_and_reports_the_batch_sum():
    x = _features()
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    reference = copy.deepcopy(model)
    loss_fn = LossNode().execute({}, {"type": "CrossEntropyLoss",
                                      "reduction": "sum"})["loss_fn"]
    out = _train(model, TensorDataset(x, y), loss_fn, lr=0.0, batch_size=16)
    expected = F.cross_entropy(reference(x), y, reduction="sum").item()
    assert out["losses"][0].item() == pytest.approx(expected, rel=1e-6)


# ── NLLLoss on raw logits ───────────────────────────────────────────────


def test_nll_loss_on_raw_logits_is_refused():
    """NLLLoss on logits reported a NEGATIVE loss (-0.04 in the audit, where
    cross-entropy was 1.15) and trained the target logit up without bound."""
    x = _features()
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="log-probabilities"):
        _train(nn.Linear(4, 3), TensorDataset(x, y), nn.NLLLoss())


def test_nll_loss_on_log_softmax_output_matches_cross_entropy():
    x = _features()
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    reference = copy.deepcopy(linear)
    out = _train(nn.Sequential(linear, nn.LogSoftmax(dim=1)),
                 TensorDataset(x, y), nn.NLLLoss(), lr=0.0, batch_size=16)
    expected = F.cross_entropy(reference(x), y).item()
    assert out["losses"][0].item() == pytest.approx(expected, abs=1e-6)


# ── a model that returns a tuple ────────────────────────────────────────


class _TupleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x), "aux"


def test_a_tuple_output_is_named_in_the_error():
    x = _features()
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(1))
    with pytest.raises(ValueError, match="returned a tuple"):
        _train(_TupleModel(), TensorDataset(x, y), nn.CrossEntropyLoss())


# ── checkpoint_every_steps files resume at an epoch ─────────────────────


def test_step_checkpoints_store_complete_epochs_and_do_not_collide(
        tmp_path, monkeypatch):
    """A step milestone used to store the STEP count as its epoch, so
    CheckpointLoader.epoch handed 5 to start_epoch for a checkpoint taken in
    epoch 2 of 4, and step 4 and epoch 4 wrote the same ``-e4.pt`` file."""
    from app.config import settings
    from app.core.execution_context import ExecutionContext
    from app.nodes.io.checkpoint_node import CheckpointLoaderNode

    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)

    x = _features(48)
    y = torch.randint(0, 3, (48,), generator=torch.Generator().manual_seed(1))
    ctx = ExecutionContext(signals_recorded=True, device="cpu")
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    # 3 batches of 16 per epoch: step 4 is inside epoch 2, step 12 ends
    # epoch 4, and checkpoint_every=4 also writes at epoch 4.
    _train(model, TensorDataset(x, y), nn.CrossEntropyLoss(), context=ctx,
           batch_size=16, epochs=4, checkpoint_every=4,
           checkpoint_every_steps=4)

    signals, _ = ctx.outbox.drain()
    written = {
        Path(s.path).name: s.meta for s in signals
        if getattr(s, "kind", None) == "checkpoint"
    }
    names = sorted(written)
    assert len(names) == 4, names  # s4, s8, s12, e4: nothing overwritten
    by_step = {meta.get("step"): meta["epoch"] for meta in written.values()}
    assert by_step == {4: 1, 8: 2, 12: 3, None: 4}

    for name, meta in written.items():
        restored = nn.Linear(4, 3)
        loaded = CheckpointLoaderNode().execute(
            {"model": restored,
             "optimizer": torch.optim.SGD(restored.parameters(), lr=0.1)},
            {"path": str(models / "periodic" / name), "device": "cpu"})
        assert loaded["epoch"] == meta["epoch"]


# ── val_every_steps does not change what training does ──────────────────


def _dropout_run(val_every_steps):
    x = _features(48)
    y = torch.randint(0, 3, (48,), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(16, 3))
    torch.manual_seed(3)
    out = _train(model, TensorDataset(x, y), nn.CrossEntropyLoss(),
                 val=TensorDataset(x[:20], y[:20]), batch_size=16, epochs=2,
                 val_every_steps=val_every_steps, context=_RecordingContext())
    return model, out


def test_val_every_steps_leaves_the_trained_weights_unchanged():
    """Starting each mid-step validation pass drew a base seed from the
    global RNG, which moved every later Dropout mask."""
    with_steps, out_steps = _dropout_run(1)
    without, out_plain = _dropout_run(0)
    for a, b in zip(with_steps.state_dict().values(),
                    without.state_dict().values()):
        assert torch.equal(a, b)
    assert out_steps["losses"].tolist() == out_plain["losses"].tolist()


# ── val_accuracy for a single-logit binary classifier ───────────────────


def test_single_logit_bce_records_the_accuracy_evaluate_model_reports():
    """Before: no val_accuracy at all for BCEWithLogitsLoss, and
    monitor=val_accuracy fell back to val_loss, although EvaluateModel
    scores the same [B, 1] model as binary at logit 0."""
    from app.nodes.training.evaluate_model_node import EvaluateModelNode

    x = _features(40)
    labels = (x[:, 0] + 0.3 * x[:, 1] > 0).float().unsqueeze(1)
    val = TensorDataset(x[:24], labels[:24])
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    ctx = _RecordingContext()
    out = _train(model, TensorDataset(x, labels), nn.BCEWithLogitsLoss(),
                 val=val, epochs=3, early_stopping_patience=5,
                 monitor="val_accuracy", context=ctx)

    assert out["metrics"]["monitor"] == "val_accuracy"
    recorded = [v for name, v, _ in ctx.metrics if name == "val_accuracy"]
    assert len(recorded) == 3
    evaluated = EvaluateModelNode().execute(
        {"model": out["model"], "dataset": val}, {"device": "cpu"})
    assert out["metrics"]["final_val_accuracy"] == evaluated["accuracy"]


def test_multi_label_bce_still_records_no_val_accuracy():
    x = _features(16)
    labels = (x[:, :3] > 0).float()
    ctx = _RecordingContext()
    out = _train(nn.Linear(4, 3), TensorDataset(x, labels),
                 nn.BCEWithLogitsLoss(), val=TensorDataset(x, labels),
                 context=ctx)
    assert out["metrics"]["final_val_accuracy"] is None
    assert not [n for n, _, _ in ctx.metrics if n == "val_accuracy"]


# ── the epoch a checkpoint stores comes from the run ────────────────────


def test_an_early_stopped_run_saves_and_resumes_at_its_real_epoch(
        tmp_path, monkeypatch):
    """CheckpointSaver.epoch was a hand-typed param only. A run that early
    stopped at epoch 4 of 30 saved whatever was typed (the shipped examples
    type the full epoch count), and the resume then trained nothing."""
    from app.config import settings
    from app.nodes.io.checkpoint_node import (
        CheckpointLoaderNode,
        CheckpointSaverNode,
    )

    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)

    x = _features(24)
    y = torch.randint(0, 3, (24,), generator=torch.Generator().manual_seed(1))
    val_x = _features(40, seed=9)
    val_y = torch.randint(0, 3, (40,), generator=torch.Generator().manual_seed(2))
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    out = _train(model, TensorDataset(x, y), nn.CrossEntropyLoss(), lr=1.0,
                 val=TensorDataset(val_x, val_y), epochs=30,
                 early_stopping_patience=2)
    assert out["metrics"]["stopped_early"]
    assert out["epoch"] == out["metrics"]["last_epoch"] < 30

    saved = CheckpointSaverNode().execute(
        {"model": out["model"], "optimizer": out["optimizer"],
         "epoch": out["epoch"]},
        {"path": "audit.pt", "epoch": 30})
    restored = nn.Linear(4, 3)
    loaded = CheckpointLoaderNode().execute(
        {"model": restored,
         "optimizer": torch.optim.SGD(restored.parameters(), lr=1.0)},
        {"path": saved["path"], "device": "cpu"})
    assert loaded["epoch"] == out["epoch"]


def test_checkpoint_saver_keeps_the_param_when_the_input_is_not_wired(
        tmp_path, monkeypatch):
    from app.config import settings
    from app.nodes.io.checkpoint_node import CheckpointSaverNode

    models = tmp_path / "data" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(settings, "MODELS_DIR", models)
    model = nn.Linear(4, 3)
    saved = CheckpointSaverNode().execute(
        {"model": model,
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.1)},
        {"path": "audit.pt", "epoch": 7})
    assert torch.load(saved["path"], weights_only=True)["epoch"] == 7


def test_training_loop_epoch_output_counts_from_start_epoch():
    x = _features(16)
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(1))
    model = nn.Linear(4, 3)
    inputs = {
        "model": model, "dataloader": DataLoader(TensorDataset(x, y), batch_size=8),
        "optimizer": torch.optim.SGD(model.parameters(), lr=0.1),
        "loss_fn": nn.CrossEntropyLoss(), "start_epoch": 3,
    }
    out = TrainingLoopNode().execute(inputs, {"device": "cpu", "epochs": 5})
    assert out["epoch"] == 5
