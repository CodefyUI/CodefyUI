"""The training nodes do not read the loss back from the device every batch.

Reading a scalar off an accelerator (``loss.item()``) is a full host/device
synchronisation. On Apple MPS it drains the Metal command queue, and doing it
once per batch was measured at ~6 ms of a 10 ms step on an M3 for the MNIST
CNN example -- the reason that example ran no faster on MPS than on CPU.
These tests pin the contract: the per-batch loss stays on the device and is
converted only for a thinned metric point, a throttled progress frame, or
the epoch average.

The device here is CPU (CI has no accelerator), so the tests count the
conversions rather than time them: every ``Tensor.item()`` and
``float(tensor)`` is one host read-back on a real accelerator.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from app.core.loop_control import ProgressThrottle
from app.nodes.training.diffusion_training_loop_node import DiffusionTrainingLoopNode
from app.nodes.training.evaluate_model_node import EvaluateModelNode
from app.nodes.training.training_loop_node import TrainingLoopNode


class _ReadBackCounter:
    """Counts host read-backs: ``Tensor.item`` and ``Tensor.__float__``/``__int__``."""

    def __init__(self, monkeypatch):
        self.count = 0
        original_item = torch.Tensor.item

        def counting_item(tensor, *args, **kwargs):
            # torch's own optimizers read their (CPU-resident) step counter
            # with ``.item()`` on every step. That is torch's business and
            # not a device read-back, so calls from inside torch.optim are
            # not counted.
            caller = sys._getframe(1).f_code.co_filename.replace("\\", "/")
            if "/torch/optim/" not in caller:
                self.count += 1
            return original_item(tensor, *args, **kwargs)

        # ``float(t)`` / ``int(t)`` on a 0-dim tensor go through ``item``
        # internally in torch, but patching ``__float__`` too keeps the
        # count honest if that ever changes.
        original_float = torch.Tensor.__float__
        original_int = torch.Tensor.__int__

        def counting_float(tensor):
            self.count += 1
            return original_float(tensor)

        def counting_int(tensor):
            self.count += 1
            return original_int(tensor)

        monkeypatch.setattr(torch.Tensor, "item", counting_item)
        monkeypatch.setattr(torch.Tensor, "__float__", counting_float)
        monkeypatch.setattr(torch.Tensor, "__int__", counting_int)


def _loader(n=64, batch_size=4, in_features=4, classes=2):
    X = torch.randn(n, in_features)
    y = torch.randint(0, classes, (n,))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=batch_size)


def test_the_throttle_builds_a_callable_payload_only_when_a_frame_goes_out():
    frames: list[dict] = []
    built = 0

    def payload():
        nonlocal built
        built += 1
        return {"event": "batch", "n": built}

    throttle = ProgressThrottle(frames.append, min_interval_s=3600.0)
    assert throttle.emit(payload) is True
    for _ in range(50):
        assert throttle.emit(payload) is False
    assert built == 1
    assert frames == [{"event": "batch", "n": 1}]


def test_the_throttle_still_accepts_a_plain_dict():
    frames: list[dict] = []
    throttle = ProgressThrottle(frames.append, min_interval_s=0.0)
    assert throttle.emit({"event": "batch"}) is True
    assert frames == [{"event": "batch"}]


def test_a_throttle_with_no_callback_never_calls_the_factory():
    def payload():
        raise AssertionError("built a frame nobody will receive")

    assert ProgressThrottle(None).emit(payload) is False


def test_training_loop_reads_the_loss_back_once_per_epoch_not_per_batch(monkeypatch):
    counter = _ReadBackCounter(monkeypatch)
    batches, epochs = 16, 2
    model = nn.Linear(4, 2)
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=batches * 4, batch_size=4),
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.01),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": epochs, "device": "cpu"},
        # A throttle interval nobody reaches: only the FIRST frame of each
        # loop goes out, so the count is not timing-dependent.
    )
    # One read-back per epoch for the average, plus one per progress frame
    # that actually went out (at most a couple over a run this short).
    assert counter.count <= epochs + 4, counter.count
    assert counter.count < batches * epochs


def test_the_batch_metric_series_is_the_one_thing_that_reads_back_per_batch(monkeypatch):
    counter = _ReadBackCounter(monkeypatch)
    batches = 16
    model = nn.Linear(4, 2)
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=batches * 4, batch_size=4),
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.01),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": 1, "device": "cpu", "batch_metrics": True, "log_interval": 4},
    )
    # Thinned by log_interval: 4 points, not 16.
    assert counter.count <= 1 + batches // 4 + 4, counter.count


def test_validation_accumulates_on_the_device(monkeypatch):
    counter = _ReadBackCounter(monkeypatch)
    model = nn.Linear(4, 2)
    res = TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=32, batch_size=4),
         "val_dataloader": _loader(n=32, batch_size=4),
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.01),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": 1, "device": "cpu"},
    )
    # train average + val average + val accuracy + a few progress frames
    assert counter.count <= 3 + 4, counter.count
    assert 0.0 <= res["metrics"]["final_val_accuracy"] <= 1.0
    assert isinstance(res["metrics"]["final_val_loss"], float)


def test_the_gradient_norm_is_read_back_only_on_thinned_steps(monkeypatch):
    counter = _ReadBackCounter(monkeypatch)
    batches = 16
    model = nn.Linear(4, 2)
    TrainingLoopNode().execute(
        {"model": model, "dataloader": _loader(n=batches * 4, batch_size=4),
         "optimizer": torch.optim.SGD(model.parameters(), lr=0.01),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": 1, "device": "cpu", "grad_clip_norm": 1.0,
         "log_grad_norm": True, "log_interval": 8},
    )
    # 2 grad-norm points + 1 epoch average + progress frames; NOT 16 norms.
    assert counter.count <= 2 + 1 + 4, counter.count


def test_evaluate_model_reads_the_count_back_once(monkeypatch):
    counter = _ReadBackCounter(monkeypatch)
    X = torch.randn(64, 4)
    y = torch.randint(0, 2, (64,))
    res = EvaluateModelNode().execute(
        {"model": nn.Linear(4, 2), "dataset": torch.utils.data.TensorDataset(X, y)},
        {"batch_size": 4, "device": "cpu"},
    )
    assert res["total"] == 64
    assert res["correct"] == int(res["correct"])
    assert counter.count <= 4, counter.count


def test_diffusion_loop_reads_the_loss_back_once_per_epoch(monkeypatch):
    counter = _ReadBackCounter(monkeypatch)

    class _Eps(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x, t):
            return self.lin(x)

    X = torch.randn(64, 8)
    epochs = 2
    res = DiffusionTrainingLoopNode().execute(
        {"model": _Eps(), "dataset": torch.utils.data.TensorDataset(X)},
        {"epochs": epochs, "batch_size": 4, "device": "cpu", "num_timesteps": 10},
    )
    assert res["losses"].shape == (epochs,)
    assert counter.count <= epochs + 4, counter.count
