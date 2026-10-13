"""Findings of the 2026-10-13 end-to-end training-pipeline audit.

Each test pins one behaviour that was wrong (or failed with a message naming
nothing useful) when TrainingLoop and its collaborators were checked against
a hand-written torch loop on the same model, data and seed.
"""

from __future__ import annotations

import copy

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
