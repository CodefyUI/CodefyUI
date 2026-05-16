"""Tests for TrainingLoopNode (end-to-end mini training)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.nodes.training.training_loop_node import TrainingLoopNode


def _make_dataset(n=8, in_features=4, out_classes=2):
    """Tiny supervised dataset (n samples)."""
    X = torch.randn(n, in_features)
    y = torch.randint(0, out_classes, (n,))
    return torch.utils.data.TensorDataset(X, y)


def _make_loader(dataset, batch_size=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def test_node_metadata():
    assert TrainingLoopNode.NODE_NAME == "TrainingLoop"
    assert TrainingLoopNode.CATEGORY == "Training"
    out_names = [p.name for p in TrainingLoopNode.define_outputs()]
    assert "model" in out_names
    assert "losses" in out_names
    assert "metrics" in out_names


def test_basic_training_returns_losses_per_epoch():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    loader = _make_loader(_make_dataset())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 3, "device": "cpu"},
    )
    assert res["losses"].shape == (3,)
    assert res["metrics"]["total_epochs_run"] == 3


def test_training_decreases_loss_when_overfitting():
    torch.manual_seed(42)
    # Set up something trivially learnable
    X = torch.randn(32, 4)
    W = torch.randn(4, 2)
    y = (X @ W).argmax(dim=-1)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = _make_loader(dataset, batch_size=8)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 20, "device": "cpu"},
    )
    losses = res["losses"]
    assert losses[-1] < losses[0]


def test_validation_loop_produces_val_losses():
    model = nn.Linear(4, 2)
    train_loader = _make_loader(_make_dataset())
    val_loader = _make_loader(_make_dataset(n=4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {
            "model": model,
            "dataloader": train_loader,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "val_dataloader": val_loader,
        },
        {"epochs": 2, "device": "cpu"},
    )
    assert res["val_losses"].shape == (2,)
    assert res["metrics"]["final_val_loss"] is not None


def test_progress_callback_invoked_per_epoch():
    model = nn.Linear(4, 2)
    loader = _make_loader(_make_dataset())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    seen_events = []
    def cb(payload):
        seen_events.append(payload.get("event"))
    TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 2, "device": "cpu"},
        progress_callback=cb,
    )
    assert "config" in seen_events
    assert seen_events.count("epoch") == 2


def test_grad_clip_does_not_error():
    model = nn.Linear(4, 2)
    loader = _make_loader(_make_dataset())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    res = TrainingLoopNode().execute(
        {"model": model, "dataloader": loader, "optimizer": optimizer, "loss_fn": loss_fn},
        {"epochs": 1, "device": "cpu", "grad_clip_norm": 1.0},
    )
    assert res["losses"].shape == (1,)
