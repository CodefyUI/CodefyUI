"""TrainingLoop: the ``lr`` series under per-step scheduling, and loss shapes.

Two findings of the 2026-10-13 functional-correctness audit.

1. With ``scheduler_step=optimizer_step`` the scheduler moves the LR inside
   the epoch, and the ``lr`` series / ``lr_history`` was read after the
   epoch's last step, so each point was the NEXT epoch's starting LR
   (OneCycleLR over 3 epochs recorded [0.9607, 0.3408, 0.0155] where the
   epochs started at [0.04, 0.9607, 0.3408]).

2. An element-wise loss (MSELoss and friends) on ``[B, 1]`` outputs against
   ``[B]`` targets broadcasts to ``[B, B]``: the reported loss and the
   gradients were wrong, with only a torch UserWarning in the server log.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.execution_context import ExecutionContext
from app.core.param_defaults import fill_missing_params
from app.nodes.training.training_loop_node import TrainingLoopNode


def _train(model, loader, optimizer, loss_fn, *, scheduler=None, val=None,
           **params):
    inputs = {"model": model, "dataloader": loader, "optimizer": optimizer,
              "loss_fn": loss_fn}
    if scheduler is not None:
        inputs["lr_scheduler"] = scheduler
    if val is not None:
        inputs["val_dataloader"] = val
    return TrainingLoopNode().execute(
        inputs,
        fill_missing_params(TrainingLoopNode, {"device": "cpu", **params}),
        context=ExecutionContext(device="cpu"),
    )


def _classification_loader(n: int = 48, batch_size: int = 8) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _onecycle_run():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.0, total_steps=18)
    loader = _classification_loader()  # 6 batches per epoch, 3 epochs
    return _train(model, loader, optimizer, nn.CrossEntropyLoss(),
                  scheduler=scheduler, epochs=3,
                  scheduler_step="optimizer_step")


def _onecycle_reference_epoch_start_lrs() -> list[float]:
    """The LR each epoch's first batch trained with, by hand."""
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1.0, total_steps=18)
    starts = []
    for _ in range(3):
        starts.append(optimizer.param_groups[0]["lr"])
        for _ in range(6):
            optimizer.step()
            scheduler.step()
    return starts


def test_per_step_lr_history_is_the_lr_each_epoch_started_with():
    result = _onecycle_run()
    expected = _onecycle_reference_epoch_start_lrs()
    assert result["metrics"]["lr_history"] == pytest.approx(expected)
    # The audit's numbers, so a regression reads as the original bug.
    assert result["metrics"]["lr_history"] == pytest.approx(
        [0.04, 0.9607, 0.3408], abs=1e-4)


def test_per_epoch_lr_history_is_unchanged():
    torch.manual_seed(1)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.5)
    result = _train(model, _classification_loader(), optimizer,
                    nn.CrossEntropyLoss(), scheduler=scheduler, epochs=4)
    assert result["metrics"]["lr_history"] == pytest.approx(
        [0.1, 0.05, 0.025, 0.0125])


def _regression_data(target_shape: str):
    torch.manual_seed(0)
    x = torch.randn(32, 3)
    y = x.sum(dim=1)
    if target_shape == "column":
        y = y.unsqueeze(1)
    return x, y


@pytest.mark.parametrize("loss_cls", [
    nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss, nn.HuberLoss])
def test_an_elementwise_loss_refuses_mismatched_shapes(loss_cls):
    x, y = _regression_data("flat")
    model = nn.Linear(3, 1)
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with pytest.raises(ValueError) as info:
        _train(model, DataLoader(TensorDataset(x, y), batch_size=8),
               optimizer, loss_cls(), epochs=2)
    message = str(info.value)
    assert "TrainingLoop" in message
    assert "[8, 1]" in message and "[8]" in message
    assert "Unsqueeze" in message
    # Refused before the first optimizer step.
    assert torch.equal(model.weight.detach(), before)


def test_bce_with_logits_mismatch_names_both_shapes():
    torch.manual_seed(0)
    x = torch.randn(16, 2)
    y = (x[:, 0] > 0).float()  # [N], model gives [B, 1]
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match=r"TrainingLoop: the model output has shape \[8, 1\]"):
        _train(model, DataLoader(TensorDataset(x, y), batch_size=8),
               optimizer, nn.BCEWithLogitsLoss(), epochs=1)


def test_the_validation_targets_are_checked_too():
    x, y_col = _regression_data("column")
    _, y_flat = _regression_data("flat")
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with pytest.raises(ValueError, match="validation targets"):
        _train(model, DataLoader(TensorDataset(x, y_col), batch_size=8),
               optimizer, nn.MSELoss(), epochs=1,
               val=DataLoader(TensorDataset(x, y_flat), batch_size=8))


def test_matching_shapes_train_and_report_the_per_sample_loss():
    x, y = _regression_data("column")
    torch.manual_seed(2)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    result = _train(model, DataLoader(TensorDataset(x, y), batch_size=32),
                    optimizer, nn.MSELoss(), epochs=40)
    with torch.no_grad():
        true_mse = float(((model(x) - y) ** 2).mean())
    # One full batch per epoch: the last reported loss is the MSE before
    # the last step, so it is close to (and not below) the final one.
    assert result["losses"][-1].item() == pytest.approx(true_mse, rel=0.2)
    assert true_mse < 0.1


def test_cross_entropy_keeps_its_class_index_targets():
    """[B, C] logits against [B] indices is CrossEntropy's correct shape."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = _classification_loader()
    result = _train(model, loader, optimizer, nn.CrossEntropyLoss(),
                    epochs=2, val=loader)
    assert len(result["metrics"]["lr_history"]) == 2
    assert "final_val_accuracy" in result["metrics"]


def test_val_accuracy_reads_column_labels_as_flat():
    from app.nodes.training.evaluate_model_node import count_correct

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    flat = torch.tensor([0, 1, 0])
    correct_flat, total_flat = count_correct(logits, flat, "TrainingLoop")
    correct_col, total_col = count_correct(
        logits, flat.view(-1, 1), "TrainingLoop")
    assert (int(correct_flat), total_flat) == (2, 3)
    assert (int(correct_col), total_col) == (2, 3)
