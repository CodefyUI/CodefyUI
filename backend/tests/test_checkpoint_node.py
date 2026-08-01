"""Tests for CheckpointSaverNode and CheckpointLoaderNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.nodes.io.checkpoint_node import CheckpointLoaderNode, CheckpointSaverNode


def _model_and_opt():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, opt


def test_saver_metadata():
    assert CheckpointSaverNode.NODE_NAME == "CheckpointSaver"
    assert CheckpointSaverNode.CATEGORY == "IO"


def test_loader_metadata():
    assert CheckpointLoaderNode.NODE_NAME == "CheckpointLoader"


def test_save_and_load_checkpoint_roundtrip():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_test.pt"
    try:
        model, opt = _model_and_opt()
        losses = torch.tensor([0.5, 0.3, 0.1])
        save_res = CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "losses": losses},
            {"path": target, "epoch": 5},
        )
        assert save_res["model"] is model

        new_model, new_opt = _model_and_opt()
        load_res = CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt},
            {"path": target, "device": "cpu"},
        )
        assert load_res["epoch"] == 5
        assert torch.equal(load_res["losses"], losses)
        # Weights should match
        x = torch.randn(1, 4)
        assert torch.allclose(model(x), new_model(x))
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_save_without_losses():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_no_loss.pt"
    try:
        model, opt = _model_and_opt()
        res = CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt},
            {"path": target, "epoch": 0},
        )
        assert res["path"].endswith(".pt")
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_load_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        model, opt = _model_and_opt()
        CheckpointLoaderNode().execute(
            {"model": model, "optimizer": opt},
            {"path": "_does_not_exist.pt", "device": "cpu"},
        )


# --- scheduler state in the payload (#118) --------------------------------


def _advanced_scheduler(opt, steps=3):
    """A StepLR that has already walked ``steps`` epochs of its schedule."""
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(steps):
        opt.step()
        sched.step()
    return sched


def test_scheduler_state_roundtrip():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_sched.pt"
    try:
        model, opt = _model_and_opt()
        sched = _advanced_scheduler(opt)

        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "lr_scheduler": sched},
            {"path": target, "epoch": 3},
        )
        raw = torch.load(settings.MODELS_DIR / target, map_location="cpu", weights_only=True)
        assert raw["scheduler_class"] == "StepLR"
        assert raw["scheduler_state_dict"]["last_epoch"] == 3

        new_model, new_opt = _model_and_opt()
        new_sched = torch.optim.lr_scheduler.StepLR(new_opt, step_size=2, gamma=0.5)
        res = CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt, "lr_scheduler": new_sched},
            {"path": target, "device": "cpu"},
        )

        assert res["lr_scheduler"] is new_sched
        assert new_sched.last_epoch == sched.last_epoch
        assert new_sched._step_count == sched._step_count
        assert new_sched.base_lrs == sched.base_lrs
        assert new_sched.get_last_lr() == sched.get_last_lr()
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_checkpoint_omits_scheduler_key_when_none_wired():
    """The key is additive: an unwired scheduler leaves the payload as it was."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_no_sched.pt"
    try:
        model, opt = _model_and_opt()
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt},
            {"path": target, "epoch": 1},
        )
        raw = torch.load(settings.MODELS_DIR / target, map_location="cpu", weights_only=True)
        assert set(raw) == {"epoch", "model_state_dict", "optimizer_state_dict"}
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_loader_refuses_scheduler_state_from_a_different_class():
    """StepLR state spliced into a CosineAnnealingLR would corrupt it."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_sched_mismatch.pt"
    try:
        model, opt = _model_and_opt()
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "lr_scheduler": _advanced_scheduler(opt)},
            {"path": target, "epoch": 3},
        )

        new_model, new_opt = _model_and_opt()
        other = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=10)
        CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt, "lr_scheduler": other},
            {"path": target, "device": "cpu"},
        )

        assert other.last_epoch == 0
        assert not hasattr(other, "step_size")
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)


def test_loader_reports_no_scheduler_when_none_wired():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = "_ckpt_sched_unused.pt"
    try:
        model, opt = _model_and_opt()
        CheckpointSaverNode().execute(
            {"model": model, "optimizer": opt, "lr_scheduler": _advanced_scheduler(opt)},
            {"path": target, "epoch": 3},
        )
        new_model, new_opt = _model_and_opt()
        res = CheckpointLoaderNode().execute(
            {"model": new_model, "optimizer": new_opt},
            {"path": target, "device": "cpu"},
        )
        assert res["lr_scheduler"] is None
        assert res["epoch"] == 3
    finally:
        (settings.MODELS_DIR / target).unlink(missing_ok=True)
