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
