"""Tests for ModelSaverNode and ModelLoaderNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.config import settings
from app.nodes.io.model_loader_node import ModelLoaderNode
from app.nodes.io.model_saver_node import ModelSaverNode


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(4, 2)


def test_saver_metadata():
    assert ModelSaverNode.NODE_NAME == "ModelSaver"
    assert ModelSaverNode.CATEGORY == "IO"


def test_loader_metadata():
    assert ModelLoaderNode.NODE_NAME == "ModelLoader"
    assert ModelLoaderNode.CATEGORY == "IO"


def test_save_and_load_state_dict_roundtrip():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = "_roundtrip_test.pt"
    try:
        original = _model(seed=42)
        save_res = ModelSaverNode().execute(
            {"model": original},
            {"path": target_path, "save_mode": "state_dict", "format": "pytorch"},
        )
        assert save_res["model"] is original

        empty_model = nn.Linear(4, 2)
        load_res = ModelLoaderNode().execute(
            {"model": empty_model},
            {"path": target_path, "load_mode": "state_dict", "device": "cpu", "strict": True},
        )
        loaded = load_res["model"]
        x = torch.randn(1, 4)
        assert torch.allclose(original(x), loaded(x))
    finally:
        (settings.MODELS_DIR / target_path).unlink(missing_ok=True)


def test_save_safetensors():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model = _model()
        res = ModelSaverNode().execute(
            {"model": model},
            {"path": "_st_test.safetensors", "save_mode": "state_dict", "format": "safetensors"},
        )
        assert res["path"].endswith(".safetensors")
    finally:
        for f in settings.MODELS_DIR.glob("_st_test*"):
            f.unlink(missing_ok=True)


def test_safetensors_with_full_model_raises():
    with pytest.raises(ValueError, match="state_dict"):
        ModelSaverNode().execute(
            {"model": _model()},
            {"path": "_should_fail.safetensors", "save_mode": "full_model", "format": "safetensors"},
        )


def test_load_state_dict_requires_model_input():
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = "_no_model_test.pt"
    try:
        ModelSaverNode().execute(
            {"model": _model()},
            {"path": target_path, "save_mode": "state_dict", "format": "pytorch"},
        )
        with pytest.raises(ValueError, match="state_dict mode"):
            ModelLoaderNode().execute(
                {},
                {"path": target_path, "load_mode": "state_dict", "device": "cpu", "strict": True},
            )
    finally:
        (settings.MODELS_DIR / target_path).unlink(missing_ok=True)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ModelLoaderNode().execute(
            {"model": _model()},
            {"path": "_definitely_missing.pt", "load_mode": "state_dict", "device": "cpu", "strict": True},
        )


def test_save_outside_data_dir_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    with pytest.raises(ValueError, match="within the project"):
        ModelSaverNode().execute(
            {"model": _model()},
            {"path": str(bad), "save_mode": "state_dict", "format": "pytorch"},
        )
