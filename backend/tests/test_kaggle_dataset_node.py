"""Tests for KaggleDatasetNode (metadata + auth check)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode, _kaggle_credentials_present


def test_node_metadata():
    assert KaggleDatasetNode.NODE_NAME == "KaggleDataset"
    assert KaggleDatasetNode.CATEGORY == "Data"


def test_no_inputs_defined():
    assert KaggleDatasetNode.define_inputs() == []


def test_output_is_dataset():
    outputs = KaggleDatasetNode.define_outputs()
    assert outputs[0].name == "dataset"


def test_credentials_present_via_env_vars():
    with patch.dict(os.environ, {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, clear=False):
        assert _kaggle_credentials_present()


def test_credentials_absent_when_no_env_and_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    # Point home to an empty directory so kaggle.json check fails
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert not _kaggle_credentials_present()


def test_missing_credentials_raises_helpful_error(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="Kaggle authentication"):
        KaggleDatasetNode().execute({}, {"dataset_slug": "x/y"})
