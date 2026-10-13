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


# ── 2026-10-13 audit: split folders read as classes ──


def _offline_download(monkeypatch, root: Path) -> None:
    import sys
    import types

    from PIL import Image

    for split in ("train", "test"):
        for cls in ("cat", "dog"):
            (root / split / cls).mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(root / split / cls / "0.png")
    monkeypatch.setitem(
        sys.modules, "kagglehub",
        types.SimpleNamespace(dataset_download=lambda slug: str(root)),
    )
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")


def test_split_folders_at_the_root_ask_for_subdir(monkeypatch, tmp_path):
    _offline_download(monkeypatch, tmp_path / "ds")
    with pytest.raises(RuntimeError, match=r"KaggleDataset: .*split folders \['test', 'train'\].*subdir"):
        KaggleDatasetNode().execute({}, {"dataset_slug": "o/n", "subdir": "", "cache_dir": ""})


def test_subdir_on_a_split_loads_its_class_folders(monkeypatch, tmp_path):
    _offline_download(monkeypatch, tmp_path / "ds")
    ds = KaggleDatasetNode().execute({}, {"dataset_slug": "o/n", "subdir": "train", "cache_dir": ""})["dataset"]
    assert ds.classes == ["cat", "dog"]
