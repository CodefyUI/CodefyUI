"""Tests for SyntheticDatasetNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.synthetic_dataset_node import SyntheticDatasetNode


def _run(**params):
    base = {
        "kind": "circles",
        "n_samples": 100,
        "noise": 0.1,
        "factor": 0.5,
        "centers": 3,
        "seed": 42,
    }
    base.update(params)
    return SyntheticDatasetNode().execute({}, base)


def test_node_metadata():
    assert SyntheticDatasetNode.NODE_NAME == "SyntheticDataset"
    assert SyntheticDatasetNode.CATEGORY == "Data"
    out_names = [p.name for p in SyntheticDatasetNode.define_outputs()]
    assert out_names == ["tensor", "labels", "columns"]


def test_circles_shape_and_types():
    res = _run(kind="circles", n_samples=120)
    assert isinstance(res["tensor"], torch.Tensor)
    assert res["tensor"].shape == (120, 2)
    assert res["tensor"].dtype == torch.float32
    assert len(res["labels"]) == 120
    assert set(res["labels"]) == {"0", "1"}
    assert res["columns"] == ["x0", "x1"]


def test_moons_returns_two_classes():
    res = _run(kind="moons", n_samples=80)
    assert res["tensor"].shape == (80, 2)
    assert set(res["labels"]) == {"0", "1"}


def test_blobs_centers_param():
    res = _run(kind="blobs", n_samples=90, centers=3)
    assert res["tensor"].shape == (90, 2)
    assert set(res["labels"]) == {"0", "1", "2"}


def test_classification_general():
    res = _run(kind="classification", n_samples=60)
    assert res["tensor"].shape == (60, 2)
    assert set(res["labels"]).issubset({"0", "1"})


def test_seed_reproducibility():
    a = _run(kind="circles", seed=7)
    b = _run(kind="circles", seed=7)
    assert torch.allclose(a["tensor"], b["tensor"])


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        _run(kind="banana")
