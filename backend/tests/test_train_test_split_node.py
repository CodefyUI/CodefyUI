"""Tests for TrainTestSplitNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.train_test_split_node import TrainTestSplitNode


def _run(features, labels, **params):
    p = {"test_size": 0.25, "seed": 42, "stratify": False}
    p.update(params)
    return TrainTestSplitNode().execute({"features": features, "labels": labels}, p)


def test_node_metadata():
    assert TrainTestSplitNode.NODE_NAME == "TrainTestSplit"
    assert TrainTestSplitNode.CATEGORY == "Data"
    out_names = [p.name for p in TrainTestSplitNode.define_outputs()]
    assert out_names == ["x_train", "y_train", "x_test", "y_test"]


def test_split_sizes_default():
    """test_size=0.25 → 75/25 split for 100 samples."""
    x = torch.randn(100, 5)
    y = ["a"] * 50 + ["b"] * 50
    res = _run(x, y, test_size=0.25, seed=42)
    assert res["x_train"].shape == (75, 5)
    assert res["x_test"].shape == (25, 5)
    assert len(res["y_train"]) == 75
    assert len(res["y_test"]) == 25


def test_split_preserves_pairings():
    """A row's label should follow it through the split."""
    x = torch.arange(40, dtype=torch.float32).view(10, 4)
    y = [f"row{i}" for i in range(10)]
    res = _run(x, y, test_size=0.3, seed=1)
    # Reconstruct full set: combined train+test should match original (any order)
    all_x = torch.cat([res["x_train"], res["x_test"]], dim=0)
    all_y = res["y_train"] + res["y_test"]
    # For each row in the combined output, find its label and check it matches
    # what was originally paired with that row's first column value.
    for row_x, lbl in zip(all_x, all_y):
        idx = int(row_x[0].item()) // 4  # original row index from 0..9
        assert lbl == f"row{idx}"


def test_seed_reproducibility():
    x = torch.randn(50, 3, generator=torch.Generator().manual_seed(0))
    y = ["a"] * 25 + ["b"] * 25
    a = _run(x, y, seed=42)
    b = _run(x, y, seed=42)
    assert torch.equal(a["x_train"], b["x_train"])
    assert a["y_train"] == b["y_train"]


def test_different_seeds_different_split():
    x = torch.randn(50, 3, generator=torch.Generator().manual_seed(0))
    y = ["a"] * 25 + ["b"] * 25
    a = _run(x, y, seed=1)
    b = _run(x, y, seed=2)
    assert not torch.equal(a["x_train"], b["x_train"])


def test_stratify_preserves_class_proportions():
    """Stratified split keeps class ratios in train and test."""
    x = torch.randn(100, 4)
    y = ["a"] * 80 + ["b"] * 20  # 80/20 imbalance
    res = _run(x, y, test_size=0.2, seed=42, stratify=True)
    train_a = sum(1 for v in res["y_train"] if v == "a")
    test_a = sum(1 for v in res["y_test"] if v == "a")
    n_train = len(res["y_train"])
    n_test = len(res["y_test"])
    # Both partitions should have ~80% class a within rounding.
    assert abs(train_a / n_train - 0.8) < 0.05
    assert abs(test_a / n_test - 0.8) < 0.05


def test_test_size_must_be_in_range():
    with pytest.raises(ValueError, match="test_size"):
        _run(torch.zeros(10, 2), ["a"] * 10, test_size=1.5)


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(10, 3), ["a"] * 5)  # 10 rows but 5 labels


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        TrainTestSplitNode().execute(
            {"features": torch.zeros(5, 2)},
            {"test_size": 0.25, "seed": 42, "stratify": False},
        )


def test_works_with_tensor_labels():
    """Labels can also be a LongTensor (integer class indices)."""
    x = torch.randn(20, 3)
    y = torch.tensor([0, 1] * 10)
    res = _run(x, y, test_size=0.25, seed=42)
    assert len(res["y_train"]) + len(res["y_test"]) == 20
