"""Tests for KNNNode (sklearn wrapper)."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.knn_node import KNNNode


def _run(x_train, y_train, x_query, **params):
    p = {"n_neighbors": 3, "weights": "uniform", "metric": "minkowski"}
    p.update(params)
    return KNNNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert KNNNode.NODE_NAME == "KNN"
    assert KNNNode.CATEGORY == "Classical"
    out_names = [p.name for p in KNNNode.define_outputs()]
    assert "predictions" in out_names
    assert "classes" in out_names


def test_predicts_majority_class():
    x_train = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
        ]
    )
    y_train = ["a", "a", "a", "b"]
    x_query = torch.tensor([[0.05, 0.05]])
    res = _run(x_train, y_train, x_query, n_neighbors=3)
    assert res["predictions"] == ["a"]


def test_returns_classes_in_sorted_order():
    x_train = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y_train = ["b", "a", "a", "b"]
    x_query = torch.tensor([[0.0]])
    res = _run(x_train, y_train, x_query, n_neighbors=1)
    # sklearn uses np.unique → sorted strings
    assert res["classes"] == ["a", "b"]


def test_distance_weighting_prefers_closest():
    """weights='distance' makes the closer point dominate even when outvoted."""
    # Three neighbours: index 0 close, indices 1 and 2 far
    x_train = torch.tensor([[0.0], [10.0], [10.5]])
    y_train = ["a", "b", "b"]
    x_query = torch.tensor([[0.1]])
    # Uniform: 2 b's vs 1 a → predicts b
    res_uni = _run(x_train, y_train, x_query, n_neighbors=3, weights="uniform")
    assert res_uni["predictions"] == ["b"]
    # Distance: a is much closer (≈0.1) vs b (≈10) → predicts a
    res_dist = _run(x_train, y_train, x_query, n_neighbors=3, weights="distance")
    assert res_dist["predictions"] == ["a"]


def test_works_with_integer_labels():
    x_train = torch.randn(20, 4, generator=torch.Generator().manual_seed(0))
    y_train = torch.tensor([0] * 10 + [1] * 10)
    x_query = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    res = _run(x_train, y_train, x_query, n_neighbors=3)
    assert len(res["predictions"]) == 5


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), ["a"] * 3, torch.zeros(1, 2))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        KNNNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"n_neighbors": 3, "weights": "uniform", "metric": "minkowski"},
        )


def test_n_neighbors_clamped_to_train_size():
    x_train = torch.randn(3, 2, generator=torch.Generator().manual_seed(0))
    y_train = ["a", "b", "c"]
    x_query = torch.randn(1, 2)
    # Should not crash even though n_neighbors > N_train
    res = _run(x_train, y_train, x_query, n_neighbors=10)
    assert len(res["predictions"]) == 1
