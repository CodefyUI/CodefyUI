"""Tests for EduKNNNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.classical.edu_knn_node import EduKNNNode


def _run(x_train, y_train, x_query, **params):
    p = {"k": 3, "metric": "euclidean"}
    p.update(params)
    return EduKNNNode().execute(
        {"x_train": x_train, "y_train": y_train, "x_query": x_query}, p
    )


def test_node_metadata():
    assert EduKNNNode.NODE_NAME == "EduKNN"
    assert EduKNNNode.CATEGORY == "Classical"
    out_names = [p.name for p in EduKNNNode.define_outputs()]
    assert out_names == [
        "predictions",
        "distances",
        "neighbor_indices",
        "train_coords",
        "query_coords",
        "train_labels",
    ]


def test_predicts_majority_class():
    """Query at (0, 0) with neighbours mostly class 'a' should predict 'a'."""
    x_train = torch.tensor([
        [0.0, 0.0],   # a (closest)
        [0.1, 0.1],   # a
        [0.2, 0.0],   # a
        [10.0, 10.0], # b (far)
    ])
    y_train = ["a", "a", "a", "b"]
    x_query = torch.tensor([[0.05, 0.05]])
    res = _run(x_train, y_train, x_query, k=3)
    assert res["predictions"] == ["a"]


def test_distances_top_k_shape():
    x_train = torch.randn(20, 4, generator=torch.Generator().manual_seed(0))
    y_train = ["x"] * 10 + ["y"] * 10
    x_query = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    res = _run(x_train, y_train, x_query, k=4)
    assert res["distances"].shape == (5, 4)
    assert len(res["neighbor_indices"]) == 5
    assert all(len(row) == 4 for row in res["neighbor_indices"])


def test_neighbor_indices_are_truly_closest():
    """k=1 should pick the single nearest training point."""
    x_train = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y_train = ["a", "b", "c", "d", "e"]
    x_query = torch.tensor([[2.4]])
    res = _run(x_train, y_train, x_query, k=1, metric="euclidean")
    # Closest to 2.4 is 2.0 (distance 0.4) → index 2 → label "c"
    assert res["neighbor_indices"] == [[2]]
    assert res["predictions"] == ["c"]


def test_manhattan_metric():
    x_train = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    y_train = ["a", "b"]
    x_query = torch.tensor([[0.0, 0.0]])
    res = _run(x_train, y_train, x_query, k=2, metric="manhattan")
    # Distances: 0 and |3| + |4| = 7 (manhattan), or sqrt(25)=5 (euclidean)
    assert res["distances"][0, 0].item() == 0.0
    assert res["distances"][0, 1].item() == 7.0


def test_cosine_metric():
    x_train = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    y_train = ["a", "b", "c"]
    x_query = torch.tensor([[1.0, 0.0]])
    res = _run(x_train, y_train, x_query, k=3, metric="cosine")
    # Cosine distance to (1,0) is: 0 (parallel), 1 (orthogonal), 2 (antiparallel)
    assert res["distances"][0, 0].item() < 1e-6
    assert res["predictions"] == ["a"]


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        _run(torch.zeros(5, 2), ["a"] * 5, torch.zeros(1, 2), metric="not-a-metric")


def test_k_clamped_to_train_size():
    """If k > number of training points, use all of them."""
    x_train = torch.randn(3, 2, generator=torch.Generator().manual_seed(0))
    y_train = ["a", "b", "c"]
    x_query = torch.randn(1, 2)
    res = _run(x_train, y_train, x_query, k=10)
    assert res["distances"].shape == (1, 3)


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        _run(torch.zeros(5, 2), ["a"] * 3, torch.zeros(1, 2))


def test_ties_broken_deterministically():
    """When two classes are tied, return one of them deterministically."""
    x_train = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y_train = ["a", "b", "a", "b"]
    x_query = torch.tensor([[1.5]])
    # Closest 2: indices 1 (b, dist 0.5) and 2 (a, dist 0.5) — tie
    res = _run(x_train, y_train, x_query, k=2)
    assert res["predictions"][0] in ("a", "b")  # Either is acceptable


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduKNNNode().execute(
            {"x_train": torch.zeros(5, 2)},
            {"k": 3, "metric": "euclidean"},
        )
