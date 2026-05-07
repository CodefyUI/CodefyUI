"""Tests for AttentionHeatmapNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.llm.attention_heatmap_node import AttentionHeatmapNode


def _run(weights, *, labels=None, **params):
    p = {"head_index": -1, "colormap": "viridis"}
    p.update(params)
    inputs: dict = {"weights": weights}
    if labels is not None:
        inputs["labels"] = labels
    return AttentionHeatmapNode().execute(inputs, p)


def test_node_metadata():
    assert AttentionHeatmapNode.NODE_NAME == "AttentionHeatmap"
    assert AttentionHeatmapNode.CATEGORY == "LLM"
    out_names = [p.name for p in AttentionHeatmapNode.define_outputs()]
    assert out_names == ["weights", "labels"]


def test_2d_weights_pass_through_unchanged():
    w = torch.rand(5, 5)
    res = _run(w)
    assert torch.equal(res["weights"], w)


def test_3d_per_head_weights_pass_through():
    w = torch.rand(4, 6, 6)  # [H, seq, seq]
    res = _run(w)
    assert torch.equal(res["weights"], w)


def test_labels_pass_through():
    res = _run(torch.eye(3), labels=["a", "b", "c"])
    assert res["labels"] == ["a", "b", "c"]


def test_labels_empty_when_not_provided():
    res = _run(torch.eye(3))
    assert res["labels"] == []


def test_head_index_selects_single_head_from_3d():
    w = torch.stack([torch.eye(4), torch.zeros(4, 4)])  # [2, 4, 4]
    res = _run(w, head_index=0)
    assert torch.equal(res["weights"], torch.eye(4))


def test_head_index_negative_returns_all_heads():
    w = torch.rand(3, 4, 4)
    res = _run(w, head_index=-1)
    assert res["weights"].shape == (3, 4, 4)


def test_head_index_out_of_range_raises():
    w = torch.rand(2, 4, 4)
    with pytest.raises(ValueError, match="head_index"):
        _run(w, head_index=5)


def test_head_index_on_2d_input_is_ignored():
    """2D weights aren't per-head, so head_index doesn't apply."""
    w = torch.eye(4)
    res = _run(w, head_index=0)
    assert torch.equal(res["weights"], w)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        AttentionHeatmapNode().execute({}, {"head_index": -1, "colormap": "viridis"})


def test_4d_batch_per_head_weights_pass_through():
    """[batch, H, seq, seq] should also pass through."""
    w = torch.rand(2, 3, 5, 5)
    res = _run(w)
    assert torch.equal(res["weights"], w)
