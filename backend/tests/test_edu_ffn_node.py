"""Tests for EduFFNNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.llm.edu_ffn_node import EduFFNNode


def _run(tensor, **params):
    p = {
        "embed_dim": 8,
        "hidden_dim": 16,
        "activation": "relu",
        "seed": 42,
    }
    p.update(params)
    return EduFFNNode().execute({"tensor": tensor}, p)


def test_node_metadata():
    assert EduFFNNode.NODE_NAME == "EduFFN"
    assert EduFFNNode.CATEGORY == "LLM"
    out_names = [p.name for p in EduFFNNode.define_outputs()]
    assert out_names == ["tensor", "activations"]


def test_output_shape_2d():
    res = _run(torch.zeros(4, 8))
    assert res["tensor"].shape == (4, 8)
    assert res["activations"].shape == (4, 16)


def test_output_shape_3d_batch():
    res = _run(torch.zeros(4, 2, 8))
    assert res["tensor"].shape == (4, 2, 8)
    assert res["activations"].shape == (4, 2, 16)


def test_relu_activations_are_non_negative():
    res = _run(torch.randn(8, 8), activation="relu")
    assert torch.all(res["activations"] >= 0)


def test_gelu_activations_can_be_negative():
    """GELU is smooth, allows small negative values."""
    res = _run(torch.randn(8, 8) * 5.0, activation="gelu")
    # GELU should produce some negative values for negative inputs.
    assert (res["activations"] < 0).any()


def test_deterministic_given_seed():
    a = _run(torch.ones(4, 8), seed=1)
    b = _run(torch.ones(4, 8), seed=1)
    assert torch.allclose(a["tensor"], b["tensor"])
    assert torch.allclose(a["activations"], b["activations"])


def test_different_seeds_give_different_outputs():
    a = _run(torch.ones(4, 8), seed=1)
    b = _run(torch.ones(4, 8), seed=2)
    assert not torch.allclose(a["tensor"], b["tensor"])


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="activation"):
        _run(torch.zeros(4, 8), activation="not-a-fn")


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduFFNNode().execute({}, {"embed_dim": 8, "hidden_dim": 16, "activation": "relu", "seed": 42})


def test_embed_dim_mismatch_raises():
    """Input D must match embed_dim param."""
    with pytest.raises(ValueError, match="embed_dim"):
        _run(torch.zeros(4, 4), embed_dim=8)
