"""Tests for ActivationNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.cnn.activation_node import ActivationNode


def _run(tensor, **params):
    return ActivationNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert ActivationNode.NODE_NAME == "Activation"
    assert ActivationNode.CATEGORY == "CNN"
    options = [p for p in ActivationNode.define_params() if p.name == "function"][0].options
    assert "relu" in options
    assert "gelu" in options


def test_relu_clamps_negative_to_zero():
    x = torch.tensor([-1.0, 0.0, 1.0])
    res = _run(x, function="relu")
    assert torch.allclose(res["tensor"], torch.tensor([0.0, 0.0, 1.0]))


def test_relu_is_default():
    x = torch.tensor([-1.0, 2.0])
    res = _run(x)
    assert torch.allclose(res["tensor"], torch.tensor([0.0, 2.0]))


def test_sigmoid_in_zero_one():
    x = torch.tensor([-100.0, 0.0, 100.0])
    res = _run(x, function="sigmoid")
    assert torch.isclose(res["tensor"][0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(res["tensor"][1], torch.tensor(0.5))
    assert torch.isclose(res["tensor"][2], torch.tensor(1.0), atol=1e-6)


def test_tanh_in_minus_one_to_one():
    x = torch.tensor([-100.0, 0.0, 100.0])
    res = _run(x, function="tanh")
    assert torch.isclose(res["tensor"][0], torch.tensor(-1.0), atol=1e-6)
    assert torch.isclose(res["tensor"][1], torch.tensor(0.0))
    assert torch.isclose(res["tensor"][2], torch.tensor(1.0), atol=1e-6)


def test_leaky_relu_keeps_small_negative():
    x = torch.tensor([-10.0, 5.0])
    res = _run(x, function="leaky_relu")
    # negative_slope=0.01 default
    assert torch.isclose(res["tensor"][0], torch.tensor(-0.1))
    assert torch.isclose(res["tensor"][1], torch.tensor(5.0))


def test_gelu_smooth_near_origin():
    x = torch.tensor([0.0])
    res = _run(x, function="gelu")
    assert torch.isclose(res["tensor"], torch.tensor(0.0))


def test_silu_x_times_sigmoid_x():
    x = torch.tensor([1.0, 2.0])
    res = _run(x, function="silu")
    expected = x * torch.sigmoid(x)
    assert torch.allclose(res["tensor"], expected)


def test_softmax_normalizes_along_last_dim():
    x = torch.tensor([1.0, 2.0, 3.0])
    res = _run(x, function="softmax")
    assert torch.isclose(res["tensor"].sum(), torch.tensor(1.0))


def test_elu_smooth_negative():
    x = torch.tensor([-1.0, 1.0])
    res = _run(x, function="elu")
    # elu(-1) = exp(-1) - 1 ≈ -0.6321
    assert res["tensor"][0] < 0
    assert res["tensor"][0] > -1
    assert res["tensor"][1] == 1.0


def test_unknown_function_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        _run(torch.zeros(3), function="bogus")


def test_preserves_shape():
    x = torch.randn(2, 3, 4)
    res = _run(x, function="relu")
    assert res["tensor"].shape == x.shape
