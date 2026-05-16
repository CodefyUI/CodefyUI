"""Tests for DropoutNode."""

from __future__ import annotations

import torch

from app.nodes.cnn.dropout_node import DropoutNode


def _run(tensor, **params):
    return DropoutNode().execute({"tensor": tensor}, params)


def test_node_metadata():
    assert DropoutNode.NODE_NAME == "Dropout"
    p = [pp for pp in DropoutNode.define_params() if pp.name == "p"][0]
    assert p.default == 0.5
    assert p.min_value == 0.0
    assert p.max_value == 1.0


def test_dropout_p_zero_is_identity():
    x = torch.ones(100)
    res = _run(x, p=0.0)
    assert torch.equal(res["tensor"], x)


def test_dropout_p_one_zeros_everything():
    # In training mode (default after building nn.Dropout), p=1 zeros all
    x = torch.ones(100)
    res = _run(x, p=1.0)
    assert torch.all(res["tensor"] == 0.0)


def test_dropout_preserves_shape():
    x = torch.randn(4, 8, 16, 16)
    res = _run(x, p=0.5)
    assert res["tensor"].shape == x.shape


def test_dropout_p_default_is_half():
    """When p is default 0.5, output should differ from input for large tensors."""
    torch.manual_seed(0)
    x = torch.ones(1000)
    res = _run(x)
    # Some entries should be zeroed
    zeros = (res["tensor"] == 0).sum().item()
    assert zeros > 0
    assert zeros < 1000
