"""Tests for BackwardOnceNode (pass-through marker)."""

from __future__ import annotations

import torch

from app.nodes.training.backward_once_node import BackwardOnceNode


def test_node_metadata():
    assert BackwardOnceNode.NODE_NAME == "BackwardOnce"
    assert BackwardOnceNode.CATEGORY == "Training"
    assert BackwardOnceNode.cacheable is False


def test_pass_through_returns_input_unchanged():
    x = torch.randn(2, 3)
    res = BackwardOnceNode().execute({"tensor": x}, {})
    assert torch.equal(res["tensor"], x)


def test_pass_through_preserves_identity():
    """BackwardOnce should be a no-op — same tensor object."""
    x = torch.randn(4, 4)
    res = BackwardOnceNode().execute({"tensor": x}, {})
    assert res["tensor"] is x


def test_preserves_grad_state():
    x = torch.randn(2, 3, requires_grad=True)
    res = BackwardOnceNode().execute({"tensor": x}, {})
    assert res["tensor"].requires_grad


def test_no_params_defined():
    assert BackwardOnceNode.define_params() == []
