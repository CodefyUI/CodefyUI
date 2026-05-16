"""Tests for SoftmaxNode."""

from __future__ import annotations

import torch

from app.nodes.tensor_ops.softmax_node import SoftmaxNode


def _run(tensor, context=None, **params):
    return SoftmaxNode().execute({"tensor": tensor}, params, context=context)


def test_node_metadata():
    assert SoftmaxNode.NODE_NAME == "Softmax"
    assert SoftmaxNode.CATEGORY == "Tensor Operations"


def test_output_sums_to_one():
    x = torch.tensor([1.0, 2.0, 3.0])
    res = _run(x, dim=-1)
    assert torch.isclose(res["tensor"].sum(), torch.tensor(1.0))


def test_output_is_non_negative():
    x = torch.tensor([-100.0, 0.0, 100.0])
    res = _run(x, dim=-1)
    assert (res["tensor"] >= 0).all()


def test_uniform_input_yields_uniform_output():
    x = torch.ones(5)
    res = _run(x, dim=-1)
    assert torch.allclose(res["tensor"], torch.full((5,), 0.2))


def test_along_dim_zero():
    x = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    res = _run(x, dim=0)
    # Each column independently softmax — identical columns yield 0.5
    assert torch.allclose(res["tensor"], torch.full((2, 2), 0.5))


def test_numerical_stability_large_values():
    # Large positive values must not overflow
    x = torch.tensor([1000.0, 1000.0, 1000.0])
    res = _run(x, dim=-1)
    assert torch.isclose(res["tensor"].sum(), torch.tensor(1.0))
    assert not torch.isnan(res["tensor"]).any()


def test_verbose_mode_records_steps():
    class _Ctx:
        verbose = True

    x = torch.tensor([1.0, 2.0, 3.0])
    res = _run(x, dim=-1, context=_Ctx())
    assert "__steps__" in res
    step_names = [s.name for s in res["__steps__"]]
    assert "logits" in step_names
    assert "exp" in step_names
    assert "divide" in step_names


def test_non_verbose_no_steps():
    x = torch.tensor([1.0, 2.0, 3.0])
    res = _run(x, dim=-1)
    assert "__steps__" not in res
