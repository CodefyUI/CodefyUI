"""Tests for LerpNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.diffusion.lerp_node import LerpNode


def _run(a, b, *, alpha=None, **params):
    p = {"alpha": 0.5}
    p.update(params)
    inputs: dict = {"tensor_a": a, "tensor_b": b}
    if alpha is not None:
        inputs["alpha"] = alpha
    return LerpNode().execute(inputs, p)


def test_node_metadata():
    assert LerpNode.NODE_NAME == "Lerp"
    assert LerpNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in LerpNode.define_outputs()]
    assert out_names == ["tensor"]


def test_alpha_zero_returns_b():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    res = _run(a, b, alpha=0.0)
    assert torch.allclose(res["tensor"], b)


def test_alpha_one_returns_a():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    res = _run(a, b, alpha=1.0)
    assert torch.allclose(res["tensor"], a)


def test_alpha_half_averages():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    res = _run(a, b, alpha=0.5)
    assert torch.allclose(res["tensor"], torch.tensor([2.5, 3.5, 4.5]))


def test_alpha_param_used_when_no_input():
    """When alpha input not connected, use the alpha param."""
    a = torch.tensor([0.0])
    b = torch.tensor([10.0])
    res = LerpNode().execute(
        {"tensor_a": a, "tensor_b": b},
        {"alpha": 0.3},
    )
    # 0.3*0 + 0.7*10 = 7.0
    assert torch.allclose(res["tensor"], torch.tensor([7.0]))


def test_scalar_tensor_alpha_input():
    """alpha input as a 0-d tensor scalar."""
    a = torch.tensor([0.0])
    b = torch.tensor([10.0])
    res = _run(a, b, alpha=torch.tensor(0.25))
    assert torch.allclose(res["tensor"], torch.tensor([7.5]))


def test_broadcasted_alpha_per_sample():
    """alpha as a [B, 1, 1, 1] tensor for batch-wise interpolation."""
    a = torch.zeros(2, 3, 2, 2)
    b = torch.ones(2, 3, 2, 2)
    alpha = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)
    res = _run(a, b, alpha=alpha)
    # Sample 0: 0*0 + 1*1 = 1.0; sample 1: 1*0 + 0*1 = 0.0
    assert torch.all(res["tensor"][0] == 1.0)
    assert torch.all(res["tensor"][1] == 0.0)


def test_diffusion_forward_equation():
    """Canonical diffusion forward: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise.

    Lerp models this when we set alpha = sqrt(alpha_bar):
        x_t = alpha*x_0 + (1-alpha)*noise   # NOTE: this is sqrt(α̅)x₀ + (1-sqrt(α̅))ε
    The exact diffusion formula uses sqrt(1-α̅), not 1-sqrt(α̅), so Lerp
    is a *teaching* approximation. This test just verifies the basic
    convex-combination property, not the diffusion equation.
    """
    x0 = torch.randn(1, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    eps = torch.randn(1, 3, 4, 4, generator=torch.Generator().manual_seed(1))
    res = _run(x0, eps, alpha=0.7)
    # Output should be a convex combination — pixel-wise within [min, max].
    expected = 0.7 * x0 + 0.3 * eps
    assert torch.allclose(res["tensor"], expected)


def test_shape_mismatch_via_broadcast_works_for_compatible_dims():
    """[3] + [1] → broadcast to [3]."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0])  # broadcastable
    res = _run(a, b, alpha=0.5)
    # 0.5*[1,2,3] + 0.5*[10,10,10] = [5.5, 6, 6.5]
    assert torch.allclose(res["tensor"], torch.tensor([5.5, 6.0, 6.5]))


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        LerpNode().execute({"tensor_a": torch.zeros(1)}, {"alpha": 0.5})
