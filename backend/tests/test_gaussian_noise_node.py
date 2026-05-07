"""Tests for GaussianNoiseNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.diffusion.gaussian_noise_node import GaussianNoiseNode


def _run(*, shape_ref=None, **params):
    p = {"shape": "1,3,4,4", "mean": 0.0, "std": 1.0, "seed": 42}
    p.update(params)
    inputs: dict = {}
    if shape_ref is not None:
        inputs["shape_ref"] = shape_ref
    return GaussianNoiseNode().execute(inputs, p)


def test_node_metadata():
    assert GaussianNoiseNode.NODE_NAME == "GaussianNoise"
    assert GaussianNoiseNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in GaussianNoiseNode.define_outputs()]
    assert out_names == ["noise"]


def test_default_shape_from_param():
    res = _run(shape="2,3,8,8")
    assert res["noise"].shape == (2, 3, 8, 8)


def test_shape_ref_input_takes_precedence():
    """When shape_ref input connected, use its shape regardless of param."""
    ref = torch.zeros(4, 5, 6)
    res = _run(shape_ref=ref, shape="1,1,1,1")
    assert res["noise"].shape == (4, 5, 6)


def test_seed_reproducibility():
    a = _run(seed=123)
    b = _run(seed=123)
    assert torch.equal(a["noise"], b["noise"])


def test_different_seeds_produce_different_noise():
    a = _run(seed=1)
    b = _run(seed=2)
    assert not torch.equal(a["noise"], b["noise"])


def test_default_mean_zero_std_one():
    """Sample many values to verify approximate N(0, 1) distribution."""
    res = _run(shape="10000", seed=42)
    n = res["noise"]
    assert abs(n.mean().item()) < 0.05  # near zero
    assert abs(n.std().item() - 1.0) < 0.05  # near one


def test_custom_mean_and_std():
    res = _run(shape="10000", mean=5.0, std=2.0, seed=42)
    n = res["noise"]
    assert abs(n.mean().item() - 5.0) < 0.1
    assert abs(n.std().item() - 2.0) < 0.1


def test_does_not_perturb_global_rng():
    """The seeded local generator shouldn't disturb the global RNG state."""
    torch.manual_seed(99)
    before = torch.randn(3)
    torch.manual_seed(99)
    _ = _run(seed=42)  # should not move global RNG
    after = torch.randn(3)
    assert torch.allclose(before, after)


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        _run(shape="not,valid,nope")


def test_empty_shape_string_raises():
    with pytest.raises(ValueError, match="shape"):
        _run(shape="")


def test_dtype_is_float32():
    res = _run()
    assert res["noise"].dtype == torch.float32
