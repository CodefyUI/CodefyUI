"""Tests for UpsampleNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.diffusion.upsample_node import UpsampleNode


def _run(tensor, **params):
    p = {"mode": "nearest", "scale_factor": 2.0}
    p.update(params)
    return UpsampleNode().execute({"tensor": tensor}, p)


def test_node_metadata():
    assert UpsampleNode.NODE_NAME == "Upsample"
    assert UpsampleNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in UpsampleNode.define_outputs()]
    assert out_names == ["tensor"]


def test_default_doubles_spatial_dims_nchw():
    res = _run(torch.zeros(1, 3, 4, 4), scale_factor=2.0)
    assert res["tensor"].shape == (1, 3, 8, 8)


def test_scale_factor_3_triples_spatial_dims():
    res = _run(torch.zeros(2, 8, 4, 4), scale_factor=3.0)
    assert res["tensor"].shape == (2, 8, 12, 12)


def test_scale_factor_half_downsamples():
    res = _run(torch.zeros(1, 3, 8, 8), scale_factor=0.5)
    assert res["tensor"].shape == (1, 3, 4, 4)


def test_nearest_mode_replicates_pixels():
    """Nearest-neighbor upsample of a 1x1x2x2 tensor should give a 4x4 with each value 4-replicated."""
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    res = _run(x, mode="nearest", scale_factor=2.0)
    # Top-left 2x2 block should all be 1.0
    assert torch.all(res["tensor"][0, 0, 0:2, 0:2] == 1.0)
    # Bottom-right 2x2 block should all be 4.0
    assert torch.all(res["tensor"][0, 0, 2:4, 2:4] == 4.0)


def test_bilinear_mode_blends_neighbors():
    """Bilinear should produce intermediate values, unlike nearest."""
    x = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    res = _run(x, mode="bilinear", scale_factor=2.0)
    # Some cells should land between 0 and 1 (smooth interpolation).
    # Nearest would only ever produce 0.0 or 1.0; bilinear breaks that.
    vals = res["tensor"].unique()
    assert len(vals) > 2


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        _run(torch.zeros(1, 3, 4, 4), mode="not-a-mode")


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        UpsampleNode().execute({}, {"mode": "nearest", "scale_factor": 2.0})


def test_handles_3d_tensor_one_spatial_dim():
    """[N, C, L] input — 1D upsampling."""
    x = torch.zeros(1, 4, 8)
    res = _run(x, mode="nearest", scale_factor=2.0)
    assert res["tensor"].shape == (1, 4, 16)


def test_dtype_preserved():
    x = torch.zeros(1, 3, 4, 4, dtype=torch.float32)
    res = _run(x)
    assert res["tensor"].dtype == torch.float32
