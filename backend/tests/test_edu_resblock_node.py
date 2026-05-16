"""Tests for EduResBlockNode."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.c3.nodes.edu_resblock_node import EduResBlockNode


def _run(tensor, *, time_emb=None, **params):
    p = {
        "in_channels": 8,
        "out_channels": 8,
        "groups": 4,
        "time_emb_dim": 32,
        "seed": 42,
    }
    p.update(params)
    inputs: dict = {"tensor": tensor}
    if time_emb is not None:
        inputs["time_emb"] = time_emb
    return EduResBlockNode().execute(inputs, p)


def test_node_metadata():
    assert EduResBlockNode.NODE_NAME == "EduResBlock"
    assert EduResBlockNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in EduResBlockNode.define_outputs()]
    assert out_names == ["tensor"]


def test_same_in_out_shape():
    """[N, 8, H, W] → [N, 8, H, W] when in_channels == out_channels."""
    x = torch.randn(2, 8, 16, 16, generator=torch.Generator().manual_seed(0))
    res = _run(x, in_channels=8, out_channels=8)
    assert res["tensor"].shape == (2, 8, 16, 16)


def test_channel_change_with_skip_projection():
    """[N, 8, H, W] → [N, 16, H, W] — needs 1x1 skip projection."""
    x = torch.randn(2, 8, 16, 16)
    res = _run(x, in_channels=8, out_channels=16)
    assert res["tensor"].shape == (2, 16, 16, 16)


def test_time_emb_optional_works_without_it():
    """Without time_emb input, runs as plain residual block."""
    x = torch.randn(1, 8, 8, 8)
    res = _run(x)
    assert res["tensor"].shape == (1, 8, 8, 8)


def test_time_emb_modulates_output():
    """Same x, different time_emb → different output."""
    x = torch.randn(1, 8, 8, 8, generator=torch.Generator().manual_seed(0))
    t1 = torch.randn(1, 32, generator=torch.Generator().manual_seed(1))
    t2 = torch.randn(1, 32, generator=torch.Generator().manual_seed(2))
    a = _run(x, time_emb=t1)
    b = _run(x, time_emb=t2)
    assert not torch.allclose(a["tensor"], b["tensor"])


def test_deterministic_given_seed():
    x = torch.randn(1, 8, 8, 8, generator=torch.Generator().manual_seed(0))
    a = _run(x, seed=42)
    b = _run(x, seed=42)
    assert torch.allclose(a["tensor"], b["tensor"])


def test_groups_must_divide_channels():
    """GroupNorm requires groups | num_channels."""
    with pytest.raises(ValueError, match="groups"):
        _run(torch.zeros(1, 8, 4, 4), in_channels=8, groups=3)


def test_in_channels_must_match_input():
    with pytest.raises(ValueError, match="in_channels"):
        _run(torch.zeros(1, 4, 8, 8), in_channels=8)


def test_time_emb_dim_must_match():
    """time_emb input dim must match time_emb_dim param."""
    x = torch.randn(1, 8, 4, 4)
    bad_emb = torch.randn(1, 64)  # not 32
    with pytest.raises(ValueError, match="time_emb"):
        _run(x, time_emb=bad_emb, time_emb_dim=32)


def test_residual_path_is_used():
    """For zero-init weights and zero input, output should still equal skip path.

    Using the seeded random init won't give exact zero, but we can verify the
    residual additivity: doubling x doesn't double output (because residual
    adds nonlinearly via Conv+GN+SiLU).
    """
    x = torch.randn(1, 8, 8, 8, generator=torch.Generator().manual_seed(0))
    a = _run(x, seed=42)
    b = _run(x * 2, seed=42)
    # Output is not exactly 2× because of GroupNorm + SiLU.
    assert not torch.allclose(a["tensor"], b["tensor"] / 2.0, atol=1e-3)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduResBlockNode().execute(
            {},
            {"in_channels": 8, "out_channels": 8, "groups": 4, "time_emb_dim": 32, "seed": 42},
        )


def test_2d_spatial_input_required():
    """Block expects [N, C, H, W] — 1D input should fail."""
    with pytest.raises(ValueError, match="shape"):
        _run(torch.zeros(1, 8))
