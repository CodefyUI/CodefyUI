"""Tests for DiffusionUNetNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.diffusion.diffusion_unet_node import DiffusionUNetNode


def _run(**params):
    p = {
        "in_channels": 3,
        "base_channels": 16,
        "channel_mult": "1,2,4",
        "time_emb_dim": 32,
        "num_groups": 4,
        "seed": 42,
    }
    p.update(params)
    return DiffusionUNetNode().execute({}, p)


def test_node_metadata():
    assert DiffusionUNetNode.NODE_NAME == "DiffusionUNet"
    assert DiffusionUNetNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in DiffusionUNetNode.define_outputs()]
    assert out_names == ["model"]


def test_outputs_callable_module():
    res = _run()
    model = res["model"]
    assert callable(model)


def test_forward_preserves_shape():
    """U-Net is a noise predictor: output same shape as input."""
    res = _run(in_channels=3, base_channels=16)
    model = res["model"]
    x = torch.randn(2, 3, 16, 16)
    t = torch.tensor([5, 5])
    pred = model(x, t)
    assert pred.shape == x.shape


def test_works_at_multiple_resolutions():
    """Power-of-2 resolutions divisible by 2^(num_levels-1) should work."""
    res = _run(channel_mult="1,2,4")
    model = res["model"]
    # 3 levels means 2 downsamples → input must be divisible by 4
    for h in (16, 32, 64):
        x = torch.randn(1, 3, h, h)
        t = torch.tensor([0])
        out = model(x, t)
        assert out.shape == x.shape


def test_different_in_out_channels():
    """When in_channels=4 (e.g. latent SD), output should also be 4."""
    res = _run(in_channels=4, base_channels=8)
    model = res["model"]
    x = torch.randn(1, 4, 8, 8)
    t = torch.tensor([0])
    out = model(x, t)
    assert out.shape == (1, 4, 8, 8)


def test_timestep_conditioning_changes_output():
    """Same x, different t → different prediction."""
    res = _run(seed=42)
    model = res["model"]
    x = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    a = model(x, torch.tensor([0]))
    b = model(x, torch.tensor([100]))
    assert not torch.allclose(a, b, atol=1e-4)


def test_deterministic_given_seed():
    """Same seed twice → same model behaviour."""
    a_model = _run(seed=42)["model"]
    b_model = _run(seed=42)["model"]
    x = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    t = torch.tensor([0])
    assert torch.allclose(a_model(x, t), b_model(x, t))


def test_invalid_channel_mult_raises():
    with pytest.raises(ValueError, match="channel_mult"):
        _run(channel_mult="not,valid")


def test_empty_channel_mult_raises():
    with pytest.raises(ValueError, match="channel_mult"):
        _run(channel_mult="")


def test_resolution_not_divisible_by_downsample_raises():
    """If channel_mult has 3 levels (2 downsamples), spatial dim must be divisible by 4."""
    res = _run(channel_mult="1,2,4")
    model = res["model"]
    # 16 → 8 → 4 ok; 17 → 8 (truncates) → 4 — pytorch will silently round.
    # Be strict: insist on cleanly divisible dimensions.
    x = torch.randn(1, 3, 13, 13)  # not divisible by 4
    t = torch.tensor([0])
    with pytest.raises((RuntimeError, ValueError)):
        model(x, t)
