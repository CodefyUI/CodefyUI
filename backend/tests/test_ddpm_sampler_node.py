"""Tests for DDPMSamplerNode."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.diffusion.ddpm_sampler_node import DDPMSamplerNode
from app.nodes.diffusion.diffusion_unet_node import DiffusionUNetNode


class _ZeroNoisePredictor(nn.Module):
    """Trivial 'model' that predicts zero noise — useful for testing the loop logic without UNet noise."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class _IdentityPredictor(nn.Module):
    """Returns x itself as 'predicted noise' — exercises the math but in a known way."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x


def _run(*, model, noise, condition=None, **params):
    p = {
        "num_steps": 5,
        "schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "seed": 42,
    }
    p.update(params)
    inputs: dict = {"model": model, "noise": noise}
    if condition is not None:
        inputs["condition"] = condition
    return DDPMSamplerNode().execute(inputs, p)


def test_node_metadata():
    assert DDPMSamplerNode.NODE_NAME == "DDPMSampler"
    assert DDPMSamplerNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in DDPMSamplerNode.define_outputs()]
    assert out_names == ["image"]


def test_output_shape_matches_noise():
    """Output shape should equal start-noise shape (no resampling)."""
    noise = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    res = _run(model=_ZeroNoisePredictor(), noise=noise, num_steps=10)
    assert res["image"].shape == noise.shape


def test_zero_noise_predictor_keeps_signal_finite():
    """Pure-zero model still runs the loop without NaN/Inf."""
    noise = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    res = _run(model=_ZeroNoisePredictor(), noise=noise, num_steps=10)
    assert torch.isfinite(res["image"]).all()


def test_deterministic_given_seed():
    """Same noise + same model + same seed → same final image."""
    noise = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    a = _run(model=_ZeroNoisePredictor(), noise=noise, seed=42, num_steps=5)
    b = _run(model=_ZeroNoisePredictor(), noise=noise, seed=42, num_steps=5)
    assert torch.allclose(a["image"], b["image"])


def test_different_num_steps_changes_result():
    """Different number of denoise steps → different image (more steps = different trajectory)."""
    noise = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    a = _run(model=_IdentityPredictor(), noise=noise, num_steps=3)
    b = _run(model=_IdentityPredictor(), noise=noise, num_steps=10)
    assert not torch.allclose(a["image"], b["image"], atol=1e-4)


def test_works_with_real_diffusion_unet():
    """End-to-end smoke: DiffusionUNet → DDPMSampler."""
    unet_res = DiffusionUNetNode().execute(
        {},
        {
            "in_channels": 3,
            "base_channels": 8,
            "channel_mult": "1,2",
            "time_emb_dim": 16,
            "num_groups": 4,
            "seed": 42,
        },
    )
    noise = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    res = _run(model=unet_res["model"], noise=noise, num_steps=5)
    assert res["image"].shape == (1, 3, 8, 8)
    assert torch.isfinite(res["image"]).all()


def test_cosine_schedule_works():
    """Just smoke-test the cosine schedule path."""
    noise = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    res = _run(model=_ZeroNoisePredictor(), noise=noise, schedule="cosine", num_steps=5)
    assert torch.isfinite(res["image"]).all()


def test_unknown_schedule_raises():
    noise = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="schedule"):
        _run(model=_ZeroNoisePredictor(), noise=noise, schedule="not-a-schedule")


def test_missing_model_raises():
    with pytest.raises(ValueError, match="requires"):
        DDPMSamplerNode().execute(
            {"noise": torch.zeros(1, 3, 8, 8)},
            {"num_steps": 5, "schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02, "seed": 42},
        )


def test_missing_noise_raises():
    with pytest.raises(ValueError, match="requires"):
        DDPMSamplerNode().execute(
            {"model": _ZeroNoisePredictor()},
            {"num_steps": 5, "schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02, "seed": 42},
        )


def test_num_steps_at_least_one():
    noise = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="num_steps"):
        _run(model=_ZeroNoisePredictor(), noise=noise, num_steps=0)
