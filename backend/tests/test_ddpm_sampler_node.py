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


# ── #691: the sampler samples with the schedule the model was trained on ──


class _StepRecorder(nn.Module):
    """Predicts zero noise and records every timestep it is asked about."""

    def __init__(self) -> None:
        super().__init__()
        self.timesteps: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.timesteps.append(int(t[0]))
        return torch.zeros_like(x)


def _reference_zero_eps_sample(noise, betas, seed):
    """Plain-PyTorch reverse DDPM for a model that predicts zero noise."""
    alphas = 1.0 - betas
    gen = torch.Generator().manual_seed(seed)
    x = noise.clone()
    for t in reversed(range(len(betas))):
        x = x / torch.sqrt(alphas[t])
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn(x.shape, generator=gen)
    return x


def _training_loop_defaults() -> dict:
    from app.nodes.training.diffusion_training_loop_node import DiffusionTrainingLoopNode

    return {p.name: p.default for p in DiffusionTrainingLoopNode.define_params()}


def test_a_fresh_sampler_samples_with_a_fresh_training_loop_schedule():
    """The issue's repro: a sampler left at its defaults must run the
    schedule a training loop left at its defaults trained on."""
    train = _training_loop_defaults()
    model = _StepRecorder()
    noise = torch.randn(2, 1, 4, 4, generator=torch.Generator().manual_seed(0))

    out = DDPMSamplerNode().execute({"model": model, "noise": noise}, {"seed": 3})

    assert model.timesteps == list(reversed(range(train["num_timesteps"])))
    betas = torch.linspace(train["beta_start"], train["beta_end"],
                           train["num_timesteps"], dtype=torch.float32)
    expected = _reference_zero_eps_sample(noise, betas, seed=3)
    assert torch.allclose(out["image"], expected, rtol=1e-4, atol=1e-5)


def _train_tiny_unet(**schedule):
    from app.nodes.training.diffusion_training_loop_node import DiffusionTrainingLoopNode

    unet = DiffusionUNetNode().execute({}, {
        "in_channels": 1, "base_channels": 8, "channel_mult": "1,2",
        "time_emb_dim": 16, "num_groups": 4, "seed": 0,
    })["model"]
    images = torch.randn(8, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    dataset = torch.utils.data.TensorDataset(images, torch.zeros(8))
    params = {"epochs": 1, "batch_size": 4, "device": "cpu", "seed": 0, **schedule}
    return DiffusionTrainingLoopNode().execute(
        {"model": unet, "dataset": dataset}, params)["model"]


class _WarningContext:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, str]] = []

    def log_warning(self, kind: str, detail: str, node_id=None) -> None:
        self.warnings.append((kind, detail))


def test_a_mismatched_sampler_follows_the_training_schedule_and_warns(caplog):
    schedule = {"num_timesteps": 12, "schedule": "linear",
                "beta_start": 0.0002, "beta_end": 0.03}
    model = _train_tiny_unet(**schedule)
    noise = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(2))

    matched = _run(model=model, noise=noise, num_steps=12,
                   beta_start=0.0002, beta_end=0.03, seed=5)
    context = _WarningContext()
    with caplog.at_level("WARNING"):
        mismatched = DDPMSamplerNode().execute(
            {"model": model, "noise": noise},
            {"num_steps": 20, "schedule": "linear", "beta_start": 0.0001,
             "beta_end": 0.02, "seed": 5},
            context=context,
        )

    assert torch.equal(mismatched["image"], matched["image"])
    [(kind, detail)] = context.warnings
    assert kind == "diffusion_schedule_mismatch"
    for fragment in ("num_steps=20", "trained with 12", "beta_end=0.02",
                     "beta_start=0.0001"):
        assert fragment in detail
    assert detail in caplog.text


def test_a_matching_sampler_does_not_warn(caplog):
    model = _train_tiny_unet(num_timesteps=6, beta_end=0.04)
    noise = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(2))
    context = _WarningContext()
    with caplog.at_level("WARNING"):
        DDPMSamplerNode().execute(
            {"model": model, "noise": noise},
            {"num_steps": 6, "schedule": "linear", "beta_start": 0.0001,
             "beta_end": 0.04, "seed": 0},
            context=context,
        )
    assert context.warnings == []
    assert "schedule" not in caplog.text


def test_a_cosine_training_schedule_ignores_the_sampler_betas():
    """beta_start/beta_end do not enter a cosine schedule, so differing
    values there are no mismatch."""
    model = _train_tiny_unet(num_timesteps=6, schedule="cosine")
    noise = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(2))
    context = _WarningContext()
    DDPMSamplerNode().execute(
        {"model": model, "noise": noise},
        {"num_steps": 6, "schedule": "cosine", "beta_start": 0.5,
         "beta_end": 0.9, "seed": 0},
        context=context,
    )
    assert context.warnings == []
