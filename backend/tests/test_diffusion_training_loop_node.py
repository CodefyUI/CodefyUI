"""DiffusionTrainingLoop: its `seed` holds beside other nodes, and a float64
dataset trains on MPS (2026-10-13 audit)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from app.nodes.training.diffusion_training_loop_node import DiffusionTrainingLoopNode


class _TinyDenoiser(nn.Module):
    """``forward(x, t)`` -> predicted noise; optionally draws from the global
    RNG on every call, the way a node running beside the loop would."""

    def __init__(self, draw_global: bool = False) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=generator) * 0.1)
            self.conv.bias.zero_()
        self.time = nn.Embedding(10, 1)
        with torch.no_grad():
            self.time.weight.copy_(torch.randn(self.time.weight.shape, generator=generator))
        self.draw_global = draw_global

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.draw_global:
            torch.rand(17)
        return self.conv(x) + self.time(t)[:, :, None, None]


def _dataset(dtype=torch.float32):
    images = torch.linspace(-1, 1, 24 * 16, dtype=dtype).reshape(24, 1, 4, 4)
    return torch.utils.data.TensorDataset(images, torch.zeros(24))


def _train(model, dataset, **params):
    p = {"epochs": 3, "batch_size": 8, "lr": 0.01, "num_timesteps": 10,
         "seed": 0, "device": "cpu"}
    p.update(params)
    return DiffusionTrainingLoopNode().execute(
        {"model": model, "dataset": dataset}, p)


def test_the_seed_holds_when_something_else_draws_from_the_global_rng():
    # A concurrent node drawing from the shared stream (simulated by the
    # model's own draws) used to shift the shuffle, the timesteps and the
    # noise, so the same seed gave a different loss curve.
    quiet = _train(_TinyDenoiser(draw_global=False), _dataset())["losses"]
    torch.manual_seed(12345)
    noisy = _train(_TinyDenoiser(draw_global=True), _dataset())["losses"]
    assert torch.equal(quiet, noisy)


def test_the_loop_never_reseeds_the_global_rng(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("reseeded the process-wide RNG")

    monkeypatch.setattr(torch, "manual_seed", refuse)
    first = _train(_TinyDenoiser(), _dataset())["losses"]
    second = _train(_TinyDenoiser(), _dataset())["losses"]
    assert torch.equal(first, second)


def test_different_seeds_give_different_curves():
    a = _train(_TinyDenoiser(), _dataset(), seed=0)["losses"]
    b = _train(_TinyDenoiser(), _dataset(), seed=1)["losses"]
    assert not torch.equal(a, b)


def test_a_float64_dataset_trains_on_cpu():
    out = _train(_TinyDenoiser(), _dataset(torch.float64))
    assert out["losses"].shape == (3,)
    assert torch.isfinite(out["losses"]).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_a_float64_dataset_trains_on_mps():
    # ``x0.to("mps")`` on float64 raised "Cannot convert a MPS Tensor to
    # float64 dtype"; the batch is cast on the host first now.
    out = _train(_TinyDenoiser(), _dataset(torch.float64), device="mps")
    assert torch.isfinite(out["losses"]).all()
