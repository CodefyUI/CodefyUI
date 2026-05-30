"""Tests for EduDenoiseStepNode (chapter pack I3-3, one DDIM reverse step)."""

from __future__ import annotations

import pytest
import torch

from cdui_plugins.deep.nodes.edu_denoise_step_node import EduDenoiseStepNode


def _run(x_t, noise_pred, *, context=None, **params):
    base = {"t": 50, "num_steps": 100, "beta_start": 0.0001, "beta_end": 0.02}
    base.update(params)
    return EduDenoiseStepNode().execute(
        {"x_t": x_t, "noise_pred": noise_pred}, base, context=context
    )


def _abars(num_steps, beta_start, beta_end, t):
    """Reference schedule: returns (abar_t, abar_prev)."""
    betas = torch.linspace(beta_start, beta_end, num_steps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return alphas_cumprod[t], alphas_cumprod[t - 1]


class _VerboseCtx:
    verbose = True


class _QuietCtx:
    verbose = False


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_node_metadata():
    assert EduDenoiseStepNode.NODE_NAME == "Edu-DenoiseStep"
    assert EduDenoiseStepNode.CATEGORY == "Diffusion"
    out_names = [p.name for p in EduDenoiseStepNode.define_outputs()]
    assert out_names == ["x_prev", "pred_x0"]
    in_names = [p.name for p in EduDenoiseStepNode.define_inputs()]
    assert in_names == ["x_t", "noise_pred"]


# --------------------------------------------------------------------------- #
# Hand-computed numeric correctness
# --------------------------------------------------------------------------- #
def test_hand_computed_step():
    # Small schedule we can fully recompute by the DDIM formulas.
    num_steps, beta_start, beta_end, t = 4, 0.1, 0.4, 2
    x_t = torch.tensor([[[[3.0]]]])
    noise_pred = torch.tensor([[[[0.5]]]])

    abar_t, abar_prev = _abars(num_steps, beta_start, beta_end, t)
    expected_pred_x0 = (x_t - torch.sqrt(1.0 - abar_t) * noise_pred) / torch.sqrt(abar_t)
    expected_x_prev = (
        torch.sqrt(abar_prev) * expected_pred_x0
        + torch.sqrt(1.0 - abar_prev) * noise_pred
    )

    res = _run(
        x_t,
        noise_pred,
        t=t,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    assert torch.allclose(res["pred_x0"], expected_pred_x0, atol=1e-6)
    assert torch.allclose(res["x_prev"], expected_x_prev, atol=1e-6)


def test_shape_preserved():
    x_t = torch.randn(2, 3, 5, 7)
    noise_pred = torch.randn(2, 3, 5, 7)
    res = _run(x_t, noise_pred, t=10, num_steps=50)
    assert res["x_prev"].shape == x_t.shape
    assert res["pred_x0"].shape == x_t.shape


def test_arbitrary_shape_supported():
    # The node operates elementwise; a flat vector must work too.
    x_t = torch.randn(6)
    noise_pred = torch.randn(6)
    res = _run(x_t, noise_pred, t=3, num_steps=20)
    assert res["x_prev"].shape == (6,)
    assert res["pred_x0"].shape == (6,)


def test_zero_noise_gives_scaled_x0():
    # With eps = 0:  pred_x0 = x_t / sqrt(abar_t).
    num_steps, beta_start, beta_end, t = 8, 0.0001, 0.02, 5
    x_t = torch.randn(1, 2, 3, 3)
    noise_pred = torch.zeros_like(x_t)

    abar_t, _ = _abars(num_steps, beta_start, beta_end, t)
    expected = x_t / torch.sqrt(abar_t)

    res = _run(
        x_t,
        noise_pred,
        t=t,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    assert torch.allclose(res["pred_x0"], expected, atol=1e-6)
    # And with zero noise the re-noising reduces to sqrt(abar_prev) * pred_x0.
    _, abar_prev = _abars(num_steps, beta_start, beta_end, t)
    assert torch.allclose(
        res["x_prev"], torch.sqrt(abar_prev) * expected, atol=1e-6
    )


def test_deterministic_repeated_calls():
    x_t = torch.randn(1, 1, 4, 4)
    noise_pred = torch.randn(1, 1, 4, 4)
    a = _run(x_t, noise_pred, t=7, num_steps=30)
    b = _run(x_t, noise_pred, t=7, num_steps=30)
    assert torch.equal(a["x_prev"], b["x_prev"])
    assert torch.equal(a["pred_x0"], b["pred_x0"])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_shape_mismatch_raises():
    x_t = torch.randn(1, 1, 4, 4)
    noise_pred = torch.randn(1, 1, 4, 5)
    with pytest.raises(ValueError, match="shape"):
        _run(x_t, noise_pred, t=5, num_steps=20)


def test_t_too_large_raises():
    x_t = torch.randn(1, 1, 1, 1)
    noise_pred = torch.zeros_like(x_t)
    # t == num_steps is out of range (valid indices are 1..num_steps-1).
    with pytest.raises(ValueError, match="t must satisfy"):
        _run(x_t, noise_pred, t=20, num_steps=20)


def test_t_too_small_raises():
    x_t = torch.randn(1, 1, 1, 1)
    noise_pred = torch.zeros_like(x_t)
    with pytest.raises(ValueError, match="t must satisfy"):
        _run(x_t, noise_pred, t=0, num_steps=20)


def test_missing_input_raises():
    with pytest.raises(ValueError, match="requires"):
        EduDenoiseStepNode().execute({"x_t": torch.zeros(1, 1, 1, 1)}, {})


# --------------------------------------------------------------------------- #
# Verbose step trace
# --------------------------------------------------------------------------- #
def test_steps_absent_without_verbose():
    x_t = torch.randn(1, 1, 2, 2)
    noise_pred = torch.randn(1, 1, 2, 2)
    # No context at all.
    assert "__steps__" not in _run(x_t, noise_pred, t=3, num_steps=20)
    # Context present but verbose=False.
    assert "__steps__" not in _run(
        x_t, noise_pred, t=3, num_steps=20, context=_QuietCtx()
    )


def test_steps_emitted_when_verbose():
    x_t = torch.randn(1, 1, 2, 2)
    noise_pred = torch.randn(1, 1, 2, 2)
    res = _run(x_t, noise_pred, t=3, num_steps=20, context=_VerboseCtx())
    assert "__steps__" in res
    steps = res["__steps__"]
    names = [s.name for s in steps]
    assert names == ["schedule", "predict_x0", "x_prev"]

    schedule = steps[0]
    for key in ("t", "abar_t", "abar_prev", "sqrt_abar_t", "sqrt_one_minus_abar_t"):
        assert key in schedule.scalars
    assert schedule.scalars["t"] == 3.0

    # predict_x0 carries the pred_x0 tensor; x_prev carries x_prev + posterior coeffs.
    assert "pred_x0" in steps[1].tensors
    assert "x_prev" in steps[2].tensors
    for key in ("sqrt_abar_prev", "sqrt_one_minus_abar_prev"):
        assert key in steps[2].scalars

    # Steps must agree with the returned outputs.
    assert torch.allclose(steps[1].tensors["pred_x0"], res["pred_x0"])
    assert torch.allclose(steps[2].tensors["x_prev"], res["x_prev"])
