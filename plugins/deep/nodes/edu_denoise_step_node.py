"""EduDenoiseStepNode — one deterministic DDIM reverse diffusion step.

Supports textbook lesson **I3-3 (反向擴散 / sampling)**: instead of a
black-box sampler loop, expose a SINGLE reverse step and every coefficient
that goes into it.

Given a noisy latent ``x_t`` and the network's predicted noise ``eps`` at
timestep ``t``, a DDIM (eta = 0, deterministic) step is:

    1. Build a linear beta schedule  betas = linspace(beta_start, beta_end, num_steps)
       alphas          = 1 − betas
       alphas_cumprod  = cumprod(alphas)
       abar_t          = alphas_cumprod[t]
       abar_prev       = alphas_cumprod[t-1]

    2. Recover the predicted clean image (the "x0 prediction"):
           pred_x0 = (x_t − sqrt(1 − abar_t) · eps) / sqrt(abar_t)

    3. Re-noise toward the previous (less noisy) step:
           x_prev  = sqrt(abar_prev) · pred_x0 + sqrt(1 − abar_prev) · eps

There is no randomness — with eta = 0 the step is fully deterministic, so the
same inputs always give the same ``x_prev``. ``pred_x0`` is the model's
current best guess of the final image and is mainly there to *watch* the
denoising sharpen over the course of a sampling loop; it is display-only and
not fed forward.

The node operates elementwise and keeps the input shape, so it works on any
tensor (a ``[N, C, H, W]`` latent, a flat vector, a scalar — whatever the
upstream produces) as long as ``noise_pred`` matches ``x_t`` exactly.
"""

from __future__ import annotations

from typing import Any

import torch

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder


class EduDenoiseStepNode(BaseNode):
    NODE_NAME = "Edu-DenoiseStep"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "One deterministic DDIM (eta=0) reverse diffusion step. From a noisy "
        "latent x_t and predicted noise eps it recovers pred_x0 = (x_t − "
        "sqrt(1−ᾱ_t)·eps) / sqrt(ᾱ_t), then re-noises to x_{t-1} = "
        "sqrt(ᾱ_{t-1})·pred_x0 + sqrt(1−ᾱ_{t-1})·eps. Verbose mode exposes the "
        "schedule coefficients (ᾱ_t, ᾱ_{t-1}) and the two posterior weights so "
        "students see exactly how a sampler walks back down the noise schedule."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x_t",
                data_type=DataType.TENSOR,
                description="Noisy latent at timestep t, shape [N, C, H, W] (or any shape).",
            ),
            PortDefinition(
                name="noise_pred",
                data_type=DataType.TENSOR,
                description="Predicted noise epsilon, the SAME shape as x_t.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x_prev",
                data_type=DataType.TENSOR,
                description="Estimate of x_{t-1} (one step less noisy), same shape as x_t.",
            ),
            PortDefinition(
                name="pred_x0",
                data_type=DataType.TENSOR,
                description="Predicted clean image x0 from this step. Display-only — not fed forward.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="t",
                param_type=ParamType.INT,
                default=50,
                min_value=1,
                description="Current timestep index. Must satisfy 1 ≤ t < num_steps.",
            ),
            ParamDefinition(
                name="num_steps",
                param_type=ParamType.INT,
                default=100,
                min_value=2,
                description="Length of the (linear) beta schedule, i.e. total diffusion steps.",
            ),
            ParamDefinition(
                name="beta_start",
                param_type=ParamType.FLOAT,
                default=0.0001,
                description="First beta in the linear schedule.",
            ),
            ParamDefinition(
                name="beta_end",
                param_type=ParamType.FLOAT,
                default=0.02,
                description="Last beta in the linear schedule.",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        x_t = inputs.get("x_t")
        noise_pred = inputs.get("noise_pred")
        if x_t is None or noise_pred is None:
            raise ValueError(
                "EduDenoiseStep requires both `x_t` and `noise_pred` inputs."
            )

        if not isinstance(x_t, torch.Tensor):
            x_t = torch.as_tensor(x_t, dtype=torch.float32)
        if not isinstance(noise_pred, torch.Tensor):
            noise_pred = torch.as_tensor(noise_pred, dtype=torch.float32)
        x_t = x_t.float()
        noise_pred = noise_pred.float()

        if noise_pred.shape != x_t.shape:
            raise ValueError(
                "EduDenoiseStep: noise_pred shape "
                f"{tuple(noise_pred.shape)} must equal x_t shape "
                f"{tuple(x_t.shape)}."
            )

        t = int(params.get("t", 50))
        num_steps = int(params.get("num_steps", 100))
        beta_start = float(params.get("beta_start", 0.0001))
        beta_end = float(params.get("beta_end", 0.02))

        if num_steps < 2:
            raise ValueError(
                f"EduDenoiseStep: num_steps must be ≥ 2; got {num_steps}."
            )
        if not (1 <= t < num_steps):
            raise ValueError(
                f"EduDenoiseStep: t must satisfy 1 ≤ t < num_steps ({num_steps}); "
                f"got t={t}."
            )

        # Linear beta schedule → alphas → cumulative product ᾱ.
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        abar_t = alphas_cumprod[t]
        abar_prev = alphas_cumprod[t - 1]

        sqrt_abar_t = torch.sqrt(abar_t)
        sqrt_one_minus_abar_t = torch.sqrt(1.0 - abar_t)
        sqrt_abar_prev = torch.sqrt(abar_prev)
        sqrt_one_minus_abar_prev = torch.sqrt(1.0 - abar_prev)

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        if recorder is not None:
            recorder.record(
                "schedule",
                "Linear beta schedule → ᾱ. Read off ᾱ_t and ᾱ_{t-1} for this step.",
                scalars={
                    "t": float(t),
                    "abar_t": float(abar_t.item()),
                    "abar_prev": float(abar_prev.item()),
                    "sqrt_abar_t": float(sqrt_abar_t.item()),
                    "sqrt_one_minus_abar_t": float(sqrt_one_minus_abar_t.item()),
                },
            )

        # Step 2: recover the predicted clean image x0.
        pred_x0 = (x_t - sqrt_one_minus_abar_t * noise_pred) / sqrt_abar_t
        if recorder is not None:
            recorder.record(
                "predict_x0",
                "pred_x0 = (x_t − sqrt(1−ᾱ_t)·eps) / sqrt(ᾱ_t): invert the forward "
                "noising to guess the clean image.",
                pred_x0=pred_x0,
            )

        # Step 3: deterministic DDIM re-noising toward x_{t-1}.
        x_prev = sqrt_abar_prev * pred_x0 + sqrt_one_minus_abar_prev * noise_pred
        if recorder is not None:
            recorder.record(
                "x_prev",
                "x_{t-1} = sqrt(ᾱ_{t-1})·pred_x0 + sqrt(1−ᾱ_{t-1})·eps: re-noise to "
                "one step less noisy (DDIM, eta=0, deterministic).",
                scalars={
                    "sqrt_abar_prev": float(sqrt_abar_prev.item()),
                    "sqrt_one_minus_abar_prev": float(sqrt_one_minus_abar_prev.item()),
                },
                x_prev=x_prev,
            )

        result: dict[str, Any] = {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
