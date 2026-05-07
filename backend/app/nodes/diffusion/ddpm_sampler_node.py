"""DDPMSamplerNode — reverse the diffusion process to denoise an image.

Given a U-Net that predicts noise and a starting noise tensor, iterate
the reverse-DDPM update for ``num_steps`` steps:

    Pre-compute schedule
        β_t          = β_start .. β_end (linear) or cosine schedule
        α_t          = 1 - β_t
        α̅_t          = ∏ α_s for s ≤ t

    Each step (large t → 0):
        ε̂           = model(x_t, t)
        x_{t-1}      = (1/√α_t) (x_t - ((1-α_t)/√(1-α̅_t)) ε̂) + σ_t z
                       where z ~ 𝒩(0, I) for t > 0, else 0

CodefyUI's DAG is forward-only, so the loop must live *inside a node*
(per the design discussion: "DDPMSampler 內部跑迴圈"). The node accepts
the U-Net via the `model` input, the start noise via `noise`, and runs
the schedule + iteration entirely in execute(). Output is the final
denoised image.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


def _linear_betas(num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def _cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021) — slower noising near the data manifold."""
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, dtype=torch.float32) / num_steps
    alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 0.0001, 0.999)


class DDPMSamplerNode(BaseNode):
    NODE_NAME = "DDPMSampler"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Run reverse DDPM denoising. Iterates a schedule of timesteps, "
        "calling `model(x_t, t)` to predict noise, then applying the DDPM "
        "update rule. Encapsulates the entire reverse loop so the graph "
        "stays acyclic — see the verbose step trace for trajectory snapshots."
    )

    cacheable = False  # Has internal randomness; conservative to skip cache.

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="A noise-predicting U-Net (e.g. from `DiffusionUNet`).",
            ),
            PortDefinition(
                name="noise",
                data_type=DataType.TENSOR,
                description="Starting noise $x_T$, shape [N, C, H, W].",
            ),
            PortDefinition(
                name="condition",
                data_type=DataType.TENSOR,
                description="Optional conditioning tensor for cross-attention. Currently unused — reserved for the next PR's text-conditioning support.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="image",
                data_type=DataType.TENSOR,
                description="Denoised image $x_0$, same shape as the input noise.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="num_steps",
                param_type=ParamType.INT,
                default=20,
                min_value=1,
                description="Number of reverse-diffusion steps. More steps = smoother trajectory but slower.",
            ),
            ParamDefinition(
                name="schedule",
                param_type=ParamType.SELECT,
                default="linear",
                options=["linear", "cosine"],
                description="Noise schedule. `linear` is the original DDPM; `cosine` (Nichol & Dhariwal 2021) noises more slowly near the data.",
            ),
            ParamDefinition(
                name="beta_start",
                param_type=ParamType.FLOAT,
                default=0.0001,
                min_value=0.0,
                description="Starting variance for the linear schedule. Ignored for cosine.",
            ),
            ParamDefinition(
                name="beta_end",
                param_type=ParamType.FLOAT,
                default=0.02,
                min_value=0.0,
                description="Ending variance for the linear schedule. Ignored for cosine.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for the per-step Gaussian noise z added during sampling.",
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
        model = inputs.get("model")
        noise = inputs.get("noise")
        if model is None:
            raise ValueError("DDPMSampler requires a `model` input.")
        if noise is None:
            raise ValueError("DDPMSampler requires a `noise` input (starting x_T).")
        if not callable(model):
            raise ValueError("DDPMSampler: `model` input is not callable.")

        num_steps = int(params.get("num_steps", 20))
        if num_steps < 1:
            raise ValueError(f"DDPMSampler: num_steps must be ≥ 1, got {num_steps}.")
        schedule = str(params.get("schedule", "linear"))
        beta_start = float(params.get("beta_start", 0.0001))
        beta_end = float(params.get("beta_end", 0.02))
        seed = int(params.get("seed", 42))

        # Build the schedule.
        if schedule == "linear":
            betas = _linear_betas(num_steps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = _cosine_betas(num_steps)
        else:
            raise ValueError(f"DDPMSampler: unknown schedule {schedule!r}.")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Run the reverse loop. We use a local generator for the per-step
        # Gaussian noise so the loop is fully reproducible given the seed.
        gen = torch.Generator()
        gen.manual_seed(seed)
        x = noise.clone().float()
        if hasattr(model, "eval"):
            model.eval()

        with torch.no_grad():
            for t in reversed(range(num_steps)):
                t_tensor = torch.full(
                    (x.shape[0],), t, dtype=torch.long, device=x.device
                )
                eps_hat = model(x, t_tensor)

                alpha_t = alphas[t]
                alpha_bar_t = alpha_bars[t]
                beta_t = betas[t]
                inv_sqrt_alpha = 1.0 / torch.sqrt(alpha_t)
                eps_coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)

                mean = inv_sqrt_alpha * (x - eps_coef * eps_hat)

                if t > 0:
                    z = torch.randn(x.shape, generator=gen, dtype=x.dtype)
                    sigma = torch.sqrt(beta_t)
                    x = mean + sigma * z
                else:
                    x = mean

        return {"image": x}
