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

import logging
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

logger = logging.getLogger(__name__)

# The defaults DiffusionTrainingLoop trains with, shared so a fresh sampler
# samples with the schedule a fresh training loop trained on (#691).
DEFAULT_NUM_TIMESTEPS = 160
DEFAULT_BETA_START = 0.0001
DEFAULT_BETA_END = 0.05

# DiffusionTrainingLoop records the schedule it trained with on the model
# under this attribute; DDPMSampler reads it back.
SCHEDULE_ATTR = "diffusion_schedule"


def record_schedule(model: Any, num_timesteps: int, schedule: str,
                    beta_start: float, beta_end: float) -> None:
    """Store the training noise schedule on ``model`` for DDPMSampler."""
    try:
        setattr(model, SCHEDULE_ATTR, {
            "num_timesteps": int(num_timesteps),
            "schedule": str(schedule),
            "beta_start": float(beta_start),
            "beta_end": float(beta_end),
        })
    except (AttributeError, TypeError):
        # A model that refuses new attributes still trains; the sampler
        # then falls back to its own params.
        pass


def _recorded_schedule(model: Any) -> dict[str, Any] | None:
    recorded = getattr(model, SCHEDULE_ATTR, None)
    if not isinstance(recorded, dict):
        return None
    try:
        return {
            "num_timesteps": int(recorded["num_timesteps"]),
            "schedule": str(recorded["schedule"]),
            "beta_start": float(recorded["beta_start"]),
            "beta_end": float(recorded["beta_end"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _schedule_differences(recorded: dict[str, Any], num_steps: int, schedule: str,
                          beta_start: float, beta_end: float) -> list[str]:
    """Describe each sampler param that differs from the recorded schedule."""
    diffs: list[str] = []
    if num_steps != recorded["num_timesteps"]:
        diffs.append(f"num_steps={num_steps} (trained with {recorded['num_timesteps']})")
    if schedule != recorded["schedule"]:
        diffs.append(f"schedule={schedule} (trained with {recorded['schedule']})")
    if recorded["schedule"] == "linear":
        if not math.isclose(beta_start, recorded["beta_start"], rel_tol=1e-6, abs_tol=1e-12):
            diffs.append(f"beta_start={beta_start} (trained with {recorded['beta_start']})")
        if not math.isclose(beta_end, recorded["beta_end"], rel_tol=1e-6, abs_tol=1e-12):
            diffs.append(f"beta_end={beta_end} (trained with {recorded['beta_end']})")
    return diffs


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
    DESCRIPTION = "Run the reverse DDPM loop: noise tensor to image"
    DETAILS = (
        "Each step calls `model(x_t, t)` to predict the noise, then applies the "
        "DDPM update; `schedule` is the original linear one or the cosine variant. "
        "The whole loop runs inside the node, which keeps the graph acyclic, and "
        "`seed` fixes the Gaussian noise added at each step. A model trained by "
        "DiffusionTrainingLoop carries the schedule it was trained with; the "
        "sampler uses that schedule and logs a warning naming any of its own "
        "schedule params that differ. The defaults match DiffusionTrainingLoop's."
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
                default=DEFAULT_NUM_TIMESTEPS,
                min_value=1,
                description=(
                    "Number of reverse-diffusion steps. Must equal the training "
                    "`num_timesteps`; a model from DiffusionTrainingLoop overrides it."
                ),
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
                default=DEFAULT_BETA_START,
                min_value=0.0,
                description="Starting variance for the linear schedule. Ignored for cosine.",
            ),
            ParamDefinition(
                name="beta_end",
                param_type=ParamType.FLOAT,
                default=DEFAULT_BETA_END,
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
        from ...core.loop_control import (
            EVENT_BATCH,
            ProgressThrottle,
            interrupted_result,
            stop_checker,
        )

        model = inputs.get("model")
        noise = inputs.get("noise")
        if model is None:
            raise ValueError("DDPMSampler requires a `model` input.")
        if noise is None:
            raise ValueError("DDPMSampler requires a `noise` input (starting x_T).")
        if not callable(model):
            raise ValueError("DDPMSampler: `model` input is not callable.")

        num_steps = int(params.get("num_steps", DEFAULT_NUM_TIMESTEPS))
        if num_steps < 1:
            raise ValueError(f"DDPMSampler: num_steps must be ≥ 1, got {num_steps}.")
        schedule = str(params.get("schedule", "linear"))
        beta_start = float(params.get("beta_start", DEFAULT_BETA_START))
        beta_end = float(params.get("beta_end", DEFAULT_BETA_END))
        seed = int(params.get("seed", 42))

        # #691: sampling with a schedule other than the one the model was
        # trained on gives washed-out samples and no error, so a schedule
        # recorded by DiffusionTrainingLoop wins over the params.
        recorded = _recorded_schedule(model)
        if recorded is not None:
            diffs = _schedule_differences(recorded, num_steps, schedule,
                                          beta_start, beta_end)
            if diffs:
                detail = (
                    "DDPMSampler: the model was trained with a different noise "
                    "schedule; sampling with the training schedule and ignoring "
                    + ", ".join(diffs) + "."
                )
                logger.warning(detail)
                if context is not None and hasattr(context, "log_warning"):
                    context.log_warning("diffusion_schedule_mismatch", detail)
            num_steps = recorded["num_timesteps"]
            schedule = recorded["schedule"]
            beta_start = recorded["beta_start"]
            beta_end = recorded["beta_end"]

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

        # Align the schedule tensors with x's (global) device so the per-step
        # scalar arithmetic doesn't mix cpu/mps. The model was already moved to
        # the device by its layer node.
        betas = betas.to(x.device)
        alphas = alphas.to(x.device)
        alpha_bars = alpha_bars.to(x.device)

        if hasattr(model, "eval"):
            model.eval()

        # #122: one U-Net forward per step and ``num_steps`` has no upper
        # bound, so this is a long loop like any other -- and until now it
        # reported nothing at all despite accepting a progress_callback.
        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)
        stopped_at_step: int | None = None

        with torch.no_grad():
            for done, t in enumerate(reversed(range(num_steps))):
                if should_stop():
                    stopped_at_step = done
                    break
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
                    # CPU generator (reproducible) → move to x's device.
                    z = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
                    sigma = torch.sqrt(beta_t)
                    x = mean + sigma * z
                else:
                    x = mean

                throttle.emit({"event": EVENT_BATCH, "batch": done + 1,
                               "total_batches": num_steps, "timestep": t})

        result: dict[str, Any] = {"image": x}
        if stopped_at_step is not None:
            # A partially denoised x is still an image, and returning it
            # beats returning nothing -- but it is not the sample that was
            # asked for, so the run is not allowed to call this a success.
            result.update(interrupted_result(batch=stopped_at_step,
                                             total_steps=num_steps))
        return result
