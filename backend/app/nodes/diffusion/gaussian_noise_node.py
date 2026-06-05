"""GaussianNoiseNode — seeded Gaussian noise tensor.

The diffusion forward process needs i.i.d. Gaussian noise of the same
shape as the data. ``TensorCreate`` can do this with ``fill=randn`` but
isn't quite as clear pedagogically — students reading a graph see the
generic primitive. This node says exactly what it is, and accepts an
optional ``shape_ref`` input so the noise tensor automatically tracks
upstream shape (the common case in a Lerp(x_0, ε, alpha) chain).
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


def _parse_shape(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("GaussianNoise: shape param is empty.")
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"GaussianNoise: invalid shape {s!r} — must be comma-separated ints.") from exc
    if any(d <= 0 for d in dims):
        raise ValueError(f"GaussianNoise: shape dims must be positive; got {dims}.")
    return dims


class GaussianNoiseNode(BaseNode):
    NODE_NAME = "GaussianNoise"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Generate i.i.d. Gaussian noise $\\epsilon \\sim \\mathcal{N}(\\mu, \\sigma^2)$. "
        "Connect a tensor to `shape_ref` to mirror upstream shape (typical "
        "for diffusion x_0 → noise pairing); otherwise parses `shape` from "
        "the param. Seeded for reproducibility."
    )

    cacheable = False  # output depends on global state being unchanged but seeded gen makes it deterministic; keep True? Set False to be safe with chain reuse.

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="shape_ref",
                data_type=DataType.TENSOR,
                description="Optional reference tensor — noise will match its shape.",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="noise",
                data_type=DataType.TENSOR,
                description="Float32 tensor sampled from N(mean, std²).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="shape",
                param_type=ParamType.STRING,
                default="1,3,8,8",
                description=(
                    "Comma-separated dimensions, e.g. '1,3,32,32'. Used "
                    "only when `shape_ref` is not connected."
                ),
            ),
            ParamDefinition(
                name="mean",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="Mean of the Gaussian. Defaults to 0 (standard normal).",
            ),
            ParamDefinition(
                name="std",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.0,
                description="Standard deviation. Defaults to 1 (standard normal).",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for reproducibility. Same seed → same noise.",
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
        ref = inputs.get("shape_ref")
        if ref is not None and isinstance(ref, torch.Tensor):
            shape = tuple(ref.shape)
        else:
            shape = _parse_shape(str(params.get("shape", "")))

        mean = float(params.get("mean", 0.0))
        std = max(0.0, float(params.get("std", 1.0)))
        seed = int(params.get("seed", 42))

        # Use a local generator so we don't touch the global RNG state — the
        # graph engine relies on global RNG being stable for cacheable nodes.
        gen = torch.Generator()
        gen.manual_seed(seed)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32)
        if std != 1.0:
            noise = noise * std
        if mean != 0.0:
            noise = noise + mean

        # CPU generator above keeps the seed reproducible; move the result to
        # the reference tensor's device (if any) or the global run device so it
        # composes with on-device tensors downstream (e.g. Lerp, DDPMSampler).
        from ...core.device_utils import context_device, to_device
        dev = ref.device if (ref is not None and isinstance(ref, torch.Tensor)) else context_device(context)
        return {"noise": to_device(noise, dev)}
