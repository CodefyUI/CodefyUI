"""UpsampleNode — pure (non-learned) spatial upsampling.

Counterpart to :class:`ConvTranspose2dNode`: ConvTranspose has *learnable*
weights and is what diffusion U-Nets typically use to upsample, but for
teaching it's clarifying to separate the two operations:

* ``Upsample`` — fixed geometric resampling (nearest / bilinear). No
  parameters, no gradients to learn. Doubles the spatial dimensions by
  default so an encoder-decoder U-Net mirrors its downsampling stages.
* ``ConvTranspose2d`` — strided transposed convolution that *learns* the
  upsampling kernel.

Showing both lets students see "upsampling can be a fixed op" before
introducing the learnable variant.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class UpsampleNode(BaseNode):
    NODE_NAME = "Upsample"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Pure spatial upsampling via F.interpolate — no learnable weights. "
        "Doubles spatial dims by default. Use this for U-Net decoder paths "
        "when you don't want the upsampling step to learn (compare with "
        "`ConvTranspose2d`, which does)."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input tensor [N, C, ...] — 1D, 2D, or 3D spatial dims.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Upsampled tensor with spatial dims scaled by `scale_factor`.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="nearest",
                options=["nearest", "bilinear", "area"],
                description=(
                    "Interpolation mode. nearest=replicate pixels, "
                    "bilinear=blend neighbours (smoother), area=averaging "
                    "(useful for downsampling)."
                ),
            ),
            ParamDefinition(
                name="scale_factor",
                param_type=ParamType.FLOAT,
                default=2.0,
                min_value=0.1,
                description="Multiply spatial dims by this. 2.0 doubles, 0.5 halves.",
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
        x = inputs.get("tensor")
        if x is None:
            raise ValueError("Upsample requires a `tensor` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        mode = str(params.get("mode", "nearest"))
        if mode not in ("nearest", "bilinear", "area"):
            raise ValueError(f"Unknown Upsample mode: {mode!r}")
        scale = float(params.get("scale_factor", 2.0))

        # F.interpolate requires align_corners=False for bilinear; nearest
        # and area don't accept it. Branch accordingly.
        kwargs: dict[str, Any] = {"scale_factor": scale, "mode": mode}
        if mode == "bilinear":
            kwargs["align_corners"] = False
        out = F.interpolate(x.float(), **kwargs)
        return {"tensor": out.to(x.dtype)}
