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


#: What ``mode="bilinear"`` means for each input rank (spatial dims = rank - 2).
_BLEND_MODE_BY_RANK = {3: "linear", 4: "bilinear", 5: "trilinear"}


class UpsampleNode(BaseNode):
    NODE_NAME = "Upsample"
    CATEGORY = "Diffusion"
    DESCRIPTION = "Scale spatial dims by a factor, no learnable weights"
    DETAILS = (
        "The resampling is `F.interpolate`: `mode` picks nearest, bilinear or "
        "area, and `scale_factor` defaults to 2.0 — a value below 1 shrinks the "
        "tensor instead. The first two axes are batch and channel: [N, C, L], "
        "[N, C, H, W] or [N, C, D, H, W], and bilinear blends along every "
        "spatial axis (linear for one, trilinear for three). A single image "
        "(C, H, W) needs a batch axis first (Unsqueeze at dim 0), or its "
        "channels are read as the batch. ConvTranspose2d is the counterpart "
        "that learns its upsampling kernel."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description=(
                    "Input tensor [N, C, L], [N, C, H, W] or [N, C, D, H, W]. A 3-D input "
                    "is [N, C, L]; add a batch axis to a single (C, H, W) image first."
                ),
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
                    "bilinear=blend neighbours (smoother; linear or trilinear "
                    "for 1 or 3 spatial dims), area=averaging (useful for "
                    "downsampling)."
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
        if x.dim() not in _BLEND_MODE_BY_RANK:
            raise ValueError(
                f"Upsample: the tensor input must be [N, C, L], [N, C, H, W] or "
                f"[N, C, D, H, W] (3 to 5 dims), got shape {tuple(x.shape)}. "
                "Add the batch and channel axes with Unsqueeze first.")

        # ``bilinear`` is the smooth option at every rank: F.interpolate names
        # it linear for one spatial axis and trilinear for three, and raises
        # NotImplementedError when handed "bilinear" on those. 4-D input
        # (the U-Net case) is unchanged.
        torch_mode = _BLEND_MODE_BY_RANK[x.dim()] if mode == "bilinear" else mode
        # F.interpolate requires align_corners=False for the blending modes;
        # nearest and area don't accept it. Branch accordingly.
        kwargs: dict[str, Any] = {"scale_factor": scale, "mode": torch_mode}
        if mode == "bilinear":
            kwargs["align_corners"] = False
        out = F.interpolate(x.float(), **kwargs)
        return {"tensor": out.to(x.dtype)}
