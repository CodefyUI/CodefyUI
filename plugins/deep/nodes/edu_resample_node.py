"""EduResampleNode — the spatial-shape change at the heart of a U-Net.

Supports textbook lesson **I3-2 (U-Net down/up sampling)**: a U-Net is, at its
core, a sequence of *resolution changes*. The encoder repeatedly halves the
spatial size (down) while the decoder doubles it back (up), and skip
connections splice the matching-resolution encoder features back into the
decoder. This node makes that single move explicit:

    down: [N, C, H, W] → [N, C, H/factor, W/factor]
    up:   [N, C, H, W] → [N, C, H*factor, W*factor]
          (+ optional channel-concat of a skip tensor of the same resolution)

Down-sampling uses ``F.avg_pool2d`` (a learnable-free pooling step) or a
strided ``F.interpolate`` depending on ``mode``; up-sampling always uses
``F.interpolate``. Nothing here is learned — the lesson is purely about how
the spatial shape (and, with a skip, the channel count) changes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder


class EduResampleNode(BaseNode):
    NODE_NAME = "Edu-Resample"
    CATEGORY = "Diffusion"
    DESCRIPTION = (
        "Down- or up-sample a feature map [N, C, H, W] by an integer factor — "
        "the core move of a U-Net. Down halves the resolution (avg-pool or "
        "strided interpolate); up doubles it (interpolate). Connect a same-"
        "resolution `skip` while direction=up to channel-concatenate it and "
        "demonstrate the skip connection. The spatial-shape change is exposed "
        "step by step in verbose mode."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x",
                data_type=DataType.TENSOR,
                description="Feature map [N, C, H, W] (or [C, H, W], promoted to batch 1).",
            ),
            PortDefinition(
                name="skip",
                data_type=DataType.TENSOR,
                description=(
                    "Optional encoder feature map [N, C, H, W]. When provided and "
                    "direction=up, it is channel-concatenated onto the upsampled "
                    "output (the U-Net skip connection). Spatial dims must match "
                    "the upsampled result."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="y",
                data_type=DataType.TENSOR,
                description=(
                    "Resampled feature map. Shape is [N, C, H/factor, W/factor] for "
                    "down and [N, C, H*factor, W*factor] for up; channels double when "
                    "a skip is concatenated."
                ),
            ),
            PortDefinition(
                name="shape",
                data_type=DataType.TENSOR,
                description="Output dims as a 1-D long tensor [N, C, Hout, Wout] (display-only).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="direction",
                param_type=ParamType.SELECT,
                default="down",
                options=["down", "up"],
                description=(
                    "'down' shrinks the resolution by `factor` (encoder side); "
                    "'up' grows it by `factor` (decoder side)."
                ),
            ),
            ParamDefinition(
                name="factor",
                param_type=ParamType.INT,
                default=2,
                min_value=2,
                description="Integer resampling factor. 2 = halve (down) / double (up) each spatial dim.",
            ),
            ParamDefinition(
                name="mode",
                param_type=ParamType.SELECT,
                default="nearest",
                options=["nearest", "bilinear", "avgpool"],
                description=(
                    "Resampling kernel. For up: 'nearest' (default) or 'bilinear' "
                    "interpolation. For down: 'avgpool' uses F.avg_pool2d(kernel=factor); "
                    "'nearest'/'bilinear' use a strided F.interpolate(scale_factor=1/factor)."
                ),
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
        x = inputs.get("x")
        if x is None:
            raise ValueError("EduResample requires an `x` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        # Accept [C, H, W] by promoting to [1, C, H, W]; reject anything else.
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(
                f"EduResample expects x of shape [N, C, H, W] (or [C, H, W]); "
                f"got {tuple(x.shape)}."
            )

        direction = str(params.get("direction", "down"))
        if direction not in ("down", "up"):
            raise ValueError(
                f"EduResample: direction must be 'down' or 'up'; got {direction!r}."
            )
        factor = int(params.get("factor", 2))
        if factor < 2:
            raise ValueError(
                f"EduResample: factor must be >= 2; got {factor}."
            )
        mode = str(params.get("mode", "nearest"))
        if mode not in ("nearest", "bilinear", "avgpool"):
            raise ValueError(
                f"EduResample: mode must be one of 'nearest', 'bilinear', 'avgpool'; "
                f"got {mode!r}."
            )

        N, C, H, W = (int(d) for d in x.shape)

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        if direction == "down":
            if mode == "avgpool":
                y = F.avg_pool2d(x, kernel_size=factor)
            else:
                # nearest/bilinear strided downsample via fractional scale.
                interp_kwargs: dict[str, Any] = {"scale_factor": 1.0 / factor, "mode": mode}
                if mode == "bilinear":
                    interp_kwargs["align_corners"] = False
                y = F.interpolate(x, **interp_kwargs)
        else:  # up
            interp_kwargs = {"scale_factor": float(factor), "mode": mode}
            if mode == "bilinear":
                interp_kwargs["align_corners"] = False
            y = F.interpolate(x, **interp_kwargs)

        Hout, Wout = int(y.shape[2]), int(y.shape[3])
        if recorder is not None:
            recorder.record(
                "resample",
                f"{direction}-sample by factor {factor} using mode '{mode}': "
                f"[H={H}, W={W}] → [H={Hout}, W={Wout}].",
                scalars={
                    "factor": float(factor),
                    "H_before": float(H),
                    "W_before": float(W),
                    "H_after": float(Hout),
                    "W_after": float(Wout),
                },
                x=x,
                y=y,
            )

        # Optional skip connection — only meaningful on the decoder (up) side.
        skip = inputs.get("skip")
        if skip is not None and direction == "up":
            if not isinstance(skip, torch.Tensor):
                skip = torch.as_tensor(skip, dtype=torch.float32)
            skip = skip.float()
            if skip.ndim == 3:
                skip = skip.unsqueeze(0)
            if skip.ndim != 4:
                raise ValueError(
                    f"EduResample: skip must be [N, C, H, W] (or [C, H, W]); "
                    f"got {tuple(skip.shape)}."
                )
            if skip.shape[2] != Hout or skip.shape[3] != Wout:
                raise ValueError(
                    f"EduResample: skip spatial dims {tuple(skip.shape[2:])} must match "
                    f"the upsampled output {(Hout, Wout)} to concatenate. Check that the "
                    f"skip was taken at the matching encoder resolution."
                )
            if skip.shape[0] != y.shape[0]:
                raise ValueError(
                    f"EduResample: skip batch {skip.shape[0]} must match output batch "
                    f"{y.shape[0]} to concatenate."
                )
            c_before = int(y.shape[1])
            c_skip = int(skip.shape[1])
            y = torch.cat([y, skip], dim=1)
            c_after = int(y.shape[1])
            if recorder is not None:
                recorder.record(
                    "skip_concat",
                    "U-Net skip connection: concatenate the encoder feature map onto "
                    f"the upsampled output along the channel axis: {c_before} + {c_skip} "
                    f"= {c_after} channels.",
                    scalars={
                        "channels_before": float(c_before),
                        "channels_skip": float(c_skip),
                        "channels_after": float(c_after),
                    },
                    y=y,
                )

        shape = torch.tensor(list(y.shape), dtype=torch.long)
        result: dict[str, Any] = {"y": y, "shape": shape}
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
