"""EduConv2dNode — a 2-D convolution unrolled into im2col → matmul → reshape.

Supports textbook lesson **I3-1 (CNN / 卷積)**: a convolution layer is *not*
a mysterious sliding stamp — it is a plain matrix multiply once you lay the
receptive fields out as columns. This node exposes that identity explicitly:

    1. cols       = im2col(x)              # F.unfold → [N, Cin·kH·kW, L]
    2. weight_mat = weight.reshape(...)    # [Cout, Cin·kH·kW]
    3. out_cols   = weight_mat @ cols (+b) # [N, Cout, L]
    4. y          = fold(out_cols)         # [N, Cout, Hout, Wout]

The numerical result is bit-for-bit what ``F.conv2d`` produces — the point of
the node is to let students *see* the column matrix and the matmul that a real
conv layer hides behind a single fused kernel.
"""

from __future__ import annotations

import math
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


class EduConv2dNode(BaseNode):
    NODE_NAME = "Edu-Conv2d"
    CATEGORY = "CNN"
    DESCRIPTION = (
        "Multi-channel 2-D convolution unrolled as im2col → matmul → reshape. "
        "F.unfold lays every receptive field out as a column ([N, Cin·kH·kW, L]); "
        "the weight reshapes to a [Cout, Cin·kH·kW] matrix; one matmul (+bias) "
        "gives [N, Cout, L]; folding back yields the feature map [N, Cout, Hout, "
        "Wout]. Numerically identical to F.conv2d, but every intermediate is "
        "exposed in verbose mode."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x",
                data_type=DataType.TENSOR,
                description="Input feature map [N, Cin, H, W] (or [Cin, H, W], a batch dim is added).",
            ),
            PortDefinition(
                name="weight",
                data_type=DataType.TENSOR,
                description="Conv weight [Cout, Cin, kH, kW]. Optional — if absent, random weights are drawn from the `seed` param.",
                optional=True,
            ),
            PortDefinition(
                name="bias",
                data_type=DataType.TENSOR,
                description="Optional per-output-channel bias [Cout].",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="y",
                data_type=DataType.TENSOR,
                description="Output feature map [N, Cout, Hout, Wout].",
            ),
            PortDefinition(
                name="cols",
                data_type=DataType.TENSOR,
                description="The im2col matrix [N, Cin·kH·kW, L] (display-only — the receptive fields laid out as columns).",
            ),
            PortDefinition(
                name="weight",
                data_type=DataType.TENSOR,
                description="The weight actually used [Cout, Cin, kH, kW] (the supplied one, or the freshly initialised random one).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="out_channels",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                description="Number of output channels (Cout). Only used when no `weight` input is connected.",
            ),
            ParamDefinition(
                name="kernel_size",
                param_type=ParamType.INT,
                default=3,
                min_value=1,
                description="Side length of the square kernel (kH = kW). Only used when no `weight` input is connected.",
            ),
            ParamDefinition(
                name="stride",
                param_type=ParamType.INT,
                default=1,
                min_value=1,
                description="Stride of the sliding window in both H and W.",
            ),
            ParamDefinition(
                name="padding",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Zero-padding added to all four sides of the input before unfolding.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="Seed for the random weight initialisation when no `weight` is supplied.",
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
            raise ValueError("EduConv2d requires an `x` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        # Accept [Cin, H, W] by promoting to [1, Cin, H, W]; reject anything else.
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(
                f"EduConv2d expects `x` of shape [N, Cin, H, W] (or [Cin, H, W]); got {tuple(x.shape)}."
            )
        N, Cin, H, W = x.shape

        stride = int(params.get("stride", 1))
        padding = int(params.get("padding", 0))
        if stride < 1:
            raise ValueError(f"EduConv2d: stride must be >= 1; got {stride}.")
        if padding < 0:
            raise ValueError(f"EduConv2d: padding must be >= 0; got {padding}.")

        # Weight: use the supplied one, else initialise a random [Cout, Cin, kH, kW].
        weight = inputs.get("weight")
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.as_tensor(weight, dtype=torch.float32)
            weight = weight.float()
            if weight.ndim != 4:
                raise ValueError(
                    f"EduConv2d: `weight` must be 4-D [Cout, Cin, kH, kW]; got {tuple(weight.shape)}."
                )
        else:
            out_channels = int(params.get("out_channels", 2))
            kernel_size = int(params.get("kernel_size", 3))
            if out_channels < 1:
                raise ValueError(
                    f"EduConv2d: out_channels must be >= 1; got {out_channels}."
                )
            if kernel_size < 1:
                raise ValueError(
                    f"EduConv2d: kernel_size must be >= 1; got {kernel_size}."
                )
            generator = torch.Generator().manual_seed(int(params.get("seed", 0)))
            # Kaiming-style fan_in scaling so values stay O(1) regardless of size.
            fan_in = Cin * kernel_size * kernel_size
            std = 1.0 / math.sqrt(fan_in)
            weight = torch.randn(
                out_channels, Cin, kernel_size, kernel_size, generator=generator
            ) * std

        Cout, w_cin, kH, kW = weight.shape
        if w_cin != Cin:
            raise ValueError(
                f"EduConv2d: weight Cin={w_cin} doesn't match input Cin={Cin}."
            )

        # Bias: optional [Cout].
        bias = inputs.get("bias")
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                bias = torch.as_tensor(bias, dtype=torch.float32)
            bias = bias.float()
            if bias.ndim != 1 or bias.shape[0] != Cout:
                raise ValueError(
                    f"EduConv2d: `bias` must have shape [Cout={Cout}]; got {tuple(bias.shape)}."
                )

        # Output spatial dims (standard conv formula).
        Hout = (H + 2 * padding - kH) // stride + 1
        Wout = (W + 2 * padding - kW) // stride + 1
        if Hout < 1 or Wout < 1:
            raise ValueError(
                f"EduConv2d: output dims must be >= 1, got Hout={Hout}, Wout={Wout} "
                f"(input {H}x{W}, kernel {kH}x{kW}, stride {stride}, padding {padding}). "
                "Reduce the kernel/stride or add padding."
            )
        L = Hout * Wout

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        # --- 1. im2col: lay every receptive field out as a column. ---
        # cols: [N, Cin*kH*kW, L], where L = Hout*Wout.
        cols = F.unfold(
            x, kernel_size=(kH, kW), stride=stride, padding=padding
        )
        if recorder is not None:
            recorder.record(
                "im2col",
                "cols = unfold(x): each of the L output positions becomes a "
                "column holding its Cin·kH·kW receptive-field values.",
                scalars={
                    "Cin": float(Cin), "kH": float(kH), "kW": float(kW),
                    "L": float(L), "Hout": float(Hout), "Wout": float(Wout),
                },
                cols=cols,
            )

        # --- 2. reshape the weight into a [Cout, Cin*kH*kW] matrix. ---
        weight_mat = weight.reshape(Cout, Cin * kH * kW)
        if recorder is not None:
            recorder.record(
                "weight_matrix",
                "weight_mat = weight.reshape(Cout, Cin·kH·kW): one row per output "
                "channel, matching the column layout of cols.",
                scalars={
                    "Cout": float(Cout),
                    "Cin*kH*kW": float(Cin * kH * kW),
                },
                weight_mat=weight_mat,
            )

        # --- 3. the convolution IS this matmul (+ bias). ---
        # [Cout, K] @ [N, K, L] broadcasts over N → [N, Cout, L].
        out_cols = torch.matmul(weight_mat, cols)
        if bias is not None:
            out_cols = out_cols + bias.view(1, Cout, 1)
        if recorder is not None:
            recorder.record(
                "matmul",
                "out_cols = weight_mat @ cols (+ bias): a single matrix multiply "
                "produces every output channel at every position → [N, Cout, L].",
                scalars={"has_bias": 1.0 if bias is not None else 0.0},
                out_cols=out_cols,
            )

        # --- 4. fold the columns back into a feature map. ---
        y = out_cols.reshape(N, Cout, Hout, Wout)
        if recorder is not None:
            recorder.record(
                "feature_map",
                "y = out_cols.reshape(N, Cout, Hout, Wout): the L columns fold "
                "back into the spatial grid — the finished feature map.",
                scalars={
                    "N": float(N), "Cout": float(Cout),
                    "Hout": float(Hout), "Wout": float(Wout),
                },
                y=y,
            )

        result: dict[str, Any] = {"y": y, "cols": cols, "weight": weight}
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
