"""EduMaxPool2dNode — max pooling / spatial downsampling, exposed window by window.

Supports textbook lesson **I3-1 (池化 / 下採樣)**: instead of a black-box
``nn.MaxPool2d``, slide a ``kernel_size × kernel_size`` window across each
channel and keep only the maximum inside every window. The feature map shrinks
from ``[N, C, H, W]`` to ``[N, C, Hout, Wout]``:

    Hout = floor((H + 2·padding − kernel_size) / stride) + 1
    Wout = floor((W + 2·padding − kernel_size) / stride) + 1

The two ideas a student should take away are visible in verbose mode:

    1. *Local maximum* — each output value is just the largest activation in
       its window, so pooling keeps the strongest response and discards the
       rest. We record a handful of sampled windows together with the value we
       kept and the flat ``argmax`` (which position inside the window won).
    2. *Downsampling* — with ``stride = kernel_size`` (the default) the windows
       tile without overlap, so the spatial size is divided by the kernel. The
       final ``downsample`` step shows the ``[H, W] → [Hout, Wout]`` shrink.

``argmax`` is computed by ``F.max_pool2d(..., return_indices=True)`` and is
display-only (the indices that ``nn.MaxUnpool2d`` would consume); the forward
result ``y`` matches ``F.max_pool2d`` exactly.
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

# How many output windows to sample as individual verbose steps. Pooling a
# large map produces thousands of windows; surfacing every one would bury the
# Teaching Inspector, so we cap the per-window steps and add one summary step.
_MAX_WINDOW_STEPS = 6


class EduMaxPool2dNode(BaseNode):
    NODE_NAME = "Edu-MaxPool2d"
    CATEGORY = "CNN"
    DESCRIPTION = (
        "Max pooling: slide a kernel_size×kernel_size window across each channel "
        "and keep only the maximum activation inside it, shrinking [N, C, H, W] → "
        "[N, C, Hout, Wout]. Verbose mode samples a few windows (their values, the "
        "max kept, and the winning argmax) and shows the overall downsample."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x",
                data_type=DataType.TENSOR,
                description="Feature map of shape [N, C, H, W] (a bare [C, H, W] gains a batch dim).",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="y",
                data_type=DataType.TENSOR,
                description="Pooled feature map, shape [N, C, Hout, Wout].",
            ),
            PortDefinition(
                name="argmax",
                data_type=DataType.TENSOR,
                description=(
                    "Integer indices (into the flattened H×W plane) of the max in "
                    "each window, shape [N, C, Hout, Wout]. Display-only — the input "
                    "MaxUnpool2d would consume to invert the pooling."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="kernel_size",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                description="Side length of the square pooling window, in pixels.",
            ),
            ParamDefinition(
                name="stride",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Step between consecutive windows. 0 (the default) means "
                    "'= kernel_size', i.e. non-overlapping windows that tile the map."
                ),
            ),
            ParamDefinition(
                name="padding",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Implicit −inf padding added to each spatial side before pooling. "
                    "Must not exceed kernel_size / 2."
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
            raise ValueError("EduMaxPool2d requires an `x` input.")
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()

        # Accept [C, H, W] by promoting to [1, C, H, W]; reject anything else.
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(
                f"EduMaxPool2d expects [N, C, H, W] (or [C, H, W]); got shape {tuple(x.shape)}."
            )

        N, C, H, W = x.shape

        kernel_size = int(params.get("kernel_size", 2))
        stride_param = int(params.get("stride", 0))
        padding = int(params.get("padding", 0))
        if kernel_size < 1:
            raise ValueError(f"EduMaxPool2d: kernel_size must be >= 1; got {kernel_size}.")
        # stride defaults to kernel_size when 0.
        stride = kernel_size if stride_param == 0 else stride_param
        if stride < 1:
            raise ValueError(f"EduMaxPool2d: stride must be >= 1; got {stride}.")
        if padding < 0:
            raise ValueError(f"EduMaxPool2d: padding must be >= 0; got {padding}.")
        if padding > kernel_size // 2:
            raise ValueError(
                f"EduMaxPool2d: padding={padding} must not exceed kernel_size//2="
                f"{kernel_size // 2}."
            )

        h_out = math.floor((H + 2 * padding - kernel_size) / stride) + 1
        w_out = math.floor((W + 2 * padding - kernel_size) / stride) + 1
        if h_out < 1 or w_out < 1:
            raise ValueError(
                f"EduMaxPool2d: window too large for input — output dims must be >= 1 but "
                f"got Hout={h_out}, Wout={w_out} for input [H={H}, W={W}], "
                f"kernel_size={kernel_size}, stride={stride}, padding={padding}."
            )

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        y, argmax = F.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True,
        )

        if recorder is not None:
            self._record_windows(
                recorder,
                x=x,
                y=y,
                argmax=argmax,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                h_out=h_out,
                w_out=w_out,
            )
            recorder.record(
                "downsample",
                "Keeping one max per window shrinks the spatial size: "
                "[H, W] → [Hout, Wout].",
                scalars={
                    "kernel_size": float(kernel_size),
                    "stride": float(stride),
                    "padding": float(padding),
                    "H": float(H),
                    "W": float(W),
                    "Hout": float(h_out),
                    "Wout": float(w_out),
                },
                y=y,
            )

        result: dict[str, Any] = {"y": y, "argmax": argmax}
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result

    @staticmethod
    def _record_windows(
        recorder: StepRecorder,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        argmax: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
        h_out: int,
        w_out: int,
    ) -> None:
        """Record up to ``_MAX_WINDOW_STEPS`` sampled output windows.

        For each sampled output cell ``(oh, ow)`` of batch/channel ``(0, 0)`` we
        slice the corresponding ``kernel_size × kernel_size`` patch out of the
        (padded) input, then surface that patch, the max we kept, and the flat
        ``argmax`` index (position within the window). Windows are sampled
        evenly across the output grid so the steps span the whole map rather
        than clustering in the top-left corner.
        """
        N, C, H, W = x.shape
        # Pad with −inf so out-of-bounds positions never win the max — this
        # mirrors what F.max_pool2d does internally for padding > 0.
        if padding > 0:
            xp = F.pad(x, (padding, padding, padding, padding), value=float("-inf"))
        else:
            xp = x

        total = h_out * w_out
        n_steps = min(_MAX_WINDOW_STEPS, total)
        # Evenly spaced flat output positions across [0, total).
        if n_steps <= 1:
            flat_positions = [0]
        else:
            flat_positions = [
                round(i * (total - 1) / (n_steps - 1)) for i in range(n_steps)
            ]

        for step_idx, flat in enumerate(flat_positions):
            oh = flat // w_out
            ow = flat % w_out
            top = oh * stride
            left = ow * stride
            window = xp[0, 0, top : top + kernel_size, left : left + kernel_size]
            max_val = float(y[0, 0, oh, ow].item())
            # argmax holds the index into the flattened *input* H×W plane; the
            # position *inside the window* is more instructive, so derive it.
            flat_in_input = int(argmax[0, 0, oh, ow].item())
            src_row = flat_in_input // W
            src_col = flat_in_input % W
            win_row = src_row - top + padding
            win_col = src_col - left + padding
            flat_in_window = win_row * kernel_size + win_col
            recorder.record(
                f"window_{step_idx}",
                f"Output cell (row={oh}, col={ow}): take the max of its "
                f"{kernel_size}×{kernel_size} window.",
                scalars={
                    "out_row": float(oh),
                    "out_col": float(ow),
                    "max": max_val,
                    "argmax_in_window": float(flat_in_window),
                },
                window=window,
            )
