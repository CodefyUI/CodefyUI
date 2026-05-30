"""EduSlidingWindowNode — the convolution sliding-window prototype, step by step.

Supports textbook lesson **I1-2 (image sliding window)**: instead of a
black-box ``Conv2d``, slide a small kernel across one 2-D grayscale image and,
for every output position, expose the three things a convolution actually does:

    1. receptive_field = image[r : r+kH, c : c+kW]   (the patch under the kernel)
    2. product         = receptive_field * kernel     (elementwise)
    3. value           = product.sum()                (one feature-map cell)

The feature map is just the grid of those summed values. ``windows`` stacks
every receptive field so the lesson can scrub through the patches; ``kernel``
echoes the kernel actually used (preset or override) so it can be drawn.

This is the hand-written sibling of the production ``Conv2d`` node — same
maths (a valid cross-correlation), but with each intermediate patch / product /
sum captured in verbose mode so students see how one number on the feature map
is built from one window of the image.
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

# 3x3 kernel presets. Each is a plain Python list-of-lists so the construction
# is readable; they're turned into float tensors at execute time.
_KERNEL_PRESETS: dict[str, list[list[float]]] = {
    "identity": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    "edge": [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
    "sharpen": [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
    "blur": [[1.0 / 9.0] * 3 for _ in range(3)],
    "sobel_x": [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    # sobel_y is the transpose of sobel_x.
    "sobel_y": [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
}

# Cap on how many per-position steps we record in verbose mode. Beyond this we
# sample evenly so the Teaching Inspector stays scrubbable on large images.
_MAX_POSITION_STEPS = 9


class EduSlidingWindowNode(BaseNode):
    NODE_NAME = "Edu-SlidingWindow"
    CATEGORY = "Vision"
    DESCRIPTION = (
        "Hand-written convolution sliding window over one 2-D grayscale image. "
        "Zero-pads, then for every output position extracts the receptive "
        "field, multiplies elementwise by the kernel, and sums to one "
        "feature-map cell. Verbose mode records the patch, the product, and "
        "the summed value for each position (sampled if there are many)."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="image",
                data_type=DataType.TENSOR,
                description=(
                    "2-D grayscale image [H, W]. A 3-D [1,H,W] or [C,H,W] "
                    "tensor is auto-reduced to channel 0."
                ),
            ),
            PortDefinition(
                name="kernel",
                data_type=DataType.TENSOR,
                description=(
                    "Optional 2-D kernel [kH, kW]. If supplied it overrides "
                    "the kernel_preset param."
                ),
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="feature_map",
                data_type=DataType.TENSOR,
                description="Convolved output [Hout, Wout]; each cell is one window·kernel sum.",
            ),
            PortDefinition(
                name="kernel",
                data_type=DataType.TENSOR,
                description="The kernel actually used (preset or input), so the lesson can display it.",
            ),
            PortDefinition(
                name="windows",
                data_type=DataType.TENSOR,
                description=(
                    "Stacked receptive fields [Hout*Wout, kH, kW] in row-major "
                    "order — display-only, for inspecting every patch."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="kernel_preset",
                param_type=ParamType.SELECT,
                default="edge",
                options=["identity", "edge", "sharpen", "blur", "sobel_x", "sobel_y"],
                description=(
                    "Built-in 3x3 kernel to slide. identity copies the centre "
                    "pixel; edge is the Laplacian; blur is a 3x3 mean; sobel_x / "
                    "sobel_y are gradient filters. Ignored if a `kernel` input "
                    "is connected."
                ),
            ),
            ParamDefinition(
                name="stride",
                param_type=ParamType.INT,
                default=1,
                min_value=1,
                description="Step (in pixels) between consecutive window positions.",
            ),
            ParamDefinition(
                name="padding",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Number of zero pixels added on every side before sliding.",
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
        image = inputs.get("image")
        if image is None:
            raise ValueError("EduSlidingWindow requires an `image` input.")

        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.float32)
        image = image.float()

        # --- Reduce the image to 2-D [H, W] -------------------------------
        reduced_note = ""
        if image.ndim == 3:
            # [1,H,W] or [C,H,W] -> take channel 0.
            reduced_note = (
                f"Reduced 3-D image {tuple(image.shape)} to 2-D by taking channel 0."
            )
            image = image[0]
        if image.ndim != 2:
            raise ValueError(
                "EduSlidingWindow expects a 2-D image [H, W] (or 3-D [C,H,W] "
                f"to auto-reduce); got shape {tuple(image.shape)} after reduction."
            )

        # --- Resolve the kernel (input overrides preset) ------------------
        kernel_in = inputs.get("kernel")
        preset_name = str(params.get("kernel_preset", "edge"))
        if kernel_in is not None:
            if not isinstance(kernel_in, torch.Tensor):
                kernel_in = torch.as_tensor(kernel_in, dtype=torch.float32)
            kernel = kernel_in.float()
            kernel_source = "input"
        else:
            if preset_name not in _KERNEL_PRESETS:
                raise ValueError(
                    f"EduSlidingWindow: unknown kernel_preset {preset_name!r}. "
                    f"Choose one of {sorted(_KERNEL_PRESETS)}."
                )
            kernel = torch.tensor(_KERNEL_PRESETS[preset_name], dtype=torch.float32)
            kernel_source = f"preset:{preset_name}"

        if kernel.ndim != 2:
            raise ValueError(
                "EduSlidingWindow expects a 2-D kernel [kH, kW]; got shape "
                f"{tuple(kernel.shape)}."
            )
        kH, kW = kernel.shape
        if kH < 1 or kW < 1:
            raise ValueError(
                f"EduSlidingWindow: kernel must be non-empty; got {kH}x{kW}."
            )

        # --- Validate stride / padding ------------------------------------
        stride = int(params.get("stride", 1))
        padding = int(params.get("padding", 0))
        if stride < 1:
            raise ValueError(f"EduSlidingWindow: stride must be >= 1; got {stride}.")
        if padding < 0:
            raise ValueError(f"EduSlidingWindow: padding must be >= 0; got {padding}.")

        # --- Zero-pad the image -------------------------------------------
        # F.pad pads the last dims as (left, right, top, bottom).
        if padding > 0:
            padded = F.pad(image, (padding, padding, padding, padding), value=0.0)
        else:
            padded = image
        H_pad, W_pad = padded.shape

        # Kernel must fit inside the padded image.
        if kH > H_pad or kW > W_pad:
            raise ValueError(
                f"EduSlidingWindow: kernel {kH}x{kW} does not fit in the padded "
                f"image {H_pad}x{W_pad} (original {tuple(image.shape)}, "
                f"padding={padding}). Reduce the kernel or increase padding."
            )

        # --- Output dimensions --------------------------------------------
        Hout = (H_pad - kH) // stride + 1
        Wout = (W_pad - kW) // stride + 1
        if Hout < 1 or Wout < 1:
            raise ValueError(
                f"EduSlidingWindow: computed output dims {Hout}x{Wout} are not "
                f"both >= 1 (padded image {H_pad}x{W_pad}, kernel {kH}x{kW}, "
                f"stride={stride}). Reduce stride/kernel or add padding."
            )

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None

        if recorder is not None:
            recorder.record(
                "kernel",
                f"Kernel actually used ({kernel_source})."
                + (f" {reduced_note}" if reduced_note else ""),
                scalars={
                    "preset": preset_name,
                    "kernel_source": kernel_source,
                    "kH": float(kH),
                    "kW": float(kW),
                    "stride": float(stride),
                    "padding": float(padding),
                },
                kernel=kernel,
            )

        # Which flat output positions get a recorded step. If there are more
        # than the cap, sample evenly across the whole grid (always including
        # the first and last) and flag that in the description.
        total_positions = Hout * Wout
        if recorder is not None:
            if total_positions <= _MAX_POSITION_STEPS:
                sampled_flat = set(range(total_positions))
                sampling_note = ""
            else:
                sampled_flat = {
                    (i * (total_positions - 1)) // (_MAX_POSITION_STEPS - 1)
                    for i in range(_MAX_POSITION_STEPS)
                }
                sampling_note = (
                    f" (showing {_MAX_POSITION_STEPS} of {total_positions} "
                    "positions, sampled evenly)"
                )
        else:
            sampled_flat = set()
            sampling_note = ""

        # --- Slide the kernel ---------------------------------------------
        feature_map = torch.empty((Hout, Wout), dtype=torch.float32)
        windows = torch.empty((total_positions, kH, kW), dtype=torch.float32)

        flat = 0
        for out_r in range(Hout):
            r0 = out_r * stride
            for out_c in range(Wout):
                c0 = out_c * stride
                field = padded[r0 : r0 + kH, c0 : c0 + kW]
                product = field * kernel
                value = product.sum()
                feature_map[out_r, out_c] = value
                windows[flat] = field

                if recorder is not None and flat in sampled_flat:
                    recorder.record(
                        f"pos_r{out_r}_c{out_c}",
                        (
                            f"Window at output ({out_r}, {out_c}) = padded image "
                            f"rows {r0}:{r0 + kH}, cols {c0}:{c0 + kW}. "
                            "value = sum(receptive_field * kernel)." + sampling_note
                        ),
                        scalars={
                            "out_r": float(out_r),
                            "out_c": float(out_c),
                            "value": float(value.item()),
                        },
                        receptive_field=field,
                        product=product,
                        value=value,
                    )
                flat += 1

        if recorder is not None:
            recorder.record(
                "feature_map",
                f"Assembled feature map [{Hout} x {Wout}] from every window sum.",
                scalars={"Hout": float(Hout), "Wout": float(Wout)},
                feature_map=feature_map,
            )

        result: dict[str, Any] = {
            "feature_map": feature_map,
            "kernel": kernel,
            "windows": windows,
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
