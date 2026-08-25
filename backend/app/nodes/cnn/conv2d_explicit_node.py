from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition

# Built-in 3x3 kernels. Names mirror what students see in image-processing
# textbooks; the suffix "3x3" makes the shape explicit at the dropdown level.
PRESETS_3X3: dict[str, list[list[float]]] = {
    # Laplacian — center +8, neighbours -1. Highlights pixels whose value
    # differs strongly from the surrounding 8.
    "EdgeDetection3x3": [
        [-1.0, -1.0, -1.0],
        [-1.0,  8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ],
    # Classic sharpening — center +5, axis-aligned neighbours -1, corners 0.
    "Sharpen3x3": [
        [ 0.0, -1.0,  0.0],
        [-1.0,  5.0, -1.0],
        [ 0.0, -1.0,  0.0],
    ],
    # Prewitt-X — responds to horizontal intensity changes (vertical edges).
    "VerticalEdge3x3": [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ],
}

CUSTOM_OPTION = "Custom"
PRESET_OPTIONS: list[str] = [*PRESETS_3X3.keys(), CUSTOM_OPTION]
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 15


def _flatten(x: Any) -> list[float]:
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


class Conv2dExplicitNode(BaseNode):
    """Convolution with a kernel you choose, not one the network learns.

    The kernel used to live in a separate ``Conv2dKernel`` node wired into a
    second input port. Two nodes and an edge to apply one textbook filter is
    a lot of canvas for a lesson whose point is "look what this 3x3 matrix
    does to an image", and the split had no payoff: nothing else consumed a
    kernel, so the producer existed solely to feed this consumer. Picking the
    kernel here makes the node self-contained and the node card readable --
    one input, one output, and the filter named on its face.
    """

    NODE_NAME = "Conv2dExplicit"
    CATEGORY = "CNN"
    DESCRIPTION = (
        "2D convolution with a kernel you pick, not one the network learns — "
        "no learnable weights, no random init. Choose a built-in 3x3 filter "
        "(EdgeDetection / Sharpen / VerticalEdge) or set preset to 'Custom' "
        "and author an NxN matrix by hand. The same kernel is applied to "
        "every input channel via grouped (depthwise) convolution, so a "
        "(N, C, H, W) input produces a (N, C, H, W) output with channels "
        "untouched."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Input tensor (N, C, H, W)",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Convolved output (N, C, H_out, W_out)",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        # Kernel choice first: it is the reason someone reaches for this node,
        # and stride/padding are the trim around it.
        return [
            ParamDefinition(
                name="preset",
                param_type=ParamType.SELECT,
                default="EdgeDetection3x3",
                description="Built-in 3x3 kernel, or 'Custom' to author your own matrix",
                options=PRESET_OPTIONS,
            ),
            ParamDefinition(
                name="kernel_size",
                param_type=ParamType.INT,
                default=3,
                description="Kernel side length N for an NxN kernel (Custom preset only)",
                min_value=MIN_KERNEL_SIZE,
                max_value=MAX_KERNEL_SIZE,
                visible_when={"preset": CUSTOM_OPTION},
            ),
            ParamDefinition(
                name="weights",
                param_type=ParamType.TENSOR_GRID,
                default=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                description="Custom kernel matrix (NxN); grid size follows kernel_size",
                visible_when={"preset": CUSTOM_OPTION},
            ),
            ParamDefinition(
                name="stride",
                param_type=ParamType.INT,
                default=1,
                description="Convolution stride",
                min_value=1,
            ),
            ParamDefinition(
                name="padding",
                param_type=ParamType.INT,
                default=1,
                description="Zero-padding on each side of the spatial dims",
                min_value=0,
            ),
        ]

    def _resolve_kernel(self, params: dict[str, Any]) -> tuple[list[float], int]:
        """The chosen kernel as (flat values, side length).

        A preset is a fixed 3x3; ``Custom`` reads the hand-authored grid and
        insists it matches ``kernel_size``. The mismatch is worth an error
        rather than a reshape: a grid with the wrong element count means the
        user changed the size and did not re-fill it, and silently padding or
        truncating would convolve with a filter they never wrote.
        """
        preset = str(params.get("preset", "EdgeDetection3x3"))
        if preset == CUSTOM_OPTION:
            raw_k = params.get("kernel_size", 3)
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                k = 3
            k = max(MIN_KERNEL_SIZE, min(MAX_KERNEL_SIZE, k))
            raw_weights = params.get("weights")
            if raw_weights is None:
                raise ValueError("preset=Custom requires `weights` to be set (an NxN matrix).")
            flat = _flatten(raw_weights)
            expected = k * k
            if len(flat) != expected:
                raise ValueError(
                    f"`weights` has {len(flat)} elements but kernel_size={k} expects {expected} "
                    f"({k}x{k}). Adjust kernel_size or re-fill the grid."
                )
            return flat, k
        kernel = PRESETS_3X3.get(preset)
        if kernel is None:
            raise ValueError(
                f"Unknown preset: {preset!r}. Choose one of {PRESET_OPTIONS}."
            )
        return _flatten(kernel), 3

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        tensor = inputs["tensor"]

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Conv2dExplicit `tensor` input must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )
        if tensor.dim() != 4:
            raise ValueError(
                f"Conv2dExplicit `tensor` must be 4D (N, C, H, W); "
                f"got shape {list(tensor.shape)}. Use an Unsqueeze on dim=0 "
                f"if you have a (C, H, W) tensor."
            )

        c_in = tensor.size(1)
        flat, k = self._resolve_kernel(params)

        # Built on the input's device from the start rather than on the CPU
        # and moved: the engine aligns what arrives on a wire, not what a node
        # conjures inside itself.
        weight = torch.tensor(flat, dtype=torch.float32, device=tensor.device)
        weight = weight.to(dtype=tensor.dtype).reshape(1, 1, k, k)
        # One kernel broadcast across every channel — grouped conv with
        # groups=C is what keeps the channels independent and the count
        # unchanged.
        weight = weight.expand(c_in, 1, k, k).contiguous()

        stride = max(1, int(params.get("stride", 1) or 1))
        padding = max(0, int(params.get("padding", 1) or 0))

        output = F.conv2d(
            tensor,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            groups=c_in,
        )
        return {"tensor": output}
