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


class Conv2dKernelNode(BaseNode):
    NODE_NAME = "Conv2dKernel"
    CATEGORY = "CNN"
    DESCRIPTION = (
        "Emits a 2D convolution kernel as a tensor (no convolution is "
        "performed here). Pick a built-in 3x3 preset (EdgeDetection / "
        "Sharpen / VerticalEdge) or switch preset to 'Custom' to enter an "
        "NxN matrix by hand. Output shape is (kernel_size, kernel_size). "
        "Wire the output into a downstream convolution node that accepts "
        "an explicit kernel."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="tensor",
                data_type=DataType.TENSOR,
                description="Kernel as a 2D tensor of shape (kernel_size, kernel_size)",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
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
        ]

    def _resolve_kernel(self, params: dict[str, Any]) -> tuple[list[float], int]:
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

        flat, k = self._resolve_kernel(params)
        tensor = torch.tensor(flat, dtype=torch.float32).reshape(k, k)
        return {"tensor": tensor}
