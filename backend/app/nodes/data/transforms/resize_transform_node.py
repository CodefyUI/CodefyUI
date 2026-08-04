"""ResizeTransform — the chain's resize step."""

from __future__ import annotations

from typing import Any

from ....core.node_base import ParamDefinition, ParamType
from ._base import TransformStepNode


class ResizeTransformNode(TransformStepNode):
    NODE_NAME = "ResizeTransform"
    DESCRIPTION = (
        "Resize every sample to a square of the given size. Put it before "
        "ToTensorTransform."
    )

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="size",
                param_type=ParamType.INT,
                default=32,
                description="Side length in pixels of the resized square",
                min_value=1,
            ),
            ParamDefinition(
                name="interpolation",
                param_type=ParamType.SELECT,
                default="bilinear",
                description=(
                    "Resampling filter. 'nearest' keeps hard edges (masks, "
                    "label maps); 'bicubic' is sharper on photographs."
                ),
                options=["nearest", "bilinear", "bicubic"],
                advanced=True,
            ),
        ]

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode

        modes = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
        }
        size = max(1, int(params.get("size", 32) or 32))
        mode = modes.get(str(params.get("interpolation", "bilinear")),
                         InterpolationMode.BILINEAR)
        # A (h, w) pair rather than a bare int: torchvision reads the int
        # form as "shortest side", which leaves a non-square image
        # non-square and then breaks the first layer that assumed a shape.
        return T.Resize((size, size), interpolation=mode)
