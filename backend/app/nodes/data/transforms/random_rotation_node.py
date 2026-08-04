"""RandomRotation — rotate by a random angle from a symmetric range."""

from __future__ import annotations

from typing import Any

from ....core.node_base import ParamDefinition, ParamType
from ._base import TransformStepNode


class RandomRotationNode(TransformStepNode):
    NODE_NAME = "RandomRotation"
    DESCRIPTION = (
        "Rotate each sample by an angle drawn uniformly from "
        "[-degrees, +degrees]. Small angles help handwriting and satellite "
        "imagery; large ones destroy any class whose orientation matters."
    )

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="degrees",
                param_type=ParamType.FLOAT,
                default=15.0,
                description=(
                    "Half-width of the rotation range, in degrees. 15 means "
                    "each sample turns by up to 15 degrees either way."
                ),
                min_value=0.0,
                max_value=180.0,
            ),
            ParamDefinition(
                name="expand",
                param_type=ParamType.BOOL,
                default=False,
                description=(
                    "Grow the output canvas so no corner is cut off. "
                    "Changes the image size, so anything downstream that "
                    "assumed a fixed shape needs a resize after it."
                ),
                advanced=True,
            ),
            ParamDefinition(
                name="fill",
                param_type=ParamType.FLOAT,
                default=0.0,
                description=(
                    "Value for the corners the rotation leaves empty. 0 is "
                    "black."
                ),
                advanced=True,
            ),
        ]

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T

        degrees = params.get("degrees", 15.0)
        degrees = 15.0 if degrees is None else abs(float(degrees))
        fill = params.get("fill", 0.0)
        return T.RandomRotation(
            degrees,
            expand=bool(params.get("expand", False)),
            fill=0.0 if fill is None else float(fill),
        )
