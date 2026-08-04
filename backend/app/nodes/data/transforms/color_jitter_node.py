"""ColorJitter — random brightness / contrast / saturation / hue shifts."""

from __future__ import annotations

from typing import Any

from ....core.node_base import ParamDefinition, ParamType
from ._base import TransformStepNode


class ColorJitterNode(TransformStepNode):
    NODE_NAME = "ColorJitter"
    DESCRIPTION = (
        "Randomly shift brightness, contrast, saturation and hue. Teaches "
        "the model that a cat under a warm lamp is still a cat. The "
        "defaults are the recipe most ImageNet training scripts use."
    )

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="brightness",
                param_type=ParamType.FLOAT,
                default=0.4,
                description=(
                    "Brightness is scaled by a factor drawn from "
                    "[1-b, 1+b]. 0 disables it."
                ),
                min_value=0.0,
                max_value=1.0,
            ),
            ParamDefinition(
                name="contrast",
                param_type=ParamType.FLOAT,
                default=0.4,
                description="Same range rule as brightness. 0 disables it.",
                min_value=0.0,
                max_value=1.0,
            ),
            ParamDefinition(
                name="saturation",
                param_type=ParamType.FLOAT,
                default=0.4,
                description="Same range rule as brightness. 0 disables it.",
                min_value=0.0,
                max_value=1.0,
            ),
            ParamDefinition(
                name="hue",
                param_type=ParamType.FLOAT,
                default=0.1,
                description=(
                    "Hue is shifted by an offset drawn from [-h, +h] on a "
                    "wheel of width 1, so it maxes out at 0.5. Keep it "
                    "small: past about 0.2 the colours a class depends on "
                    "stop being that class's colours."
                ),
                min_value=0.0,
                max_value=0.5,
            ),
        ]

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T

        def value(name: str, fallback: float) -> float:
            raw = params.get(name, fallback)
            return fallback if raw is None else float(raw)

        return T.ColorJitter(
            brightness=value("brightness", 0.4),
            contrast=value("contrast", 0.4),
            saturation=value("saturation", 0.4),
            hue=value("hue", 0.1),
        )
