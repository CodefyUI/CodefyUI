"""RandomHorizontalFlip — mirror the image, sometimes."""

from __future__ import annotations

from typing import Any

from ....core.node_base import ParamDefinition, ParamType
from ._base import TransformStepNode


class RandomHorizontalFlipNode(TransformStepNode):
    NODE_NAME = "RandomHorizontalFlip"
    DESCRIPTION = (
        "Mirror the image left-to-right with probability p. Free accuracy "
        "on photographs; wrong for anything where handedness carries "
        "meaning, such as digits or text."
    )

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="p",
                param_type=ParamType.FLOAT,
                default=0.5,
                description="Probability of flipping each sample",
                min_value=0.0,
                max_value=1.0,
            ),
        ]

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T

        p = params.get("p", 0.5)
        return T.RandomHorizontalFlip(p=0.5 if p is None else float(p))
