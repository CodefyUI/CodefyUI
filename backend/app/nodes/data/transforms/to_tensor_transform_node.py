"""ToTensorTransform — the chain's PIL-to-tensor step."""

from __future__ import annotations

from typing import Any

from ._base import TransformStepNode


class ToTensorTransformNode(TransformStepNode):
    NODE_NAME = "ToTensorTransform"
    DESCRIPTION = (
        "Convert a PIL image to a CxHxW float tensor scaled to [0, 1]. The "
        "hinge of most chains: geometric and colour steps go before it, "
        "NormalizeTransform after it."
    )

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T

        return T.ToTensor()
