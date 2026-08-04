"""RandomCrop — the padded random crop every CIFAR baseline starts with."""

from __future__ import annotations

from typing import Any

from ....core.node_base import ParamDefinition, ParamType
from ._base import TransformStepNode


class RandomCropNode(TransformStepNode):
    NODE_NAME = "RandomCrop"
    DESCRIPTION = (
        "Pad, then take a random size x size window. With size 32 and "
        "padding 4 this is the standard CIFAR-10 augmentation: the object "
        "lands a few pixels off centre each epoch, so the model stops "
        "relying on where it was."
    )

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="size",
                param_type=ParamType.INT,
                default=32,
                description="Side length in pixels of the cropped square",
                min_value=1,
            ),
            ParamDefinition(
                name="padding",
                param_type=ParamType.INT,
                default=4,
                description=(
                    "Pixels added on every side before cropping. 0 crops a "
                    "genuinely smaller window out of the image; a padding "
                    "equal to the shift you want keeps the output the same "
                    "size as the input."
                ),
                min_value=0,
            ),
            ParamDefinition(
                name="padding_mode",
                param_type=ParamType.SELECT,
                default="constant",
                description=(
                    "What the added border contains. 'constant' is black, "
                    "'reflect' mirrors the image edge (no artificial border "
                    "for the model to learn)."
                ),
                options=["constant", "edge", "reflect", "symmetric"],
                advanced=True,
            ),
        ]

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T

        size = max(1, int(params.get("size", 32) or 32))
        padding = max(0, int(params.get("padding", 4) or 0))
        mode = str(params.get("padding_mode", "constant") or "constant")
        return T.RandomCrop(size, padding=padding, padding_mode=mode)
