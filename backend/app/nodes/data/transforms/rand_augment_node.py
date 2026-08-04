"""RandAugment — a whole augmentation policy behind two numbers."""

from __future__ import annotations

from typing import Any

from ....core.node_base import ParamDefinition, ParamType
from ._base import TransformStepNode


class RandAugmentNode(TransformStepNode):
    NODE_NAME = "RandAugment"
    DESCRIPTION = (
        "Apply num_ops operations picked at random from a fixed set "
        "(shear, translate, rotate, posterize, solarize, colour, contrast, "
        "brightness, sharpness, equalize, autocontrast, identity), each at "
        "the same magnitude. Needs a PIL image or a uint8 tensor, so it "
        "goes before ToTensorTransform."
    )

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="num_ops",
                param_type=ParamType.INT,
                default=2,
                description=(
                    "How many operations to apply to each sample. The "
                    "paper's default is 2."
                ),
                min_value=0,
                max_value=8,
            ),
            ParamDefinition(
                name="magnitude",
                param_type=ParamType.INT,
                default=9,
                description=(
                    "Strength of every operation, on a scale of 0 to "
                    "num_magnitude_bins - 1. Raise it for larger models and "
                    "larger datasets; a small model on a small dataset "
                    "underfits long before it needs 15."
                ),
                min_value=0,
            ),
            ParamDefinition(
                name="num_magnitude_bins",
                param_type=ParamType.INT,
                default=31,
                description=(
                    "Resolution of the magnitude scale. torchvision's "
                    "default is 31; changing it re-scales what magnitude "
                    "means."
                ),
                min_value=2,
                advanced=True,
            ),
        ]

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T

        bins = max(2, int(params.get("num_magnitude_bins", 31) or 31))
        magnitude = max(0, int(params.get("magnitude", 9) or 0))
        if magnitude >= bins:
            # torchvision indexes its magnitude table with this value, so
            # the alternative is an IndexError from inside the DataLoader
            # that names neither parameter.
            raise ValueError(
                f"magnitude must be less than num_magnitude_bins "
                f"({bins}); got {magnitude}.")
        return T.RandAugment(
            num_ops=max(0, int(params.get("num_ops", 2) or 0)),
            magnitude=magnitude,
            num_magnitude_bins=bins,
        )
