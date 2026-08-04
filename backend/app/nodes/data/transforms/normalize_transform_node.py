"""NormalizeTransform — per-channel standardisation, with real dataset stats.

Before core#136 the only normalisation in CodefyUI was hardcoded
``Normalize((0.5,), (0.5,))``, which maps [0, 1] onto [-1, 1] and is the
right answer for nothing in particular. Reproducing a published vision
result needs that dataset's actual channel statistics, so they ship as
presets here.
"""

from __future__ import annotations

from typing import Any

from ....core.node_base import ParamDefinition, ParamType
from ....core.param_values import parse_float_sequence
from ._base import TransformStepNode

#: (mean, std) per preset, in torchvision's channel order (R, G, B).
#:
#: ImageNet's are the numbers every torchvision pretrained checkpoint was
#: trained with -- using anything else with one of those models shifts the
#: input distribution away from what the weights expect. The CIFAR values are
#: the training-split statistics quoted throughout the CIFAR literature.
#: "Half" is the [-1, 1] mapping CodefyUI used everywhere before this node
#: existed, kept so an existing graph can be rebuilt as a chain unchanged.
PRESETS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CIFAR-10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "CIFAR-100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    # One element, not three: torchvision broadcasts a single-channel mean
    # across however many channels the tensor has, so this one preset is
    # correct for MNIST and for RGB alike.
    "Half": ((0.5,), (0.5,)),
}

PRESET_OPTIONS = [*PRESETS, "Custom"]


class NormalizeTransformNode(TransformStepNode):
    NODE_NAME = "NormalizeTransform"
    DESCRIPTION = (
        "Standardise each channel to (x - mean) / std. Needs a tensor, so "
        "it goes after ToTensorTransform."
    )

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="preset",
                param_type=ParamType.SELECT,
                default="Half",
                description=(
                    "Channel statistics to standardise with. 'Half' maps "
                    "[0, 1] to [-1, 1] and is what CodefyUI used before "
                    "presets existed; pick the dataset you are actually "
                    "training on when you want to reproduce a published "
                    "result."
                ),
                options=PRESET_OPTIONS,
            ),
            ParamDefinition(
                name="mean",
                param_type=ParamType.STRING,
                default="0.5",
                description=(
                    "Per-channel mean, comma separated. One value applies "
                    "to every channel."
                ),
                visible_when={"preset": "Custom"},
            ),
            ParamDefinition(
                name="std",
                param_type=ParamType.STRING,
                default="0.5",
                description=(
                    "Per-channel standard deviation, comma separated. One "
                    "value applies to every channel."
                ),
                visible_when={"preset": "Custom"},
            ),
        ]

    def build(self, params: dict[str, Any]) -> Any:
        from torchvision import transforms as T

        preset = str(params.get("preset", "Half"))
        if preset in PRESETS:
            mean, std = PRESETS[preset]
            return T.Normalize(mean, std)

        mean = parse_float_sequence(params.get("mean"), name="mean")
        std = parse_float_sequence(params.get("std"), name="std")
        if not mean or not std:
            raise ValueError(
                "NormalizeTransform preset 'Custom' needs both mean and "
                "std; give one value per channel, or a single value to "
                "apply to every channel.")
        if len(mean) != len(std):
            raise ValueError(
                f"mean has {len(mean)} value(s) and std has {len(std)}; "
                "they must describe the same number of channels.")
        if any(value == 0.0 for value in std):
            # Division by zero produces inf/nan for the whole channel, and
            # the failure only shows up as a loss of nan several nodes
            # later -- say it here, where the number was entered.
            raise ValueError("std cannot contain 0; it divides every pixel.")
        return T.Normalize(mean, std)
