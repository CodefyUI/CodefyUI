from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class InstanceNorm2dNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "InstanceNorm2d"
    CATEGORY = "Normalization"
    DESCRIPTION = "Normalises each sample's channel over height and width"
    DETAILS = (
        "Backed by nn.InstanceNorm2d, which is GroupNorm with one group per "
        "channel; it is the normalisation used in style transfer and image "
        "generation. affine is off by default, so there is no learnable $\\gamma$ "
        "or $\\beta$ unless you turn it on."
    )

    structural_params = ("num_features", "affine")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_features", param_type=ParamType.INT, default=64, description="Number of features (channels)"),
            ParamDefinition(name="affine", param_type=ParamType.BOOL, default=False, description="Whether to use learnable affine parameters"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.InstanceNorm2d(
            num_features=params.get("num_features", 64),
            affine=params.get("affine", False),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        norm = self.get_or_build_module(context, params)
        tensor = self.match_input_dtype(tensor, norm)
        num_features = norm.num_features
        if tensor.dim() in (3, 4) and tensor.shape[-3] != num_features:
            # nn.InstanceNorm2d checks this only when affine is on; without
            # it a 3-channel input ran under num_features=7.
            raise ValueError(
                f"InstanceNorm2d: the input has {tensor.shape[-3]} channels "
                f"but num_features is {num_features}. Set num_features to "
                f"{tensor.shape[-3]}."
            )
        return {"tensor": norm(tensor)}
