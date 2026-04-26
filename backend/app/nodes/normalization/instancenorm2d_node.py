from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class InstanceNorm2dNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "InstanceNorm2d"
    CATEGORY = "Normalization"
    DESCRIPTION = "Apply 2D instance normalization (wraps nn.InstanceNorm2d). Used in style transfer and image generation."

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
        return {"tensor": norm(tensor)}
