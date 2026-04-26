from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class GroupNormNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "GroupNorm"
    CATEGORY = "Normalization"
    DESCRIPTION = "Apply group normalization (wraps nn.GroupNorm). Used in modern CNN architectures."

    structural_params = ("num_groups", "num_channels")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, *)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_groups", param_type=ParamType.INT, default=32, description="Number of groups to divide channels into"),
            ParamDefinition(name="num_channels", param_type=ParamType.INT, default=256, description="Number of channels (must be divisible by num_groups)"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.GroupNorm(
            num_groups=params.get("num_groups", 32),
            num_channels=params.get("num_channels", 256),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        gn = self.get_or_build_module(context, params)
        return {"tensor": gn(tensor)}
