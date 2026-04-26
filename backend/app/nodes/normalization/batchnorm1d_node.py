from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class BatchNorm1dNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "BatchNorm1d"
    CATEGORY = "Normalization"
    DESCRIPTION = "Apply 1D batch normalization (wraps nn.BatchNorm1d). Used after Linear layers."

    structural_params = ("num_features",)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C) or (N, C, L)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_features", param_type=ParamType.INT, default=128, description="Number of features to normalize"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.BatchNorm1d(num_features=params.get("num_features", 128))

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        bn = self.get_or_build_module(context, params)
        return {"tensor": bn(tensor)}
