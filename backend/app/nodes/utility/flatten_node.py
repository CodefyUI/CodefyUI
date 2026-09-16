from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class FlattenNode(BaseNode):
    NODE_NAME = "Flatten"
    CATEGORY = "Utility"
    DESCRIPTION = "Flatten a tensor's dimensions from start_dim onward"
    DETAILS = (
        "Backed by nn.Flatten, so every dimension from start_dim to the last "
        "collapses into one."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Flattened tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="start_dim", param_type=ParamType.INT, default=1, description="First dim to flatten"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        flatten = nn.Flatten(start_dim=params.get("start_dim", 1))
        return {"tensor": flatten(tensor)}
