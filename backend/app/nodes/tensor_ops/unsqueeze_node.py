from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class UnsqueezeNode(BaseNode):
    NODE_NAME = "Unsqueeze"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Insert a size-1 dimension at a position"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Tensor with added dimension"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="dim", param_type=ParamType.INT, default=0, description="Dimension at which to insert"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        tensor = inputs["tensor"]
        dim = params.get("dim", 0)
        return {"tensor": tensor.unsqueeze(dim)}
