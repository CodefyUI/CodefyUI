from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class SqueezeNode(BaseNode):
    NODE_NAME = "Squeeze"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Remove size-1 dimensions from a tensor"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Squeezed tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="dim", param_type=ParamType.INT, default=-1, description="Dimension to squeeze (-1 for all)"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        tensor = inputs["tensor"]
        dim = params.get("dim", -1)
        if dim == -1:
            return {"tensor": tensor.squeeze()}
        return {"tensor": tensor.squeeze(dim)}
