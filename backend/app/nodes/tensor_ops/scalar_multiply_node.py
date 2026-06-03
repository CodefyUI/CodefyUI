from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ScalarMultiplyNode(BaseNode):
    NODE_NAME = "ScalarMultiply"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Multiply a tensor by a constant scalar (no second input needed)."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="tensor * scalar"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="scalar",
                param_type=ParamType.FLOAT,
                default=1.0,
                description="Constant scalar multiplied with every element of the input tensor",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        tensor = inputs["tensor"]
        scalar = float(params.get("scalar", 1.0))
        return {"tensor": tensor * scalar}
