from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ReshapeNode(BaseNode):
    NODE_NAME = "Reshape"
    CATEGORY = "Utility"
    DESCRIPTION = "Reshape a tensor to a given shape"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor to reshape"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Reshaped tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="shape",
                param_type=ParamType.STRING,
                default="-1,784",
                description="Target shape as comma-separated ints (e.g. '-1,784')",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        if "tensor" not in inputs:
            raise ValueError("Reshape node requires a 'tensor' input. Please connect a tensor source to this node.")
        tensor = inputs["tensor"]
        shape_str = params.get("shape", "-1,784")

        shape = tuple(int(s.strip()) for s in shape_str.split(","))
        output = tensor.reshape(shape)

        return {"tensor": output}
