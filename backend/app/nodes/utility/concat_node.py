from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ConcatNode(BaseNode):
    NODE_NAME = "Concat"
    CATEGORY = "Utility"
    DESCRIPTION = "Concatenate two tensors along a dimension"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor_a", data_type=DataType.TENSOR, description="First input tensor"),
            PortDefinition(name="tensor_b", data_type=DataType.TENSOR, description="Second input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Concatenated output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="dim", param_type=ParamType.INT, default=0, description="Dimension along which to concatenate"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        tensor_a = inputs["tensor_a"]
        tensor_b = inputs["tensor_b"]
        dim = params.get("dim", 0)

        output = torch.cat([tensor_a, tensor_b], dim=dim)

        return {"tensor": output}
