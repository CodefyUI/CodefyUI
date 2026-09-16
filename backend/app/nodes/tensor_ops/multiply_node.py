from typing import Any

from ...core.node_base import BaseNode, DataType, PortDefinition


class MultiplyNode(BaseNode):
    NODE_NAME = "Multiply"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Element-wise product of two tensors, with broadcasting"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor_a", data_type=DataType.TENSOR, description="First tensor"),
            PortDefinition(name="tensor_b", data_type=DataType.TENSOR, description="Second tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Result of a * b"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        a = inputs["tensor_a"]
        b = inputs["tensor_b"]
        return {"tensor": torch.mul(a, b)}
