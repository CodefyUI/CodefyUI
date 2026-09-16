from typing import Any

from ...core.node_base import BaseNode, DataType, PortDefinition


class MatMulNode(BaseNode):
    NODE_NAME = "MatMul"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Batched matrix product of two tensors"
    DETAILS = (
        "Follows `torch.matmul`: the last two dimensions multiply as matrices "
        "while any leading dimensions broadcast as batch. Two 1-D inputs give a "
        "dot product instead."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor_a", data_type=DataType.TENSOR, description="First tensor"),
            PortDefinition(name="tensor_b", data_type=DataType.TENSOR, description="Second tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Result of matmul(a, b)"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        a = inputs["tensor_a"]
        b = inputs["tensor_b"]
        return {"tensor": torch.matmul(a, b)}
