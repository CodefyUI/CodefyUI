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
        try:
            return {"tensor": torch.matmul(a, b)}
        except RuntimeError as exc:
            a_shape = tuple(getattr(a, "shape", ()))
            b_shape = tuple(getattr(b, "shape", ()))
            if a_shape and b_shape and len(b_shape) >= 1 and a_shape[-1] != b_shape[-2 if len(b_shape) >= 2 else 0]:
                raise RuntimeError(
                    f"MatMul: tensor_a {a_shape} and tensor_b {b_shape} do not fit -- the "
                    f"last dim of tensor_a ({a_shape[-1]}) must equal the "
                    f"{'second-to-last' if len(b_shape) >= 2 else 'only'} dim of tensor_b "
                    f"({b_shape[-2] if len(b_shape) >= 2 else b_shape[0]}). Check which "
                    f"input goes to which port, or transpose one with Permute."
                ) from exc
            raise
