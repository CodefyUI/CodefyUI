from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class StackNode(BaseNode):
    NODE_NAME = "Stack"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Join two tensors along a new dimension"
    DETAILS = "Both tensors must have the same shape."

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor_a", data_type=DataType.TENSOR, description="First tensor"),
            PortDefinition(name="tensor_b", data_type=DataType.TENSOR, description="Second tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Stacked tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="dim", param_type=ParamType.INT, default=0, description="Dimension to stack along"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        a = inputs["tensor_a"]
        b = inputs["tensor_b"]
        dim = params.get("dim", 0)
        return {"tensor": torch.stack([a, b], dim=dim)}
