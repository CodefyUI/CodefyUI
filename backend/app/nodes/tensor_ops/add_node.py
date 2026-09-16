from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class AddNode(BaseNode):
    NODE_NAME = "Add"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Element-wise sum of two tensors, with broadcasting"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor_a", data_type=DataType.TENSOR, description="First tensor"),
            PortDefinition(name="tensor_b", data_type=DataType.TENSOR, description="Second tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Result of a + b"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="alpha", param_type=ParamType.FLOAT, default=1.0, description="Multiplier for tensor_b: a + alpha * b"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        a = inputs["tensor_a"]
        b = inputs["tensor_b"]
        alpha = params.get("alpha", 1.0)
        return {"tensor": torch.add(a, b, alpha=alpha)}
