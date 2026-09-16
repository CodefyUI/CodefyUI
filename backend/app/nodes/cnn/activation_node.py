from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ActivationNode(BaseNode):
    NODE_NAME = "Activation"
    CATEGORY = "CNN"
    DESCRIPTION = "Applies the selected nonlinearity; shape unchanged"
    DETAILS = (
        "Twelve functions are available: relu, leaky_relu, elu, gelu, silu, mish, "
        "selu, prelu, sigmoid, tanh, hardswish and softmax. softmax normalises "
        "over the last dimension, and prelu builds a fresh, untrained slope of "
        "0.25 on every run."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Activated output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="function",
                param_type=ParamType.SELECT,
                default="relu",
                description="Activation function to apply",
                options=["relu", "sigmoid", "tanh", "leaky_relu", "elu", "gelu", "silu", "mish", "selu", "softmax", "hardswish", "prelu"],
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        tensor = inputs["tensor"]
        function = params.get("function", "relu")

        activations = {
            "relu": lambda x: F.relu(x),
            "sigmoid": lambda x: torch.sigmoid(x),
            "tanh": lambda x: torch.tanh(x),
            "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.01),
            "elu": lambda x: F.elu(x),
            "gelu": lambda x: F.gelu(x),
            "silu": lambda x: F.silu(x),
            "mish": lambda x: F.mish(x),
            "selu": lambda x: F.selu(x),
            "softmax": lambda x: F.softmax(x, dim=-1),
            "hardswish": lambda x: F.hardswish(x),
            "prelu": lambda x: nn.PReLU()(x),
        }
        act_fn = activations.get(function)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {function}")
        output = act_fn(tensor)

        return {"tensor": output}
