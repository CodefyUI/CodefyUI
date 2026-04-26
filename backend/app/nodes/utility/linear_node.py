from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class LinearNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "Linear"
    CATEGORY = "Utility"
    DESCRIPTION = (
        "Fully-connected (dense) layer: $y = xW^T + b$. Wraps nn.Linear(in_features, out_features)."
    )

    structural_params = ("in_features", "out_features")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (..., in_features)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Output tensor (..., out_features)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="in_features", param_type=ParamType.INT, default=512, description="Input feature size"),
            ParamDefinition(name="out_features", param_type=ParamType.INT, default=10, description="Output feature size"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.Linear(
            in_features=params.get("in_features", 512),
            out_features=params.get("out_features", 10),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        linear = self.get_or_build_module(context, params)
        output = linear(tensor)
        result: dict[str, Any] = {"tensor": output}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder
            recorder = StepRecorder()
            recorder.record(
                "input",
                "Input tensor with shape (..., in_features).",
                input=tensor,
            )
            recorder.record(
                "weight_matrix",
                "Learnable weight $W$ with shape (out_features, in_features).",
                weight=linear.weight,
            )
            if linear.bias is not None:
                recorder.record(
                    "bias_vector",
                    "Learnable bias $b$ with shape (out_features,).",
                    bias=linear.bias,
                )
            recorder.record(
                "linear_output",
                "Compute $y = x W^T + b$.",
                output=output,
            )
            result["__steps__"] = recorder.steps

        return result
