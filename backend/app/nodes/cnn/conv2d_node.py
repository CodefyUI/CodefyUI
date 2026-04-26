from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class Conv2dNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "Conv2d"
    CATEGORY = "CNN"
    DESCRIPTION = (
        "Apply 2D convolution to input tensor (wraps nn.Conv2d). "
        "$y[i,j]=\\sum_{k,l} x[i+k,j+l]\\cdot w[k,l] + b$"
    )

    structural_params = (
        "in_channels", "out_channels", "kernel_size", "stride", "padding",
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Convolved output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="in_channels", param_type=ParamType.INT, default=1, description="Number of input channels"),
            ParamDefinition(name="out_channels", param_type=ParamType.INT, default=32, description="Number of output channels"),
            ParamDefinition(name="kernel_size", param_type=ParamType.INT, default=3, description="Size of the convolving kernel"),
            ParamDefinition(name="stride", param_type=ParamType.INT, default=1, description="Stride of the convolution"),
            ParamDefinition(name="padding", param_type=ParamType.INT, default=1, description="Zero-padding added to both sides"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.Conv2d(
            in_channels=params.get("in_channels", 1),
            out_channels=params.get("out_channels", 32),
            kernel_size=params.get("kernel_size", 3),
            stride=params.get("stride", 1),
            padding=params.get("padding", 1),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        conv = self.get_or_build_module(context, params)
        output = conv(tensor)
        result: dict[str, Any] = {"tensor": output}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder
            recorder = StepRecorder()
            recorder.record(
                "input_tensor",
                "Input image batch (N, C, H, W).",
                input=tensor,
            )
            bias = conv.bias if conv.bias is not None else conv.weight.new_zeros(conv.out_channels)
            recorder.record(
                "kernel_weights",
                "Learnable convolution kernel: shape $(C_{out}, C_{in}, k, k)$.",
                weight=conv.weight,
                bias=bias,
            )
            recorder.record(
                "convolved_output",
                "Output: each spatial position is a dot product of kernel and a sliding window of input plus bias.",
                output=output,
            )
            result["__steps__"] = recorder.steps

        return result
