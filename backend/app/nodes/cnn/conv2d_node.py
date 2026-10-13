from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class Conv2dNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "Conv2d"
    CATEGORY = "CNN"
    DESCRIPTION = "Slides learnable kernels over an image, one map each"
    DETAILS = (
        "Backed by nn.Conv2d: $y[i,j]=\\sum_{k,l} x[i+k,j+l]\\cdot w[k,l] + b$. "
        "in_channels must equal the input's channel count; the weights start "
        "random and are learned during training."
    )

    structural_params = (
        "in_channels", "out_channels", "kernel_size", "stride", "padding",
        "dilation", "groups", "bias",
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
            ParamDefinition(name="dilation", param_type=ParamType.INT, default=1, min_value=1, description="Spacing between kernel elements (1 = a plain convolution)"),
            ParamDefinition(
                name="groups",
                param_type=ParamType.INT,
                default=1,
                min_value=1,
                description=(
                    "Split the channels into this many independent groups. "
                    "Both in_channels and out_channels must divide by it; "
                    "groups = in_channels is a depthwise convolution."
                ),
            ),
            ParamDefinition(name="bias", param_type=ParamType.BOOL, default=True, description="Add a learnable bias per output channel. Off is usual right before a BatchNorm."),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn

        from ...core.param_values import bool_param, int_param

        def whole(name: str, default: int, minimum: int) -> int:
            return int_param(params, name, node="Conv2d", default=default,
                             min_value=minimum)

        return nn.Conv2d(
            in_channels=whole("in_channels", 1, 1),
            out_channels=whole("out_channels", 32, 1),
            kernel_size=whole("kernel_size", 3, 1),
            stride=whole("stride", 1, 1),
            padding=whole("padding", 1, 0),
            dilation=whole("dilation", 1, 1),
            groups=whole("groups", 1, 1),
            # bool("false") is True; bool_param reads the word.
            bias=bool_param(params, "bias", node="Conv2d", default=True),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        conv = self.get_or_build_module(context, params)
        tensor = self.match_input_dtype(tensor, conv)
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
