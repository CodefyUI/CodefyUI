from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class BatchNormNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "BatchNorm2d"
    CATEGORY = "CNN"
    DESCRIPTION = (
        "Apply 2D batch normalization (wraps nn.BatchNorm2d). "
        "Normalises each channel: $y = \\frac{x - \\mu_C}{\\sqrt{\\sigma_C^2 + \\epsilon}} \\gamma + \\beta$ "
        "where $\\mu_C, \\sigma_C^2$ are per-channel statistics over (N, H, W)."
    )

    structural_params = ("num_features",)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_features", param_type=ParamType.INT, default=32, description="Number of features (channels) to normalize"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.BatchNorm2d(num_features=params.get("num_features", 32))

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        tensor = inputs["tensor"]
        batchnorm = self.get_or_build_module(context, params)
        output = batchnorm(tensor)
        result: dict[str, Any] = {"tensor": output}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder
            recorder = StepRecorder()
            mean_c = tensor.mean(dim=(0, 2, 3), keepdim=True)
            var_c = tensor.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            eps = batchnorm.eps
            normalized = (tensor - mean_c) / torch.sqrt(var_c + eps)
            scaled = normalized * batchnorm.weight.view(1, -1, 1, 1) + batchnorm.bias.view(1, -1, 1, 1)
            recorder.record(
                "input",
                "Input tensor (N, C, H, W).",
                input=tensor,
            )
            recorder.record(
                "per_channel_mean",
                "Mean per channel, computed over (N, H, W): $\\mu_C$.",
                scalars={"eps": float(eps)},
                mean=mean_c,
            )
            recorder.record(
                "per_channel_var",
                "Variance per channel: $\\sigma_C^2$.",
                var=var_c,
            )
            recorder.record(
                "normalize",
                "Centre and scale: $\\hat{x} = (x - \\mu_C)/\\sqrt{\\sigma_C^2 + \\epsilon}$.",
                normalized=normalized,
            )
            recorder.record(
                "scale_shift",
                "Apply learnable affine: $y = \\hat{x}\\gamma_C + \\beta_C$.",
                gamma=batchnorm.weight,
                beta=batchnorm.bias,
                output=scaled,
            )
            result["__steps__"] = recorder.steps

        return result
