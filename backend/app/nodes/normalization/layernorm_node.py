from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class LayerNormNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "LayerNorm"
    CATEGORY = "Normalization"
    DESCRIPTION = (
        "Apply layer normalization (wraps nn.LayerNorm). "
        "$y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta$"
    )

    structural_params = ("normalized_shape", "eps")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="normalized_shape",
                param_type=ParamType.STRING,
                default="512",
                description="Shape to normalize over as comma-separated ints (e.g. '512' or '64,32')",
            ),
            ParamDefinition(name="eps", param_type=ParamType.FLOAT, default=1e-5, description="Epsilon for numerical stability"),
        ]

    @staticmethod
    def _parse_shape(shape_str: str) -> list[int]:
        return [int(s.strip()) for s in shape_str.split(",") if s.strip()]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        shape_str = params.get("normalized_shape", "512")
        return nn.LayerNorm(
            normalized_shape=self._parse_shape(shape_str),
            eps=params.get("eps", 1e-5),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        tensor = inputs["tensor"]
        ln = self.get_or_build_module(context, params)
        output = ln(tensor)
        result: dict[str, Any] = {"tensor": output}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder
            recorder = StepRecorder()
            normalized_shape = self._parse_shape(params.get("normalized_shape", "512"))
            eps = params.get("eps", 1e-5)
            dims = tuple(range(-len(normalized_shape), 0))
            mean = tensor.mean(dim=dims, keepdim=True)
            var = tensor.var(dim=dims, keepdim=True, unbiased=False)
            normalized = (tensor - mean) / torch.sqrt(var + eps)
            scaled = normalized * ln.weight + ln.bias
            recorder.record(
                "input",
                "Input tensor before normalisation.",
                input=tensor,
            )
            recorder.record(
                "compute_mean",
                "Per-sample mean: $\\mu = \\text{mean}(x)$ over the normalised dims.",
                scalars={"eps": float(eps)},
                mean=mean,
            )
            recorder.record(
                "compute_var",
                "Per-sample variance: $\\sigma^2 = \\text{var}(x)$ over the normalised dims.",
                var=var,
            )
            recorder.record(
                "normalize",
                "Centre and rescale: $\\hat{x} = (x-\\mu)/\\sqrt{\\sigma^2+\\epsilon}$.",
                normalized=normalized,
            )
            recorder.record(
                "scale_shift",
                "Apply learnable affine: $y = \\hat{x} \\gamma + \\beta$.",
                gamma=ln.weight,
                beta=ln.bias,
                output=scaled,
            )
            result["__steps__"] = recorder.steps

        return result
