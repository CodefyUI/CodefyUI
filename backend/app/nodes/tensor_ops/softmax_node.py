from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class SoftmaxNode(BaseNode):
    NODE_NAME = "Softmax"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = (
        "Apply softmax along a dimension: $\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$. "
        "Numerically stable: subtract $\\max(x)$ before exponentiating."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (logits)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Softmax probabilities"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="dim", param_type=ParamType.INT, default=-1, description="Dimension along which to apply softmax"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        tensor = inputs["tensor"]
        dim = params.get("dim", -1)
        output = F.softmax(tensor, dim=dim)
        result: dict[str, Any] = {"tensor": output}

        if context is not None and getattr(context, "verbose", False):
            from ...core.step_trace import StepRecorder
            recorder = StepRecorder()
            max_vals = tensor.max(dim=dim, keepdim=True).values
            shifted = tensor - max_vals
            exped = torch.exp(shifted)
            partition = exped.sum(dim=dim, keepdim=True)
            divided = exped / partition
            recorder.record(
                "logits",
                "Raw input logits.",
                logits=tensor,
            )
            recorder.record(
                "max_subtract",
                "Subtract the max for numerical stability: $x' = x - \\max(x)$. Doesn't change the result of softmax but prevents overflow.",
                max=max_vals,
                shifted=shifted,
            )
            recorder.record(
                "exp",
                "Element-wise exponential: $e^{x'}$.",
                exped=exped,
            )
            recorder.record(
                "partition",
                "Sum along the softmax dimension: $Z = \\sum_j e^{x'_j}$.",
                partition=partition,
            )
            recorder.record(
                "divide",
                "Divide each element by the partition: $\\text{softmax}(x)_i = e^{x'_i} / Z$.",
                output=divided,
            )
            result["__steps__"] = recorder.steps

        return result
