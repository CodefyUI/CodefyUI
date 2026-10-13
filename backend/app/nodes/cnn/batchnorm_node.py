from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


def _require_batch_statistics(name: str, tensor: Any) -> None:
    """Refuse an input with one value per channel.

    The canvas node normalises with the statistics of the batch it is given
    (see ``build_module``), and one value has no variance. torch says
    "Expected more than 1 value per channel when training", which names
    neither the node nor the fix.
    """
    if tensor.dim() < 2:
        return
    per_channel = tensor.numel() // max(1, tensor.shape[1])
    if per_channel <= 1:
        raise ValueError(
            f"{name}: the input {tuple(tensor.shape)} has one value per "
            f"channel, and batch statistics need at least two. Feed a batch "
            f"of two or more samples (or a larger spatial size)."
        )


class BatchNormNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "BatchNorm2d"
    CATEGORY = "CNN"
    DESCRIPTION = "Normalises each channel over batch, height and width"
    DETAILS = (
        "Backed by nn.BatchNorm2d: $y = \\frac{x - \\mu_C}{\\sqrt{\\sigma_C^2 + "
        "\\epsilon}} \\gamma + \\beta$, where $\\mu_C$ and $\\sigma_C^2$ are "
        "per-channel statistics over (N, H, W). num_features must equal the "
        "input's channel count. The statistics always come from the batch "
        "being run (no running averages are kept), so the input needs more "
        "than one value per channel."
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
        # Batch statistics on every run and no running buffers. Nothing
        # puts a canvas layer into eval mode, so the running mean/var were
        # never read; they only drifted on every run with persistence on
        # (hidden state the canvas cannot show). The output is the same as
        # before: training-mode BatchNorm already normalised with the
        # batch's own statistics.
        return nn.BatchNorm2d(
            num_features=params.get("num_features", 32),
            track_running_stats=False,
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        import torch

        tensor = inputs["tensor"]
        batchnorm = self.get_or_build_module(context, params)
        tensor = self.match_input_dtype(tensor, batchnorm)
        _require_batch_statistics(self.NODE_NAME, tensor)
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
