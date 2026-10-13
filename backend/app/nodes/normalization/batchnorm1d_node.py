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


class BatchNorm1dNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "BatchNorm1d"
    CATEGORY = "Normalization"
    DESCRIPTION = "Normalises each feature over the batch"
    DETAILS = (
        "Backed by nn.BatchNorm1d. The input is (N, C) or (N, C, L) and "
        "num_features must equal C; for (N, C, H, W) image tensors use BatchNorm2d "
        "instead. A learnable scale and shift are applied after normalising. The "
        "statistics always come from the batch being run (no running averages "
        "are kept), so the input needs more than one value per feature."
    )

    structural_params = ("num_features",)

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C) or (N, C, L)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Normalized output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="num_features", param_type=ParamType.INT, default=128, description="Number of features to normalize"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        # Batch statistics on every run and no running buffers. Nothing
        # puts a canvas layer into eval mode, so the running mean/var were
        # never read; they only drifted on every run with persistence on
        # (hidden state the canvas cannot show). The output is the same as
        # before: training-mode BatchNorm already normalised with the
        # batch's own statistics.
        return nn.BatchNorm1d(
            num_features=params.get("num_features", 128),
            track_running_stats=False,
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        bn = self.get_or_build_module(context, params)
        tensor = self.match_input_dtype(tensor, bn)
        _require_batch_statistics(self.NODE_NAME, tensor)
        return {"tensor": bn(tensor)}
