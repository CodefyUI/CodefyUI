from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class GRUNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "GRU"
    CATEGORY = "RNN"
    DESCRIPTION = "Apply GRU recurrent layer to input sequence (wraps nn.GRU). Gates: reset, update."

    structural_params = (
        "input_size", "hidden_size", "num_layers", "batch_first", "bidirectional",
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (batch, seq_len, input_size) if batch_first=True"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="output", data_type=DataType.TENSOR, description="Output tensor containing hidden states for each time step"),
            PortDefinition(name="hidden", data_type=DataType.TENSOR, description="Final hidden state (h_n)"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="input_size", param_type=ParamType.INT, default=128, description="Number of expected features in the input"),
            ParamDefinition(name="hidden_size", param_type=ParamType.INT, default=256, description="Number of features in the hidden state"),
            ParamDefinition(name="num_layers", param_type=ParamType.INT, default=1, description="Number of recurrent layers"),
            ParamDefinition(name="batch_first", param_type=ParamType.BOOL, default=True, description="If True, input/output shape is (batch, seq, feature)"),
            ParamDefinition(name="bidirectional", param_type=ParamType.BOOL, default=False, description="If True, becomes a bidirectional GRU"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        return nn.GRU(
            input_size=params.get("input_size", 128),
            hidden_size=params.get("hidden_size", 256),
            num_layers=params.get("num_layers", 1),
            batch_first=params.get("batch_first", True),
            bidirectional=params.get("bidirectional", False),
        )

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        gru = self.get_or_build_module(context, params)
        output, h_n = gru(tensor)
        return {"output": output, "hidden": h_n}
