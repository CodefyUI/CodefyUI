from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition
from ...core.stateful_module import StatefulModuleMixin


class TransformerDecoderNode(StatefulModuleMixin, BaseNode):
    NODE_NAME = "TransformerDecoder"
    CATEGORY = "Transformer"
    DESCRIPTION = "Apply Transformer decoder stack to input tensor with encoder memory"

    structural_params = ("d_model", "nhead", "num_layers", "dim_feedforward")

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Target tensor (seq_len, batch, d_model)"),
            PortDefinition(name="memory", data_type=DataType.TENSOR, description="Encoder output / memory tensor"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Decoded output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="d_model", param_type=ParamType.INT, default=512, description="Dimension of the model"),
            ParamDefinition(name="nhead", param_type=ParamType.INT, default=8, description="Number of attention heads"),
            ParamDefinition(name="num_layers", param_type=ParamType.INT, default=6, description="Number of decoder layers"),
            ParamDefinition(name="dim_feedforward", param_type=ParamType.INT, default=2048, description="Dimension of feedforward network"),
        ]

    def build_module(self, params: dict[str, Any]) -> Any:
        import torch.nn as nn
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=params.get("d_model", 512),
            nhead=params.get("nhead", 8),
            dim_feedforward=params.get("dim_feedforward", 2048),
            batch_first=False,
        )
        return nn.TransformerDecoder(decoder_layer, num_layers=params.get("num_layers", 6))

    def execute(self, inputs: dict[str, Any], params: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        tensor = inputs["tensor"]
        memory = inputs["memory"]
        decoder = self.get_or_build_module(context, params)
        return {"tensor": decoder(tensor, memory)}
