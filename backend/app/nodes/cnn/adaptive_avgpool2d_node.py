from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class AdaptiveAvgPool2dNode(BaseNode):
    NODE_NAME = "AdaptiveAvgPool2d"
    CATEGORY = "CNN"
    DESCRIPTION = "Averages each channel to a fixed height and width"
    DETAILS = (
        "Backed by nn.AdaptiveAvgPool2d, which derives the window size from the "
        "input, so any input becomes (N, C, output_height, output_width). The 1x1 "
        "default reduces each channel to a single number."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Pooled output tensor with fixed spatial size"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="output_height", param_type=ParamType.INT, default=1, description="Target output height"),
            ParamDefinition(name="output_width", param_type=ParamType.INT, default=1, description="Target output width"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        output_size = (params.get("output_height", 1), params.get("output_width", 1))
        pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        return {"tensor": pool(tensor)}
