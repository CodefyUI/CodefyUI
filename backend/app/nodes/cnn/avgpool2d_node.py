from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class AvgPool2dNode(BaseNode):
    NODE_NAME = "AvgPool2d"
    CATEGORY = "CNN"
    DESCRIPTION = "Average of each window; height and width shrink"
    DETAILS = (
        "Backed by nn.AvgPool2d. Every value in the window contributes, so the "
        "result is smoother than MaxPool2d. With the defaults, kernel_size 2 and "
        "stride 2, height and width are halved."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (N, C, H, W)"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Pooled output tensor"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="kernel_size", param_type=ParamType.INT, default=2, description="Size of the pooling window"),
            ParamDefinition(name="stride", param_type=ParamType.INT, default=2, description="Stride of the pooling window"),
            ParamDefinition(name="padding", param_type=ParamType.INT, default=0, description="Zero-padding added to both sides"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        pool = nn.AvgPool2d(
            kernel_size=params.get("kernel_size", 2),
            stride=params.get("stride", 2),
            padding=params.get("padding", 0),
        )
        return {"tensor": pool(tensor)}
