from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class MaxPool2dNode(BaseNode):
    NODE_NAME = "MaxPool2d"
    CATEGORY = "CNN"
    DESCRIPTION = "Largest value in each window; height and width shrink"
    DETAILS = (
        "Backed by nn.MaxPool2d; only kernel_size and stride are exposed, with no "
        "padding. With the defaults, kernel_size 2 and stride 2, height and width "
        "are halved."
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
            ParamDefinition(name="kernel_size", param_type=ParamType.INT, default=2, description="Size of the max pooling window"),
            ParamDefinition(name="stride", param_type=ParamType.INT, default=2, description="Stride of the pooling window"),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch.nn as nn

        tensor = inputs["tensor"]
        kernel_size = params.get("kernel_size", 2)
        stride = params.get("stride", 2)

        pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        output = pool(tensor)
        return {"tensor": output}
