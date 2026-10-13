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
        import torch
        import torch.nn as nn

        from ...core.param_values import int_param, is_blank

        node = "AvgPool2d"
        tensor = inputs["tensor"]
        kernel_size = int_param(params, "kernel_size", node=node, default=2, min_value=1)
        # A blank stride means "same as kernel_size", as in torch.
        stride = (None if "stride" in params and is_blank(params["stride"])
                  else int_param(params, "stride", node=node, default=2, min_value=1))
        padding = int_param(params, "padding", node=node, default=0, min_value=0)
        if isinstance(tensor, torch.Tensor) and tensor.dim() not in (3, 4):
            raise ValueError(
                f"{node}: the input must be (C, H, W) or (N, C, H, W), got "
                f"shape {tuple(tensor.shape)}. For a single (H, W) image add "
                f"the channel dimension with an Unsqueeze (dim=0)."
            )
        pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        return {"tensor": pool(tensor)}
