from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ArgmaxNode(BaseNode):
    NODE_NAME = "Argmax"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = "Index of the largest value along a dimension"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Input tensor (usually logits)."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="Tensor of argmax indices, without the reduced dimension."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dim",
                param_type=ParamType.INT,
                default=-1,
                description=(
                    "Dimension to take the argmax over. Use dim=1 for the classes of a "
                    "segmentation output (N,C,H,W)."
                ),
            ),
            ParamDefinition(
                name="keepdim",
                param_type=ParamType.BOOL,
                default=False,
                description="Keep the reduced dimension with length 1.",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        tensor = inputs["tensor"]
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        dim = int(params.get("dim", -1))
        keepdim = bool(params.get("keepdim", False))

        output = torch.argmax(tensor, dim=dim, keepdim=keepdim)
        return {"tensor": output}
