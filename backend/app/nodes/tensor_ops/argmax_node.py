from typing import Any

from ...core.node_base import BaseNode, DataType, ParamDefinition, ParamType, PortDefinition


class ArgmaxNode(BaseNode):
    NODE_NAME = "Argmax"
    CATEGORY = "Tensor Operations"
    DESCRIPTION = (
        "沿指定維度取最大值的索引（argmax）。常用在分類 / 分割輸出："
        "對 logits 取 argmax 得到預測類別。例如分割輸出 (N, C, H, W) 沿 dim=1 取 argmax，"
        "得到每像素的類別 (N, H, W)。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="輸入張量（通常是 logits）。"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="tensor", data_type=DataType.TENSOR, description="argmax 後的索引張量（少掉被取的那一維）。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dim",
                param_type=ParamType.INT,
                default=-1,
                description="沿哪一維取 argmax。分割輸出 (N,C,H,W) 取類別用 dim=1。",
            ),
            ParamDefinition(
                name="keepdim",
                param_type=ParamType.BOOL,
                default=False,
                description="是否保留被取的那一維（長度變 1）。",
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
