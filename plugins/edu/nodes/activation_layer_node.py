"""ActivationLayerNode — 在 MLP 堆疊裡夾一個激活函數（非線性）。

對應教材 **I2-3**：神經網路「有意義的深度」全靠層與層之間那個非線性。這顆就是
那個非線性積木——把它夾在兩個 `FFNLayer` 之間，整個堆疊才彎得起來。

故意做成**獨立一顆**（而不是塞進 FFNLayer），這樣「拿掉激活」這個關鍵實驗就只是
把 `function` 設成 `identity`——你會親眼看到不管堆幾層，網路都塌回一條直線、同心圓
退回約 50%。

跟 FFNLayer 一樣傳「正在組裝的網路」：往尾端加一個激活模組。
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class ActivationLayerNode(BaseNode):
    NODE_NAME = "ActivationLayer"
    CATEGORY = "EDU"
    DESCRIPTION = (
        "在 MLP 堆疊裡夾一個激活函數（非線性）。夾在兩個 FFNLayer 之間，堆疊才彎得起來。"
        "function 設 identity = 等於沒有激活，堆幾層都會塌回一條直線（I2-3 的關鍵實驗）。"
        "跟 FFNLayer 一樣傳「正在組裝的網路」。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="上游正在組裝的網路（通常接在一個 FFNLayer 後面）。",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="尾端加上一個激活函數後的網路。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="function",
                param_type=ParamType.SELECT,
                default="relu",
                options=["relu", "tanh", "sigmoid", "identity"],
                description="激活函數。relu / tanh 帶非線性；sigmoid 兩端梯度易消失；identity = 沒有激活（讓堆疊塌成線性）。",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch.nn as nn

        prev = inputs.get("model")
        if not isinstance(prev, nn.Module):
            raise ValueError(
                "ActivationLayer needs a `model` input — connect it after an FFNLayer."
            )
        fn = str(params.get("function", "relu"))
        act = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity(),
        }.get(fn)
        if act is None:
            raise ValueError(f"ActivationLayer: unknown function {fn!r}.")
        modules = list(prev.children())
        modules.append(act)
        return {"model": nn.Sequential(*modules)}
