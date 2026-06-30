"""FFNLayerNode — 一個全連接（線性）層，串成多層 MLP 的積木。

對應教材 **I2-3**：C2-4 說「神經網路 = 把線性單元堆很多層、層之間夾非線性」。這顆
就是那個「線性」積木——一個全連接層。把它跟 `ActivationLayer` 交錯串起來，就堆出
一個多層 MLP，最後接 `TrainAndEvaluate` 訓練。

它**傳的不是 tensor、而是「正在組裝的網路」**（一個 nn.Sequential）：每接一顆
FFNLayer 就往網路尾端加一個 `nn.Linear(in_features, out_features)`。輸入維度會**自動
從上一層的輸出維度推得**，所以你通常只要設 `out_features`；只有第一顆（沒有上游）
要在 `in_features` 指定輸入維度（2D 玩具資料就設 2）。

每層參數量 = `out_features x in_features + out_features`，節點面板會直接顯示——這正是
C2-5 要你會算的東西。`out_features` 就是這一層的 hidden size，越大表達力越強、參數越多。
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


class FFNLayerNode(BaseNode):
    NODE_NAME = "FFNLayer"
    CATEGORY = "EDU"
    DESCRIPTION = (
        "一個全連接（線性）層，串成多層 MLP 的積木。傳的是「正在組裝的網路」：每接一顆"
        "就往尾端加一個 nn.Linear(in_features, out_features)，輸入維度自動從上一層推得"
        "（第一顆用 in_features 參數）。跟 ActivationLayer 交錯串、最後接 TrainAndEvaluate。"
        "面板會顯示每層參數量 = out_features x in_features + out_features。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="model",
                data_type=DataType.MODEL,
                description="上游正在組裝的網路；第一顆 FFNLayer 不用接，會自己開一個新網路。",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="尾端加上一個線性層後的網路。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="in_features",
                param_type=ParamType.INT,
                default=2,
                min_value=1,
                description="輸入維度。只有第一顆 FFNLayer 會用到（2D 資料設 2）；接在別層後面時自動從上一層推得、此值忽略。",
            ),
            ParamDefinition(
                name="out_features",
                param_type=ParamType.INT,
                default=16,
                min_value=1,
                description="這一層的輸出維度（hidden size）。越大表達力越強、參數越多。",
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
        modules = list(prev.children()) if isinstance(prev, nn.Module) else []
        # 輸入維度：從上一層的最後一個 Linear 推得；第一顆沒有上游就用 in_features 參數。
        in_features = None
        for m in modules:
            if isinstance(m, nn.Linear):
                in_features = m.out_features
        if in_features is None:
            in_features = max(1, int(params.get("in_features", 2)))
        out_features = max(1, int(params.get("out_features", 16)))
        modules.append(nn.Linear(in_features, out_features))
        return {"model": nn.Sequential(*modules)}
