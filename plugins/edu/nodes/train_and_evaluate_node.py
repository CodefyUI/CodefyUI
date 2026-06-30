"""TrainAndEvaluateNode — 把你堆好的 MLP 拿來訓練 + 預測（I2-3 的「result」節點）。

對應教材 **I2-3**：`FFNLayer` / `ActivationLayer` 串出來的只是一個**還沒訓練**的網路
（隨機權重，直接用會 ~50%）。這顆把它接過來，在訓練資料上跑梯度下降把它**練到會**，
再對查詢資料預測。它把通用訓練流（Loss + Optimizer + 訓練迴圈）包在一顆裡，畫布上
不用拉那一整坨。

介面刻意跟 I2-1 的 `Classifier`、I2-2 的 `AdvancedClassifier` 一致——吐 `predictions`
（→ Accuracy）和 `model`（→ DecisionBoundary），所以 I2-3 跟前兩章用的是同一條 pipeline，
只是中間那塊從「現成分類器」換成「你堆的網路 + 這顆訓練節點」。

自動補一個輸出層：你的 FFNLayer 都當隱藏層，這顆會在尾端補一個線性層壓到「類別數」，
用 CrossEntropy 訓練（多類、二類都通用）。
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


class _FittedNet:
    """包住訓練好的網路，提供統一的 ``predict(X) -> 標籤list`` 給 DecisionBoundary。"""

    def __init__(self, net: Any, classes: list[str]) -> None:
        self.net = net
        self.classes = list(classes)

    def predict(self, x: Any) -> list[str]:
        import numpy as np
        import torch

        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.as_tensor(np.asarray(x, dtype="float32")))
            idx = logits.argmax(dim=1).tolist()
        return [self.classes[int(i)] for i in idx]


class TrainAndEvaluateNode(BaseNode):
    NODE_NAME = "TrainAndEvaluate"
    CATEGORY = "EDU"
    DESCRIPTION = (
        "把 FFNLayer / ActivationLayer 堆好的網路拿來訓練 + 預測。在訓練資料上跑梯度下降"
        "把網路練到會，再對 x_query 預測。介面跟 Classifier / AdvancedClassifier 一致："
        "吐 predictions(→Accuracy) 和 model(→DecisionBoundary)。輸出層自動補、用 CrossEntropy 訓練。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL, description="FFNLayer / ActivationLayer 串好的網路（隱藏層堆疊）。"),
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="訓練特徵 [N, F]。"),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="訓練標籤（長度 N）。"),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="要分類的查詢特徵 [M, F]。"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.LIST, description="每筆查詢的預測標籤。"),
            PortDefinition(name="classes", data_type=DataType.LIST, description="類別標籤（排序後）。"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="訓練好的網路包裝，接給 DecisionBoundary 畫邊界。"),
            PortDefinition(name="losses", data_type=DataType.TENSOR, description="每個 epoch 的訓練 loss（可接 Visualize 看收斂）。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="epochs",
                param_type=ParamType.INT,
                default=400,
                min_value=1,
                description="訓練輪數。越多通常越收斂、越慢。",
            ),
            ParamDefinition(
                name="lr",
                param_type=ParamType.FLOAT,
                default=0.05,
                min_value=0.0,
                description="學習率（梯度下降的步伐大小）。",
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
        import torch
        import torch.nn as nn

        net_in = inputs.get("model")
        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if net_in is None or not isinstance(net_in, nn.Module):
            raise ValueError(
                "TrainAndEvaluate needs a `model` input — connect your FFNLayer / ActivationLayer stack."
            )
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("TrainAndEvaluate requires `x_train`, `y_train`, and `x_query` inputs.")

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        x_train = x_train.float()
        x_query = x_query.float()

        labels = y_train.tolist() if isinstance(y_train, torch.Tensor) else list(y_train)
        labels = [str(l) for l in labels]
        classes = sorted(set(labels))
        idx = {c: i for i, c in enumerate(classes)}
        y_idx = torch.tensor([idx[l] for l in labels], dtype=torch.long)

        # 把隱藏堆疊接上一個輸出層（壓到類別數）。輸入維度從堆疊最後一個 Linear 推得。
        modules = list(net_in.children())
        last_out = None
        for m in modules:
            if isinstance(m, nn.Linear):
                last_out = m.out_features
        if last_out is None:
            raise ValueError(
                "TrainAndEvaluate: the stacked model has no Linear layer — add at least one FFNLayer."
            )
        modules.append(nn.Linear(last_out, len(classes)))
        net = nn.Sequential(*modules)

        epochs = max(1, int(params.get("epochs", 400)))
        lr = float(params.get("lr", 0.05))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        losses: list[float] = []
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = net(x_train)
            loss = loss_fn(out, y_idx)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        net.eval()
        with torch.no_grad():
            pred_idx = net(x_query).argmax(dim=1).tolist()
        predictions = [classes[int(i)] for i in pred_idx]

        return {
            "predictions": predictions,
            "classes": classes,
            "model": _FittedNet(net, classes),
            "losses": torch.tensor(losses, dtype=torch.float32),
        }
