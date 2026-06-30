"""ClassifierNode — 一顆可切換 KNN / 線性 / 邏輯斯的分類器。

對應教材 **I2-1**：C2-2 的核心是「很多分類器共用同一條 pipeline，只差中間那塊」。
這顆把三種經典分類器收成**一顆節點 + 一個下拉選單**——`kind` 選 knn / linear /
logistic，行為就跟對應的單獨節點一模一樣。學生只要改下拉、不必重拉節點，就能驗證
「只換中間這顆、其餘不動」。

介面：吃 `x_train, y_train(標籤), x_query`，吐 `predictions(標籤)`。另外吐一個
`model` 輸出（訓練好的分類器包裝），餵給 `DecisionBoundary` 就能畫出「這一顆」的
決策邊界——所以邊界節點接在分類器後面、換 kind 邊界也跟著換。

三種 kind：
* knn      — sklearn KNeighborsClassifier（吃 k 個鄰居投票）
* logistic — sklearn LogisticRegression（內積 + sigmoid）
* linear   — 線性回歸當分類器（對 one-hot 標籤做最小平方 + argmax）
"""

from __future__ import annotations

from typing import Any

import torch

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class _FittedClassifier:
    """包住已訓練的分類器，提供統一的 ``predict(X) -> 標籤list``。

    給 ``DecisionBoundary`` 用：不管底層是 knn / logistic（sklearn 直接吐標籤）
    還是 linear（one-hot 回歸 + argmax），對外都是「吃 [M, F]、吐 M 個標籤」。
    """

    def __init__(self, kind: str, estimator: Any, classes: list[str]) -> None:
        self.kind = kind
        self.estimator = estimator
        self.classes = list(classes)

    def predict(self, x: Any) -> list[str]:
        import numpy as np

        feats = np.asarray(x, dtype=float)
        if self.kind == "linear":
            scores = np.atleast_2d(self.estimator.predict(feats))
            idx = scores.argmax(axis=1)
            return [self.classes[int(i)] for i in idx]
        return [str(p) for p in self.estimator.predict(feats)]


class ClassifierNode(BaseNode):
    NODE_NAME = "Classifier"
    CATEGORY = "EDU"
    DESCRIPTION = (
        "一顆可切換的分類器：kind 選 knn / linear / logistic，行為就跟對應的單獨節點"
        "一樣（吃 x_train,y_train(標籤),x_query → predictions(標籤)）。另吐 model 給"
        "DecisionBoundary 畫邊界。對應 I2-1「只換中間這顆、其餘不動」。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="訓練特徵 [N, F]。"),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="訓練標籤（長度 N）。"),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="要分類的查詢特徵 [M, F]。"),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.LIST, description="每筆查詢的預測標籤。"),
            PortDefinition(name="probabilities", data_type=DataType.TENSOR, description="每類分數 [M, C]（knn/logistic 是機率，linear 是回歸分數）。"),
            PortDefinition(name="classes", data_type=DataType.LIST, description="類別標籤，與 probabilities 欄序一致（排序後）。"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="訓練好的分類器包裝，接給 DecisionBoundary 畫邊界。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="kind",
                param_type=ParamType.SELECT,
                default="knn",
                options=["knn", "linear", "logistic"],
                description="要用哪一種分類器：knn 鄰居投票、linear 線性回歸當分類器、logistic 邏輯斯回歸。",
            ),
            ParamDefinition(
                name="n_neighbors",
                param_type=ParamType.INT,
                default=5,
                min_value=1,
                description="KNN 的鄰居數 k（只有 kind=knn 時用到）。",
                visible_when={"kind": "knn"},
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
        import numpy as np

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("Classifier requires `x_train`, `y_train`, and `x_query` inputs.")

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        x_train_np = x_train.detach().cpu().float().numpy()
        x_query_np = x_query.detach().cpu().float().numpy()

        labels = y_train.tolist() if isinstance(y_train, torch.Tensor) else list(y_train)
        labels = [str(l) for l in labels]
        if x_train_np.shape[0] != len(labels):
            raise ValueError(
                f"Classifier: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )

        kind = str(params.get("kind", "knn"))
        classes = sorted(set(labels))

        if kind == "knn":
            from sklearn.neighbors import KNeighborsClassifier

            k = max(1, min(int(params.get("n_neighbors", 5)), len(labels)))
            est = KNeighborsClassifier(n_neighbors=k)
            est.fit(x_train_np, labels)
            predictions = [str(p) for p in est.predict(x_query_np)]
            proba = est.predict_proba(x_query_np)
        elif kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            est = LogisticRegression(max_iter=1000)
            est.fit(x_train_np, labels)
            predictions = [str(p) for p in est.predict(x_query_np)]
            proba = est.predict_proba(x_query_np)
        elif kind == "linear":
            from sklearn.linear_model import LinearRegression

            one_hot = np.zeros((len(labels), len(classes)), dtype=np.float32)
            idx = {c: i for i, c in enumerate(classes)}
            for r, lab in enumerate(labels):
                one_hot[r, idx[lab]] = 1.0
            est = LinearRegression()
            est.fit(x_train_np, one_hot)
            proba = np.atleast_2d(est.predict(x_query_np))
            predictions = [classes[int(i)] for i in proba.argmax(axis=1)]
        else:
            raise ValueError(f"Classifier: unknown kind {kind!r}. Choose knn / linear / logistic.")

        model = _FittedClassifier(kind, est, classes)
        return {
            "predictions": [str(p) for p in predictions],
            "probabilities": torch.from_numpy(np.asarray(proba, dtype=np.float32)).float(),
            "classes": classes,
            "model": model,
        }
