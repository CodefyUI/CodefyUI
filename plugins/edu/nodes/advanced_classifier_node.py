"""AdvancedClassifierNode — 一顆可切換 SVM / 決策樹 / 隨機森林的進階分類器。

對應教材 **I2-2**：C2-3 的兩種「比線性聰明」的世界觀——SVM（升維後一刀切，
邊界是平滑曲線）與決策樹（軸對齊切很多刀，邊界是階梯方框），再加上決策樹的
集成版隨機森林。跟 I2-1 的 `Classifier`（knn / linear / logistic）刻意分成兩顆，
這樣 I2-2 的圖一眼就跟 I2-1 不同。

介面跟 `Classifier` 一致：吃 `x_train, y_train(標籤), x_query`，吐
`predictions(標籤)` 與 `model`。`model` 餵給 `DecisionBoundary` 就能畫邊界；
SVM 的 model 還帶著支持向量座標，搭配 DecisionBoundary 的「標出支持向量」選配
就能把決定邊界的那幾顆點圈出來。

三種 kind：
* svm  — sklearn SVC（kernel 選 rbf / linear / poly；rbf 解同心圓、linear 退回直線）
* tree — sklearn DecisionTreeClassifier（max_depth 控制切幾層，0=不限；太深會過擬合）
* rf   — sklearn RandomForestClassifier（n_estimators 棵樹投票，比單棵穩）
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


class _FittedAdvanced:
    """包住已訓練的進階分類器，統一 ``predict(X) -> 標籤list``，並在 SVM 時帶出支持向量。"""

    def __init__(self, estimator: Any, classes: list[str]) -> None:
        self.estimator = estimator
        self.classes = list(classes)
        # SVC 才有 support_vectors_；其餘分類器是 None，DecisionBoundary 會自動忽略。
        sv = getattr(estimator, "support_vectors_", None)
        self.support_vectors = None if sv is None else sv

    def predict(self, x: Any) -> list[str]:
        import numpy as np

        return [str(p) for p in self.estimator.predict(np.asarray(x, dtype=float))]


class AdvancedClassifierNode(BaseNode):
    NODE_NAME = "AdvancedClassifier"
    CATEGORY = "EDU"
    DESCRIPTION = (
        "一顆可切換的進階分類器：kind 選 svm / tree / rf。吃 x_train,y_train(標籤),x_query "
        "→ predictions(標籤)，另吐 model 給 DecisionBoundary 畫邊界（SVM 的 model 還帶支持"
        "向量）。對應 I2-2 SVM 的 kernel 與決策樹 / 隨機森林。"
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
            PortDefinition(name="classes", data_type=DataType.LIST, description="類別標籤（排序後）。"),
            PortDefinition(name="model", data_type=DataType.MODEL, description="訓練好的分類器包裝，接給 DecisionBoundary 畫邊界。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="kind",
                param_type=ParamType.SELECT,
                default="SVM",
                options=["SVM", "Decision Tree", "Random Forest"],
                description="進階分類器種類：SVM 支持向量機、Decision Tree 決策樹、Random Forest 隨機森林。",
            ),
            ParamDefinition(
                name="kernel",
                param_type=ParamType.SELECT,
                default="rbf",
                options=["rbf", "linear", "poly"],
                description="SVM 的 kernel（只有 kind=SVM 用到）。rbf 能解同心圓、linear 退回直線。",
                visible_when={"kind": "SVM"},
            ),
            ParamDefinition(
                name="max_depth",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="決策樹最大深度，0 = 不限（只有 kind=Decision Tree 用到）。太深會過擬合。",
                visible_when={"kind": "Decision Tree"},
            ),
            ParamDefinition(
                name="n_estimators",
                param_type=ParamType.INT,
                default=50,
                min_value=1,
                description="隨機森林的樹數量（只有 kind=Random Forest 用到）。越多通常越穩、越慢。",
                visible_when={"kind": "Random Forest"},
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
        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("AdvancedClassifier requires `x_train`, `y_train`, and `x_query` inputs.")

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
                f"AdvancedClassifier: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )

        kind = str(params.get("kind", "SVM"))
        classes = sorted(set(labels))

        if kind == "SVM":
            from sklearn.svm import SVC

            kernel = str(params.get("kernel", "rbf"))
            est = SVC(kernel=kernel)
        elif kind == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier

            depth = int(params.get("max_depth", 0))
            est = DecisionTreeClassifier(max_depth=(None if depth <= 0 else depth), random_state=0)
        elif kind == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier

            n = max(1, int(params.get("n_estimators", 50)))
            est = RandomForestClassifier(n_estimators=n, random_state=0)
        else:
            raise ValueError(
                f"AdvancedClassifier: unknown kind {kind!r}. Choose 'SVM' / 'Decision Tree' / 'Random Forest'."
            )

        est.fit(x_train_np, labels)
        predictions = [str(p) for p in est.predict(x_query_np)]
        model = _FittedAdvanced(est, classes)

        return {
            "predictions": predictions,
            "classes": classes,
            "model": model,
        }
