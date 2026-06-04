"""RandomForestClassifierNode — 隨機森林分類器（sklearn）。

補上 classical 類原本缺的隨機森林：多棵決策樹各自在隨機子集上訓練、預測時投票。
比單棵決策樹平滑、穩定（重跑種子形狀變化小），是決策樹家族最常用的集成版。

介面與其他 classical 分類器一致（`SVMClassifier` / `DecisionTreeClassifier` /
`KNN`）：吃 `x_train, y_train(標籤), x_query`，吐 `predictions` / `probabilities` /
`classes`，可在同一條 pipeline 上直接互換。
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class RandomForestClassifierNode(BaseNode):
    NODE_NAME = "RandomForestClassifier"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "隨機森林分類器（sklearn）：多棵決策樹投票，比單棵樹平滑、穩定。介面與 "
        "SVMClassifier / DecisionTreeClassifier / KNN 一致（吃標籤、吐標籤），可直接互換。"
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
            PortDefinition(name="probabilities", data_type=DataType.TENSOR, description="各類別的投票機率 [M, C]。"),
            PortDefinition(name="classes", data_type=DataType.LIST, description="類別標籤，與 probabilities 欄序一致（排序後）。"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="n_estimators",
                param_type=ParamType.INT,
                default=100,
                min_value=1,
                description="森林裡的樹數量。越多通常越穩、越慢。",
            ),
            ParamDefinition(
                name="max_depth",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="每棵樹的最大深度，0 = 不限。",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="隨機種子，固定後結果可重現。",
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
        from sklearn.ensemble import RandomForestClassifier

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("RandomForestClassifier requires `x_train`, `y_train`, and `x_query` inputs.")

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
                f"RandomForestClassifier: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )

        n_estimators = max(1, int(params.get("n_estimators", 100)))
        depth = int(params.get("max_depth", 0))
        seed = int(params.get("seed", 0))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=(None if depth <= 0 else depth),
            random_state=seed,
        )
        clf.fit(x_train_np, labels)
        predictions = [str(p) for p in clf.predict(x_query_np)]
        proba = clf.predict_proba(x_query_np)
        classes = [str(c) for c in clf.classes_.tolist()]

        return {
            "predictions": predictions,
            "probabilities": torch.from_numpy(np.asarray(proba, dtype=np.float32)).float(),
            "classes": classes,
        }
