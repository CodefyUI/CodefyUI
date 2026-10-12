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

from ._labels import class_labels


class RandomForestClassifierNode(BaseNode):
    NODE_NAME = "RandomForestClassifier"
    CATEGORY = "Classical"
    DESCRIPTION = "Labels each query by a vote of many decision trees"
    DETAILS = (
        "sklearn's RandomForestClassifier: every tree fits a bootstrap resample of "
        "the rows and their votes come out as per-class probabilities, which makes "
        "the boundary smoother than a single decision tree's. n_estimators sets "
        "how many trees, more being steadier and slower, and max_depth 0 leaves "
        "their depth unlimited. Its inputs and predictions port match the other "
        "classical classifiers."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features [N, F]."),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="Training labels (length N)."),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="Query features to classify [M, F]."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.LIST, description="Predicted label for each query."),
            PortDefinition(name="probabilities", data_type=DataType.TENSOR, description="Per-class vote probabilities [M, C]."),
            PortDefinition(name="classes", data_type=DataType.LIST, description="Class labels, sorted, in the column order of probabilities."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="n_estimators",
                param_type=ParamType.INT,
                default=100,
                min_value=1,
                description="Number of trees in the forest. More trees are usually steadier and slower.",
            ),
            ParamDefinition(
                name="max_depth",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description="Maximum depth of each tree. 0 = unlimited.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="Random seed. A fixed seed makes the result reproducible.",
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

        labels = class_labels(y_train, "RandomForestClassifier y_train")
        labels = [str(label) for label in labels]
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
