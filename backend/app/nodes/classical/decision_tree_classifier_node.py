"""DecisionTreeClassifierNode — CART decision tree via sklearn.

Recursively splits the feature space by maximising information gain (or Gini
purity) at each node. Outputs:

- ``predictions``: per-query class label.
- ``feature_importances``: how much each feature contributed to the splits.
- ``tree_text``: sklearn's human-readable rule dump — the killer feature
  for teaching (you can literally read off the rules).
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


class DecisionTreeClassifierNode(BaseNode):
    NODE_NAME = "DecisionTreeClassifier"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "CART decision tree (sklearn). Recursively splits features to "
        "maximise purity; the killer feature for teaching is tree_text — a "
        "readable dump of the learned if/else rules."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features [N, F]."),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="Training labels (length N)."),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="Query features [M, F]."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.LIST, description="Predicted label per query."),
            PortDefinition(name="feature_importances", data_type=DataType.TENSOR, description="Importance score per feature [F], summing to 1."),
            PortDefinition(name="tree_text", data_type=DataType.STRING, description="Human-readable dump of the learned rules."),
            PortDefinition(name="classes", data_type=DataType.LIST, description="Class labels in sorted order."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="max_depth",
                param_type=ParamType.INT,
                default=5,
                min_value=0,
                description="Maximum tree depth. 0 = no limit (grow until pure).",
            ),
            ParamDefinition(
                name="criterion",
                param_type=ParamType.SELECT,
                default="gini",
                options=["gini", "entropy", "log_loss"],
                description="Split quality function: gini impurity, entropy, or log loss.",
            ),
            ParamDefinition(
                name="random_state",
                param_type=ParamType.INT,
                default=42,
                description="Seed for tie-breaking; controls reproducibility on equal-quality splits.",
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
        from sklearn.tree import DecisionTreeClassifier, export_text

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError(
                "DecisionTreeClassifier requires `x_train`, `y_train`, and `x_query` inputs."
            )

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)

        x_train_np = x_train.detach().cpu().float().numpy()
        x_query_np = x_query.detach().cpu().float().numpy()

        labels = y_train.tolist() if isinstance(y_train, torch.Tensor) else list(y_train)
        labels = [str(v) for v in labels]
        if x_train_np.shape[0] != len(labels):
            raise ValueError(
                f"DecisionTreeClassifier: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )

        max_depth_raw = int(params.get("max_depth", 5))
        max_depth = None if max_depth_raw <= 0 else max_depth_raw
        criterion = str(params.get("criterion", "gini"))
        random_state = int(params.get("random_state", 42))

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
        )
        clf.fit(x_train_np, labels)

        preds = clf.predict(x_query_np).tolist()
        classes = [str(c) for c in clf.classes_.tolist()]
        tree_text = export_text(clf)

        return {
            "predictions": [str(p) for p in preds],
            "feature_importances": torch.from_numpy(clf.feature_importances_).float(),
            "tree_text": tree_text,
            "classes": classes,
        }
