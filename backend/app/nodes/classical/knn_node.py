"""KNNNode — production k-NN classifier wrapping ``sklearn.neighbors``.

Same interface as :class:`EduKNNNode`, different implementation:

- ``EduKNN``: hand-written O(N_train · N_query) brute force, three-line math,
  exposes intermediate distances/indices for visualisation. Right for ≤ 200-row
  teaching datasets.
- ``KNN`` (this node): wraps ``sklearn.neighbors.KNeighborsClassifier``. Builds
  a KD-tree / ball-tree internally, scales to 100k+ rows, supports the full
  ``weights``/``metric`` knob set. Right for "I want to actually use this on
  real data."

The pair lets the textbook teach the algorithm with EduKNN, then swap in KNN
without rewiring the graph — a recurring theme of the dual-track design.
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


class KNNNode(BaseNode):
    NODE_NAME = "KNN"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "k-Nearest-Neighbours classifier (sklearn). Drop-in replacement for "
        "EduKNN with KD-tree indexing, distance weighting, and a choice of "
        "metrics. Use this when you want production scaling; use EduKNN when "
        "you want to see the math."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features [N, F]."),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="Training labels (length N)."),
            PortDefinition(name="x_query", data_type=DataType.TENSOR, description="Query features [M, F] to classify."),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="predictions", data_type=DataType.LIST, description="Predicted label per query."),
            PortDefinition(name="probabilities", data_type=DataType.TENSOR, description="Class probabilities [M, C] from the k-NN vote."),
            PortDefinition(name="classes", data_type=DataType.LIST, description="Class labels in column order (sklearn-sorted)."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(name="n_neighbors", param_type=ParamType.INT, default=5, min_value=1, description="Number of neighbours k."),
            ParamDefinition(
                name="weights",
                param_type=ParamType.SELECT,
                default="uniform",
                options=["uniform", "distance"],
                description="Vote weighting: uniform = each neighbour counts the same; distance = closer neighbours count more.",
            ),
            ParamDefinition(
                name="metric",
                param_type=ParamType.SELECT,
                default="minkowski",
                options=["minkowski", "euclidean", "manhattan", "chebyshev", "cosine"],
                description="Distance metric. minkowski with default p=2 == euclidean.",
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
        from sklearn.neighbors import KNeighborsClassifier

        x_train = inputs.get("x_train")
        y_train = inputs.get("y_train")
        x_query = inputs.get("x_query")
        if x_train is None or y_train is None or x_query is None:
            raise ValueError("KNN requires `x_train`, `y_train`, and `x_query` inputs.")

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        x_train_np = x_train.detach().cpu().float().numpy()
        x_query_np = x_query.detach().cpu().float().numpy()

        labels = y_train.tolist() if isinstance(y_train, torch.Tensor) else list(y_train)
        if x_train_np.shape[0] != len(labels):
            raise ValueError(
                f"KNN: features and labels length mismatch — "
                f"{x_train_np.shape[0]} rows vs {len(labels)} labels."
            )

        n_neighbors = max(1, min(int(params.get("n_neighbors", 5)), x_train_np.shape[0]))
        weights = str(params.get("weights", "uniform"))
        metric = str(params.get("metric", "minkowski"))

        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )
        clf.fit(x_train_np, labels)
        preds = clf.predict(x_query_np).tolist()
        proba = clf.predict_proba(x_query_np)
        classes = [str(c) for c in clf.classes_.tolist()]

        return {
            "predictions": [str(p) for p in preds],
            "probabilities": torch.from_numpy(proba).float(),
            "classes": classes,
        }
