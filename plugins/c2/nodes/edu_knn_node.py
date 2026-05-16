"""EduKNNNode — hand-written k-nearest-neighbours classifier.

Three lines of math, three lines of code, one of the most teachable
classifiers in ML:

    distances[i, j] = ||x_query[i] - x_train[j]||   (or |·|, or 1 - cos)
    nearest[i]      = argpartition(distances[i])[:k]
    pred[i]         = mode(y_train[nearest[i]])

Output exposes the intermediate ``distances`` and ``neighbor_indices``
so the visualisation node can draw the lines from each query to its
k nearest training points.

Compare with the production ``KNN`` node (next PR) that wraps
``sklearn.neighbors.KNeighborsClassifier`` — same interface, different
trade-offs:

* ``EduKNN`` runs in O(N_train · N_query) with O(N) memory per query;
  fine for the ≤200-point datasets in lessons.
* ``KNN`` (sklearn) builds a KD-tree / ball-tree, scales to 100k+ rows.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from app.core.step_trace import StepRecorder


class EduKNNNode(BaseNode):
    NODE_NAME = "EduKNN"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Hand-written k-NN classifier. Computes per-query distances to all "
        "training points, picks the k smallest, votes by majority. Outputs "
        "the distances and neighbour indices so a viz can draw lines from "
        "each query to its k nearest training samples."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x_train",
                data_type=DataType.TENSOR,
                description="Training features [N_train, F].",
            ),
            PortDefinition(
                name="y_train",
                data_type=DataType.LIST,
                description="Training labels (length N_train).",
            ),
            PortDefinition(
                name="x_query",
                data_type=DataType.TENSOR,
                description="Query features [N_query, F] to classify.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.LIST,
                description="Predicted label per query (length N_query).",
            ),
            PortDefinition(
                name="distances",
                data_type=DataType.TENSOR,
                description="Top-k distances per query, [N_query, k].",
            ),
            PortDefinition(
                name="neighbor_indices",
                data_type=DataType.LIST,
                description="List of length N_query; each entry is the k nearest training indices (sorted closest-first).",
            ),
            PortDefinition(
                name="train_coords",
                data_type=DataType.TENSOR,
                description="Pass-through of x_train so the viz can plot training points alongside neighbour lines.",
            ),
            PortDefinition(
                name="query_coords",
                data_type=DataType.TENSOR,
                description="Pass-through of x_query so the viz can plot query points.",
            ),
            PortDefinition(
                name="train_labels",
                data_type=DataType.LIST,
                description="Pass-through of y_train so the viz can colour-code training classes.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="k",
                param_type=ParamType.INT,
                default=5,
                min_value=1,
                description="Number of nearest neighbours to consider. Clamped to N_train when too large.",
            ),
            ParamDefinition(
                name="metric",
                param_type=ParamType.SELECT,
                default="euclidean",
                options=["euclidean", "manhattan", "cosine"],
                description="Distance metric. Cosine returns 1 - cos(query, train).",
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
            raise ValueError(
                "EduKNN requires `x_train`, `y_train`, and `x_query` inputs."
            )

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        x_train = x_train.float()
        x_query = x_query.float()
        y_train_list = list(y_train)

        if x_train.shape[0] != len(y_train_list):
            raise ValueError(
                f"EduKNN: x_train and y_train length mismatch — "
                f"{x_train.shape[0]} rows vs {len(y_train_list)} labels."
            )

        metric = str(params.get("metric", "euclidean"))
        k = max(1, min(int(params.get("k", 5)), x_train.shape[0]))

        # Pairwise distance matrix [N_query, N_train]
        if metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            diff = x_query.unsqueeze(1) - x_train.unsqueeze(0)
            distances = torch.linalg.vector_norm(diff, dim=-1)
        elif metric == "manhattan":
            diff = x_query.unsqueeze(1) - x_train.unsqueeze(0)
            distances = torch.sum(torch.abs(diff), dim=-1)
        elif metric == "cosine":
            q_norm = torch.nn.functional.normalize(x_query, dim=-1, eps=1e-12)
            t_norm = torch.nn.functional.normalize(x_train, dim=-1, eps=1e-12)
            distances = 1.0 - q_norm @ t_norm.T
        else:
            raise ValueError(f"EduKNN: unknown metric {metric!r}")

        # Top-k smallest distances (= nearest neighbours)
        topk_d, topk_idx = torch.topk(distances, k=k, dim=1, largest=False)

        # Majority vote per query
        predictions: list[str] = []
        neighbor_indices: list[list[int]] = []
        for i in range(x_query.shape[0]):
            idx_row = topk_idx[i].tolist()
            neighbor_indices.append(idx_row)
            votes = Counter(str(y_train_list[j]) for j in idx_row)
            # Counter.most_common is deterministic given Python's dict order;
            # ties resolved by first-seen.
            predictions.append(votes.most_common(1)[0][0])

        verbose = context is not None and getattr(context, "verbose", False)
        if verbose:
            recorder = StepRecorder()
            recorder.record(
                "compute_distances",
                f"Pairwise {metric} distances [{x_query.shape[0]} queries × {x_train.shape[0]} train].",
                distances=distances,
            )
            recorder.record(
                "topk",
                f"Pick the k={k} nearest neighbours per query.",
                topk_distances=topk_d,
                topk_indices=topk_idx,
            )
            recorder.record(
                "vote",
                "Majority vote among the k nearest labels.",
                scalars={"k": float(k), "n_classes": float(len(set(y_train_list)))},
            )
            return {
                "predictions": predictions,
                "distances": topk_d,
                "neighbor_indices": neighbor_indices,
                "train_coords": x_train,
                "query_coords": x_query,
                "train_labels": y_train_list,
                "__steps__": recorder.steps,
            }

        return {
            "predictions": predictions,
            "distances": topk_d,
            "neighbor_indices": neighbor_indices,
            "train_coords": x_train,
            "query_coords": x_query,
            "train_labels": y_train_list,
        }
