"""EduDecisionTreeNode — from-scratch CART classification tree.

Textbook lesson **I2-2 (CART classification tree)**: instead of calling
``sklearn.tree.DecisionTreeClassifier`` as a black box, build a binary
decision tree by hand and expose, for *every* internal node, the choice
that was made and *why*:

    impurity(node)   = gini  : 1 − Σ_c p_c²
                       entropy: − Σ_c p_c log2 p_c
    gain(feature, t) = impurity(parent)
                       − (n_left/n)·impurity(left)
                       − (n_right/n)·impurity(right)

At each node we scan every feature, try every candidate threshold (the
midpoints between consecutive sorted unique values), and keep the
(feature, threshold) with the largest impurity gain. Left children hold
the samples with ``x[feature] <= threshold``. Recursion stops when the
node hits ``max_depth``, has fewer than ``min_samples_split`` samples,
is already pure, or no split yields positive gain — then it becomes a
leaf predicting the majority class.

Outputs:
- ``predictions`` — predicted integer class label per query, ``[M]``.
- ``tree`` — a JSON-serialisable nested dict describing the learned tree
  (feature / threshold / impurity / gain / sample counts at every node).
- ``node_count`` — scalar count of nodes in the tree (display-only).

Compare with the production ``DecisionTree`` node (sklearn) — same
interface, but this one is fully transparent and deterministic
(stable tie-breaking: lowest feature index, then lowest threshold) so
students can trace a split by hand and get the same answer.
"""

from __future__ import annotations

import math
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


class EduDecisionTreeNode(BaseNode):
    NODE_NAME = "Edu-DecisionTree"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Hand-written CART classification tree (binary splits, Gini or "
        "entropy). For every internal node it exposes the chosen feature, "
        "threshold, impurity before/after the split, and the information "
        "gain, plus the final nested-dict tree structure. Deterministic "
        "tie-breaking (lowest feature index, then lowest threshold)."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="x_train",
                data_type=DataType.TENSOR,
                description="Training features [N, D].",
            ),
            PortDefinition(
                name="y_train",
                data_type=DataType.TENSOR,
                description="Training labels [N] (integer class labels, any number of classes).",
            ),
            PortDefinition(
                name="x_query",
                data_type=DataType.TENSOR,
                description="Query features [M, D] to classify.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.TENSOR,
                description="Predicted integer class label per query, [M].",
            ),
            PortDefinition(
                name="tree",
                data_type=DataType.ANY,
                description=(
                    "JSON-serialisable nested dict. Internal node: "
                    "{feature, threshold, impurity, gain, n_samples, left, right}. "
                    "Leaf: {leaf: true, prediction, n_samples, class_counts}. "
                    "Left = samples with x[feature] <= threshold."
                ),
            ),
            PortDefinition(
                name="node_count",
                data_type=DataType.TENSOR,
                description="Scalar count of nodes in the tree (display-only).",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="max_depth",
                param_type=ParamType.INT,
                default=3,
                min_value=1,
                description="Maximum tree depth. The root is depth 0.",
            ),
            ParamDefinition(
                name="min_samples_split",
                param_type=ParamType.INT,
                default=2,
                min_value=2,
                description="A node with fewer than this many samples becomes a leaf.",
            ),
            ParamDefinition(
                name="criterion",
                param_type=ParamType.SELECT,
                default="gini",
                options=["gini", "entropy"],
                description="Impurity measure. gini = 1 − Σ p²; entropy = − Σ p log2 p.",
            ),
        ]

    # ------------------------------------------------------------------ #
    # Impurity helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _impurity(y: torch.Tensor, criterion: str) -> float:
        """Gini or entropy impurity of a 1-D label tensor."""
        n = y.shape[0]
        if n == 0:
            return 0.0
        counts = torch.bincount(y)
        probs = counts[counts > 0].float() / float(n)
        if criterion == "entropy":
            return float(-(probs * torch.log2(probs)).sum().item())
        # gini
        return float((1.0 - (probs * probs).sum()).item())

    @staticmethod
    def _majority(y: torch.Tensor) -> int:
        """Majority class, ties broken by the lowest label (deterministic)."""
        counts = Counter(int(v) for v in y.tolist())
        best_count = max(counts.values())
        return min(label for label, c in counts.items() if c == best_count)

    @staticmethod
    def _class_counts(y: torch.Tensor) -> dict[str, int]:
        """JSON-serialisable {label: count}; string keys, sorted by label."""
        counts = Counter(int(v) for v in y.tolist())
        return {str(label): counts[label] for label in sorted(counts)}

    def _best_split(
        self, x: torch.Tensor, y: torch.Tensor, parent_impurity: float, criterion: str
    ) -> tuple[int, float, float] | None:
        """Return (feature, threshold, gain) with the largest positive gain.

        Candidate thresholds are midpoints between consecutive sorted unique
        values of each feature. Deterministic tie-breaking: iterate features
        in ascending index, thresholds in ascending value, and only adopt a
        candidate that *strictly* beats the current best — so the lowest
        feature index then lowest threshold wins on ties.
        """
        n = x.shape[0]
        n_features = x.shape[1]
        best_feature: int | None = None
        best_threshold = 0.0
        best_gain = 0.0  # require strictly positive gain to split

        for feature in range(n_features):
            column = x[:, feature]
            uniques = torch.unique(column, sorted=True)
            if uniques.numel() < 2:
                continue  # constant feature → no split possible
            midpoints = (uniques[:-1] + uniques[1:]) / 2.0
            for thr in midpoints.tolist():
                left_mask = column <= thr
                n_left = int(left_mask.sum().item())
                n_right = n - n_left
                if n_left == 0 or n_right == 0:
                    continue
                left_imp = self._impurity(y[left_mask], criterion)
                right_imp = self._impurity(y[~left_mask], criterion)
                weighted = (n_left / n) * left_imp + (n_right / n) * right_imp
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(thr)

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------ #
    # Tree building / prediction
    # ------------------------------------------------------------------ #
    def _build(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        depth: int,
        max_depth: int,
        min_samples_split: int,
        criterion: str,
        splits: list[dict[str, float]],
    ) -> dict[str, Any]:
        """Recursively build a node. Records each internal split in build order."""
        n = x.shape[0]
        impurity = self._impurity(y, criterion)

        def make_leaf() -> dict[str, Any]:
            return {
                "leaf": True,
                "prediction": self._majority(y),
                "n_samples": int(n),
                "class_counts": self._class_counts(y),
            }

        # Stopping conditions → leaf.
        if (
            depth >= max_depth
            or n < min_samples_split
            or impurity == 0.0  # pure node
        ):
            return make_leaf()

        split = self._best_split(x, y, impurity, criterion)
        if split is None:  # no positive-gain split
            return make_leaf()

        feature, threshold, gain = split
        left_mask = x[:, feature] <= threshold

        # Record this internal split in build (pre-order) order.
        splits.append(
            {
                "depth": float(depth),
                "feature": float(feature),
                "threshold": float(threshold),
                "impurity_before": float(impurity),
                "impurity_after": float(impurity - gain),
                "gain": float(gain),
                "n_samples": float(n),
            }
        )

        left = self._build(
            x[left_mask], y[left_mask], depth + 1, max_depth,
            min_samples_split, criterion, splits,
        )
        right = self._build(
            x[~left_mask], y[~left_mask], depth + 1, max_depth,
            min_samples_split, criterion, splits,
        )
        return {
            "feature": int(feature),
            "threshold": float(threshold),
            "impurity": float(impurity),
            "gain": float(gain),
            "n_samples": int(n),
            "left": left,
            "right": right,
        }

    @staticmethod
    def _count_nodes(node: dict[str, Any]) -> int:
        if node.get("leaf"):
            return 1
        return 1 + EduDecisionTreeNode._count_nodes(node["left"]) + EduDecisionTreeNode._count_nodes(node["right"])

    @staticmethod
    def _tree_depth(node: dict[str, Any]) -> int:
        """Number of edges on the longest root-to-leaf path (a single leaf = 0)."""
        if node.get("leaf"):
            return 0
        return 1 + max(
            EduDecisionTreeNode._tree_depth(node["left"]),
            EduDecisionTreeNode._tree_depth(node["right"]),
        )

    @staticmethod
    def _predict_one(node: dict[str, Any], row: torch.Tensor) -> int:
        while not node.get("leaf"):
            if float(row[node["feature"]].item()) <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return int(node["prediction"])

    # ------------------------------------------------------------------ #
    # execute
    # ------------------------------------------------------------------ #
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
                "EduDecisionTree requires `x_train`, `y_train`, and `x_query` inputs."
            )

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(x_train, dtype=torch.float32)
        if not isinstance(x_query, torch.Tensor):
            x_query = torch.as_tensor(x_query, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.as_tensor(y_train, dtype=torch.long)
        x_train = x_train.float()
        x_query = x_query.float()
        y_train = y_train.long().reshape(-1)

        # --- validation -------------------------------------------------
        if x_train.ndim != 2:
            raise ValueError(
                f"EduDecisionTree: x_train must be 2-D [N, D]; got shape {tuple(x_train.shape)}."
            )
        if x_query.ndim != 2:
            raise ValueError(
                f"EduDecisionTree: x_query must be 2-D [M, D]; got shape {tuple(x_query.shape)}."
            )
        n_train, n_features = x_train.shape
        if n_train == 0:
            raise ValueError("EduDecisionTree: training set is empty.")
        if y_train.shape[0] != n_train:
            raise ValueError(
                f"EduDecisionTree: x_train and y_train length mismatch — "
                f"{n_train} rows vs {y_train.shape[0]} labels."
            )
        if x_query.shape[1] != n_features:
            raise ValueError(
                f"EduDecisionTree: x_query has {x_query.shape[1]} features but "
                f"x_train has {n_features}."
            )
        if (y_train < 0).any():
            raise ValueError(
                "EduDecisionTree: y_train must contain non-negative integer class labels."
            )

        max_depth = int(params.get("max_depth", 3))
        min_samples_split = int(params.get("min_samples_split", 2))
        criterion = str(params.get("criterion", "gini"))
        if max_depth < 1:
            raise ValueError(f"EduDecisionTree: max_depth must be >= 1; got {max_depth}.")
        if min_samples_split < 2:
            raise ValueError(
                f"EduDecisionTree: min_samples_split must be >= 2; got {min_samples_split}."
            )
        if criterion not in ("gini", "entropy"):
            raise ValueError(
                f"EduDecisionTree: unknown criterion {criterion!r} (expected 'gini' or 'entropy')."
            )

        # --- build ------------------------------------------------------
        splits: list[dict[str, float]] = []
        tree = self._build(
            x_train, y_train, 0, max_depth, min_samples_split, criterion, splits
        )

        node_count = self._count_nodes(tree)
        depth = self._tree_depth(tree)

        # --- predict ----------------------------------------------------
        predictions = torch.tensor(
            [self._predict_one(tree, x_query[i]) for i in range(x_query.shape[0])],
            dtype=torch.long,
        )

        verbose = context is not None and getattr(context, "verbose", False)
        recorder = StepRecorder() if verbose else None
        if recorder is not None:
            # One step per internal split, in build (pre-order) order.
            for rec in splits:
                d = int(rec["depth"])
                recorder.record(
                    f"split_d{d}",
                    (
                        f"depth {d}: split on feature {int(rec['feature'])} "
                        f"at x <= {rec['threshold']:.4g} "
                        f"(gain {rec['gain']:.4g}, {criterion})."
                    ),
                    scalars={
                        "feature": rec["feature"],
                        "threshold": rec["threshold"],
                        "impurity_before": rec["impurity_before"],
                        "impurity_after": rec["impurity_after"],
                        "gain": rec["gain"],
                        "n_samples": rec["n_samples"],
                    },
                )
            recorder.record(
                "tree",
                f"Built a {criterion} tree with {node_count} nodes, depth {depth}.",
                scalars={"node_count": float(node_count), "depth": float(depth)},
            )

        result: dict[str, Any] = {
            "predictions": predictions,
            "tree": tree,
            "node_count": torch.tensor(node_count, dtype=torch.long),
        }
        if recorder is not None and recorder.steps:
            result["__steps__"] = recorder.steps
        return result
