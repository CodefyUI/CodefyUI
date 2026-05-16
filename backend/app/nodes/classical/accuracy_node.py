"""AccuracyNode — classification accuracy between predictions and ground truth.

Closes the typical classification pipeline:

    SyntheticDataset → TrainTestSplit → SomeClassifier → Accuracy → Print

so a graph can land on a single number ("50%") instead of asking the
student to eyeball the predictions list. Used by every C2 chapter
example, where the failure of linear methods on concentric circles
becomes visible the moment Accuracy reports ≈ 0.5.
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


class AccuracyNode(BaseNode):
    NODE_NAME = "Accuracy"
    CATEGORY = "Classical"
    DESCRIPTION = (
        "Compute classification accuracy between a predictions list and a "
        "ground-truth labels list. Outputs the accuracy as a float (0–1), "
        "the count of correct predictions, and the total count."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="predictions",
                data_type=DataType.LIST,
                description="Predicted labels (length N).",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Ground-truth labels (length N).",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="accuracy",
                data_type=DataType.SCALAR,
                description="Accuracy in [0, 1].",
            ),
            PortDefinition(
                name="correct",
                data_type=DataType.SCALAR,
                description="Number of correct predictions.",
            ),
            PortDefinition(
                name="total",
                data_type=DataType.SCALAR,
                description="Number of samples compared.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return []

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        preds = inputs.get("predictions")
        labels = inputs.get("labels")
        if preds is None or labels is None:
            raise ValueError("Accuracy requires `predictions` and `labels` inputs.")

        preds_list = preds.tolist() if isinstance(preds, torch.Tensor) else list(preds)
        labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)

        if len(preds_list) != len(labels_list):
            raise ValueError(
                f"Accuracy: length mismatch — predictions {len(preds_list)} vs labels {len(labels_list)}."
            )
        if len(preds_list) == 0:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        # Normalize to strings so '0' vs 0 mismatches don't sink the comparison.
        preds_s = [str(p) for p in preds_list]
        labels_s = [str(l) for l in labels_list]

        correct = sum(1 for p, y in zip(preds_s, labels_s) if p == y)
        total = len(preds_s)
        return {
            "accuracy": float(correct) / float(total),
            "correct": int(correct),
            "total": int(total),
        }
