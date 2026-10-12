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


from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    PortDefinition,
)

from ._labels import label_strings


class AccuracyNode(BaseNode):
    NODE_NAME = "Accuracy"
    CATEGORY = "Classical"
    DESCRIPTION = "Predictions vs true labels: accuracy, correct, total"
    DETAILS = (
        "Accuracy is correct / total, in [0, 1]. Labels are compared as strings, "
        "so 0 and '0' match; the two lists must be the same length, and two empty "
        "lists give 0. The accuracy is also recorded as the run metric "
        "`accuracy`, so a sweep can rank by it."
    )

    # The accuracy goes out through ``context.log_metric`` as well as the
    # output port, and a cache hit returns the outputs without calling
    # execute(), so the metric would be missing on every rerun (#254, #711).
    cacheable = False

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

        # One label per row, integral floats as ints: '1.0' and '[1.0]' would
        # never equal CSVReader's '1' and the accuracy would read 0.
        preds_list = label_strings(preds, "Accuracy predictions")
        labels_list = label_strings(labels, "Accuracy labels")

        if len(preds_list) != len(labels_list):
            raise ValueError(
                f"Accuracy: length mismatch — predictions {len(preds_list)} vs labels {len(labels_list)}."
            )
        if len(preds_list) == 0:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        preds_s = preds_list
        labels_s = labels_list

        correct = sum(1 for p, y in zip(preds_s, labels_s) if p == y)
        total = len(preds_s)
        accuracy = float(correct) / float(total)
        if context is not None:
            context.log_metric("accuracy", accuracy, 0)
        return {
            "accuracy": accuracy,
            "correct": int(correct),
            "total": int(total),
        }
