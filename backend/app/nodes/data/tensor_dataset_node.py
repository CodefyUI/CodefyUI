"""TensorDatasetNode — pair a feature tensor with its targets as a DATASET.

The bridge from tabular data to the neural-network side (#712):

    CSVReader → TrainTestSplit → TensorDataset → DataLoader → TrainingLoop

Without it nothing turns tensors into a DATASET, so a CSV could feed the
classical nodes but never an MLP.

``target_type`` decides what the targets become, because the two loss
families want different things:

* ``number`` — float32. A 1-D target becomes ``[N, 1]``, the shape a
  ``Linear(..., 1)`` head outputs, so ``MSELoss`` compares it element by
  element. Left at ``[N]`` it would broadcast against ``[B, 1]`` into a
  ``[B, B]`` matrix and train on a meaningless number without an error.
* ``class`` — int64 class indices for ``CrossEntropyLoss``. String labels
  are numbered in sorted order (numerically when every label is a number),
  and the order is returned on ``classes`` so a prediction can be read back.
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


def _class_order(labels: list[str]) -> list[str]:
    """Sorted unique labels; numeric ones by value, so '10' follows '9'."""
    unique = set(labels)
    try:
        return sorted(unique, key=float)
    except ValueError:
        return sorted(unique)


def _number_targets(targets: Any) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        values = targets.detach().cpu().to(torch.float32)
    else:
        try:
            values = torch.tensor([float(v) for v in targets], dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "TensorDataset: target_type is 'number' but the targets are not "
                f"all numbers ({exc}). Set target_type to 'class' for class "
                "labels, or set CSVReader's target_type to 'number'."
            ) from exc
    if values.ndim == 1:
        values = values.unsqueeze(1)
    return values


def _class_targets(targets: Any) -> tuple[torch.Tensor, list[str]]:
    if isinstance(targets, torch.Tensor) and not torch.is_floating_point(targets):
        # Already class indices: kept as they are, so index i names class i.
        values = targets.detach().cpu().to(torch.long).reshape(-1)
        count = int(values.max()) + 1 if values.numel() else 0
        return values, [str(i) for i in range(count)]
    raw = targets.tolist() if isinstance(targets, torch.Tensor) else list(targets)
    labels = [str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
              for v in raw]
    classes = _class_order(labels)
    index = {label: i for i, label in enumerate(classes)}
    return torch.tensor([index[label] for label in labels], dtype=torch.long), classes


class TensorDatasetNode(BaseNode):
    NODE_NAME = "TensorDataset"
    CATEGORY = "Data"
    DESCRIPTION = "Pair a feature tensor with its targets as a dataset"
    DETAILS = (
        "Wraps torch.utils.data.TensorDataset, so DataLoader and TrainingLoop "
        "can train on tensors, such as CSVReader → TrainTestSplit output. With "
        "target_type `number` the targets become float32, and a 1-D target "
        "becomes [N, 1] to match a model with one output, so MSELoss compares "
        "element by element. With `class` they become int64 class indices for "
        "CrossEntropyLoss: labels are numbered in sorted order, numerically "
        "when every label is a number, and `classes` lists that order; an "
        "integer tensor is taken as class indices already. "
        "Features and targets must have the same number of rows."
    )

    # Host-side: the dataset is handed to DataLoader, and TrainingLoop moves
    # each batch to the run's device itself.
    align_inputs = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="features",
                data_type=DataType.TENSOR,
                description="[N, F] feature tensor, e.g. TrainTestSplit's x_train.",
            ),
            PortDefinition(
                name="targets",
                data_type=DataType.ANY,
                description="N targets: a LIST (e.g. TrainTestSplit's y_train) or a TENSOR.",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="(features, target) pairs for DataLoader.",
            ),
            PortDefinition(
                name="classes",
                data_type=DataType.LIST,
                description="Class label for each index (class targets); empty for number targets.",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="target_type",
                param_type=ParamType.SELECT,
                default="number",
                options=["number", "class"],
                description=(
                    "number: float32 targets for regression losses. class: "
                    "int64 class indices for CrossEntropyLoss."
                ),
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
        features = inputs.get("features")
        targets = inputs.get("targets")
        if features is None or targets is None:
            raise ValueError("TensorDataset requires `features` and `targets` inputs.")
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32)
        features = features.detach().cpu()
        if torch.is_floating_point(features):
            features = features.to(torch.float32)

        if str(params.get("target_type", "number")) == "class":
            target_tensor, classes = _class_targets(targets)
        else:
            target_tensor, classes = _number_targets(targets), []

        if features.shape[0] != target_tensor.shape[0]:
            raise ValueError(
                f"TensorDataset: features have {features.shape[0]} rows but "
                f"there are {target_tensor.shape[0]} targets."
            )
        return {
            "dataset": torch.utils.data.TensorDataset(features, target_tensor),
            "classes": classes,
        }
