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
  A test split wires the training node's ``classes`` output into its own
  ``classes`` input, so both splits number a class the same way even when
  the test split lacks a class.
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


def _label_text(value: Any) -> str:
    """A label as text, keeping integral floats integral ('1', not '1.0')."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _class_targets(
    targets: Any, given: list[str] | None = None
) -> tuple[torch.Tensor, list[str]]:
    """Class indices for *targets*, and the class order they index.

    *given* is a class order from another TensorDataset (its ``classes``
    output, wired into this node's ``classes`` input). A test split rarely
    holds every class, so numbering its own labels would give a class a
    different index than the training split did; the given order keeps
    them aligned.
    """
    if isinstance(targets, torch.Tensor) and not torch.is_floating_point(targets):
        # Already class indices: kept as they are, so index i names class i.
        values = targets.detach().cpu().to(torch.long).reshape(-1)
        count = int(values.max()) + 1 if values.numel() else 0
        if given is None:
            return values, [str(i) for i in range(count)]
        if values.numel() and (int(values.min()) < 0 or count > len(given)):
            raise ValueError(
                f"TensorDataset: the targets are class indices up to {count - 1}, "
                f"but the `classes` input lists {len(given)} class(es). Wire the "
                f"`classes` output of the TensorDataset that built the training set."
            )
        return values, list(given)
    raw = targets.tolist() if isinstance(targets, torch.Tensor) else list(targets)
    labels = [_label_text(v) for v in raw]
    classes = _class_order(labels) if given is None else list(given)
    index = {label: i for i, label in enumerate(classes)}
    unknown = sorted(set(labels) - index.keys())
    if unknown:
        shown = ", ".join(repr(u) for u in unknown[:5])
        more = f" and {len(unknown) - 5} more" if len(unknown) > 5 else ""
        raise ValueError(
            f"TensorDataset: label(s) {shown}{more} are not in the `classes` "
            f"input {classes}. The `classes` input must come from the "
            f"TensorDataset built on the training split, and every label here "
            f"must appear there."
        )
    return torch.tensor([index[label] for label in labels], dtype=torch.long), classes


def _given_classes(value: Any) -> list[str] | None:
    """The ``classes`` input as label strings, or None when it is not wired."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(
            "TensorDataset: the `classes` input must be a LIST of class labels, "
            "such as another TensorDataset's `classes` output."
        )
    classes = [_label_text(v) for v in value]
    if len(set(classes)) != len(classes):
        raise ValueError(
            f"TensorDataset: the `classes` input {classes} repeats a label; "
            f"each class must appear once."
        )
    return classes


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
        "integer tensor is taken as class indices already. For a test or "
        "validation split, wire the training TensorDataset's `classes` output "
        "into this node's `classes` input: the split then uses the training "
        "numbering, even when it lacks a class, and a label the training split "
        "never had stops the run. "
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
            PortDefinition(
                name="classes",
                data_type=DataType.LIST,
                description=(
                    "Optional class order to number the labels with: wire the "
                    "training TensorDataset's `classes` output here on the test split."
                ),
                optional=True,
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
            target_tensor, classes = _class_targets(
                targets, _given_classes(inputs.get("classes"))
            )
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
