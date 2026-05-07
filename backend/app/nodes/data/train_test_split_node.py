"""TrainTestSplitNode — split a dataset into training and test partitions.

Wraps ``sklearn.model_selection.train_test_split``. The whole dataset
flows in as one ``features`` tensor + matching ``labels`` list, and
flows out as four streams that the rest of the pipeline can wire
independently — typical pattern is:

    CSVReader → ColumnSelector → Normalize ──→ TrainTestSplit ──→ x_train ──→ Classifier.fit
                                                              ╰── x_test  ──→ Classifier.predict
                                                              ╰── y_test  ──→ accuracy

Stratified splits preserve class proportions in both partitions, which
matters for imbalanced datasets — a 90/10 binary classifier could
easily end up with zero positive samples in test under random splitting.
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


class TrainTestSplitNode(BaseNode):
    NODE_NAME = "TrainTestSplit"
    CATEGORY = "Data"
    DESCRIPTION = (
        "Partition (features, labels) into a training set and a test set "
        "using sklearn.train_test_split. `stratify=True` keeps class "
        "proportions identical across both partitions — essential for "
        "imbalanced labels."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="features",
                data_type=DataType.TENSOR,
                description="Feature tensor [N, F].",
            ),
            PortDefinition(
                name="labels",
                data_type=DataType.LIST,
                description="Per-sample labels (LIST or LongTensor).",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR, description="Training features."),
            PortDefinition(name="y_train", data_type=DataType.LIST, description="Training labels."),
            PortDefinition(name="x_test", data_type=DataType.TENSOR, description="Test features."),
            PortDefinition(name="y_test", data_type=DataType.LIST, description="Test labels."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="test_size",
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.0,
                max_value=1.0,
                description="Fraction of samples reserved for testing. Must be in (0, 1).",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed for reproducible shuffling.",
            ),
            ParamDefinition(
                name="stratify",
                param_type=ParamType.BOOL,
                default=False,
                description="Preserve per-class proportions in both partitions.",
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
        from sklearn.model_selection import train_test_split

        features = inputs.get("features")
        labels = inputs.get("labels")
        if features is None or labels is None:
            raise ValueError("TrainTestSplit requires both `features` and `labels` inputs.")

        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32)
        labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)

        if features.shape[0] != len(labels_list):
            raise ValueError(
                f"TrainTestSplit: features and labels length mismatch — "
                f"{features.shape[0]} rows vs {len(labels_list)} labels."
            )

        test_size = float(params.get("test_size", 0.2))
        if not (0 < test_size < 1):
            raise ValueError(
                f"TrainTestSplit: test_size must be in (0, 1), got {test_size}."
            )
        seed = int(params.get("seed", 42))
        stratify = bool(params.get("stratify", False))

        x_train, x_test, y_train, y_test = train_test_split(
            features.numpy(),
            labels_list,
            test_size=test_size,
            random_state=seed,
            stratify=labels_list if stratify else None,
        )

        return {
            "x_train": torch.from_numpy(x_train).float(),
            "y_train": list(y_train),
            "x_test": torch.from_numpy(x_test).float(),
            "y_test": list(y_test),
        }
