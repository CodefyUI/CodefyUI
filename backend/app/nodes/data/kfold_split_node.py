"""KFoldSplitNode — hold out one fold of a k-fold split (#712).

Wraps ``sklearn.model_selection.KFold`` (``StratifiedKFold`` with
``stratify`` on). It takes the same ``features`` + ``labels`` pair as
``TrainTestSplit`` and returns the rows of fold ``fold`` as the validation
part and the other ``n_folds - 1`` folds as the training part:

    CSVReader → KFoldSplit ──→ x_train, y_train → TensorDataset / LinearRegression
                           ╰── x_val,   y_val   → RegressionMetrics / Accuracy

One run evaluates one fold. Cross-validation is a sweep over ``fold`` from
0 to ``n_folds - 1``: each variant holds out a different fold, and the mean
of the recorded metric across the variants is the cross-validation score.
The folds depend only on ``n_folds``, ``shuffle``, ``seed`` and the rows,
never on the run seed, so every variant of that sweep cuts the same folds.

A separate node rather than ``n_folds``/``fold`` params on TrainTestSplit:
TrainTestSplit's ``test_size`` would be dead under k-fold, its ``x_test``
outputs would carry validation rows, and sklearn itself keeps the two as
separate splitters.
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


class KFoldSplitNode(BaseNode):
    NODE_NAME = "KFoldSplit"
    CATEGORY = "Data"
    DESCRIPTION = "Hold out one fold of a k-fold split for validation"
    DETAILS = (
        "sklearn.model_selection.KFold cuts the rows into `n_folds` folds; fold "
        "`fold` (counted from 0) becomes x_val/y_val and the other folds "
        "x_train/y_train. Cross-validation is a sweep over `fold` from 0 to "
        "n_folds - 1, ranked by a recorded metric such as RegressionMetrics' "
        "`r2` or Accuracy's `accuracy`; the mean across the variants is the "
        "cross-validation score. The folds depend only on n_folds, shuffle and "
        "seed, never on the run seed, so every variant cuts the same folds. "
        "With `shuffle` off the folds are consecutive blocks in row order. "
        "`stratify` uses StratifiedKFold, which keeps each class's share in "
        "every fold and needs class labels."
    )

    # sklearn works on host memory; see TrainTestSplit.
    align_inputs = False

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
                description="Per-sample labels or targets (LIST or tensor).",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="x_train", data_type=DataType.TENSOR,
                           description="Features of every fold except `fold`."),
            PortDefinition(name="y_train", data_type=DataType.LIST,
                           description="Labels of every fold except `fold`."),
            PortDefinition(name="x_val", data_type=DataType.TENSOR,
                           description="Features of fold `fold`."),
            PortDefinition(name="y_val", data_type=DataType.LIST,
                           description="Labels of fold `fold`."),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="n_folds",
                param_type=ParamType.INT,
                default=5,
                min_value=2,
                description="Number of folds, at least 2 and at most the number of rows.",
            ),
            ParamDefinition(
                name="fold",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "The fold held out for validation, from 0 to n_folds - 1. "
                    "Sweep it over that range to cross-validate."
                ),
            ),
            ParamDefinition(
                name="shuffle",
                param_type=ParamType.BOOL,
                default=True,
                description="Shuffle the rows before cutting folds. Off: folds are consecutive blocks.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description="Seed of the shuffle. The same seed gives the same folds.",
            ),
            ParamDefinition(
                name="stratify",
                param_type=ParamType.BOOL,
                default=False,
                description="Keep each class's share in every fold (StratifiedKFold). Needs class labels.",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from sklearn.model_selection import KFold, StratifiedKFold

        features = inputs.get("features")
        labels = inputs.get("labels")
        if features is None or labels is None:
            raise ValueError("KFoldSplit requires both `features` and `labels` inputs.")

        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32)
        # A label column picked with ColumnSelector arrives as [N, 1].
        if isinstance(labels, torch.Tensor) and labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.reshape(-1)
        labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)

        rows = features.shape[0]
        if rows != len(labels_list):
            raise ValueError(
                f"KFoldSplit: features and labels length mismatch — "
                f"{rows} rows vs {len(labels_list)} labels."
            )

        n_folds = int(params.get("n_folds", 5))
        fold = int(params.get("fold", 0))
        shuffle = bool(params.get("shuffle", True))
        seed = int(params.get("seed", 42))
        stratify = bool(params.get("stratify", False))
        if n_folds < 2:
            raise ValueError(f"KFoldSplit: n_folds must be at least 2, got {n_folds}.")
        if n_folds > rows:
            raise ValueError(
                f"KFoldSplit: n_folds={n_folds} is more than the {rows} rows; "
                f"every fold needs at least one row."
            )
        if not 0 <= fold < n_folds:
            raise ValueError(
                f"KFoldSplit: fold must be from 0 to {n_folds - 1} "
                f"(n_folds - 1), got {fold}."
            )

        # sklearn refuses a random_state when shuffle is off.
        splitter_cls = StratifiedKFold if stratify else KFold
        splitter = splitter_cls(n_splits=n_folds, shuffle=shuffle,
                                random_state=seed if shuffle else None)
        try:
            splits = splitter.split(np.zeros((rows, 1)), labels_list if stratify else None)
            for _ in range(fold):
                next(splits)
            train_idx, val_idx = next(splits)
        except ValueError as exc:
            if stratify:
                raise ValueError(
                    f"KFoldSplit: stratify needs class labels with at least "
                    f"n_folds={n_folds} rows in some class ({exc}). Turn "
                    f"stratify off for a numeric target."
                ) from exc
            raise

        x = features.detach().cpu()
        return {
            "x_train": x[torch.from_numpy(train_idx)].float(),
            "y_train": [labels_list[i] for i in train_idx],
            "x_val": x[torch.from_numpy(val_idx)].float(),
            "y_val": [labels_list[i] for i in val_idx],
        }
