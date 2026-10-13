"""KFoldSplit holds out exactly the fold sklearn's KFold would (#712).

Every test compares row sets with ``sklearn.model_selection.KFold`` or
``StratifiedKFold`` run on the same rows, so the node's semantics are
sklearn's and not a reimplementation that drifts. Each feature row carries
its own row number, which is how the tests read back which rows went where.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold, StratifiedKFold

from app.core.node_registry import registry
from app.core.sweep_compiler import SWEEPABLE_PARAM_TYPES
from app.nodes.data.kfold_split_node import KFoldSplitNode

_ROWS = 23  # Not a multiple of 5, so the folds differ in size as sklearn's do.


def _rows(n: int = _ROWS) -> torch.Tensor:
    """Row i is [i, 10 * i], so a row's first value is its row number."""
    index = torch.arange(n, dtype=torch.float32)
    return torch.stack([index, 10 * index], dim=1)


def _run(features, labels, **params):
    p = {"n_folds": 5, "fold": 0, "shuffle": True, "seed": 42, "stratify": False}
    p.update(params)
    return KFoldSplitNode().execute({"features": features, "labels": labels}, p)


def _row_numbers(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x[:, 0].tolist()]


def test_node_metadata_and_registration():
    assert KFoldSplitNode.NODE_NAME == "KFoldSplit"
    assert KFoldSplitNode.CATEGORY == "Data"
    assert registry.get("KFoldSplit") is KFoldSplitNode
    assert [p.name for p in KFoldSplitNode.define_inputs()] == ["features", "labels"]
    assert [p.name for p in KFoldSplitNode.define_outputs()] == [
        "x_train", "y_train", "x_val", "y_val"]


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_folds", [2, 5, 7])
def test_every_fold_holds_out_the_rows_sklearn_kfold_does(n_folds, shuffle):
    x = _rows()
    labels = [f"r{i}" for i in range(_ROWS)]
    reference = KFold(n_splits=n_folds, shuffle=shuffle,
                      random_state=3 if shuffle else None)
    held_out: list[int] = []
    for fold, (train_idx, val_idx) in enumerate(reference.split(np.zeros((_ROWS, 1)))):
        out = _run(x, labels, n_folds=n_folds, fold=fold, shuffle=shuffle, seed=3)
        assert _row_numbers(out["x_val"]) == val_idx.tolist()
        assert _row_numbers(out["x_train"]) == train_idx.tolist()
        # Labels travel with their rows.
        assert out["y_val"] == [f"r{i}" for i in val_idx]
        assert out["y_train"] == [f"r{i}" for i in train_idx]
        assert out["x_val"].dtype == torch.float32
        held_out += val_idx.tolist()
    # Across the folds every row is held out exactly once.
    assert sorted(held_out) == list(range(_ROWS))


def test_shuffle_off_cuts_consecutive_blocks_and_the_seed_changes_shuffled_folds():
    x = _rows(10)
    labels = list(range(10))
    assert _run(x, labels, n_folds=5, fold=1, shuffle=False)["y_val"] == [2, 3]
    # shuffle off ignores the seed, as sklearn requires.
    assert _run(x, labels, n_folds=5, fold=1, shuffle=False, seed=7)["y_val"] == [2, 3]
    a = _run(x, labels, n_folds=5, fold=1, seed=0)["y_val"]
    b = _run(x, labels, n_folds=5, fold=1, seed=1)["y_val"]
    assert a != b
    assert a == _run(x, labels, n_folds=5, fold=1, seed=0)["y_val"]


@pytest.mark.parametrize("shuffle", [True, False])
def test_stratify_holds_out_the_rows_sklearn_stratified_kfold_does(shuffle):
    x = _rows(30)
    labels = ["a"] * 18 + ["b"] * 9 + ["c"] * 3
    reference = StratifiedKFold(n_splits=3, shuffle=shuffle,
                                random_state=11 if shuffle else None)
    for fold, (train_idx, val_idx) in enumerate(
            reference.split(np.zeros((30, 1)), labels)):
        out = _run(x, labels, n_folds=3, fold=fold, shuffle=shuffle, seed=11,
                   stratify=True)
        assert _row_numbers(out["x_val"]) == val_idx.tolist()
        assert _row_numbers(out["x_train"]) == train_idx.tolist()
        # 18/9/3 split three ways keeps 6/3/1 of each class in every fold.
        assert sorted(out["y_val"]) == ["a"] * 6 + ["b"] * 3 + ["c"]


def test_numeric_targets_and_a_label_column_tensor_are_split_as_values():
    x = _rows(6)
    targets = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    out = _run(x, targets, n_folds=3, fold=2, shuffle=False)
    assert out["y_val"] == [4.5, 5.5]
    assert out["y_train"] == [0.5, 1.5, 2.5, 3.5]
    # ColumnSelector's [N, 1] column splits as N values, as in TrainTestSplit.
    column = torch.tensor(targets).reshape(-1, 1)
    assert _run(x, column, n_folds=3, fold=2, shuffle=False)["y_val"] == [4.5, 5.5]


def test_refusals_name_what_is_wrong():
    x = _rows(6)
    labels = list(range(6))
    with pytest.raises(ValueError, match=r"fold must be from 0 to 2 \(n_folds - 1\), got 3"):
        _run(x, labels, n_folds=3, fold=3)
    with pytest.raises(ValueError, match="got -1"):
        _run(x, labels, n_folds=3, fold=-1)
    with pytest.raises(ValueError, match="n_folds=7 is more than the 6 rows"):
        _run(x, labels, n_folds=7)
    with pytest.raises(ValueError, match="at least 2"):
        _run(x, labels, n_folds=1)
    with pytest.raises(ValueError, match="6 rows vs 5 labels"):
        _run(x, labels[:5])
    with pytest.raises(ValueError, match="requires both"):
        KFoldSplitNode().execute({"features": x}, {})
    # StratifiedKFold refuses a continuous target; the message says what to do.
    with pytest.raises(ValueError, match="Turn stratify off for a numeric target"):
        _run(x, [0.1, 0.7, 1.3, 2.2, 3.9, 4.4], n_folds=2, stratify=True)


def test_fold_is_a_sweepable_param():
    """A sweep addresses ``{node_id, param}`` and accepts INT params; ``fold``
    has no max, so a sweep over 0..n_folds-1 is never refused for the range."""
    fold = next(p for p in KFoldSplitNode.define_params() if p.name == "fold")
    assert fold.param_type in SWEEPABLE_PARAM_TYPES
    assert fold.min_value == 0 and fold.max_value is None
