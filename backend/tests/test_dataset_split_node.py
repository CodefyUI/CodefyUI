"""DatasetSplit: a seeded, reproducible train/val split of a DATASET (#703)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.core.graph_engine import invoke_node
from app.nodes.data.dataset_split_node import DatasetSplitNode


def _dataset(n: int = 100) -> TensorDataset:
    # Each sample's feature is its own index, so a split can be read back.
    return TensorDataset(torch.arange(n, dtype=torch.float32).unsqueeze(1), torch.arange(n))


def _ids(part) -> list[int]:
    return [int(part[i][1]) for i in range(len(part))]


def _split(dataset, **params):
    return invoke_node(DatasetSplitNode(), {"dataset": dataset}, params)


def test_the_two_parts_partition_the_dataset():
    out = _split(_dataset(), val_fraction=0.2, seed=0)
    train, val = _ids(out["train"]), _ids(out["val"])
    assert len(train) == 80 and len(val) == 20
    assert sorted(train + val) == list(range(100))


def test_the_same_seed_gives_the_same_split():
    a = _split(_dataset(), val_fraction=0.2, seed=7)
    b = _split(_dataset(), val_fraction=0.2, seed=7)
    assert _ids(a["val"]) == _ids(b["val"])
    assert _ids(a["train"]) == _ids(b["train"])


def test_the_split_ignores_the_global_rng():
    torch.manual_seed(1)
    a = _split(_dataset(), val_fraction=0.2, seed=7)
    torch.manual_seed(2)
    b = _split(_dataset(), val_fraction=0.2, seed=7)
    assert _ids(a["val"]) == _ids(b["val"])


def test_a_different_seed_gives_a_different_split():
    a = _split(_dataset(), val_fraction=0.2, seed=1)
    b = _split(_dataset(), val_fraction=0.2, seed=2)
    assert _ids(a["val"]) != _ids(b["val"])


def test_the_split_is_shuffled_rather_than_a_tail_cut():
    val = _ids(_split(_dataset(), val_fraction=0.2, seed=0)["val"])
    assert val != list(range(80, 100))
    assert val != list(range(20))


def test_subset_keeps_that_many_samples_before_splitting():
    out = _split(_dataset(), val_fraction=0.25, subset=40, seed=3)
    train, val = _ids(out["train"]), _ids(out["val"])
    assert len(train) == 30 and len(val) == 10
    assert not set(train) & set(val)


def test_val_fraction_zero_sends_everything_to_train():
    out = _split(_dataset(10), val_fraction=0.0, seed=0)
    assert len(out["train"]) == 10 and len(out["val"]) == 0


def test_declared_defaults_hold_out_ten_percent():
    out = _split(_dataset(50))
    assert len(out["val"]) == 5 and len(out["train"]) == 45


def test_a_part_feeds_a_dataloader():
    out = _split(_dataset(), val_fraction=0.1, seed=0)
    xs, ys = next(iter(DataLoader(out["val"], batch_size=10)))
    assert xs.shape == (10, 1)
    assert torch.equal(xs.squeeze(1).long(), ys)


def test_val_fraction_of_one_is_rejected():
    with pytest.raises(ValueError, match="val_fraction"):
        DatasetSplitNode().execute({"dataset": _dataset()}, {"val_fraction": 1.0})
