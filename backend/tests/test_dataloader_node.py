"""Tests for DataLoaderNode."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.nodes.data.dataloader_node import DataLoaderNode


def _dataset(n=16, in_features=4):
    X = torch.randn(n, in_features)
    y = torch.randint(0, 2, (n,))
    return TensorDataset(X, y)


def test_node_metadata():
    assert DataLoaderNode.NODE_NAME == "DataLoader"
    assert DataLoaderNode.CATEGORY == "Data"


def test_wraps_dataset_in_dataloader():
    ds = _dataset()
    res = DataLoaderNode().execute(
        {"dataset": ds},
        {"batch_size": 4, "shuffle": False, "num_workers": 0},
    )
    assert isinstance(res["dataloader"], DataLoader)


def test_batch_size_param():
    ds = _dataset(n=16)
    res = DataLoaderNode().execute({"dataset": ds}, {"batch_size": 8})
    loader = res["dataloader"]
    assert loader.batch_size == 8
    first_batch = next(iter(loader))
    assert first_batch[0].shape[0] == 8


def test_shuffle_false_yields_deterministic_order():
    ds = _dataset()
    res = DataLoaderNode().execute(
        {"dataset": ds},
        {"batch_size": 4, "shuffle": False, "num_workers": 0},
    )
    batches_1 = [b[1].tolist() for b in res["dataloader"]]
    res2 = DataLoaderNode().execute(
        {"dataset": ds},
        {"batch_size": 4, "shuffle": False, "num_workers": 0},
    )
    batches_2 = [b[1].tolist() for b in res2["dataloader"]]
    assert batches_1 == batches_2


def test_default_batch_size_is_32():
    ds = _dataset(n=64)
    res = DataLoaderNode().execute({"dataset": ds}, {})
    assert res["dataloader"].batch_size == 32
