"""Tests for DatasetNode metadata.

The full dataset download is expensive and depends on network; we only verify
metadata and parameter handling here. Integration coverage of an actual MNIST
download is provided indirectly by `test_dataset_extension.py`.
"""

from __future__ import annotations

import pytest

from app.nodes.data.dataset_node import DatasetNode


def test_node_metadata():
    assert DatasetNode.NODE_NAME == "Dataset"
    assert DatasetNode.CATEGORY == "Data"
    out_names = [p.name for p in DatasetNode.define_outputs()]
    assert out_names == ["dataset"]


def test_unsupported_name_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        DatasetNode().execute({}, {"name": "Bogus", "split": "train", "data_dir": "./data"})


def test_options_listed_in_param():
    params = DatasetNode.define_params()
    name_param = [p for p in params if p.name == "name"][0]
    assert set(name_param.options) == {"MNIST", "CIFAR10", "FashionMNIST"}
    split_param = [p for p in params if p.name == "split"][0]
    assert set(split_param.options) == {"train", "test"}
