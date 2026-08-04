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
    assert set(name_param.options) == {
        "MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "SVHN", "STL10"}
    split_param = [p for p in params if p.name == "split"][0]
    assert set(split_param.options) == {"train", "test"}


def test_every_listed_dataset_exists_in_torchvision():
    """A typo in the options list is a run failure, not a validation error.

    ``execute`` looks the name up in its own map, so a name that is offered
    but not mapped -- or mapped to something torchvision does not have --
    would only surface when someone picks it and waits for a download.
    """
    from torchvision import datasets

    from app.nodes.data.dataset_node import DATASET_NAMES

    for name in DATASET_NAMES:
        assert hasattr(datasets, name), name


def test_split_datasets_take_a_split_kwarg():
    """The two constructors that differ, checked against torchvision itself.

    ``SVHN`` and ``STL10`` take ``split="train"`` where the rest take
    ``train=True``; passing the wrong one is a TypeError from inside
    torchvision after the download has already run.
    """
    import inspect

    from torchvision import datasets

    from app.nodes.data.dataset_node import DATASET_NAMES, SPLIT_KWARG_DATASETS

    for name in DATASET_NAMES:
        signature = inspect.signature(getattr(datasets, name).__init__)
        expected = "split" if name in SPLIT_KWARG_DATASETS else "train"
        assert expected in signature.parameters, name
