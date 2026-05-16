"""Tests for TransformNode."""

from __future__ import annotations

import pytest
import torch

from app.nodes.data.transform_node import TransformNode


class _FakeDataset:
    """A minimal stand-in for a torchvision dataset — just needs a `transform` attribute."""
    def __init__(self):
        self.transform = None


def test_node_metadata():
    assert TransformNode.NODE_NAME == "Transform"
    assert TransformNode.CATEGORY == "Data"


def test_assigns_transform_to_dataset():
    ds = _FakeDataset()
    res = TransformNode().execute(
        {"dataset": ds},
        {"resize": 0, "normalize": True, "to_tensor": True},
    )
    assert res["dataset"] is ds
    assert ds.transform is not None


def test_resize_zero_skipped():
    ds = _FakeDataset()
    res = TransformNode().execute(
        {"dataset": ds},
        {"resize": 0, "normalize": False, "to_tensor": True},
    )
    # No resize transform was added but transform is set
    assert res["dataset"].transform is not None


def test_no_transforms_leaves_dataset_untouched():
    """When all toggles are off, no transform should be assigned."""
    ds = _FakeDataset()
    res = TransformNode().execute(
        {"dataset": ds},
        {"resize": 0, "normalize": False, "to_tensor": False},
    )
    assert res["dataset"].transform is None


def test_resize_param_adds_resize_step():
    """Resize > 0 should construct a working transform pipeline."""
    from PIL import Image
    img = Image.new("RGB", (100, 100))
    ds = _FakeDataset()
    TransformNode().execute(
        {"dataset": ds},
        {"resize": 32, "normalize": True, "to_tensor": True},
    )
    # Test the transform pipeline actually works
    tensor = ds.transform(img)
    assert tensor.shape == (3, 32, 32)
