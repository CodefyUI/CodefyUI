"""Tests for TransformNode."""

from __future__ import annotations


from app.nodes.data.transform_node import TransformNode


class _FakeDataset:
    """A minimal stand-in for a torchvision dataset — just needs a `transform` attribute."""
    def __init__(self):
        self.transform = None


def test_node_metadata():
    assert TransformNode.NODE_NAME == "Transform"
    assert TransformNode.CATEGORY == "Data"


def test_assigns_transform_to_a_copy_of_the_dataset():
    ds = _FakeDataset()
    res = TransformNode().execute(
        {"dataset": ds},
        {"resize": 0, "normalize": True, "to_tensor": True},
    )
    assert res["dataset"].transform is not None
    # #697: the dataset it was handed is shared with sibling branches.
    assert res["dataset"] is not ds
    assert ds.transform is None


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
    out = TransformNode().execute(
        {"dataset": ds},
        {"resize": 32, "normalize": True, "to_tensor": True},
    )["dataset"]
    # Test the transform pipeline actually works
    tensor = out.transform(img)
    assert tensor.shape == (3, 32, 32)


def test_transform_is_not_cacheable():
    """Re-running is one shallow copy, and a fresh seeded wrapper per run is
    what a seeded run expects."""
    assert TransformNode.cacheable is False


def test_a_torchvision_dataset_copy_keeps_its_target_transform():
    """#697: the copy is still a working torchvision dataset.

    ``FakeData`` reads ``transform`` and ``target_transform`` in
    ``__getitem__``; the copy applies the new pipeline and the source's
    target transform, and the source still returns PIL images.
    """
    from PIL import Image
    from torchvision.datasets import FakeData

    source = FakeData(size=2, image_size=(1, 28, 28), num_classes=3,
                      target_transform=lambda label: label + 100)
    out = TransformNode().execute(
        {"dataset": source},
        {"resize": 14, "normalize": False, "to_tensor": True},
    )["dataset"]

    image, label = out[0]
    assert tuple(image.shape) == (1, 14, 14)
    assert label >= 100
    raw_image, raw_label = source[0]
    assert isinstance(raw_image, Image.Image)
    assert raw_image.size == (28, 28)
    assert raw_label == label
