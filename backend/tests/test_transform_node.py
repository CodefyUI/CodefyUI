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


# ── 2026-10-13 audit: a Transform on a DatasetSplit output ──


class _PilDataset:
    """Reads ``transform`` in ``__getitem__`` like a torchvision dataset."""

    def __init__(self, n: int = 6):
        from PIL import Image

        self.transform = None
        # Left half black, right half white, so a flip is visible.
        self.images = []
        for _ in range(n):
            img = Image.new("L", (4, 4), 0)
            for x in range(2, 4):
                for y in range(4):
                    img.putpixel((x, y), 255)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        return (self.transform(img) if self.transform else img), 0


def _flip_chain():
    from torchvision import transforms as T

    return T.Compose([T.RandomHorizontalFlip(p=1.0), T.ToTensor()])


def test_a_transform_on_a_split_reaches_its_samples():
    from app.nodes.data.dataset_split_node import DatasetSplitNode

    source = _PilDataset()
    split = DatasetSplitNode().execute({"dataset": source}, {"val_fraction": 0.5, "subset": 0, "seed": 0})
    out = TransformNode().execute(
        {"dataset": split["train"], "transform": _flip_chain()}, {}
    )["dataset"]
    sample, _ = out[0]
    # Flipped: the white half is now on the left.
    assert sample[0, 0, 0].item() == 1.0 and sample[0, 0, 3].item() == 0.0
    assert len(out) == len(split["train"])
    assert list(out.indices) == list(split["train"].indices)


def test_a_transform_on_a_split_leaves_the_source_and_sibling_untouched():
    from PIL import Image

    from app.nodes.data.dataset_split_node import DatasetSplitNode

    source = _PilDataset()
    split = DatasetSplitNode().execute({"dataset": source}, {"val_fraction": 0.5, "subset": 0, "seed": 0})
    TransformNode().execute({"dataset": split["train"], "transform": _flip_chain()}, {})
    assert source.transform is None
    assert isinstance(split["val"][0][0], Image.Image)
    assert isinstance(split["train"][0][0], Image.Image)


def test_a_transform_reaches_through_nested_subsets():
    from torch.utils.data import Subset

    source = _PilDataset()
    nested = Subset(Subset(source, [5, 4, 3, 2]), [1, 0])
    out = TransformNode().execute({"dataset": nested, "transform": _flip_chain()}, {})["dataset"]
    assert out[0][0][0, 0, 0].item() == 1.0
    assert isinstance(out, Subset) and isinstance(out.dataset, Subset)
    assert out.dataset.dataset is not source and source.transform is None


def test_a_dataset_that_ignores_transforms_is_reported_in_the_run_log():
    import torch
    from torch.utils.data import TensorDataset

    notes: list[tuple[str, str]] = []

    class _Ctx:
        current_node_id = "t1"

        def log_warning(self, kind, detail):
            notes.append((kind, detail))

        def make_generator(self, label):
            return None

        def derive_seed(self, label):
            return 0

    ds = TensorDataset(torch.zeros(2, 1))
    res = TransformNode().execute({"dataset": ds}, {"resize": 0, "normalize": True, "to_tensor": False}, context=_Ctx())
    assert "no 'transform' attribute" in res["__log__"]
    assert notes and notes[0][0] == "transform_ignored"
