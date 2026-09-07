"""The Dataset node's batched default transform is bit-identical and stays out of the way.

See ``app/nodes/data/_batched_vision.py``. These tests use torchvision's
classes with the download skipped and the storage filled in by hand, so
they need no dataset on disk.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.nodes.data import _batched_vision as bv
from app.nodes.data.dataset_node import DatasetNode


def _mnist_like(cls, n=40, seed=0):
    """A MNIST-family instance with synthetic uint8 storage, no files read."""
    ds = cls.__new__(cls)
    datasets.VisionDataset.__init__(ds, root="/nonexistent", transform=bv.DefaultVisionTransform())
    g = torch.Generator().manual_seed(seed)
    ds.train = True
    ds.data = torch.randint(0, 256, (n, 28, 28), generator=g, dtype=torch.uint8)
    ds.targets = torch.randint(0, 10, (n,), generator=g)
    return ds


def _cifar_like(cls, n=40, seed=0):
    ds = cls.__new__(cls)
    datasets.VisionDataset.__init__(ds, root="/nonexistent", transform=bv.DefaultVisionTransform())
    rng = np.random.default_rng(seed)
    ds.train = True
    ds.data = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
    ds.targets = [int(t) for t in rng.integers(0, 10, n)]
    return ds


@pytest.mark.parametrize("make, batched, plain", [
    (_mnist_like, bv.MNIST, datasets.MNIST),
    (_mnist_like, bv.FashionMNIST, datasets.FashionMNIST),
    (_cifar_like, bv.CIFAR10, datasets.CIFAR10),
    (_cifar_like, bv.CIFAR100, datasets.CIFAR100),
])
def test_the_batched_path_is_bit_identical_to_the_per_sample_path(make, batched, plain):
    fast = make(batched)
    slow = make(plain)
    indices = [3, 0, 17, 39, 5]
    got = fast.__getitems__(indices)
    for i, (x, y) in zip(indices, got):
        sx, sy = slow[i]
        assert torch.equal(x, sx), i
        assert x.dtype == sx.dtype == torch.float32
        assert y == sy and isinstance(y, int)


def test_the_dataloader_uses_the_batched_path_and_collates_the_same():
    fast = _mnist_like(bv.MNIST)
    slow = _mnist_like(datasets.MNIST)
    fx, fy = next(iter(DataLoader(fast, batch_size=16)))
    sx, sy = next(iter(DataLoader(slow, batch_size=16)))
    assert torch.equal(fx, sx)
    assert torch.equal(fy, sy)
    assert fx.shape == (16, 1, 28, 28)


def test_a_transform_installed_afterwards_takes_the_per_sample_path():
    """``TransformNode`` assigns ``dataset.transform``; the fast path must notice."""
    fast = _mnist_like(bv.MNIST)
    fast.transform = transforms.Compose([transforms.Resize(14), transforms.ToTensor()])
    x, _ = fast.__getitems__([0])[0]
    assert x.shape == (1, 14, 14)


def test_a_target_transform_takes_the_per_sample_path():
    fast = _mnist_like(bv.MNIST)
    fast.target_transform = lambda t: t + 100
    _, y = fast.__getitems__([0])[0]
    assert y == int(fast.targets[0]) + 100


def test_per_sample_indexing_is_untouched():
    fast = _mnist_like(bv.MNIST)
    slow = _mnist_like(datasets.MNIST)
    x, y = fast[7]
    sx, sy = slow[7]
    assert torch.equal(x, sx) and y == sy
    assert len(fast) == 40


def test_the_subclasses_pickle_by_name_for_spawned_workers():
    fast = _mnist_like(bv.MNIST)
    clone = pickle.loads(pickle.dumps(fast))
    assert type(clone) is bv.MNIST
    assert torch.equal(clone.__getitems__([1])[0][0], fast.__getitems__([1])[0][0])


def test_the_node_picks_the_batched_class_only_for_the_default_pipeline(monkeypatch):
    import app.nodes.data.dataset_node as node_module

    built: list[str] = []

    def fake(label):
        def ctor(**kwargs):
            built.append(label)
            return object()
        return ctor

    plain_mnist = fake("torchvision MNIST")
    monkeypatch.setattr(datasets, "MNIST", plain_mnist)
    monkeypatch.setattr(datasets, "SVHN", fake("torchvision SVHN"))
    monkeypatch.setattr(node_module, "BATCHED_FOR",
                        {plain_mnist: fake("batched MNIST")})

    node = DatasetNode()
    node.execute({}, {"name": "MNIST", "split": "train", "data_dir": "/nonexistent"})
    node.execute({"eval_transform": transforms.ToTensor()},
                 {"name": "MNIST", "split": "test", "data_dir": "/nonexistent"})
    node.execute({}, {"name": "SVHN", "split": "train", "data_dir": "/nonexistent"})
    assert built == ["batched MNIST", "torchvision MNIST", "torchvision SVHN"]


def test_a_substituted_torchvision_class_is_used_as_is(monkeypatch):
    """Keyed by class object: a test double for ``datasets.MNIST`` is honoured."""
    captured = {}

    class _Fake:
        def __init__(self, root, train, download, transform):
            captured["root"] = root

    monkeypatch.setattr(datasets, "MNIST", _Fake)
    DatasetNode().execute({}, {"name": "MNIST", "split": "train", "data_dir": "/nonexistent"})
    assert captured["root"] == "/nonexistent"


def test_the_default_transform_is_still_a_plain_compose_of_the_two_steps():
    t = bv.DefaultVisionTransform()
    assert isinstance(t, transforms.Compose)
    assert [type(s) for s in t.transforms] == [transforms.ToTensor, transforms.Normalize]
