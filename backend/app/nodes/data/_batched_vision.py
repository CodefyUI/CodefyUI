"""Batch-level ``ToTensor`` + ``Normalize`` for torchvision's in-memory datasets.

Why
---
``Dataset``'s default pipeline (``ToTensor`` then ``Normalize((0.5,), (0.5,))``)
runs per SAMPLE through torchvision's ``__getitem__``: a uint8 array becomes
a PIL image, the image becomes a tensor, the tensor is normalised. Measured
on an M3 that is ~60 us per MNIST sample -- ~5 us of PIL, the rest per-call
op overhead on a 1x28x28 tensor -- so a 60k-sample epoch spends ~1.5 s on
the training thread doing arithmetic that is one op each when done on the
whole batch. Same story for CIFAR. With ``num_workers`` at its default of 0
(the right default on a Mac, where workers are spawned processes and were
measured SLOWER for these datasets) all of it sits on the critical path.

How
---
``torch.utils.data.DataLoader`` calls ``dataset.__getitems__(indices)`` when
the dataset defines it (``_MapDatasetFetcher.fetch``, torch >= 2.0) and
collates whatever comes back exactly as it would a list of ``__getitem__``
results. So each supported class gets a subclass with one extra method:
slice the uint8 storage for the batch, ``float()/255``, ``(x - 0.5)/0.5``,
hand back one ``(tensor, label)`` per index. Same ops in the same dtype as
the per-sample path, so the result is bit-for-bit identical (a test pins
it), and ~11x faster per epoch. Everything else -- ``__getitem__``,
``.data``, ``.targets``, ``.classes``, ``len``, download and checksum --
is torchvision's own, untouched.

What it does NOT cover, deliberately
------------------------------------
* A wired transform chain. Only the node's DEFAULT pipeline is
  reproduced; anything wired into ``train_transform`` / ``eval_transform``
  may resize, augment or expect PIL, and stays on torchvision's path.
* A pipeline replaced after construction (``TransformNode`` assigns
  ``dataset.transform``). ``__getitems__`` checks that ``transform`` is
  still the very object it was built for and falls back per sample if not.
* ``SVHN`` and ``STL10``. Their storage layouts differ (and STL10's
  unlabeled split has no targets at all); they keep the per-sample path.

The subclasses are defined at module level, not built by a factory, so a
dataset pickles into a spawned DataLoader worker by its importable name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchvision import datasets, transforms


class DefaultVisionTransform(transforms.Compose):
    """The node's default pipeline, as a distinguishable type.

    A plain ``Compose`` of the same two steps. The subclass exists so that
    :meth:`_BatchedDefaultTransform.__getitems__` can tell "still the
    default I was built for" from "something else was assigned later"
    with an identity check rather than by introspecting the steps.
    """

    def __init__(self) -> None:
        super().__init__([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])


def _batched_default(raw: Any) -> torch.Tensor:
    """``ToTensor`` + ``Normalize((0.5,), (0.5,))`` over a batch of uint8.

    ``raw`` is either an ``[N, H, W]`` uint8 tensor (the MNIST family) or
    an ``[N, H, W, C]`` uint8 ndarray (the CIFAR family). Both come out as
    ``[N, C, H, W]`` float32, exactly what ``ToTensor`` produces per sample.
    """
    if isinstance(raw, np.ndarray):
        x = torch.from_numpy(raw).permute(0, 3, 1, 2)
    else:
        x = raw.unsqueeze(1)
    # Same arithmetic as the per-sample path: ToTensor divides by 255 in
    # float32, Normalize subtracts then divides. Keeping the order and the
    # dtype is what makes the two paths bit-identical.
    x = x.to(torch.float32).div_(255.0)
    return x.sub_(0.5).div_(0.5)


class _BatchedDefaultTransform:
    """Mixin: ``__getitems__`` for a torchvision dataset holding uint8 data."""

    data: Any
    targets: Any
    transform: Any
    target_transform: Any

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, int]]:
        if (not isinstance(self.transform, DefaultVisionTransform)
                or self.target_transform is not None):
            # Someone installed a different pipeline after construction.
            # torchvision's per-sample path honours it; this one would not.
            return [self[i] for i in indices]  # type: ignore[index]
        index = torch.as_tensor(indices, dtype=torch.int64)
        data = self.data
        raw = data[index.numpy()] if isinstance(data, np.ndarray) else data[index]
        images = _batched_default(raw)
        targets = self.targets
        if isinstance(targets, torch.Tensor):
            labels = targets[index].tolist()
        else:
            labels = [int(targets[i]) for i in indices]
        return list(zip(images.unbind(0), labels))


class MNIST(_BatchedDefaultTransform, datasets.MNIST):
    pass


class FashionMNIST(_BatchedDefaultTransform, datasets.FashionMNIST):
    pass


class CIFAR10(_BatchedDefaultTransform, datasets.CIFAR10):
    pass


class CIFAR100(_BatchedDefaultTransform, datasets.CIFAR100):
    pass


#: torchvision class -> its batched subclass. Keyed by the class object,
#: not the name: the fast path applies to torchvision's own classes, and a
#: class substituted for one of them (a test double, a monkeypatch) is used
#: as it is. Only these four: see the module docstring for SVHN and STL10.
BATCHED_FOR: dict[type, type] = {
    datasets.MNIST: MNIST,
    datasets.FashionMNIST: FashionMNIST,
    datasets.CIFAR10: CIFAR10,
    datasets.CIFAR100: CIFAR100,
}
