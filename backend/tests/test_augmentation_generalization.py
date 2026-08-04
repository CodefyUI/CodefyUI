"""Augmentation measurably beats no augmentation (core#136).

The acceptance criterion is directional: RandomCrop + HorizontalFlip +
Normalize(CIFAR stats) has to train and beat the un-augmented baseline. This
is the test for it, and two things about it are worth stating plainly.

**Why not CIFAR-10 itself.** A real CIFAR-10 run is minutes of CPU per arm
and needs 170 MB downloaded, neither of which belongs in a unit suite. The
fixture is synthetic instead -- but it is not a toy that has been tuned
until the numbers looked right. It isolates exactly the invariance the two
augmentations teach, and it isolates it in the only way a directional test
can be honest about:

* the training split shows two glyphs at ONE position and in ONE
  orientation; the test split shows the same two glyphs SHIFTED and
  MIRRORED. Nothing else differs.
* ``RandomCrop(size, padding=3)`` covers the shift; ``RandomHorizontalFlip``
  covers the mirror. So the augmented model can learn what the plain one
  cannot see.
* both arms start from the SAME initial weights and run the same optimizer
  for the same epochs, so the only variable in the experiment is whether
  the two transforms are in the chain.
* both arms are asserted to fit the TRAINING split completely. Without that
  the test could pass because the baseline failed to train at all, which
  would say nothing about augmentation.

**How big the margin is.** Measured over the three seeds below, the
un-augmented model lands at 0.50-0.70 on the mirrored, shifted test split
(chance is 0.50) while the augmented one lands at 1.00 -- a margin of +0.30
to +0.50. The assertion asks for +0.15 averaged, which is well inside that
and is deliberately not fitted to the observation.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.nodes.data.transforms.normalize_transform_node import (
    NormalizeTransformNode,
)
from app.nodes.data.transforms.random_crop_node import RandomCropNode
from app.nodes.data.transforms.random_horizontal_flip_node import (
    RandomHorizontalFlipNode,
)
from app.nodes.data.transforms.to_tensor_transform_node import (
    ToTensorTransformNode,
)
from app.nodes.training.training_loop_node import TrainingLoopNode

SIZE = 16
EPOCHS = 25
SEEDS = (0, 1, 2)

#: Minimum average accuracy gain the augmented arm must show. See the module
#: docstring for the measured spread this sits inside.
MIN_MEAN_MARGIN = 0.15


def _stamp(kind: str, mirrored: bool) -> torch.Tensor:
    """A 7x7 asymmetric glyph. Mirroring one never produces the other.

    That property is what keeps horizontal flipping a legitimate
    augmentation here rather than a label-destroying one: flipping an 'L'
    gives a shape that is still an 'L' and never a '7'.
    """
    glyph = torch.zeros(7, 7)
    if kind == "L":
        glyph[:, 0] = 1.0        # vertical bar down the left
        glyph[6, :4] = 1.0       # foot along the bottom
    else:                        # "7"
        glyph[0, :] = 1.0        # horizontal bar across the top
        glyph[:4, 6] = 1.0       # stem down the right
    return torch.flip(glyph, dims=[1]) if mirrored else glyph


def _image(kind: str, mirrored: bool, dx: int, dy: int) -> Image.Image:
    canvas = torch.zeros(SIZE, SIZE)
    top, left = 4 + dy, 4 + dx
    canvas[top:top + 7, left:left + 7] = _stamp(kind, mirrored)
    rgb = (canvas.unsqueeze(-1).repeat(1, 1, 3) * 255).to(torch.uint8)
    return Image.fromarray(rgb.numpy(), mode="RGB")


class _Glyphs(Dataset):
    """Alternating L / 7 glyphs, with a writable ``transform`` attribute."""

    def __init__(self, *, mirrored: bool, shifts, count: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.items = []
        for index in range(count):
            kind = "L" if index % 2 == 0 else "7"
            dx = int(shifts[torch.randint(len(shifts), (1,),
                                          generator=generator)])
            dy = int(shifts[torch.randint(len(shifts), (1,),
                                          generator=generator)])
            self.items.append(
                (_image(kind, mirrored, dx, dy), 0 if kind == "L" else 1))
        self.transform = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image, label = self.items[index]
        return (self.transform(image) if self.transform is not None
                else image), label


def _chain(augment: bool):
    """The chain from the acceptance criterion, built out of the real nodes."""
    upstream = {}
    if augment:
        step = RandomCropNode().execute(
            {}, {"size": SIZE, "padding": 3})["transform"]
        step = RandomHorizontalFlipNode().execute(
            {"transform": step}, {"p": 0.5})["transform"]
        upstream = {"transform": step}
    step = ToTensorTransformNode().execute(upstream, {})["transform"]
    return NormalizeTransformNode().execute(
        {"transform": step}, {"preset": "CIFAR-10"})["transform"]


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 4 * 4, 2),
    )


def _accuracy(model: nn.Module, dataset: Dataset) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in DataLoader(dataset, batch_size=32):
            correct += int((model(data).argmax(1) == target).sum())
            total += int(target.numel())
    return correct / total


def _arm(augment: bool, seed: int, initial_state) -> tuple[float, float]:
    """Train one arm; return (test accuracy, training accuracy)."""
    torch.manual_seed(seed)
    train = _Glyphs(mirrored=False, shifts=[0], count=64, seed=seed)
    train.transform = _chain(augment)

    # Held out by a DISTRIBUTION SHIFT, not by a random split: the glyphs
    # are mirrored and off-centre, which is precisely what the training
    # split never shows and what the two augmentations put back.
    test = _Glyphs(mirrored=True, shifts=[-3, -2, 2, 3], count=64,
                   seed=seed + 1000)
    test.transform = _chain(False)

    # The same images the model trained on, WITHOUT the augmentation, so
    # "did this arm actually learn its training set" is a fair question.
    seen = _Glyphs(mirrored=False, shifts=[0], count=64, seed=seed)
    seen.transform = _chain(False)

    model = _model()
    model.load_state_dict(initial_state)
    TrainingLoopNode().execute(
        {"model": model,
         "dataloader": DataLoader(train, batch_size=16, shuffle=True),
         "optimizer": torch.optim.Adam(model.parameters(), lr=0.01),
         "loss_fn": nn.CrossEntropyLoss()},
        {"epochs": EPOCHS, "device": "cpu"},
    )
    return _accuracy(model, test), _accuracy(model, seen)


def test_augmentation_beats_the_un_augmented_baseline():
    margins: list[float] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial_state = copy.deepcopy(_model().state_dict())

        plain_test, plain_train = _arm(False, seed, initial_state)
        aug_test, aug_train = _arm(True, seed, initial_state)

        # Neither arm may have simply failed to train -- otherwise the
        # comparison is between two models and not between two chains.
        assert plain_train == 1.0, f"seed {seed}: baseline did not fit train"
        assert aug_train == 1.0, f"seed {seed}: augmented did not fit train"

        assert aug_test > plain_test, (
            f"seed {seed}: augmented {aug_test:.3f} did not beat baseline "
            f"{plain_test:.3f}")
        margins.append(aug_test - plain_test)

    mean_margin = sum(margins) / len(margins)
    assert mean_margin >= MIN_MEAN_MARGIN, (
        f"mean margin {mean_margin:.3f} over seeds {SEEDS} is below "
        f"{MIN_MEAN_MARGIN}; margins were "
        f"{[round(m, 3) for m in margins]}")


def test_the_baseline_is_genuinely_near_chance_on_the_shifted_split():
    """Guards the fixture, not the feature.

    If someone later makes the test split easier, the margin would shrink
    for a reason that has nothing to do with augmentation. This says the
    un-augmented model really cannot do the held-out task.
    """
    torch.manual_seed(0)
    initial_state = copy.deepcopy(_model().state_dict())
    plain_test, _ = _arm(False, 0, initial_state)
    assert plain_test < 0.85
