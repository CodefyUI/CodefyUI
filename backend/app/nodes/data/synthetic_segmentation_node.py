"""SyntheticSegmentationNode — tiny synthetic image-segmentation dataset.

Counterpart to :class:`SyntheticDatasetNode` (which makes 2D point clouds for
classifiers). This makes small single-channel images with per-pixel class
masks, so a UNet can be trained end-to-end on CPU with zero downloads:

    each sample = (image  : float32 (1, H, W)  in [0, 1],
                   mask   : int64   (H, W)     class id per pixel)

Three classes by design: 0 = background, 1 = circle, 2 = rectangle. Each
image draws a couple of random shapes; the mask labels every pixel. Designed
for I3-2 (UNet): train a small segmentation model, then compare with / without
skip connections.

Outputs a torch ``Dataset`` so it drops straight into DataLoader / TrainingLoop
/ EvaluateModel / DatasetBatch, exactly like the built-in ``Dataset`` node.
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class _SyntheticSegDataset:
    """A pre-generated (image, mask) dataset. Deterministic given seed."""

    def __init__(self, n_samples: int, image_size: int, noise: float, seed: int):
        import numpy as np
        import torch

        rng = np.random.default_rng(seed)
        H = W = image_size
        ys, xs = np.mgrid[0:H, 0:W]

        self.images: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []

        for _ in range(n_samples):
            img = np.zeros((H, W), dtype=np.float32)
            mask = np.zeros((H, W), dtype=np.int64)

            n_shapes = int(rng.integers(1, 3))  # 1 or 2 shapes
            for _s in range(n_shapes):
                if rng.random() < 0.5:
                    # circle -> class 1
                    r = int(rng.integers(H // 8, H // 4))
                    cy = int(rng.integers(r, H - r))
                    cx = int(rng.integers(r, W - r))
                    region = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
                    img[region] = 0.7
                    mask[region] = 1
                else:
                    # rectangle -> class 2
                    h = int(rng.integers(H // 6, H // 3))
                    w = int(rng.integers(W // 6, W // 3))
                    top = int(rng.integers(0, H - h))
                    left = int(rng.integers(0, W - w))
                    img[top:top + h, left:left + w] = 1.0
                    mask[top:top + h, left:left + w] = 2

            if noise > 0:
                img = img + rng.normal(0.0, noise, size=img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

            self.images.append(torch.from_numpy(img).unsqueeze(0))  # (1, H, W)
            self.masks.append(torch.from_numpy(mask))               # (H, W)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.masks[idx]


class SyntheticSegmentationNode(BaseNode):
    NODE_NAME = "SyntheticSegmentation"
    CATEGORY = "Data"
    DESCRIPTION = "Generate a dataset of images with per-pixel masks"
    DETAILS = (
        "Each image is single-channel with one or two random shapes on it, and the "
        "mask labels every pixel: 0 background, 1 circle, 2 rectangle. It stays "
        "small enough to train a UNet on CPU. Everything is generated in memory "
        "from `seed`, with nothing downloaded — give the training and test sets "
        "different seeds."
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description=(
                    "Segmentation dataset: each item is (image (1,H,W) float, mask (H,W) int "
                    "with classes 0/1/2)."
                ),
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="image_size",
                param_type=ParamType.INT,
                default=64,
                min_value=16,
                description="Side length of the square images in pixels. 32-96 suits a CPU.",
            ),
            ParamDefinition(
                name="n_samples",
                param_type=ParamType.INT,
                default=200,
                min_value=4,
                description="Number of images to generate.",
            ),
            ParamDefinition(
                name="noise",
                param_type=ParamType.FLOAT,
                default=0.05,
                min_value=0.0,
                description="Strength of the Gaussian noise added to the images. The masks are unaffected.",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=42,
                description=(
                    "Random seed. A different seed gives a different set of images; use "
                    "different seeds for train and test."
                ),
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_size = int(params.get("image_size", 64))
        n_samples = int(params.get("n_samples", 200))
        noise = float(params.get("noise", 0.05))
        seed = int(params.get("seed", 42))

        dataset = _SyntheticSegDataset(
            n_samples=n_samples,
            image_size=image_size,
            noise=noise,
            seed=seed,
        )
        return {"dataset": dataset}
