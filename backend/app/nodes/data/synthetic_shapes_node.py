"""SyntheticShapesNode — tiny image dataset for *generative* models.

Where SyntheticSegmentation pairs an image with a per-pixel mask (for
training a segmentation UNet), this node produces just images — a simple
distribution a small diffusion model can learn to *generate*: soft
gaussian blobs of varied position and size, on a dark background.

Images are normalised to [-1, 1] (the usual range for diffusion training),
single channel. Each sample is (image (1, H, W) float, 0) — the dummy label
lets it flow through the standard DataLoader; diffusion training ignores it.
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


class _SyntheticShapesDataset:
    def __init__(self, n_samples: int, image_size: int, seed: int):
        import numpy as np
        import torch

        rng = np.random.default_rng(seed)
        H = image_size
        ys, xs = np.mgrid[0:H, 0:H].astype("float32")
        self.images: list[torch.Tensor] = []
        margin = max(2, H // 3)
        sig_lo, sig_hi = H / 5.0, H / 3.5
        for _ in range(n_samples):
            cy = rng.uniform(margin, H - margin)
            cx = rng.uniform(margin, H - margin)
            sig = rng.uniform(sig_lo, sig_hi)
            blob = np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig * sig))).astype("float32")
            img = blob * 2.0 - 1.0  # [0,1] -> [-1,1]
            self.images.append(torch.from_numpy(img).unsqueeze(0))  # (1, H, W)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], 0


class SyntheticShapesNode(BaseNode):
    NODE_NAME = "SyntheticShapes"
    CATEGORY = "Data"
    DESCRIPTION = (
        "產生一個小型「影像生成」資料集（CPU 友善、免下載）：一批單通道小圖，每張上面有一個"
        "位置、大小隨機的柔和光斑（gaussian blob），背景為暗。影像正規化到 [-1, 1]（擴散模型訓練"
        "的慣用範圍）。用來訓練一個小型擴散模型，學會『從雜訊生成這種形狀』。輸出資料集，可接 "
        "DataLoader / DiffusionTrainingLoop。"
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
                description="影像生成資料集：每筆是 (影像 (1,H,W) float、範圍 [-1,1])。",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="image_size",
                param_type=ParamType.INT,
                default=24,
                min_value=8,
                description="正方形影像邊長（像素）。擴散在 CPU 上偏慢，建議 16–32。",
            ),
            ParamDefinition(
                name="n_samples",
                param_type=ParamType.INT,
                default=384,
                min_value=8,
                description="產生幾張影像。",
            ),
            ParamDefinition(
                name="seed",
                param_type=ParamType.INT,
                default=0,
                description="亂數種子，決定這批光斑的位置與大小。",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        dataset = _SyntheticShapesDataset(
            n_samples=int(params.get("n_samples", 384)),
            image_size=int(params.get("image_size", 24)),
            seed=int(params.get("seed", 0)),
        )
        return {"dataset": dataset}
