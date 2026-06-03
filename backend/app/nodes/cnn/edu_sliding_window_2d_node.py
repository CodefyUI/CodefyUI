"""EduSlidingWindow2DNode — 在影像上滑動一個 kernel 做加權加總。

對應教材 **I1-2**。把一個 kxk 的小視窗（kernel）在影像上一格一格滑過去，
每個位置做「視窗內的數字 × 對應像素，再加總」。換不同的 kernel
（模糊 / 邊緣 / 銳化 / 垂直邊緣）就抽出不同的形狀——這是整個影像 AI
最小的拼塊。

* 吃 **3D `(C, H, W)`** 影像（單通道 `(H, W)` 也接，當 1 通道處理）。
* 對每個通道套用**同一個** kernel（depthwise），輸出同階的影像；
  `padding=0` 時邊長各縮 `k-1`（3x3 → 邊長 -2）。
* batch 維 `N` 是 CNN / 訓練才需要的設計，這裡刻意不暴露——內部自己
  包 / 拆 batch，學生只跟 2D / 3D 影像打交道。

命名用 SlidingWindow2D 而非 Conv2D：它做的是 cross-correlation（kernel 不翻轉），
跟嚴謹數學定義的 convolution（會先翻轉 kernel）略有差異；深度學習裡的
「Conv2D」其實也是這個 cross-correlation。
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

# 內建 3x3 kernel。名稱對應影像處理課本，後綴 3x3 把形狀寫進下拉選單。
PRESETS_3X3: dict[str, list[list[float]]] = {
    # 模糊：9 格各 1/9，就是鄰域平均，把尖銳細節抹平。
    "Blur3x3": [[1.0 / 9.0] * 3 for _ in range(3)],
    # 邊緣偵測（Laplacian）：中心 +8、周圍 -1，只有亮度突變處留強訊號。
    "EdgeDetection3x3": [
        [-1.0, -1.0, -1.0],
        [-1.0, 8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ],
    # 銳化：中心 +5、上下左右 -1、四角 0，強化邊緣。
    "Sharpen3x3": [
        [0.0, -1.0, 0.0],
        [-1.0, 5.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    # 垂直邊緣（Prewitt-X）：左 -1、中 0、右 +1，抓左右方向亮度變化。
    "VerticalEdge3x3": [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ],
}

CUSTOM_OPTION = "Custom"
PRESET_OPTIONS: list[str] = [*PRESETS_3X3.keys(), CUSTOM_OPTION]
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 15


def _flatten(x: Any) -> list[float]:
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [float(x)]


class EduSlidingWindow2DNode(BaseNode):
    NODE_NAME = "Edu-SlidingWindow2D"
    CATEGORY = "Custom"
    DESCRIPTION = (
        "在影像上滑動一個 kernel 做加權加總（cross-correlation）。吃 (C,H,W) 影像"
        "（單通道 (H,W) 也接），對每個通道套用同一個 kernel，輸出同階影像。"
        "選內建 3x3 preset（模糊 Blur / 邊緣 EdgeDetection / 銳化 Sharpen / 垂直邊緣 "
        "VerticalEdge）或切到 Custom 自填 NxN 數字。padding=0 時邊長各縮 k-1。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="image",
                data_type=DataType.TENSOR,
                description="影像 tensor，(C, H, W) 或單通道 (H, W)。",
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="image",
                data_type=DataType.TENSOR,
                description="卷積後的影像，與輸入同階；padding=0 時邊長各縮 k-1。",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="preset",
                param_type=ParamType.SELECT,
                default="Blur3x3",
                description="內建 3x3 kernel，或選 'Custom' 自己填數字。",
                options=PRESET_OPTIONS,
            ),
            ParamDefinition(
                name="kernel_size",
                param_type=ParamType.INT,
                default=3,
                description="kernel 邊長 N（NxN）；僅 Custom 時可調。",
                min_value=MIN_KERNEL_SIZE,
                max_value=MAX_KERNEL_SIZE,
                visible_when={"preset": CUSTOM_OPTION},
            ),
            ParamDefinition(
                name="weights",
                param_type=ParamType.TENSOR_GRID,
                default=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                description="自訂 kernel 矩陣（NxN），格子數跟著 kernel_size。",
                visible_when={"preset": CUSTOM_OPTION},
            ),
            ParamDefinition(
                name="padding",
                param_type=ParamType.INT,
                default=0,
                description="四周補零的圈數。0 = 邊長各縮 k-1（224→222）；(k-1)/2 可維持尺寸。",
                min_value=0,
            ),
        ]

    def _resolve_kernel(self, params: dict[str, Any]) -> tuple[list[float], int]:
        preset = str(params.get("preset", "Blur3x3"))
        if preset == CUSTOM_OPTION:
            try:
                k = int(params.get("kernel_size", 3))
            except (TypeError, ValueError):
                k = 3
            k = max(MIN_KERNEL_SIZE, min(MAX_KERNEL_SIZE, k))
            raw_weights = params.get("weights")
            if raw_weights is None:
                raise ValueError("preset=Custom requires `weights` to be set (an NxN matrix).")
            flat = _flatten(raw_weights)
            if len(flat) != k * k:
                raise ValueError(
                    f"`weights` has {len(flat)} elements but kernel_size={k} expects {k * k} "
                    f"({k}x{k}). Adjust kernel_size or re-fill the grid."
                )
            return flat, k
        kernel = PRESETS_3X3.get(preset)
        if kernel is None:
            raise ValueError(f"Unknown preset: {preset!r}. Choose one of {PRESET_OPTIONS}.")
        return _flatten(kernel), 3

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        image = inputs.get("image")
        if image is None:
            raise ValueError("Edu-SlidingWindow2D requires an `image` input.")
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.float32)
        image = image.float()

        # 接 (C,H,W) 或單通道 (H,W)。記住是否為 2D，最後還原回去。
        was_2d = image.dim() == 2
        if was_2d:
            image = image.unsqueeze(0)  # (1, H, W)
        if image.dim() != 3:
            raise ValueError(
                f"Edu-SlidingWindow2D expects a (C, H, W) or (H, W) image; got shape {list(image.shape)}."
            )

        c = image.size(0)
        flat, k = self._resolve_kernel(params)
        padding = max(0, int(params.get("padding", 0) or 0))

        kernel = torch.tensor(flat, dtype=image.dtype, device=image.device).reshape(1, 1, k, k)
        weight = kernel.expand(c, 1, k, k).contiguous()  # depthwise：每通道同一個 kernel

        x = image.unsqueeze(0)  # (1, C, H, W) — batch 維只在內部存在
        out = F.conv2d(x, weight, bias=None, stride=1, padding=padding, groups=c)
        out = out.squeeze(0)  # (C, H_out, W_out)

        if was_2d:
            out = out.squeeze(0)  # 還原成 (H_out, W_out)
        return {"image": out}
