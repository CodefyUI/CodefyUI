#!/usr/bin/env python3
"""Phase-3 spike：原生 MLX 推論子集 proof-of-concept（非阻塞、僅推論）。

背景：Apple 加速在 CodefyUI 的圖引擎是用 **PyTorch MPS**（已接好）。MLX 是
Apple 另一套陣列框架，**不是 torch backend**（沒有 `tensor.to("mlx")`），所以
不能直接塞進現有 torch 執行路徑。本 spike 驗證「把一個已訓練好的小模型的
**前向推論**搬到 MLX 跑」是否可行、數值是否一致、效能如何——這是未來把 MLX
當成可選推論加速器的最小起點。

做法：
  1. 用 torch 建一個小 MLP（Linear→ReLU→Linear→ReLU→Linear），在 CPU 跑推論。
  2. 把 torch 的權重（state_dict）轉成 MLX 陣列。
  3. 用 mlx.core 重寫同一條前向（matmul + relu），在 MLX(GPU/Metal) 上跑。
  4. 比對兩邊輸出的數值誤差，並各量一次 wall-clock。

範圍與限制（刻意）：
  - **只做推論**：不追求 autograd/訓練在 MLX 上的對等，那是更大的工程。
  - 權重以 float32 轉移（MLX 在 Apple 上原生 float32）。
  - 這是 spike，不是正式 backend；圖引擎仍以 MPS 為 Apple 預設。

用法：
    python scripts/mlx_spike.py
未安裝 mlx 時會印出安裝指引並以非錯誤狀態結束。
"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn


def build_torch_mlp(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def mlx_forward(state: dict, x_np):
    """Reimplement the Sequential MLP forward with mlx.core ops.

    torch ``nn.Linear`` stores weight as [out, in] and computes
    ``y = x @ W.T + b``; we mirror that with mx.matmul.
    """
    import mlx.core as mx

    def linear(x, w, b):
        return mx.matmul(x, w.T) + b

    def relu(x):
        return mx.maximum(x, 0)

    x = mx.array(x_np)
    # Sequential indices: 0=Linear, 2=Linear, 4=Linear (1,3 are ReLU).
    x = relu(linear(x, mx.array(state["0.weight"]), mx.array(state["0.bias"])))
    x = relu(linear(x, mx.array(state["2.weight"]), mx.array(state["2.bias"])))
    x = linear(x, mx.array(state["4.weight"]), mx.array(state["4.bias"]))
    mx.eval(x)  # force evaluation (MLX is lazy) before we read/time it
    return x


def main() -> int:
    from importlib.util import find_spec

    if find_spec("mlx") is None:
        print("MLX 尚未安裝。這個 spike 需要 Apple 的 MLX 框架。")
        print("安裝： uv pip install mlx   （或 pip install mlx），僅限 Apple Silicon。")
        print("未安裝不影響主程式：Apple 加速在圖引擎是走 PyTorch MPS。")
        return 0

    import mlx.core as mx
    import numpy as np

    print(f"MLX default device: {mx.default_device()}")
    print(f"torch MPS available: {torch.backends.mps.is_available()}\n")

    model = build_torch_mlp().eval()
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((256, 64)).astype(np.float32)

    # ── torch (CPU) reference ──
    with torch.no_grad():
        t0 = time.perf_counter()
        torch_out = model(torch.from_numpy(x_np)).numpy()
        t_torch = time.perf_counter() - t0

    # ── MLX (GPU/Metal) ──
    mlx_forward(state, x_np)  # warm-up (compile kernels)
    t0 = time.perf_counter()
    mlx_out = mlx_forward(state, x_np)
    t_mlx = time.perf_counter() - t0
    mlx_np = np.array(mlx_out)

    max_abs = float(np.max(np.abs(torch_out - mlx_np)))
    print(f"output shape       : torch {torch_out.shape}  |  mlx {mlx_np.shape}")
    print(f"max abs difference : {max_abs:.3e}")
    print(f"torch CPU forward  : {t_torch * 1e3:.2f} ms")
    print(f"mlx   GPU forward  : {t_mlx * 1e3:.2f} ms")

    ok = max_abs < 1e-4
    print(f"\n{'PASS' if ok else 'FAIL'} — inference-subset parity "
          f"{'within' if ok else 'EXCEEDS'} 1e-4 tolerance.")
    print("結論：把已訓練模型的前向推論搬到 MLX 可行且數值一致；"
          "可作為未來可選 MLX 推論加速器的基礎。圖引擎 Apple 預設仍為 MPS。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
