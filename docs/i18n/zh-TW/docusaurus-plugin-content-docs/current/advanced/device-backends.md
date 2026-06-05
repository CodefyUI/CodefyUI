---
sidebar_position: 4
title: 裝置後端
description: CodefyUI 如何在 CPU、CUDA、MPS 與 ROCm 之間選擇與退回——以及實驗性的原生 MLX 推論 spike。
---

# 裝置後端

CodefyUI 執行於 PyTorch 之上，因此繼承了 PyTorch 的裝置後端：**CPU**、**NVIDIA CUDA**、**Apple Silicon（MPS）**與 **AMD ROCm**（Linux）。關於安裝正確的 wheel，請參閱 **[GPU 與裝置設定](/getting-started/gpu-device)**；本頁說明裝置選擇在執行時的行為。

## 全域裝置選擇

單一全域 **device** 設定會驅動所有張量來源節點，因此你只需設定一次，而不必逐節點設定。後端會暴露 PyTorch 實際能看到的裝置（透過 `device_utils.get_available_devices()`），UI 則依該清單填入每個裝置下拉選單。被請求的裝置會與可用的裝置比對，若不存在則**退回 CPU 並發出警告**。

## float64 + MPS 的限制

MPS 是 **float32 原生**的，會拒絕 float64 張量。CodefyUI 在 `device_utils.to_device` 中將其正規化，但如果你撰寫一個直接建立張量的[自訂節點](./custom-nodes)，請在 Apple GPU 上將它們維持為 float32，以避免執行時錯誤。

## ROCm 呈現為 CUDA

在 AMD + Linux 上搭配 ROCm 版本的 PyTorch 時，`torch.cuda.is_available()` 會回傳 `True`，因為 ROCm 暴露了一個與 CUDA 相容的介面。該裝置在下拉選單中會顯示為 `cuda`；這是預期的行為。

## 實驗性：原生 MLX（spike）

有一個**概念驗證 (proof-of-concept)**，把一個小型 MLP 的*前向推論*從 PyTorch 移植到 Apple 的 [MLX](https://github.com/ml-explore/mlx) 框架，產生數值上完全相同的結果（最大絕對差約 1.9e-7）。重點如下：

- **真正圖引擎中的 Apple 加速是 PyTorch MPS**，它已接好並完成端到端驗證。MLX **並非**已交付的執行後端。
- MLX 是一個*獨立的陣列框架*，並非 PyTorch 後端——並沒有 `torch.device("mlx")`——所以它無法成為全域裝置選擇器（驅動 `torch`）中的一個值。
- 此 spike 僅供**推論**且為 **float32**，可臨時執行：

  ```bash
  uv pip install mlx        # Apple Silicon only
  python scripts/mlx_spike.py
  ```

- `mlx` **並非**已納入的相依套件；主應用程式從不匯入它。只透過 `device_utils.mlx_available()`（偵測）與 spike 腳本來呈現它。

**建議：**將 **MPS** 維持為所有執行（訓練 + 推論）的 Apple 預設；把 MLX 當作選用的推論加速器，只在推論密集的教學示範上有可量測的效益時才回頭考慮。
