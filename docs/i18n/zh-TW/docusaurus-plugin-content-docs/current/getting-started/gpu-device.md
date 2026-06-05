---
sidebar_position: 3
title: GPU 與裝置設定
description: 為 NVIDIA CUDA、Apple Silicon（MPS）或 AMD ROCm 選擇合適的 PyTorch 版本，並驗證 GPU 偵測。
---

# GPU 與裝置設定

預設的 PyTorch 安裝適用於所有平台（CPU，以及透過 MPS 的 Apple Silicon）。只有在你需要特定 CUDA 版本、AMD ROCm/DirectML，或想驗證 GPU 偵測時才需要繼續往下讀。

CodefyUI 會在執行階段從後端讀取可用的裝置，所以只要 PyTorch 看得到的，都會出現在每個節點的 **device** 下拉選單裡。全域裝置可以設定一次，並套用到所有以張量為來源的節點。

## NVIDIA CUDA（特定版本）

先確認你已安裝的 CUDA 版本：

```bash
nvidia-smi
```

看右上角的 `CUDA Version:` 欄位，然後重裝對應的 wheel。PyTorch 目前提供以下這些穩定版的 CUDA wheel 通道：

```bash
uv pip uninstall torch torchvision

# CUDA 12.8 —— RTX 50 系列（Blackwell, sm_120）必裝，RTX 30 / 40 亦可使用。
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6 —— RTX 30 / 40 系列，現代驅動的通用預設選擇
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8 —— GTX 10 / RTX 20 系列，或舊驅動
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

:::warning RTX 50 系列（Blackwell）
RTX 5090 / 5080 / 5070 **必須**使用 `cu128` —— 舊版 wheel 缺少 `sm_120` kernel，執行時會以 `no kernel image is available for execution` 失敗。
:::

驗證 CUDA 可用：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Apple Silicon (MPS)

在 M1/M2/M3/M4 Mac 上，預設安裝就已經使用 Metal Performance Shaders 後端。驗證：

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

:::note MPS 上的 float64
MPS 以 float32 為原生格式，並會拒絕 float64 張量。CodefyUI 在 `device_utils.to_device` 中處理了這點，但若你撰寫自訂節點，在 Apple GPU 上請把張量保持為 float32。（有一條實驗性的原生 MLX 推論路徑作為 spike 存在 —— 請參考 [Device Backends](/advanced/device-backends)。）
:::

## AMD 顯卡

AMD 的支援度高度取決於你的作業系統。

### Linux + AMD（ROCm，官方支援）

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

驗證：

```bash
python -c "import torch; print('CUDA (ROCm):', torch.cuda.is_available())"
```

在 ROCm 上，`torch.cuda.is_available()` 會回傳 `True`，因為 ROCm 對外以相容於 CUDA 的後端介面呈現。

### Windows + AMD（支援有限）

PyTorch **沒有**提供官方的 Windows ROCm 版本。你的選項有：

- **(a) DirectML** —— 可使用 AMD 顯卡，但效能較差，而且需要修改程式碼（內建節點預設使用 `cuda`／`cpu`）：

  ```bash
  uv pip install torch-directml
  ```

- **(b) CPU 模式** —— 上方的預設安裝已經可用。建議在 Windows + AMD 上用於學習／原型開發。

## 疑難排解

### 從 CPU 切換到 CUDA（或反向切換）

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### `uv pip install -e ".[ml]"` 裝到錯的 PyTorch 版本

`pyproject.toml` 中的 `[ml]` 選項群組**沒有**指定 index URL，所以 uv 會安裝 PyPI 的預設版本 —— 通常 Windows 上是 CPU 版，或是版本不一定符合你的 CUDA runtime。請務必使用本頁中明確指定 `--index-url` 的指令。

### 有 NVIDIA 顯卡時 `torch.cuda.is_available()` 仍回傳 False

1. 執行 `nvidia-smi` 確認驅動版本。
2. 確認你安裝的是與驅動匹配的 CUDA PyTorch wheel（例如不要在只支援到 CUDA 11.8 的驅動上安裝 `cu128`）。
3. RTX 50 系列 + `no kernel image is available for execution` → 代表你用的是舊版 wheel；重新安裝 `cu128`。
4. 若需要請更新你的 NVIDIA 驅動。

### UI 的 device 下拉選單沒有顯示 CUDA

前端會從後端讀取可用裝置。若你的 GPU 沒有列出來：

1. 確認 PyTorch 能看見它：`python -c "import torch; print(torch.cuda.is_available())"`
2. 點擊工具列的 **重新載入節點** 按鈕。
3. 重新整理頁面。

### 從 API 驗證裝置偵測

```bash
curl -s http://127.0.0.1:8000/api/nodes/TrainingLoop | python -c "import sys,json; d=json.load(sys.stdin); print([p['options'] for p in d['params'] if p['name']=='device'][0])"
```

這會印出可用的裝置，例如 NVIDIA 會顯示 `['cpu', 'cuda']`，未安裝 PyTorch 時則顯示 `['cpu']`。
