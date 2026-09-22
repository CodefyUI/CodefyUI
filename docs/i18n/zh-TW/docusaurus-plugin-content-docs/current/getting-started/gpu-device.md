---
sidebar_position: 3
title: GPU 與裝置設定
description: 為 NVIDIA CUDA、Apple Silicon（MPS）或 AMD ROCm 選擇合適的 PyTorch 版本，並驗證 GPU 偵測。
---

# GPU 與裝置設定

預設的 PyTorch 安裝適用於所有平台（CPU，以及透過 MPS 的 Apple Silicon）。只有在你需要特定 CUDA 版本、AMD ROCm/DirectML，或想驗證 GPU 偵測時才需要繼續往下讀。

CodefyUI 會在執行階段從後端讀取可用的裝置，所以只要 PyTorch 看得到的，都會出現在每個裝置下拉選單裡：設定、執行旁邊這張圖自己的裝置控制項，以及每個節點「進階」裡的 **device** 參數。在圖上或設定裡設一次即可。節點參數是為舊圖保留的；一張圖只在一個裝置上執行，需要兩個裝置的工作應拆成兩張圖。

## NVIDIA CUDA（特定版本） {/* #nvidia-cuda-specific-version */}

先確認你已安裝的 CUDA 版本：

```bash
nvidia-smi
```

查看右上角的 `CUDA Version:` 欄位，然後使用 `cdui install --gpu` 安裝相符的 wheel。此指令會替換 venv 中的 PyTorch build。之後執行 `cdui update` 時，指令會從已安裝的 wheel 讀取變體並保留該設定：

```bash
# CUDA 12.8 —— RTX 50 系列（Blackwell, sm_120）必裝，RTX 30 / 40 亦可使用。
cdui install --gpu cu128

# CUDA 12.6 —— RTX 30 / 40 系列，現代驅動的通用預設選擇
cdui install --gpu cu126

# CUDA 11.8 —— GTX 10 / RTX 20 系列，或舊驅動
cdui install --gpu cu118
```

一行指令安裝程式使用 `--gpu auto`。它會依 `nvidia-smi` 回報的 driver 版本選擇 wheel：560 以上選擇 `cu128`，555 以上選擇 `cu126`，545 以上選擇 `cu124`，530 以上選擇 `cu121`，520 以上選擇 `cu118`。較舊的 driver 會選擇 CPU build。如果伺服器是以 `cdui start` 啟動，套件中心的 **GPU 版 PyTorch** 卡片會執行相同的安裝並重新啟動伺服器。請參閱[讓伺服器重新啟動的安裝](/usage/optional-packs#installs-that-restart-the-server)。

若要手動安裝，請啟用 `backend/.venv`，並執行安裝程式使用的同一個 `uv pip` 指令：

```bash
cd backend
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux
uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --index-url https://download.pytorch.org/whl/cu128   # 或 cu126 / cu118
```

:::warning RTX 50 系列（Blackwell）
RTX 5090 / 5080 / 5070 **必須**使用 `cu128` —— 舊版 wheel 缺少 `sm_120` kernel，執行時會以 `no kernel image is available for execution` 失敗。
:::

驗證 CUDA 可用：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Apple Silicon (MPS)

在 M1/M2/M3/M4 Mac 上，預設安裝就已經附帶 Metal Performance Shaders 後端。注意「附帶」不等於「使用」：除非你在設定裡、或直接在圖上選了 `mps`，否則執行仍然留在 CPU——參見 [裝置後端](/advanced/device-backends)。驗證後端存在：

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

:::note MPS 上的 float64
MPS 以 float32 為原生格式，並會拒絕 float64 張量。CodefyUI 會在 `device_utils.to_device` 中處理，但自訂節點在 Apple GPU 上應建立 float32 張量。MPS 沒有某項運算的 kernel 時，PyTorch 會改在 CPU 上執行，而不會使整次執行失敗。[裝置後端](/advanced/device-backends)說明如何變更此退回行為，也記錄實驗性的原生 MLX 推論 spike。
:::

## AMD 顯卡 {/* #amd-gpu */}

AMD 的支援度高度取決於你的作業系統。

### Linux + AMD（ROCm，官方支援） {/* #linux--amd-rocm-officially-supported */}

```bash
cdui install --gpu rocm6.2      # 或 rocm6.1
```

`PATH` 中有 `rocm-smi` 時，`--gpu auto` 會選擇 `rocm6.2`。若要手動安裝，請啟用 `backend/.venv`，再執行 `uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2`。

驗證：

```bash
python -c "import torch; print('CUDA (ROCm):', torch.cuda.is_available())"
```

在 ROCm 上，`torch.cuda.is_available()` 會回傳 `True`，因為 ROCm 對外以相容於 CUDA 的後端介面呈現。

### Windows + AMD（支援有限） {/* #windows--amd-limited */}

PyTorch **沒有**提供官方的 Windows ROCm 版本。你的選項有：

- **(a) DirectML** —— 需要修改程式碼。套件可以安裝，但執行、圖與節點參數只接受 `cpu`、`auto`、`cuda`（或 `cuda:N`）與 `mps`，因此 CodefyUI 永遠不會選用 DirectML 裝置：

  ```bash
  uv pip install torch-directml
  ```

- **(b) CPU 模式** —— 上方的預設安裝已經可用。建議在 Windows + AMD 上用於學習／原型開發。

## 裝置無法使用時 {/* #when-a-device-is-unavailable */}

**設定**裡的**運算裝置**儲存在瀏覽器中，而不是伺服器上，因此它可能指向這台伺服器沒有的裝置，例如在另一台電腦上選的 `cuda`。這時編輯器會顯示：

- 設定的下拉選單會把儲存的值顯示為停用的選項「cuda（此伺服器沒有）」，這一列的提示也會從「最佳可用裝置：…」改為「這台伺服器沒有 cuda，執行時會退回 CPU。」
- **執行**旁邊這張圖的裝置選單中，第一個選項會從「跟隨設定（…）」改為「跟隨設定（cuda → CPU）」。
- 圖檔中儲存、但不在這台伺服器清單中的裝置，例如 `auto`，或單一 GPU 機器上的 `cuda:1`，會在該選單中以停用選項顯示儲存的值。儲存圖時會保留這個值。

選擇清單中的裝置即可解除這個狀態。伺服器沒有的裝置類型會改在 CPU 上執行；伺服器清單中沒有的 `cuda:N` 或 `mps:N` 會改用它目前的 GPU（或 `mps`），`auto` 則使用現有最好的裝置（見[裝置後端](/advanced/device-backends)）。上面的 CPU 訊息不會區分這些情況：在只列出 `cuda` 的單一 GPU 伺服器上，設定中的裝置為 `cuda:0` 或 `cuda:1` 時會顯示「執行時會退回 CPU」與「→ CPU」，但執行其實會在 GPU 上。

伺服器偵測到 GPU、但 PyTorch build 不是該 GPU 建議的版本時（通常是 CPU 版），設定的這一列會加上「偵測到 NVIDIA GeForce RTX 4080 (driver 560.94)，但這台伺服器跑的是 CPU 版。安裝：」、安裝指令（例如 `cdui install --gpu cu128`）與**套件中心**連結。只要版本不一致，訊息就會說「CPU 版」，包括已經能使用 GPU 的較舊 CUDA 版（例如 `cu126`）。伺服器若由 `cdui start` 啟動，套件中心裡的 **GPU 版 PyTorch** 卡片會安裝該版本、重新啟動伺服器並重新載入頁面。設定中的裝置不會自動改變；要使用 GPU，請在設定或圖上選擇它。

## 疑難排解 {/* #troubleshooting */}

### 從 CPU 切換到 CUDA（或反向切換） {/* #switching-from-cpu-to-cuda-or-vice-versa */}

```bash
cdui install --gpu cu128     # 改回去：cdui install --gpu cpu
```

### `uv pip install -e .` 裝到錯的 PyTorch 版本 {/* #uv-pip-install--e--installs-the-wrong-pytorch-build */}

`torch` 與 `torchvision` 是後端的核心依賴，而且沒有指定 index URL。venv 中沒有 torch，或 torch 版本低於 2.5 時，`uv pip install -e .`（或 `-e ".[dev]"`）會安裝 PyPI 的預設 wheel：Windows 上是純 CPU 版，其他平台則不一定符合你的 CUDA runtime。請先用 `cdui install --gpu <choice>` 或本頁明確指定 `--index-url` 的指令安裝需要的版本；已符合需求的 torch 不會被更動。

### 有 NVIDIA 顯卡時 `torch.cuda.is_available()` 仍回傳 False {/* #torchcudais_available-returns-false-with-an-nvidia-gpu */}

1. 執行 `nvidia-smi` 確認驅動版本。
2. 確認你安裝的是與驅動匹配的 CUDA PyTorch wheel（例如不要在只支援到 CUDA 11.8 的驅動上安裝 `cu128`）。
3. RTX 50 系列 + `no kernel image is available for execution` → 代表你用的是舊版 wheel；重新安裝 `cu128`。
4. 若需要請更新你的 NVIDIA 驅動。

### UI 的 device 下拉選單沒有顯示 CUDA {/* #the-device-dropdown-in-the-ui-doesnt-show-cuda */}

前端會從後端讀取可用裝置。若你的 GPU 沒有列出來：

1. 在啟用 `backend/.venv` 的情況下確認 PyTorch 能看見它：`python -c "import torch; print(torch.cuda.is_available())"`
2. 重新啟動伺服器（先 `cdui stop`，再 `cdui start`）。伺服器只在啟動時匯入一次 PyTorch，所以執行期間安裝的 build 它看不到，**重新載入節點**也不會重新匯入。
3. 重新整理頁面。編輯器每次載入頁面只讀取一次裝置清單。

### 從 API 驗證裝置偵測 {/* #verify-device-detection-from-the-api */}

```bash
curl -s http://127.0.0.1:8000/api/nodes/TrainingLoop | python -c "import sys,json; d=json.load(sys.stdin); print([p['options'] for p in d['params'] if p['name']=='device'][0])"
```

這會印出可用裝置。NVIDIA 系統會顯示 `['auto', 'cpu', 'cuda']`；有多張卡時，還會顯示 `cuda:0`、`cuda:1` 與其他帶索引的裝置。PyTorch 未偵測到加速器時，會顯示 `['auto', 'cpu']`。`auto` 一定可用，並會跟著圖的裝置走：工具列指定的裝置，圖沒有指定時則是設定裡的裝置。
