# 環境設定指南

[![en](https://img.shields.io/badge/lang-English-red)](./SETUP.md)

CodefyUI 提供兩種安裝方式：

1. **[快速安裝](#快速安裝)**（全平台）—— 一行指令，使用者只需 `git`、`uv` 與 Python，**不需要 Node.js**。
2. **[開發者安裝](#開發者安裝)**（全平台）—— 手動使用 [uv](https://github.com/astral-sh/uv) + pnpm 安裝，適合開發者與貢獻者。

---

## 快速安裝

自動安裝 `git`、`uv` 與 Python（由 uv 提供）。前端 bundle 會從 GitHub 最新 release 直接下載 — **一般使用者不需要 Node.js 或 pnpm**。

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
```

預設安裝到 `~/CodefyUI`（macOS/Linux）或 `%USERPROFILE%\CodefyUI`（Windows）。若想自訂路徑，可設定環境變數 `CODEFYUI_DIR`。

Windows 的 `install.ps1` 會透過 [winget](https://learn.microsoft.com/zh-tw/windows/package-manager/) 安裝缺少的 `git`。`winget` 內建於 Windows 11 與較新的 Windows 10（透過 Microsoft Store 的「App Installer」套件）。

install 腳本會把 `cdui` launcher 放到 `~/.local/bin/cdui`（Windows 為 `%USERPROFILE%\.local\bin\cdui.cmd`）。重新開啟 terminal 後，任何目錄都能執行：

```bash
cdui start
```

開啟瀏覽器至 [http://localhost:8000](http://localhost:8000)。單一 uvicorn 同時處理 API 與預編好的 React 前端。

支援指令：`install` / `update` / `build` / `start` / `dev` / `stop` / `test` / `clean` / `uninstall`。`cdui` 是一支輕量 launcher，會自動挑一個可用的 Python（優先 uv 管理的）並轉交給 `dev.py`。

### 環境變數

| 變數 | 用途 |
|------|------|
| `CODEFYUI_DIR` | 安裝目錄（預設 `~/CodefyUI`）|
| `CODEFYUI_RELEASE_TAG` | 鎖定特定 release tag 的前端 bundle（預設 `latest`）|
| `CODEFYUI_FORCE_BUILD` | 設為 `1` 強制安裝 Node + pnpm 並本地 build，不下載 release |

> 預設的 PyTorch 安裝適用於所有平台（CPU / Apple Silicon MPS）。若需特定 NVIDIA CUDA 版本、AMD ROCm，或想驗證 GPU，請參考 [GPU 加速](#gpu-加速)。

### Production 與 Developer 模式

- `cdui start` — 單一 uvicorn 跑 `:8000` 同時 serve 預編前端，**不需要 Node**。一般使用者預設用這個。
- `cdui dev` — Vite dev server 跑 `:5173`（HMR）+ uvicorn 跑 `:8000`，**需要 Node 24+ 與 pnpm**。改前端代碼時用這個。
- `cdui build` — 本地重建 `frontend/dist`（也需要 Node + pnpm）。

---

## 開發者安裝

使用 [uv](https://github.com/astral-sh/uv) 與 pnpm 的手動安裝方式 —— 支援 Windows、macOS、Linux。適合開發或貢獻 CodefyUI。

### 1. Clone 專案

```bash
git clone https://github.com/treeleaves30760/CodefyUI.git
cd CodefyUI
```

### 2. 安裝 uv

```bash
uv --version
```

若未安裝：

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

### 3. 安裝 pnpm + Node.js

```bash
pnpm --version
```

若未安裝：

```bash
# Windows (PowerShell)
iwr https://get.pnpm.io/install.ps1 -useb | iex

# macOS / Linux
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

接著讓 pnpm 安裝 Node.js runtime：

```bash
pnpm env use --global lts
```

安裝完後請重新開啟 terminal 讓 PATH 生效，然後驗證：

```bash
node -v
```

### 4. 後端設定

```bash
cd backend

# 建立虛擬環境（Python 3.10+）
uv venv --python 3.11

# 啟用虛擬環境
# Windows：
.venv\Scripts\activate
# macOS / Linux：
source .venv/bin/activate

# 安裝核心依賴 + 測試工具
uv pip install -e ".[dev]"
```

### 5. 安裝 PyTorch

預設安裝 —— 全平台通用：

```bash
uv pip install torch torchvision
uv pip install gymnasium safetensors
```

這樣就夠跑應用與測試模型。macOS 會拿到支援 MPS 的版本，Linux/Windows 會拿到 PyPI 預設版本。除非你需要特定 GPU 設定，否則直接跳到 [啟動後端與前端](#啟動後端與前端)。

---

## GPU 加速

只有在需要特定 CUDA 版本、AMD ROCm/DirectML，或想驗證 GPU 偵測時才需要展開。

<details>
<summary><b>NVIDIA CUDA / AMD / MPS 驗證</b> —— 點擊展開</summary>

### NVIDIA CUDA（特定版本）

先確認你的 CUDA 版本：

```bash
nvidia-smi
```

看右上角 `CUDA Version:` 欄位，然後重裝對應的 wheel。PyTorch 官方穩定版目前提供三個 CUDA wheel 通道：

```bash
uv pip uninstall torch torchvision

# CUDA 12.8 —— RTX 50 系列（Blackwell, sm_120）必裝，RTX 30 / 40 亦可使用
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6 —— RTX 30 / 40 系列，現代驅動的通用選擇
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8 —— GTX 10 / RTX 20 系列，或舊驅動
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> RTX 5090 / 5080 / 5070（Blackwell）**必須**使用 `cu128`，舊版 wheel 缺少 `sm_120` kernel，執行時會報 `no kernel image is available for execution`。

驗證 CUDA 可用：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Apple Silicon (MPS) 驗證

M1/M2/M3/M4 Mac 上預設安裝就已經使用 Metal Performance Shaders 後端。驗證：

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### AMD 顯卡

AMD 的支援度高度取決於你的作業系統。

#### Linux + AMD（ROCm，官方支援）

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

驗證：

```bash
python -c "import torch; print('CUDA (ROCm):', torch.cuda.is_available())"
```

註：ROCm 環境下 `torch.cuda.is_available()` 會回傳 True，因為 ROCm 對外以相容於 CUDA 的介面呈現。

#### Windows + AMD（支援有限）

PyTorch 官方沒有提供 Windows ROCm 版本。可行選項：

**(a) DirectML** — 可使用 AMD 顯卡，但效能較差，而且節點程式碼預設使用 `cuda`/`cpu`，使用時需要改程式：

```bash
uv pip install torch-directml
```

**(b) CPU 模式** — 上方的預設安裝已經可用，建議 Windows + AMD 用戶採用，適合學習與原型開發。

</details>

---

## 啟動後端與前端

### Production 模式（一般使用者建議）

先 build 前端一次，之後單一 uvicorn 全部處理：

```bash
cdui build               # 產生 frontend/dist（若 install 時已下載則略過）
cdui start               # 同時 serve API 與前端，開 http://localhost:8000
```

### Developer 模式（HMR）

**後端（終端機 1）：**

```bash
cd backend
# 若尚未啟用虛擬環境
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**前端（終端機 2）：**

```bash
cd frontend
pnpm install
pnpm dev
```

開啟瀏覽器至 [http://localhost:5173](http://localhost:5173)。Vite dev server 會把 API/WS proxy 到後端 `:8000`。

或在專案根目錄同時啟動兩者：

```bash
cdui dev                 # 若 ~/.local/bin 已在 PATH
./cdui dev               # 從專案根目錄執行
python scripts/dev.py dev
```

## 驗證環境是否正確

1. 後端健康檢查：
   ```bash
   curl http://127.0.0.1:8000/api/health
   ```
   應回傳 `{"status":"ok","nodes_loaded":69,"presets_loaded":3}`（`nodes_loaded` 數量會隨版本增加，確認非 0 即可）。

2. 驗證裝置偵測：
   ```bash
   curl -s http://127.0.0.1:8000/api/nodes/TrainingLoop | python -c "import sys,json; d=json.load(sys.stdin); print([p['options'] for p in d['params'] if p['name']=='device'][0])"
   ```
   應列出你機器上可用的裝置，例如 NVIDIA 環境會顯示 `['cpu', 'cuda']`，未安裝 PyTorch 時只會顯示 `['cpu']`。

3. 開啟前端，載入 **Train CNN on MNIST** 範例並點擊 **執行**。應該會在下方面板看到訓練進度。

## 疑難排解

### 從 CPU 切換到 CUDA（或反向切換）

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### `uv pip install -e ".[ml]"` 裝到錯的 PyTorch 版本

`pyproject.toml` 中的 `[ml]` 選項群組沒有指定 index URL，所以 uv 會安裝 PyPI 的預設版本——通常 Windows 上是 CPU 版，或是版本不一定符合你的 CUDA runtime。請務必使用上方 [GPU 加速](#gpu-加速) 區塊裡對應硬體的 `--index-url` 指令。

### CUDA 版本不匹配

若 `torch.cuda.is_available()` 在有 NVIDIA 顯卡時仍回傳 False：
1. 執行 `nvidia-smi` 確認驅動版本
2. 確認安裝的 CUDA PyTorch wheel 與驅動匹配（例如不要在只支援 CUDA 11.8 的驅動上安裝 `cu128`）
3. 若是 RTX 50 系列顯卡且看到 `no kernel image is available for execution`，代表 wheel 版本太舊 —— 重新安裝 `cu128` 即可
4. 若需要請更新 NVIDIA 驅動

### UI 的 device 下拉選單沒有顯示 CUDA

前端會從後端動態讀取可用裝置。若你的 GPU 沒有出現：
1. 確認 PyTorch 能看見它：`python -c "import torch; print(torch.cuda.is_available())"`
2. 點擊工具列的 **重新載入節點** 按鈕
3. 重新整理頁面

### `pnpm: command not found`

請參考上方 [開發者安裝 → 安裝 pnpm + Node.js](#3-安裝-pnpm--nodejs)。

## 執行測試

```bash
cd backend
.venv\Scripts\activate       # 或 source .venv/bin/activate
pytest tests/ -v
```
