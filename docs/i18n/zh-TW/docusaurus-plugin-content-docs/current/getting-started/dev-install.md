---
sidebar_position: 2
title: 開發者安裝
description: 用於開發或貢獻 CodefyUI 的手動 uv + pnpm 設定，支援熱模組重載。
---

# 開發者安裝

使用 [uv](https://github.com/astral-sh/uv) 與 pnpm 手動安裝，支援 Windows、macOS 與 Linux。需要編輯程式碼、貢獻專案，或在後端與前端使用熱重載時，請採用這種方式。

:::tip
若你只想*執行* CodefyUI，請改用[一行指令安裝程式](./installation) —— 它不需要 Node.js 或 pnpm。
:::

## 1. Clone 專案 {/* #1-clone-the-repository */}

```bash
git clone https://github.com/CodefyUI/CodefyUI.git
cd CodefyUI
```

## 2. 安裝 uv {/* #2-install-uv */}

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

## 3. 安裝 pnpm + Node.js {/* #3-install-pnpm--nodejs */}

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

接著讓 pnpm 安裝 Node.js runtime（建議 Node 24+）：

```bash
pnpm env use --global lts
```

重新開啟你的 terminal 讓 PATH 更新生效，然後驗證：

```bash
node -v
```

## 4. 後端設定 {/* #4-backend-setup */}

```bash
cd backend

# 建立虛擬環境（Python 3.10+）
uv venv --python 3.11

# 啟用虛擬環境
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

# 安裝核心依賴 + 測試工具
uv pip install -e ".[dev]"
```

## 5. 安裝 PyTorch {/* #5-install-pytorch */}

第 4 步已經從 PyPI 安裝了 `torch` 與 `torchvision`，因為它們是後端的核心依賴。macOS 會拿到支援 MPS 的版本；Linux 與 Windows 會拿到 PyPI 的預設版本，其中 Windows 版是純 CPU 版。這樣就足以執行應用與測試模型。若需要 CUDA 或 ROCm 版本，請從對應的 wheel index 重新安裝，詳見 **[GPU 與裝置設定](./gpu-device)**。

## 6. 啟動後端 + 前端 {/* #6-start-backend--frontend */}

### 開發者模式（HMR） {/* #developer-mode-hmr */}

**後端（終端機 1）：**

```bash
cd backend
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --ws-max-size 67108864
```

`--ws-max-size 67108864` 讓畫布的 WebSocket 使用與 `cdui start`、`cdui dev` 相同的 64 MB 上限。少了它，uvicorn 會套用自己的 16 MB 上限，編輯器就無法執行超過這個大小的圖。

如果變更 port 或綁定位址，請將 `CODEFYUI_PORT` 與 `CODEFYUI_HOST` 設為相同的值。伺服器會從這些變數推導 Host 允許清單，而不是從監聽中的 socket 取得。port 不一致時，每個請求都會回傳 `421`；綁定位址不一致時，只有來自其他機器的請求會回傳 `421`，因為 loopback 名稱一律允許。`cdui start --project` 會設定 `CODEFYUI_PROJECT_DIR=<absolute dir>`。直接以此指令啟動的 uvicorn 會將 session token 與其他使用者資料儲存在平台資料目錄：`%LOCALAPPDATA%\codefyui`、`~/.local/share/codefyui` 或 `~/Library/Application Support/codefyui`。它不會使用 `.codefyui_dev/`。請參閱[專案目錄](/usage/project-directories#6-create-an-api-key-invoke-needs-one)。

**前端（終端機 2）：**

```bash
cd frontend
pnpm install
pnpm dev
```

開啟 [http://localhost:5173](http://localhost:5173)。Vite dev server 會把 API/WS proxy 到後端 `:8000`。

或在專案根目錄一次啟動兩者：

```bash
./cdui dev               # Windows：.\cdui.cmd dev
python scripts/dev.py dev
```

`PATH` 上的 `cdui` 是一行指令安裝程式寫入的啟動器，它一律執行該安裝程式建立的目錄，而不是這份 clone。

## 執行測試 {/* #running-tests */}

```bash
./cdui test                  # 後端（pytest）+ 前端（vitest）；--backend / --frontend 可只跑其中一組
```

`cdui test` 需要第 4 步安裝的 `[dev]` extra。使用啟動器時，請執行 `./cdui install --dev` 來安裝。使用一行指令安裝程式時，請設定 `CODEFYUI_FORCE_BUILD=1 CODEFYUI_DEV=1`。若要直接執行後端測試套件：

```bash
cd backend
.venv\Scripts\activate       # 或 source .venv/bin/activate
pytest tests/ -v
```
