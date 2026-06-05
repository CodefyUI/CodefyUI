---
sidebar_position: 2
title: 開發者安裝
description: 用於開發或貢獻 CodefyUI 的手動 uv + pnpm 設定，支援熱模組重載。
---

# 開發者安裝

使用 [uv](https://github.com/astral-sh/uv) 與 pnpm 的手動安裝方式 —— 支援 Windows、macOS、Linux。當你想要在後端與前端都享有熱重載時，請用這種方式來開發或貢獻。

:::tip
若你只想*執行* CodefyUI，請改用[一行指令安裝程式](./installation) —— 它不需要 Node.js 或 pnpm。
:::

## 1. Clone 專案

```bash
git clone https://github.com/treeleaves30760/CodefyUI.git
cd CodefyUI
```

## 2. 安裝 uv

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

## 3. 安裝 pnpm + Node.js

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

## 4. 後端設定

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

## 5. 安裝 PyTorch

預設安裝適用於所有平台：

```bash
uv pip install torch torchvision
uv pip install gymnasium safetensors
```

macOS 會拿到支援 MPS 的版本；Linux/Windows 會拿到 PyPI 預設版本。這樣就足以執行應用與測試模型。若需特定的 GPU 設定，請參考 **[GPU 與裝置設定](./gpu-device)**。

## 6. 啟動後端 + 前端

### 開發者模式（HMR）

**後端（終端機 1）：**

```bash
cd backend
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

開啟 [http://localhost:5173](http://localhost:5173)。Vite dev server 會把 API/WS proxy 到後端 `:8000`。

或在專案根目錄一次啟動兩者：

```bash
cdui dev                 # 若 ~/.local/bin 已在 PATH
./cdui dev               # 從專案根目錄執行
python scripts/dev.py dev
```

## 執行測試

```bash
cd backend
.venv\Scripts\activate       # 或 source .venv/bin/activate
pytest tests/ -v
```
