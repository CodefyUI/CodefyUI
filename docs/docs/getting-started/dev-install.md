---
sidebar_position: 2
title: Dev Install
description: Manual uv + pnpm setup for developing or contributing to CodefyUI, with hot module reload.
---

# Dev Install

Manual setup using [uv](https://github.com/astral-sh/uv) and pnpm — works on Windows, macOS, and Linux. Use this for development or contributing, when you want hot reload on both the backend and frontend.

:::tip
If you only want to *run* CodefyUI, use the [one-line installer](./installation) instead — it needs no Node.js or pnpm.
:::

## 1. Clone the repository

```bash
git clone https://github.com/treeleaves30760/CodefyUI.git
cd CodefyUI
```

## 2. Install uv

```bash
uv --version
```

If not installed:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

## 3. Install pnpm + Node.js

```bash
pnpm --version
```

If not installed:

```bash
# Windows (PowerShell)
iwr https://get.pnpm.io/install.ps1 -useb | iex

# macOS / Linux
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

Then let pnpm install a Node.js runtime (Node 24+ is recommended):

```bash
pnpm env use --global lts
```

Restart your terminal so PATH updates take effect, then verify:

```bash
node -v
```

## 4. Backend setup

```bash
cd backend

# Create a virtual environment with Python 3.10+
uv venv --python 3.11

# Activate it
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

# Install core dependencies + test tools
uv pip install -e ".[dev]"
```

## 5. Install PyTorch

The default install works on every platform:

```bash
uv pip install torch torchvision
uv pip install gymnasium safetensors
```

macOS gets an MPS-capable build; Linux/Windows get the default PyPI build. This is enough to run the app and test models. For a specific GPU configuration, see **[GPU & Device Setup](./gpu-device)**.

## 6. Start backend + frontend

### Developer mode (HMR)

**Backend (terminal 1):**

```bash
cd backend
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend (terminal 2):**

```bash
cd frontend
pnpm install
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173). The Vite dev server proxies API/WS to the backend on `:8000`.

Or start both at once from the project root:

```bash
cdui dev                 # if ~/.local/bin is on PATH
./cdui dev               # from the project root
python scripts/dev.py dev
```

## Running tests

```bash
cd backend
.venv\Scripts\activate       # or source .venv/bin/activate
pytest tests/ -v
```
