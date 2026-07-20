---
sidebar_position: 1
title: Installation
description: Install CodefyUI with the one-line installer — end users only need git, uv, and Python. No Node.js required.
---

# Installation

The quick installer automatically sets up `git`, `uv`, and Python (via uv). The frontend bundle is downloaded prebuilt from the latest GitHub release, and the backend is checked out at that same release tag so the two stay in sync — **end users do not need Node.js or pnpm**.

:::tip Which install do I want?
- **Quick Install** (this page) — you just want to *run* CodefyUI.
- **[Dev Install](./dev-install)** — you want to edit the code or contribute (manual `uv` + pnpm setup with hot reload).
:::

## Quick Install

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
```

By default this installs to `~/CodefyUI` (macOS/Linux) or `%USERPROFILE%\CodefyUI` (Windows). Override with the `CODEFYUI_DIR` environment variable.

On Windows, `install.ps1` uses [winget](https://learn.microsoft.com/windows/package-manager/) to install `git` if it's missing. `winget` ships with Windows 11 and recent Windows 10 via the "App Installer" package.

The installer places a `cdui` launcher at `~/.local/bin/cdui` (Windows: `%USERPROFILE%\.local\bin\cdui.cmd`). **Restart your terminal**, then from any directory:

```bash
cdui start
```

Open [http://localhost:8000](http://localhost:8000). A single uvicorn process serves both the API and the prebuilt React app. `cdui start` runs in the **background** by default — you can close the terminal and the server keeps running; manage it with `cdui status` and `cdui stop`. Add `--foreground` (`-f`) to run it attached and stop with `Ctrl+C`.

:::note
This quick start assumes the default PyTorch build, which works on every platform (CPU / Apple Silicon MPS). For a specific NVIDIA CUDA version, AMD ROCm, or to verify GPU detection, see **[GPU & Device Setup](./gpu-device)**.
:::

## Install flags & environment variables

Both `install.sh`/`install.ps1` and `cdui install` (after the first install) accept the same choices, either as CLI flags or pre-set environment variables. Defaults are interactive when stdin is a TTY, and the safe choices otherwise.

| Flag | Env var | Values | Purpose |
|------|---------|--------|---------|
| `--gpu <choice>` | `CODEFYUI_GPU` | `auto` / `cu118` / `cu121` / `cu124` / `cu128` / `rocm6.1` / `rocm6.2` / `cpu` / `mps` / `skip` | Select the PyTorch wheel index. `auto` detects via `nvidia-smi` / `rocm-smi` / Apple Silicon. `skip` installs no torch (advanced). |
| `--dev` / `--no-dev` | `CODEFYUI_DEV` | `1` / `0` | Install the `[dev]` extra (pytest, httpx, httpx-ws). Required for `cdui test`. Off for end users, on for contributors. |
| `--yes` | — | — | Accept all defaults non-interactively (CI / headless). |
| `--lang <code>` | `CODEFYUI_LANG` | `en` / `zh-TW` | Localise the installer prompts. |
| — | `CODEFYUI_DIR` | path | Install directory (default `~/CodefyUI`). |
| — | `CODEFYUI_RELEASE_TAG` | tag | Pin the frontend bundle to a specific release (default `latest`). |
| — | `CODEFYUI_FORCE_BUILD` | `1` | Skip the prebuilt-dist download and build locally with pnpm (tracks `main`). |

## Production vs developer mode

- `cdui start` — single uvicorn on `:8000` serves the prebuilt frontend. **No Node needed.** This is the default end-user mode.
- `cdui dev` — Vite dev server on `:5173` with HMR + uvicorn on `:8000`. **Requires Node 24+ and pnpm.** Use this when editing frontend code — see [Dev Install](./dev-install).
- `cdui build` — rebuild `frontend/dist` locally (also needs Node + pnpm).

See the full list of launcher commands in **[CLI Commands](./cli-commands)**.

## Verify it works

```bash
curl http://127.0.0.1:8000/api/health
```

This should return something like `{"status":"ok","nodes_loaded":94,"presets_loaded":3}` (the `nodes_loaded` count grows with each release — just confirm it's non-zero).

Then open the frontend, load the **Train CNN on MNIST** example, and click **Run**. You should see training progress appear in the bottom panel.

## Updating

```bash
cdui update
```

Updates to the latest release (prebuilt path) or pulls `main` (when building from source) and re-syncs the frontend.

Unlike `cdui install`, this never prompts. It reuses the PyTorch variant and dev tooling already in the venv — reading the variant straight off the installed wheel — so a deliberately chosen torch build is left alone, and an unchanged one isn't re-downloaded. The same `--gpu` / `--dev` flags and `CODEFYUI_GPU` / `CODEFYUI_DEV` env vars still override when you do want a switch.
