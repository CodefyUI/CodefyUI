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
curl -fsSL https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.ps1 | iex"
```

By default this installs to `~/CodefyUI` (macOS/Linux) or `%USERPROFILE%\CodefyUI` (Windows). Override with the `CODEFYUI_DIR` environment variable.

On Windows, `install.ps1` uses [winget](https://learn.microsoft.com/windows/package-manager/) to install `git` if it's missing. `winget` ships with Windows 11 and recent Windows 10 via the "App Installer" package. If `winget` is unavailable or its package sources cannot be reached (corporate TLS interception makes the `msstore` source fail with `0x8a15005e`), the installer falls back to extracting [PortableGit](https://git-scm.com/download/win) into `%LOCALAPPDATA%\CodefyUI\PortableGit` — no administrator rights required.

The installer places a `cdui` launcher at `~/.local/bin/cdui` (Windows: `%USERPROFILE%\.local\bin\cdui.cmd`). **Restart your terminal**, then from any directory:

```bash
cdui start
```

Open [http://localhost:8000](http://localhost:8000). A single uvicorn process serves both the API and the prebuilt React app. `cdui start` runs in the **background** by default — you can close the terminal and the server keeps running; manage it with `cdui status` and `cdui stop`. Add `--foreground` (`-f`) to run it attached and stop with `Ctrl+C`.

:::note
This quick start assumes the default PyTorch build, which works on every platform (CPU / Apple Silicon MPS). For a specific NVIDIA CUDA version, AMD ROCm, or to verify GPU detection, see **[GPU & Device Setup](./gpu-device)**.

Switching build after the fact does not need a terminal either: on a server started with `cdui start`, the **GPU PyTorch** card in the Package Center (toolbar > Settings > Optional packs) installs the matching wheel and restarts the server for you, with the same `cdui install --gpu <choice>` line printed underneath for when you would rather run it yourself. See [Installs that restart the server](/usage/optional-packs#installs-that-restart-the-server).
:::

## Install flags & environment variables

`install.sh` and `install.ps1` read only the environment variables below. They always run `cdui install --yes`, accept no command-line flags, and do not prompt. After installation, run `cdui install` directly to pass flags or use the interactive menu. The menu appears only in a terminal when no flag or environment variable has already selected an option. In a pipe or CI, the command uses the safe defaults.

| Flag | Env var | Values | Purpose |
|------|---------|--------|---------|
| `--gpu <choice>` | `CODEFYUI_GPU` | `auto` / `cu118` / `cu121` / `cu124` / `cu126` / `cu128` / `rocm6.1` / `rocm6.2` / `cpu` / `mps` / `skip` | Select the PyTorch wheel index. `auto` detects via `nvidia-smi` / `rocm-smi` / Apple Silicon. `skip` installs no torch (advanced). |
| `--dev` / `--no-dev` | `CODEFYUI_DEV` | `1` / `0` | Install the `[dev]` extra (pytest, httpx, httpx-ws). Required for `cdui test`. Off for end users, on for contributors. |
| `--yes` | — | — | Accept all defaults non-interactively (CI / headless). |
| `--lang <code>` | `CODEFYUI_LANG` | `en` / `zh` (the environment variable also accepts `zh-TW`, `zh-HK`, `zh-CN`, `english`, and `chinese`) | The flag applies to `cdui install` and `cdui update` only; the environment variable sets the output language of every `cdui` command. |
| — | `CODEFYUI_DIR` | path | Set the installation directory. Default: `~/CodefyUI`. |
| — | `CODEFYUI_RELEASE_TAG` | tag | Pin the frontend bundle and backend checkout to the same release. Default: `latest`. |
| — | `CODEFYUI_FORCE_BUILD` | `1` | Skip the prebuilt distribution download, build locally with pnpm, and track `main`. |
| — | `CODEFYUI_UV_INSTALL_TIMEOUT` | seconds | Set the automatic `uv` download timeout when `uv` is missing from `PATH`. Default: `180`. Set to `0` for no limit. |

## Production vs developer mode

- `cdui start` — single uvicorn on `:8000` serves the prebuilt frontend. **No Node needed.** This is the default end-user mode.
- `cdui dev` — Vite dev server on `:5173` with HMR + uvicorn on `:8000`. **Requires Node 24+ and pnpm.** Use this when editing frontend code — see [Dev Install](./dev-install).
- `cdui build` — rebuild `frontend/dist` locally (also needs Node + pnpm).

See the full list of launcher commands in **[CLI Commands](./cli-commands)**.

## Installing on a server for a team

The steps above install a personal instance on `127.0.0.1`. If several people are going to share one machine, read **[Deployment Behind a Reverse Proxy](/usage/deployment)** first: CodefyUI has no user accounts, so authentication and TLS both come from a proxy in front of it, and the proxy's hostname has to be added to `CODEFYUI_EXTRA_ALLOWED_HOSTS` or every request — including the page itself — is rejected with `421` and the browser shows a blank screen.

## Verify it works

```bash
curl http://127.0.0.1:8000/api/health
```

This should return something like `{"status":"ok","nodes_loaded":152,"presets_loaded":3}` (the `nodes_loaded` count grows with each release — just confirm it's non-zero).

Then open the frontend, load the **Train CNN on MNIST** example, and click **Run**. You should see training progress appear in the bottom panel.

## Optional packs

The install above is deliberately small, so the large extras some lessons need — `sentence-transformers`, the embedding models (90 MB to 470 MB each), the 69 MB GloVe word-vector table — are not in it; install the ones you want from the **Package Center** (toolbar > Settings > Optional packs) or with `cdui packs install <id>`. Nothing else changes: a graph run never downloads pack contents on its own, so a node whose pack is missing stops with a message naming it instead of pulling half a gigabyte mid-run.

See **[Optional Packs](/usage/optional-packs)** for the catalog, where the files land, and which embedding model to pick.

## Updating

```bash
cdui update
```

Updates to the latest release (prebuilt path) or pulls `main` (when building from source) and re-syncs the frontend.

Unlike `cdui install`, this never prompts. It reuses the PyTorch variant and dev tooling already in the venv — reading the variant straight off the installed wheel — so a deliberately chosen torch build is left alone, and an unchanged one isn't re-downloaded. The same `--gpu` / `--dev` flags and `CODEFYUI_GPU` / `CODEFYUI_DEV` env vars still override when you do want a switch.
