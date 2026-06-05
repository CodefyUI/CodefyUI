---
sidebar_position: 4
title: CLI Commands
description: The cdui launcher commands — install, start, status, dev, build, plugin management, and more.
---

# CLI Commands

`cdui` is a thin launcher (`cdui.cmd` on Windows) placed at `~/.local/bin/cdui` by the installer. If you haven't restarted your terminal yet, invoke the absolute path `~/CodefyUI/cdui start`, or use `python scripts/dev.py <cmd>` — `dev.py` re-execs into the venv's Python automatically.

## Core commands

| Command | Description |
|---------|-------------|
| `cdui install` | Install backend deps; download the prebuilt frontend (or build locally if `pnpm` is available). |
| `cdui update` | Update to the latest release (prebuilt path) or pull `main` (source build) and re-sync the frontend. |
| `cdui start` | Production mode — single uvicorn on `:8000`, in the background (no Node needed). `--foreground` / `-f` runs it attached. |
| `cdui status` | btop / k9s-style dashboard: CPU, memory, disk, GPU, top processes, plus the server's PID and health. Refreshes live (every 2s; `Ctrl+C` to quit). Pass a number to set the interval (`cdui status 1`), or `--once` for a single frame. |
| `cdui dev` | Developer mode — backend `:8000` + Vite HMR `:5173` (requires Node + pnpm). |
| `cdui build` | Build the frontend bundle locally (requires Node + pnpm). |
| `cdui stop` | Stop all services (including the background server). |
| `cdui test` | Run backend tests. |
| `cdui clean` | Remove the virtualenv, `node_modules`, and `frontend/dist`. |
| `cdui uninstall` | Clean + remove the PATH launcher. |

## Plugin commands

| Command | Description |
|---------|-------------|
| `cdui plugin install <name\|url>` | Install a plugin pack (catalog name like `foundations`, `owner/repo[@ref]`, or a full GitHub URL). |
| `cdui plugin list` | List installed plugin packs. |
| `cdui plugin info <id>` | Show a pack's manifest, lessons covered, and node names. |
| `cdui plugin search <query>` | Query the plugin catalog. |
| `cdui plugin uninstall <id>` | Remove an installed plugin pack. |

See **[Plugins](/advanced/plugins)** for the full plugin workflow.

## Background vs foreground

`cdui start` runs in the **background** by default — close the terminal and the server keeps running. Manage it with:

```bash
cdui status     # live dashboard + health
cdui stop       # stop the background server
cdui start -f   # run attached instead (Ctrl+C to stop)
```

## Running a graph without the server

You don't need the web UI to execute a graph — see **[CLI Graph Runner](/usage/cli-runner)**:

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```
