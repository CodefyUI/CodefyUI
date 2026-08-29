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
| `cdui update` | Update to the latest release (prebuilt path) or pull `main` (source build) and re-sync the frontend. Never prompts — reuses the PyTorch variant and dev tooling already in the venv unless `--gpu` / `--dev` override. Refuses while a server is running (it would delete the `frontend/dist` that server is serving) — `cdui stop` first. |
| `cdui start` | Production mode — single uvicorn on `:8000`, in the background (no Node needed). `--foreground` / `-f` runs it attached. |
| `cdui status` | btop / k9s-style dashboard: CPU, memory, disk, GPU, top processes, plus the server's PID and health. Refreshes live (every 2s; `Ctrl+C` to quit). Pass a number to set the interval (`cdui status 1`), or `--once` for a single frame. It also reports a pack install that restarts the server: a `Restart install` line while a claim exists — *finishing* while its helper is still working, *abandoned* once it is not — and a `Last restart` line for an hour after one finished. |
| `cdui dev` | Developer mode — backend `:8000` + Vite HMR `:5173` (requires Node + pnpm). |
| `cdui build` | Build the frontend bundle locally (requires Node + pnpm). |
| `cdui stop` | Stop **this install's** services: the background server from the pidfile, plus leftovers started from this directory (a foreground `cdui start`, `cdui dev`'s Vite, stray workers). Add `--all` to stop every CodefyUI and Vite process on the machine instead — that reaches other people's servers and unrelated Vite dev servers, so avoid it on a shared host. |
| `cdui test` | Run the whole project's tests: backend (`pytest`) and frontend (`vitest`). Without pnpm the frontend half is reported as `SKIPPED` rather than failing — a release install has no Node by design. Both halves always finish, so one run tells you about both; the exit code is 1 if either failed. `--backend` / `--frontend` narrow it. |
| `cdui clean` | Remove the virtualenv, `node_modules`, and `frontend/dist`. |
| `cdui uninstall` | Clean + remove the PATH launcher. |

## Plugin commands

| Command | Description |
|---------|-------------|
| `cdui plugin install <name\|url>` | Install a plugin pack (catalog name like `foundations`, `owner/repo[@ref]`, or a full GitHub URL). |
| `cdui plugin sync` | Install every **built-in** pack this install has not decided about yet — the one command to run after an update that shipped a new pack. Asks for confirmation once (`--yes` skips it, and is required when there is no terminal); `--dry-run` only lists; `--prune` also drops lockfile entries for built-in packs that no longer ship. Packs you uninstalled are left alone. |
| `cdui plugin list` | List installed plugin packs, plus any built-in pack still waiting for a decision. |
| `cdui plugin info <id>` | Show a pack's manifest, lessons covered, and node names. |
| `cdui plugin search <query>` | Query the plugin catalog. |
| `cdui plugin uninstall <id>` | Remove an installed plugin pack. For a built-in pack the removal is remembered, so `cdui plugin sync` will not bring it back; `cdui plugin install <id>` undoes that. |

See **[Plugins](/advanced/plugins)** for the full plugin workflow.

## Package commands

Optional packs are the large extras a stock install deliberately leaves out — `sentence-transformers`, the embedding models, the GloVe word-vector table, an accelerated PyTorch build. The in-app **Package Center** installs them with a progress bar; these do the same from a terminal, and are the only way to install a pack that has to replace something the running server already imported.

| Command | Description |
|---------|-------------|
| `cdui packs list` | List every pack in the catalog: what is in it, what it costs to download, and what is already installed. |
| `cdui packs status` | Like `list`, plus the PyTorch build in this venv and the command to run next. |
| `cdui packs install <id>` | Install one pack. `--items a,b` downloads only those models (default: everything the pack is missing); `--yes` / `-y` skips the download-size confirmation, and is required when there is no terminal to confirm at. Only ids from the catalog are accepted — there is no way to pass a package spec, an index URL or a repo id. |
| `cdui packs remove <id> <item-id>` | Delete one downloaded model and forget it. A pack's Python packages are left alone; the `uv pip uninstall` line that would remove them is printed for you to run with the server stopped. |

Exit codes, for scripts: `0` done, `1` the install failed or was declined at the prompt, `2` refused before anything ran (unknown id, unmet dependency, no terminal to confirm at), `3` this cannot be done while the server is running — the command to type instead is printed, `130` cancelled with `Ctrl+C`.

**Installs that restart the server.** A server started with `cdui start` can install the GPU PyTorch pack — or any pack whose live install hits a resolver conflict — by going away and coming back: it writes down what to install, starts `cdui packs-run-pending` detached, and shuts down; that helper waits for the process to go, installs, records the outcome, and starts the server again with the arguments this `cdui start` was given. `packs-run-pending` is **internal** and deliberately absent from the help text — the file it is handed names a process to wait for, so running it by hand against a live server would wait two minutes and then stop it. `CODEFYUI_ENABLE_RESTART_INSTALL=0` turns the whole path off on a machine where the restart does not come back cleanly. While one is still *finishing* — the helper it recorded is alive, or its claim file is under sixty seconds old and no helper has stamped its pid in yet — `cdui start` will not start a second server into the venv that helper is rewriting: it says a restart install is finishing, points at `cdui status`, and returns. Once the helper is gone, or it never arrived and those sixty seconds have passed, the claim is *abandoned*: `cdui start` deletes it and starts normally. See **[Installs that restart the server](/usage/optional-packs#installs-that-restart-the-server)**.

## Background vs foreground

`cdui start` runs in the **background** by default — close the terminal and the server keeps running. Manage it with:

```bash
cdui status     # live dashboard + health
cdui stop       # stop the background server
cdui start -f   # run attached instead (Ctrl+C to stop)
```

## `cdui start` flags

| Flag | Description |
|------|-------------|
| `--foreground`, `-f` | Run attached instead of daemonizing. Required when a supervisor such as systemd owns the process. |
| `--host <addr>` | Bind address (default `127.0.0.1`). `0.0.0.0` or a LAN IP lets other machines connect — anyone who can reach the port controls the instance, so only on a trusted network. See [Publish](/usage/publish). |
| `--port <n>` | Port (default `8000`). |
| `--project <dir>` | Use a project directory containing `codefyui.project.toml` — see [Project Directories](/usage/project-directories). |
| `--` | Everything after a bare `--` is forwarded to uvicorn verbatim, e.g. `cdui start -- --proxy-headers --root-path /x`. `cdui start` reads its own flags only from the part before the separator, so the two sets cannot collide. `--host` and `--port` are refused there (exit code 2) because `cdui` records the bind address itself — use `cdui start --host` instead. |

```bash
# Behind a reverse proxy: bind loopback, trust the proxy's forwarded headers.
cdui start --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1
```

:::warning A proxy also needs its hostname whitelisted
The server answers `421` to any `Host` it does not recognise, including on the page itself, so a proxy in front of it produces a blank browser page until you set `CODEFYUI_EXTRA_ALLOWED_HOSTS` to the public name. Full instructions in **[Deployment Behind a Reverse Proxy](/usage/deployment)**.
:::

## Running a graph without the server

You don't need the web UI to execute a graph — see **[CLI Graph Runner](/usage/cli-runner)**:

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```
