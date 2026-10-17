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
| `cdui install` | Install backend deps; download the prebuilt frontend (or build locally if `pnpm` is available). The frontend step is skipped when `frontend/dist/index.html` already exists, unless `CODEFYUI_FORCE_BUILD=1`. |
| `cdui update` | Update the checkout, then reinstall. Which source it takes depends on whether pnpm is on `PATH` when you run it, not on how CodefyUI was installed. Without pnpm, it checks out the latest release tag (or `CODEFYUI_RELEASE_TAG`) with `git checkout -f`, which discards local changes to tracked files, and downloads that release's frontend. It looks the latest tag up through the GitHub API and then the release page, which the API's hourly limit does not cover; if both fail, it stops before changing anything and names `CODEFYUI_RELEASE_TAG=<tag>` (skips the lookup) and `CODEFYUI_FORCE_BUILD=1` (tracks `main`). With pnpm, or with `CODEFYUI_FORCE_BUILD=1`, it resets the local `main` branch to the fetched `origin/main` (`git fetch origin main --depth 1`, then `git checkout -B main FETCH_HEAD`) and rebuilds the frontend; a tag set in `CODEFYUI_RELEASE_TAG` is checked out instead, and its frontend is built locally. Do not run it in a development clone that holds local work. The command does not prompt. It reuses the PyTorch variant and development tools in the venv unless `--gpu` or `--dev` overrides them. It refuses to run while the server is active because updating removes the served `frontend/dist`; run `cdui stop` first. It also exits with code `1` while a restart install is finishing. See [Package commands](#package-commands). |
| `cdui start` | Production mode — single uvicorn on `:8000`, in the background (no Node needed). `--foreground` / `-f` runs it attached. It exits with code `1` when `frontend/dist/index.html` is missing; run `cdui install` or `cdui build`. It also exits with code `1`, launching nothing, when a server or another program already holds the port, and says which. A server on another port refuses to start while one is running on the same data directory or run database — see [Deployment](/usage/deployment#3-bind-loopback). When the dist was built from a commit whose frontend differs from the checkout's (recorded in `frontend/dist/build-info.json`), or uncommitted changes make `frontend/src` newer than the dist, it prints a warning and starts anyway. The warning says to run `cdui build` when pnpm is on `PATH`, otherwise `cdui update`. |
| `cdui run <graph.json>` | Submit a saved graph to the running server's per-device FIFO queue. The server owns the run, so closing the terminal does not stop it. Flags: `--name`, `--device` (`cpu` / `auto` / `cuda` / `cuda:N` / `mps`), `--target` (only `local` today), `--devices` (a device set of one kind, e.g. `cuda:0,cuda:1`; its first device is the run's device), `--seed` (default: the graph's `settings.seed`) or `--no-seed`, `--deterministic`, `--record-outputs`, `--wait` (default) or `--detach`, `--timeout <s>`, and `--host` / `--port`. Exit codes: `0` for success or submission with `--detach`; `1` for failure, cancellation, or submission failure; `2` for an invalid command line; and `130` for `Ctrl+C`. `Ctrl+C` stops waiting but does not stop the run. See **[Run Queue](/usage/run-queue#cdui-run)**. |
| `cdui status` | Display a btop / k9s-style dashboard with CPU, memory, disk, GPU, top processes, and the server's PID and health. It refreshes every 2 seconds by default; press `Ctrl+C` to quit. Pass a number to set the interval, as in `cdui status 1`, or use `--once` (or `-1`) for one frame. `-w` / `--watch [secs]` enables continuous output even when stdout is not a terminal, with a minimum interval of 0.5 seconds. In single-frame mode, which includes `--once` and piped output, the command exits with code `1` when the server is not running. During a restart install, it displays `Restart install` while a claim exists, with a state of `finishing` while the helper or the install it started is active, or `abandoned` after both stop. It displays `Last restart` for one hour after completion. |
| `cdui dev` | Run the backend on `:8000` and Vite HMR on `:5173` for development. Requires Node and pnpm. Accepts `--project <dir>` like `cdui start`. Exits with code `1` while a restart install is finishing, and when a server or another program already holds `:8000`. |
| `cdui build` | Build the frontend bundle locally (requires Node + pnpm). |
| `cdui stop` | Stop **this install's** services: the background server from the pidfile, plus leftovers started from this directory (a foreground `cdui start`, `cdui dev`'s Vite, stray workers). Add `--all` to stop every CodefyUI and Vite process on the machine instead — that reaches other people's servers and unrelated Vite dev servers, so avoid it on a shared host. |
| `cdui test` | Run the backend (`pytest`) and frontend (`vitest`) test suites. A release installation has no Node, so without pnpm the frontend suite is reported as `SKIPPED` instead of failing. Both suites complete, and the command exits with code `1` if either fails. When the backend suite is selected and `backend/.venv` or its `pytest` is missing (an install without `--dev`), the command exits with code `1` before either suite runs; run `cdui install --dev`. Use `--backend` or `--frontend` to run one suite. Any other argument is rejected with exit code `2`. To filter individual tests, run `pytest` or `pnpm test` directly. |
| `cdui clean` | Remove the virtualenv, `node_modules`, and `frontend/dist`. |
| `cdui uninstall` | Clean + remove the PATH launcher. The install directory stays, including your saved graphs and uploads in `backend/data/` and the plugin lockfile, session token, server log and pack downloads in `.codefyui_dev/`. To remove everything, delete that directory by hand and remove any `PATH` entry the installer added: the block marked `# >>> CodefyUI PATH (cdui) >>>` in your shell's startup files on macOS/Linux, or the user `PATH` entry on Windows. If the Windows installer fell back to PortableGit, also delete `%LOCALAPPDATA%\CodefyUI\PortableGit` and remove `%LOCALAPPDATA%\CodefyUI\PortableGit\cmd` from the user `PATH`. |
| `cdui --version` | Print `CodefyUI <version>`. The aliases are `-V` and `cdui version`. This check runs before `uv` or the venv is required, so it also works when installation is incomplete. |

## Plugin commands

The in-app **Plugin Center**, available from the sidebar's **Custom & Plugins** tab or **Settings → Optional Packs & Plugins**, uses the same installation function as these commands. See **[Plugin Center](/usage/plugin-center)**.

| Command | Description |
|---------|-------------|
| `cdui plugin install <source>...` | Install one or more plugin packs, in the order given; the command stops at the first one that fails and returns its exit code. A source is a catalog name, `owner/repo[@ref]`, or a full GitHub URL. Catalog names cover the built-in packs and the official plugins that live in their own repositories alike, so `cdui plugin install graph-copilot` fetches the repository the catalog names for you. The manifest is read and shown first — what the pack is, the Python packages it would add, the capabilities it declares — so you are asked before anything is downloaded, and `no` costs nothing. `--force` reinstalls over what is already there, `-y` / `--no-confirm` skips only the `Proceed?` question, `--accept-capabilities` grants the declared capabilities without a prompt, `--trust-author` accepts a pack that asks to import outside the allowlist. |
| `cdui plugin sync` | Install every **built-in** pack this install has not decided about yet — the one command to run after an update that shipped a new pack. Asks for confirmation once (`--yes` skips it, and is required when there is no terminal); `--dry-run` only lists; `--prune` also drops lockfile entries for built-in packs that no longer ship. Packs you uninstalled are left alone. |
| `cdui plugin update [<id>]` | Re-read a plugin's own repository at its recorded ref and reinstall if the commit moved; without an id, every installed third-party pack. The pack keeps the catalog row it was installed from, so an official plugin still reads as official after its first update, and it keeps being switched off if you had switched it off. A repository whose manifest now declares a *different* plugin id is refused (exit `1`) rather than installed under the new name — updating one plugin never installs another, or overwrites one you already have; install the renamed repository as the new plugin it is. Built-in and linked packs are skipped — a built-in pack updates with `cdui update`, and a linked directory is already whatever is on its author's disk. |
| `cdui plugin list` | List installed plugin packs, plus any built-in pack still waiting for a decision. |
| `cdui plugin info <id \| catalog-name \| owner/repo[@ref]>` | Show a plugin's manifest, covered lessons, and node names. For an uninstalled plugin, it reads only the manifest at the resolved commit and downloads no plugin files. |
| `cdui plugin search [query]` | Query the plugin catalog. Without a query, list the full catalog and mark installed entries. GitHub-hosted entries include tags; official entries use `[github, official]`. |
| `cdui plugin uninstall <id>` | Remove an installed plugin pack. For a built-in pack the removal is remembered, so `cdui plugin sync` will not bring it back. The command then prints the line that installs the plugin again, whether or not any packages were left: the catalog name for a built-in or catalog plugin, `owner/repo@ref` for a plugin installed from any other GitHub repository, and `cdui plugin link <path>` for a linked folder that still exists. A source that cannot be installed again gets no command. A pack's Python packages are left installed either way — removing packages from inside the interpreter that imported them is how you get a half-loaded server — so the command names the packages the pack declared in `[python_deps]` that nothing else still needs, and prints the `uv pip uninstall` line that removes them, to run with the server stopped. |
| `cdui plugin enable <id>` / `cdui plugin disable <id>` | Enable or disable an installed plugin without changing its files. The command updates `enabled` in the lockfile and hot-reloads the running server. It exits with code `1` if the plugin is not installed. If the plugin already has the requested state, it makes no change and exits with code `0`. |
| `cdui plugin link <path>` | Register a local directory that contains `cdui.plugin.toml` and load the plugin from that directory without copying it. `--force` replaces an existing entry with the same id. |
| `cdui plugin unlink <id>` | Remove a linked plugin's lockfile entry without changing its files. |
| `cdui plugin reload` | Request a hot reload of plugins and nodes from the running server. The `link`, `enable`, and `disable` commands request the same reload automatically. |
| `cdui plugin dev <path>` | Link a local plugin and monitor its directory. A change to its manifest, nodes, presets, or frontend triggers a reload. `--interval <s>` sets the check interval; the default is `1` second and the minimum is `0.2` seconds. `--once` links the plugin, reloads once, and exits. The editor reloads the plugin's frontend bundle automatically only when the page was opened while an enabled linked plugin was installed; after linking a plugin into an editor that is already open, reload the browser once. |
| `cdui plugin new <id>` | Create a plugin directory from the built-in template, including a manifest, example node, and test. `--ui` adds a React frontend that uses the SDK. `--name` sets the display name, which otherwise derives from the id. `--dir` sets the parent directory, which defaults to the current directory. `--force` permits writing to a non-empty directory. |

Every subcommand that changes the lockfile, and `reload`, asks the running server to hot-reload. The request connects to `127.0.0.1` on `CODEFYUI_PORT` (default `8000`); see [Environment variables](#environment-variables).

Exit codes, for scripts: `0` done — including declining at the `Proceed?` prompt, which is an answer rather than a failure; `1` the install failed, or a capability or module request was refused; `2` a source could not be parsed (the sources before it are already installed), or no source was given; `3` the plugin's Python packages cannot be installed while the server is running — the command to type instead is printed (with no server running, the install upgrades the packages instead, keeping the PyTorch build pinned); `130` cancelled with `Ctrl+C`.

See **[Plugins](/advanced/plugins)** for the full plugin workflow.

## Package commands

Optional packs are the large extras a stock install deliberately leaves out — `sentence-transformers`, the embedding models, the GloVe word-vector table, an accelerated PyTorch build. The in-app **Package Center** installs them with a progress bar; these do the same from a terminal, except for a pack that has to replace something already imported: `cdui packs` refuses the GPU PyTorch pack (exit `2`) and stops at a package conflict (exit `3`), and prints the command to run instead. On a server started with `cdui start`, the Package Center installs both kinds by restarting the server (below).

| Command | Description |
|---------|-------------|
| `cdui packs list` | List every pack in the catalog: what is in it, what it costs to download, and what is already installed. |
| `cdui packs status` | Like `list`, plus the PyTorch build in this venv and the command to run next. |
| `cdui packs install <id>` | Install one pack. `--items a,b` downloads only those models (default: everything the pack is missing); `--yes` / `-y` skips the download-size confirmation, and is required when there is no terminal to confirm at. Only ids from the catalog are accepted — there is no way to pass a package spec, an index URL or a repo id. |
| `cdui packs remove <id> <item-id>` | Delete one downloaded model and forget it. A pack's Python packages are left alone; the `uv pip uninstall` line that would remove them is printed for you to run with the server stopped. |

Script exit codes are: `0` when complete; `1` when installation fails or the user declines the prompt; `2` when validation fails before installation, including an unknown pack or `--items` id, an unmet dependency, a restart-only pack, or no terminal for confirmation; `3` when the server must be stopped before the operation; and `130` when cancelled with `Ctrl+C`. For the restart-only `gpu-torch` pack, the command prints the required `cdui install --gpu` command. For exit code `3`, it also prints the command to run instead.

**Installs that restart the server.** A server started with `cdui start` can install the GPU PyTorch pack — or any pack whose live install hits a resolver conflict — by going away and coming back: it writes down what to install, starts `cdui packs-run-pending` detached, and shuts down; that helper waits for the process to go, installs, records the outcome, and starts the server again with the arguments this `cdui start` was given, minus `--foreground`: the server always comes back in the background. `packs-run-pending` is **internal** and deliberately absent from the help text — the file it is handed names a process to wait for, so running it by hand against a live server would wait two minutes and then stop it. `CODEFYUI_ENABLE_RESTART_INSTALL=0` turns the whole path off on a machine where the restart does not come back cleanly. While one is still *finishing* — the helper it recorded, or the `uv` install that helper started, is alive, or its claim file is under sixty seconds old and no helper has stamped its pid in yet — `cdui start` will not start a second server into the venv that helper is rewriting: it says a restart install is finishing, points at `cdui status`, and returns. Once both are gone, or the helper never arrived and those sixty seconds have passed, the claim is *abandoned*: `cdui start` deletes it and starts normally. See **[Installs that restart the server](/usage/optional-packs#installs-that-restart-the-server)**.

## Cache commands

Some nodes write results they can rebuild, and until you delete them nothing else will. `LMTokenizedDataset` is the big one: it packs a whole corpus into one token stream under `<data>/cache/lm_blocks/`, one file per distinct corpus, tokenizer, `seq_len`, `append_eos` and `max_tokens`, at 8 bytes per token — around 800 MB per file for a 100M-token corpus, so sweeping `seq_len` over three values leaves three full copies behind. These commands cover derived caches only: downloaded models and assets are untouched (`cdui packs remove` deletes those), and so are run outputs, saved models and graphs.

| Command | Description |
|---------|-------------|
| `cdui cache list` | Every derived cache: how many entries it holds, how much disk that is, and the directory it lives in. |
| `cdui cache prune` | Delete those entries, after a `[y/N]` confirmation. `--older-than DAYS` keeps anything written more recently than that (last *written* — a read does not count, because a cache hit does not touch the file). `--yes` / `-y` skips the prompt, and is required when there is no terminal to confirm at. Refuses while a **background** server (`cdui start`) is running: a graph in it may be part-way through reading a block file. Only background — a foreground `cdui dev` or `cdui start -f` writes no pidfile, so nothing can detect it: stop one yourself first. |

Both take `--project <dir>`, the same flag you started the server with: project mode keeps the cache in `<dir>/assets/cache/`, and without the flag these answer about this install's `<data>/cache` instead.

Exit codes, for scripts: `0` done (including "there was nothing to delete"), `1` declined at the prompt or an entry could not be deleted, `2` refused before anything ran (a negative `--older-than`, no terminal to confirm at), `3` a background server is running — `cdui stop` first, `130` cancelled with `Ctrl+C`.

## Project commands

A project directory is a Git repository used as the service's storage. It contains one logic file and one layout file per graph, plus project-specific assets and secrets. See **[Project Directories](/usage/project-directories)** for the full workflow.

| Command | Description |
|---------|-------------|
| `cdui project init <dir>` | Create `graphs/`, `layout/`, `assets/images/`, `assets/models/`, `assets/data/`, the manifest `codefyui.project.toml`, `.gitignore`, `.gitattributes`, `.env.example`, and `README.md`, and then run `git init` without making a commit. None of these files is overwritten if it already exists. [Project Directories](/usage/project-directories#1-create-the-project) lists what `.gitattributes` holds and the line a project created before 2.6.0 must add to its own. A non-empty directory needs `--force` or `--adopt`. `--adopt <old-graphs-dir>` copies every `*.json` file from a flat graphs directory and splits each graph into logic and layout files, replacing a graph or layout file of the same name that is already there. |
| `cdui project validate <dir>` | Load the full node registry and run publish's checks on every graph under `graphs/`; [Project Directories](/usage/project-directories#4-validate-the-ci-gate) lists where the two differ. The command also fails if Git tracks `.env`. Repeat `--graph <name>` to select specific graphs. `--strict` treats missing plugin pins as errors instead of warnings. |
| `cdui project freeze <dir>` | Write the exact commit SHA of each installed GitHub plugin to the manifest's `[plugins]` table. Linked local plugins are skipped. |
| `cdui project restore <dir>` | Install the manifest's plugin pins at their exact SHAs. Run this command before `validate` in CI. |
| `cdui project publish <dir>` | Publish a graph to the local server (at `127.0.0.1` on `CODEFYUI_PORT`, default `8000`) and record the Git commit in the version. `--graph` and `--slug` override the manifest's `[publish]` target. `--note` adds an immutable version note. `--create` permits the first publish to a `--slug` that the server does not yet know. |

## Agent commands

A worker agent runs graphs for a CodefyUI server (the control plane), each run in a subprocess of its own, so a crash, an out-of-memory kill or a forced stop ends that run and nothing else. The agent connects out to the server's `/ws/worker` socket and never listens on a port. Without flags it runs beside `cdui start` on the same machine; on another machine it enrolls once with a token from the Workers panel (see [Remote Workers](/usage/remote-workers#adding-a-worker)). Send a run to it with `cdui run --target <name>` or `options.target` (see [Run Queue, Workers](/usage/run-queue#workers)).

| Command | Description |
|---------|-------------|
| `cdui agent start` | Start the agent in the background. Its pid is kept in `agent.pid` and its log in `agent.log`, both in the user-data directory (`.codefyui_dev/` in an install), with each run's output in `agent-runs/<run id>/output.log`, kept for the same number of runs as the run history. Without flags it uses the address of the last `cdui start` and that server's session token, and reads both again on every reconnect, so a restarted server is found again. If either cannot be found and the terminal is interactive, it asks for that value; without a terminal, a missing token makes `start` exit with code `2` and name the token file it looked for. `--control <url>` sets the server URL (with no recorded address and no terminal: `http://127.0.0.1:8000`), `--token <token>` the connection token (else `$CODEFYUI_AGENT_TOKEN`, else the server's token file), `--name <name>` the worker name (default: the hostname; lower-case letters, digits and `-`, at most 63 characters; `local`, `any`, `enroll` and `enrollments` are reserved), `--devices cuda:0,cuda:1` limits the devices it offers to the listed ones plus the CPU, which is always offered (without the flag it offers every device it finds, `mps` included on Apple silicon; a device the machine does not have is an error), `--mount data=<path>` (repeatable) names the shared folder this machine holds under a storage label, so runs that use files can be placed here (see [Storage: shared filesystem](/usage/remote-workers#storage-shared-filesystem)); mounts are saved in `agent.json` in the user-data directory and reused by later starts, and `--mount data=` removes one; while a background agent runs, `--mount` exits with code `1` and saves nothing. `--foreground` keeps it attached, for systemd (a unit file: [Remote Workers, Running the agent as a service](/usage/remote-workers#running-the-agent-as-a-service)). A `--token` of the enrollment-token form (`abcdef.0123456789abcdef`) enrolls with `--control` first: the worker credential it gets is stored in `<user-data>/agent/<host>-<port>.token` (0600), and every later `start --control <url>` uses it, with or without `--token`, and asks for nothing. A failed enrollment exits with code `1` and the server's reason; a start that finds an agent already running reports it before enrolling. For an `https` control plane the agent checks the certificate against this machine's CAs; `--ca-fingerprint sha256:<hex>` instead pins the server (leaf) certificate's public key, for a self-signed certificate, records it in the credential file, and refuses a different key with exit code `2`. Plain `http` to another machine prints one warning (see [Remote Workers, Trust between agent and control plane](/usage/remote-workers#trust-between-agent-and-control-plane)). `python -m app.agent start` (or `stop`), run from `backend/` in its venv, is the same entry point. A second `start` while one is running reports the running agent and exits with code `0`. The agent itself never imports PyTorch; only its run subprocesses do. A run sees only the GPUs it was assigned, in the order the assignment lists them, as `cuda:0` to `cuda:N-1`: a run given `cuda:2` and `cuda:0` calls the machine's `cuda:2` `cuda:0`. A CPU or MPS run sees none. The run records what it saw in `execution_facts` as `visible_devices` (its `CUDA_VISIBLE_DEVICES`, such as `"2,0"`) and `device_aliases` (`{"cuda:2": "cuda:0", "cuda:0": "cuda:1"}`). In the foreground, `Ctrl+C` stops the agent the same way `cdui agent stop` does. |
| `cdui agent forget <url>` | Delete the worker credential stored for that control plane (with the pinned fingerprint it recorded), so the next `start` enrolls again. |
| `cdui agent stop` | Stop the background agent. Running runs are asked to stop and are terminated after 10 seconds. On Windows the agent and its run subprocesses are terminated at once. |

## Background vs foreground

`cdui start` runs in the **background** by default — close the terminal and the server keeps running. Manage it with:

```bash
cdui status     # live dashboard + health
cdui stop       # stop the background server
cdui start -f   # run attached instead (Ctrl+C to stop)
```

In background mode, the server writes all output to `<install dir>/.codefyui_dev/server.log`. Both `cdui start` and `cdui status` print this path.

## `cdui start` flags

| Flag | Description |
|------|-------------|
| `--foreground`, `-f` | Run attached instead of daemonizing. Required when a supervisor such as systemd owns the process. |
| `--host <addr>` | Bind address (default `127.0.0.1`). `0.0.0.0` or a LAN IP lets other machines connect — anyone who can reach the port controls the instance, so only on a trusted network. See [Publish](/usage/publish). |
| `--port <n>` | Port (default `8000`). |
| `--project <dir>` | Use a project directory containing `codefyui.project.toml` — see [Project Directories](/usage/project-directories). |
| `--` | Pass all arguments after a bare `--` to uvicorn without modification. For example: `cdui start -- --proxy-headers --root-path /x`. `cdui start` parses its own flags only before the separator. It rejects `--host`, `--port`, and `--ws-max-size` after the separator with exit code `2`. CodefyUI records the bind address and derives the WebSocket limit from `CODEFYUI_WS_MAX_MESSAGE_BYTES`; use `cdui start --host` / `--port` and the environment variable instead. |

```bash
# Behind a reverse proxy: bind loopback, trust the proxy's forwarded headers.
cdui start --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1
```

:::warning A proxy also needs its hostname whitelisted
The server answers `421` to any `Host` it does not recognise, including on the page itself, so a proxy in front of it produces a blank browser page until you set `CODEFYUI_EXTRA_ALLOWED_HOSTS` to the public name. Full instructions in **[Deployment Behind a Reverse Proxy](/usage/deployment)**.
:::

## Environment variables

| Variable | Read by | Meaning |
|----------|---------|---------|
| `CODEFYUI_DIR` | the one-line installers | Installation directory. Default: `~/CodefyUI`. |
| `CODEFYUI_RELEASE_TAG` | installers, `cdui install`, `cdui update` | Release to install. Default: `latest`. The installers and `cdui update` check out this tag and download its frontend; with `CODEFYUI_FORCE_BUILD=1` (or, for `cdui update`, pnpm on `PATH`) they still check out the tag and build its frontend locally. `cdui install` only uses it to pick which release's frontend to download, when it downloads one. |
| `CODEFYUI_FORCE_BUILD` | installers, `cdui install`, `cdui update` | Set to `1` to build the frontend locally with pnpm instead of downloading the release bundle, and to track `main` when `CODEFYUI_RELEASE_TAG` names no release. |
| `CODEFYUI_GPU` | `cdui install`, `cdui update` | Default value for `--gpu`. The command-line flag takes precedence. See [Installation](/getting-started/installation) for valid values. |
| `CODEFYUI_DEV` | `cdui install`, `cdui update` | Default value for `--dev`. Enable with `1`, `true`, or `yes`; disable with `0`, `false`, or `no`. |
| `CODEFYUI_LANG` | every command | Output language for `cdui` commands. English values: `en` or `english`. Chinese values: `zh`, `zh-TW`, `zh-HK`, `zh-CN`, or `chinese`. When unset, `LANG` and the system locale determine the language. The `plugin`, `project`, `packs` and `cache` groups recognise only `zh` itself, and when the variable is unset they read `LANG` / `LC_ALL` but not the system locale. |
| `NO_COLOR` | every command | Set to a non-empty value to turn off ANSI colour. Colour is also off when output is not a terminal. |
| `CODEFYUI_UV_INSTALL_TIMEOUT` | every command that may need `uv` | Time in seconds allowed for the automatic `uv` download when `uv` is missing from `PATH`. Default: `180`. Set to `0` for no limit. The one-line installers download `uv` themselves, with no time limit. |
| `CODEFYUI_USER_DATA_DIR` | `cdui start`, `cdui dev`, `cdui run`, the `plugin` / `project` / `cache` / `packs` groups, and the server | Directory for the session token, plugin lockfile, asset cache, ChatGPT sign-in, and restart-install files. Unless already exported, those commands set it to `<install dir>/.codefyui_dev/`. See [Project Directories](/usage/project-directories#6-create-an-api-key-invoke-needs-one). |
| `CODEFYUI_HOST`, `CODEFYUI_PORT` | the server, the `plugin` group, `cdui project publish` | Bind address and port. `cdui start --host` and `--port` export them to the server they start. Set them directly for a manually launched uvicorn process. The server uses them to derive the Host allowlist and loopback-only installation restrictions. See [Dev Install](/getting-started/dev-install). The plugin commands' hot reload and `cdui project publish` connect to `127.0.0.1` on `CODEFYUI_PORT` (default `8000`); they do not read the port `cdui start` recorded, so when the server runs on another port, export `CODEFYUI_PORT` in the shell you run them from. |
| `CODEFYUI_ENABLE_RESTART_INSTALL` | the server | Set to `0` to disable installations that restart the server. |
| `CODEFYUI_AGENT_TOKEN` | `cdui agent start` | Connection token for the agent when `--token` is not given; it takes precedence over the server's token file. A token given with `--token` reaches the background agent process this way. |
| `CODEFYUI_GITHUB_TOKEN` | plugin installs | GitHub token used by `cdui plugin install`, `info`, and `update`; `cdui project restore`; and the Plugin Center. It increases GitHub's unauthenticated limit of 60 API requests per hour per IP. The token is read for each call, sent only to GitHub, and never logged. |

See [Deployment](/usage/deployment) for `CODEFYUI_EXTRA_ALLOWED_HOSTS`, `CODEFYUI_WS_MAX_MESSAGE_BYTES`, `CODEFYUI_LOG_LEVEL`, `CODEFYUI_LOG_DIR`, and `CODEFYUI_LOG_JSON`. See [Run Queue](/usage/run-queue#configuration) for queue settings.

## Running a graph without the server

You don't need the web UI to execute a graph — see **[CLI Graph Runner](/usage/cli-runner)**. The runner needs the backend's virtual environment:

```bash
cd backend
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
```
