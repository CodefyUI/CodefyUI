---
sidebar_position: 7.7
title: Version Control Your Graphs
description: Keep your graph JSON in a git service repo, validate every graph in CI with run_graph.py, and publish from a versioned source.
---

# Version Control Your Graphs

:::tip First-class project directories now exist
This page describes an older flat-directory recipe whose shape does **not**
forward-map to the split logic/layout format. For new services use
[Project directories](./project-directories); migrate an existing graphs dir
in one command: `cdui project init my-service --adopt /path/to/my-graphs`.
:::

:::tip The editor can run the git steps for you
Once the graphs live in a project directory, the editor's **Source Control**
tab does the git of [Set up a service repo](#set-up-a-service-repo) for you:
stage, commit, branch, push, settle a conflict, and read the history and the
diffs (the one-off credential setup still happens in a terminal). See
[Source Control](./source-control).
:::

CodefyUI saves each graph as a plain JSON file, so graphs are a natural fit for git: you get history, review, and rollback for the pipelines you build. This page is a recipe for keeping your graphs in their own git repository, validating them in CI, and publishing from a versioned source.

One catch up front: the default save location is **not** version-controlled. Saved graphs land in `backend/data/graphs/`, which the repo's own `.gitignore` excludes (`backend/data/graphs/*.json`, keeping only `.gitkeep`). To version your work you point CodefyUI's graph directory at a repo you own -- the rest of this page shows how.

## What to version, and what never to

**Version these:**

- Your graph JSON files -- one `<name>.json` per saved graph.
- Small data or model files you own and want reproducible, or a script that fetches the large ones.

**Never commit these.** They are machine-local state, secrets, or derived data, and none of them live in the graph JSON -- most sit outside your graphs directory entirely by default, so keep them there:

- The SQLite database `codefyui.db` (default `backend/data/codefyui.db`, overridable with `CODEFYUI_DB_PATH`). It holds published apps, versioned snapshots, API keys, and run records.
- Run records -- stored only in that database, never in a file you would commit.
- Published-app API keys (the `cdui_...` bearer tokens) -- also database-only, kept as sha256 hashes.
- `.env` files and any local secrets.
- The editor session token file: `<install dir>/.codefyui_dev/session.token` for a server started with `cdui start` or `cdui dev` (default install dir `~/CodefyUI`), or `<dir>/session.token` when `CODEFYUI_USER_DATA_DIR` is set. Only a `uvicorn app.main:app` started by hand uses the platform user-data directory (`%LOCALAPPDATA%\codefyui\session.token` on Windows). It is written outside your graphs directory and rotates on every server restart, so it should never be copied into a repo.
- LLM provider API keys -- keep them in environment variables; see [Secrets](#secrets-keep-keys-out-of-your-graphs) below.

## Set up a service repo

Create a directory for your graphs, initialize git, and point CodefyUI at it with `CODEFYUI_GRAPHS_DIR`. Then start the server, save graphs from the UI, and commit them like any other source.

```bash
mkdir my-graphs && cd my-graphs
git init
```

Point CodefyUI's graph directory at it. The variable is read once at server startup, so set it in the same shell (or session) you launch `cdui start` from.

PowerShell:

```powershell
$env:CODEFYUI_GRAPHS_DIR = "C:\path\to\my-graphs"
cdui start
```

cmd.exe:

```bat
set CODEFYUI_GRAPHS_DIR=C:\path\to\my-graphs
cdui start
```

bash:

```bash
export CODEFYUI_GRAPHS_DIR=/path/to/my-graphs
cdui start
```

Now every graph you save from the UI is written as `<name>.json` inside `my-graphs/`. Save a graph, then commit it:

```bash
git add .
git commit -m "Add my first classifier graph"
```

**`CODEFYUI_GRAPHS_DIR` must be set every time the server starts** -- it is not persisted anywhere. Launch a fresh terminal without it and `cdui start` falls back to the default `backend/data/graphs/`, so your service repo will look empty. Set it once in your shell profile (PowerShell `$PROFILE`, or `~/.bashrc` / `~/.zshrc`), or wrap the two lines in a tiny start script you keep next to the repo:

```bash
# start.sh
export CODEFYUI_GRAPHS_DIR="$(cd "$(dirname "$0")" && pwd)"
cdui start
```

## A .gitignore for your service repo

Drop this in the root of your graphs repo so weights, databases, and secrets never sneak in:

```
*.pt
*.pth
*.safetensors
*.onnx
*.ckpt
*.db
.env
__pycache__/
```

For large datasets, commit a small download script (or a URL plus a checksum) rather than the data itself -- keep the repo to graphs and the code that fetches everything else.

## Secrets: keep keys out of your graphs

`LLMChat` has two API-key fields, `openai_api_key` and `anthropic_api_key`. They are secret parameters: a value typed into one is used only while that editor session lasts, and it is blanked from saved graphs, exports, published versions and run history ([Shared Instances](./shared-instances#what-is-per-graph-instead) has the full list). A graph you commit therefore carries no key, and the field is empty again after a reload or on another machine.

Provide the key through the environment instead. The node reads the first non-empty value it finds, in this order:

1. the node's `openai_api_key` field (current session only)
2. `CODEFYUI_OPENAI_API_KEY` (environment)
3. `OPENAI_API_KEY` (environment)

Anthropic works the same way with `anthropic_api_key`, then `CODEFYUI_ANTHROPIC_API_KEY`, then `ANTHROPIC_API_KEY`. Set the environment variable before `cdui start`.

A key can still reach a commit if you paste it into a field that is not secret, such as a prompt, or write it into the JSON by hand; the publish pre-flight refuses a graph file that still carries a secret value. A node from a plugin that is not loaded is not recognised, so its secret fields are not blanked when the graph is saved.

## Validate every graph in CI

`run_graph.py` can check a graph without executing it: it discovers all nodes, validates the DAG, types, ports, and Start wiring, and exits non-zero if anything is wrong. That is exactly what you want in CI -- a broken graph fails the build. See [CLI Graph Runner](./cli-runner) for the runner itself.

```bash
# locally, from a CodefyUI checkout
cd backend
python run_graph.py /path/to/my-graphs/classifier.json --validate-only
```

The runner lives inside the CodefyUI backend, and there is no standalone package on PyPI today, so the most reliable way to get it in a *separate* graphs repo is to check CodefyUI out alongside your repo at a pinned release tag and install its backend with uv (mirroring how CodefyUI's own CI installs itself). Installing the backend pulls the full runtime, including PyTorch, so the job is not featherweight -- cache the venv or expect a few minutes on a cold run. This is the honest state today; a lightweight validate command is a natural future.

The job below assumes this repo carries the project-directory layout (a `codefyui.project.toml` manifest plus `graphs/`/`layout/`) rather than a bare flat `*.json` folder, since `cdui project validate` needs that manifest -- see [Project directories](./project-directories) to migrate with `cdui project init <dir> --adopt <this-repo>`. Staying on the flat layout instead? Keep validating file-by-file with `run_graph.py` as shown above.

```yaml
name: validate-graphs
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Check out this graphs repo
        uses: actions/checkout@v4

      - name: Check out CodefyUI (pinned)
        uses: actions/checkout@v4
        with:
          repository: CodefyUI/CodefyUI
          ref: "2.8.4" # pin a release tag, 1.4.0 or later (cdui project arrived in 1.4.0)
          path: CodefyUI

      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> "$GITHUB_PATH"

      - name: Install the CodefyUI backend
        working-directory: CodefyUI/backend
        run: |
          uv venv
          uv pip install -e .

      - name: Restore plugin pins, then validate the project
        run: |
          ./CodefyUI/cdui project restore .    # or: CodefyUI/backend/.venv/bin/python CodefyUI/scripts/dev.py project restore .
          ./CodefyUI/cdui project validate .   # checks every graph in the project
```

The job calls the `cdui` launcher by its path inside the checkout:
`uv pip install -e .` installs no `cdui` command, so a bare `cdui` is not on
`PATH` there. `cdui project validate .` checks every graph in the project
([Project directories](./project-directories#4-validate-the-ci-gate) lists the
checks). It never hands `layout/*.layout.json` files to the validator, so
there is no `*.json` glob to get wrong. Run `cdui project restore` first so
plugin-provided nodes are installed before validation (CI order: restore, then
validate).

## Publishing from a versioned graph

Version control does not change how you publish: save the graph, then [publish](./publish) it. Because publish snapshots the exact bytes of the saved graph file into the database at publish time, the version you ship is frozen independently of any later edit or commit.

To tie a published version back to its source, put the graph's git commit hash in the publish `note` field -- it is free-text, stored with the version, and echoed in the versions list:

```json
{"graph": "classifier", "create": true, "note": "git 1a2b3c4"}
```

That gives you a trail from a running app version back to the exact commit it came from -- or skip the manual `note` convention entirely: [Project directories](./project-directories) ship first-class publish provenance, where `cdui project publish` records the exact `git_commit`/`git_dirty` on every version automatically.

## Known rough edges

The flat `<name>.json` recipe on this page has two sources of diff noise:

- **Node positions add diff noise.** Dragging a node changes its saved coordinates, so rearranging the canvas produces JSON diffs even when the pipeline is unchanged.
- **Copy/paste regenerates node ids.** Duplicating nodes assigns fresh ids, which can make a small logical change look like a large diff.

A [project directory](./project-directories) reduces both. Positions are saved in `layout/<name>.layout.json`, apart from the logic file, and **Hide layout files** keeps those files out of the Source Control tab's Changes list. The tab's graph summary counts moved positions and ignores regenerated edge ids; a node with a new id still counts as one removed and one added. See [What changed in the graph](./source-control#what-changed-in-the-graph).
