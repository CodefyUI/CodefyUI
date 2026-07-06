---
sidebar_position: 7.7
title: Version Control Your Graphs
description: Keep your graph JSON in a git service repo, validate every graph in CI with run_graph.py, and publish from a versioned source.
---

# Version Control Your Graphs

CodefyUI saves each graph as a plain JSON file, so graphs are a natural fit for git: you get history, review, and rollback for the pipelines you build. This page is a recipe for keeping your graphs in their own git repository, validating them in CI, and publishing from a versioned source.

One catch up front: the default save location is **not** version-controlled. Saved graphs land in `backend/data/graphs/`, which the repo's own `.gitignore` excludes (`backend/data/graphs/*.json`, keeping only `.gitkeep`). To version your work you point CodefyUI's graph directory at a repo you own — the rest of this page shows how.

## What to version, and what never to

**Version these:**

- Your graph JSON files — one `<name>.json` per saved graph.
- Small data or model files you own and want reproducible, or a script that fetches the large ones.

**Never commit these.** They are machine-local state, secrets, or derived data, and none of them live in the graph JSON — most sit outside your graphs directory entirely by default, so keep them there:

- The SQLite database `codefyui.db` (default `backend/data/codefyui.db`, overridable with `CODEFYUI_DB_PATH`). It holds published apps, versioned snapshots, API keys, and run records.
- Run records — stored only in that database, never in a file you would commit.
- Published-app API keys (the `cdui_...` bearer tokens) — also database-only, kept as sha256 hashes.
- `.env` files and any local secrets.
- The editor session token file at `<user_data_dir>/codefyui/session.token` (Windows `%LOCALAPPDATA%\codefyui\session.token`). It is written outside your graphs directory and rotates on every server restart, so it should never be copied into a repo.
- LLM provider API keys — see [Secrets](#secrets-keep-keys-out-of-your-graphs) below, because these are the one secret that can end up *inside* a graph you commit.

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

**`CODEFYUI_GRAPHS_DIR` must be set every time the server starts** — it is not persisted anywhere. Launch a fresh terminal without it and `cdui start` falls back to the default `backend/data/graphs/`, so your service repo will look empty. Set it once in your shell profile (PowerShell `$PROFILE`, or `~/.bashrc` / `~/.zshrc`), or wrap the two lines in a tiny start script you keep next to the repo:

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

For large datasets, commit a small download script (or a URL plus a checksum) rather than the data itself — keep the repo to graphs and the code that fetches everything else.

## Secrets: keep keys out of your graphs

The LLM nodes (for example LLMChat) expose API-key parameter fields such as `openai_api_key` and `anthropic_api_key`. **A value typed into one of these fields is saved verbatim into the graph JSON** — so if you commit that graph, you commit your key. Do not do that.

Leave the field blank and provide the key through the environment instead. The node reads the first non-empty value it finds, in this order:

1. the node's `openai_api_key` field (saved in the graph — avoid)
2. a generic `api_key` field on the node (also saved in the graph — avoid)
3. `CODEFYUI_OPENAI_API_KEY` (environment)
4. `OPENAI_API_KEY` (environment)

Anthropic works the same way with `CODEFYUI_ANTHROPIC_API_KEY` then `ANTHROPIC_API_KEY`. Set the environment variable before `cdui start`, keep the node field empty, and your saved graph carries no secret. If you ever paste a key into a node to test, clear it before you save and commit.

## Validate every graph in CI

`run_graph.py` can check a graph without executing it: it discovers all nodes, validates the DAG, types, ports, and Start wiring, and exits non-zero if anything is wrong. That is exactly what you want in CI — a broken graph fails the build. See [CLI Graph Runner](./cli-runner) for the runner itself.

```bash
# locally, from a CodefyUI checkout
cd backend
python run_graph.py /path/to/my-graphs/classifier.json --validate-only
```

The runner lives inside the CodefyUI backend, and there is no standalone package on PyPI today, so the most reliable way to get it in a *separate* graphs repo is to check CodefyUI out alongside your repo at a pinned release tag and install its backend with uv (mirroring how CodefyUI's own CI installs itself). Installing the backend pulls the full runtime, including PyTorch, so the job is not featherweight — cache the venv or expect a few minutes on a cold run. This is the honest state today; a lightweight validate command is a natural future.

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
          ref: "1.3.0" # pin to a release tag for reproducibility
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

      - name: Validate every graph
        run: |
          status=0
          for f in $(git ls-files '*.json'); do
            echo "== validating $f =="
            CodefyUI/backend/.venv/bin/python \
              CodefyUI/backend/run_graph.py "$f" --validate-only || status=1
          done
          exit $status
```

`git ls-files '*.json'` lists only the JSON files tracked in *your* graphs repo, so the loop skips the nested CodefyUI checkout automatically. Any single failure sets `status=1`, and the final `exit $status` fails the whole job.

## Publishing from a versioned graph

Version control does not change how you publish: save the graph, then [publish](./publish) it. Because publish snapshots the exact bytes of the saved graph file into the database at publish time, the version you ship is frozen independently of any later edit or commit.

To tie a published version back to its source, put the graph's git commit hash in the publish `note` field — it is free-text, stored with the version, and echoed in the versions list:

```json
{"graph": "classifier", "create": true, "note": "git 1a2b3c4"}
```

That gives you a trail from a running app version back to the exact commit it came from. First-class project directories and richer publish provenance are on the roadmap.

## Known rough edges

Versioning graphs works today, but a few things produce friction and are being addressed:

- **Node positions add diff noise.** Dragging a node changes its saved coordinates, so rearranging the canvas produces JSON diffs even when the pipeline is unchanged.
- **Copy/paste regenerates node ids.** Duplicating nodes assigns fresh ids, which can make a small logical change look like a large diff.
- **Saving over an existing name overwrites silently.** Saving a graph under a name that already exists replaces the file with no warning — commit early and lean on git to recover a previous version.
