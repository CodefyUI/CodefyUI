---
sidebar_position: 7.8
title: Project Directories
description: Turn a service into a self-contained git repo (cdui project) with a logic/layout split, per-project assets and secrets, CI validation, and publish provenance.
---

# Project Directories

A **project directory** is a self-contained git repo that IS your service's
storage. The editor reads and writes its files directly: one clean logic file
per graph, positions in a sibling layout file, per-project assets and secrets,
CI-able validation, and a git commit recorded at every publish.

```
my-service/
  codefyui.project.toml   manifest: name, plugin pins, default publish target
  graphs/    <name>.graph.json    logic (nodes/edges/params/presets)
  layout/    <name>.layout.json   positions (reviewable, generated)
  assets/images/   assets/models/   assets/data/    scaffolded empty
  assets/output/                                    created on demand (e.g. ImageWriter)
  .env.example     committed template of required secret keys
  .env             your secrets (gitignored, never committed)
```

## Why the split?

`graphs/<name>.graph.json` holds only what changes the *behavior* of the graph
(nodes, edges, parameters, embedded presets). Node **positions** and note
geometry live in `layout/<name>.layout.json`. So a drag produces a diff only
in `layout/`, and a parameter edit a diff only in `graphs/` -- code review sees
the logic change, not a wall of moved-pixels noise. (Known exception:
`SequentialModel` sub-graph layer positions live inside `params.layers` and
stay in the logic file.)

A missing layout file (or a node without a saved position) makes the editor
auto-layout the graph on load and persist the result at the next save; a note
missing only its geometry entry (size/binding) simply falls back to defaults --
geometry-only absence intentionally does not count as a missing layout.

## A complete walkthrough

### 1. Create the project

```bash
cdui project init my-service
cd my-service
```

`init` scaffolds `graphs/`, `layout/`, and `assets/{images,models,data}/`
(empty, `.gitkeep`-tracked), writes `.gitignore` / `.gitattributes` /
`.env.example` / `README.md`, and runs `git init` (no commit -- it prints the
next steps). `assets/output/` is not created up front; it appears the first
time a node (for example ImageWriter) writes to it.

### 2. Add a graph

Either build it in the editor (`cdui start --project .`, drop a **Start**, a
**GraphInput** named `x`, and a **GraphOutput** named `y`, wire Start's trigger
into GraphInput and GraphInput's value into GraphOutput, then press
**Ctrl/Cmd+S** and name it `echo`), or drop this file at
`graphs/echo.graph.json`:

```json
{
  "format_version": 1,
  "name": "echo",
  "description": "Echo the input string",
  "nodes": [
    {"id": "start", "type": "Start", "data": {"params": {}}},
    {"id": "gi", "type": "GraphInput", "data": {"params": {"name": "x", "type": "string", "required": true, "default": "", "description": "text to echo"}}},
    {"id": "out", "type": "GraphOutput", "data": {"params": {"name": "y", "description": "the echoed text"}}}
  ],
  "edges": [
    {"id": "t1", "source": "start", "target": "gi", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
    {"id": "d1", "source": "gi", "target": "out", "sourceHandle": "value", "targetHandle": "value", "type": "data"}
  ],
  "presets": []
}
```

### 3. Commit

```bash
git config user.name  "You"
git config user.email "you@example.com"
git add -A
git commit -m "echo service"
```

`.env` is gitignored; `.env.example` is committed. Commit a small fetch script
for large data, never the data or weights themselves.

### 4. Validate (the CI gate)

```bash
cdui project validate .
```

`validate` initializes the FULL registry (builtin + custom + plugin nodes and
presets, exactly like the server) and runs the publish pre-flight on every
graph: the secret-in-graph check, contract, entry points, wiring, and
node/preset validity. It also errors if `.env` is tracked by git, and warns
(errors with `--strict`) on missing plugin pins. In CI, run **restore then
validate**:

```bash
cdui project restore .   # install the manifest's plugin pins by exact SHA
cdui project validate .
```

`validate` checks **every** graph under `graphs/` and prints the checked
count -- an empty `graphs/` reports `Validation passed (0 graphs checked)`
rather than a bare green. A **canvas-only** graph (say, a training graph
that declares no **GraphOutput**) fails the contract gate, because every
publishable graph needs at least one declared output. Either give it a
legitimate output (the MNIST example project publishes its checkpoint path
as a `weights_path` output) or validate only your publish targets:

```bash
cdui project validate . --graph serve   # repeatable: --graph a --graph b
```

A `--graph` name that does not exist in `graphs/` is an error, so a typo can
never turn the CI gate into a vacuous pass.

Pins come from `cdui project freeze .`: it reads your locally-installed
plugins and writes each one's exact commit SHA into `codefyui.project.toml`'s
`[plugins]` table (a plugin you installed as a local dev link is skipped --
there is no SHA to pin for a machine-specific path). Run it after installing
or updating a plugin, and commit the manifest change before your next push:

```bash
cdui project freeze .
```

### 5. Start the server on the project

```bash
cdui start --project .
```

The log prints `Project: <abs> (git <short-sha>)` and warns once, naming
`cdui project restore`, if any pinned plugin is missing.

### 6. Create an API key (invoke needs one)

The session token lives at `<user_data_dir>/codefyui/session.token` -- on
Windows `%LOCALAPPDATA%\codefyui\session.token`, on macOS `~/Library/Application
Support/codefyui/session.token`, on Linux `~/.local/share/codefyui/session.token`
(see [Graph as a Function](./graph-as-a-function.md) for the full breakdown).

PowerShell:

```powershell
# payload.json: {"name": "demo"}
$token = Get-Content "$env:LOCALAPPDATA\codefyui\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/keys" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash:

```bash
TOKEN=$(cat ~/.local/share/codefyui/session.token)   # macOS: ~/Library/Application Support/codefyui/session.token
curl -s -X POST http://127.0.0.1:8000/api/keys \
  -H "X-CodefyUI-Token: $TOKEN" -H "Content-Type: application/json" \
  --data '{"name": "demo"}'
```

`# -> {"id": 1, "name": "demo", "prefix": "cdui_xxxxxxxx", "token": "cdui_..."}` (the full key is shown ONCE, in the "token" field)

### 7. Publish (records the git commit)

`cdui project publish` wraps the same [publish](./publish.md) endpoint
(`POST /api/apps/{slug}/publish`) with a project-mode guard and automatic git
provenance. Set the default target once in `codefyui.project.toml`:

```toml
[publish]
graph = "echo"
slug = "echo-svc"
```

Commit it -- an uncommitted manifest change is exactly the kind of dirty tree
the next step warns about -- then publish:

```bash
git add -A && git commit -m "set publish target"
cdui project publish .
# -> Published echo-svc v1 (git 1a2b3c4)
```

Publish is **local-only** in v1: it confirms `GET /api/health` reports THIS
project open (so it can never record the wrong commit against foreign bytes),
computes `git rev-parse HEAD` + `git status --porcelain`, and warns LOUDLY if
the tree is dirty. Every publish from a git repo records `git_dirty` as
`true` or `false` alongside the commit -- a dirty tree additionally prints
the warning banner above. If `git status` itself fails after the commit was
resolved, `git_dirty` is recorded as `null` (= unknown), never a fabricated
`false`.

Creating the app on first publish is automatic **only** for the manifest's
committed `[publish].slug` target. An explicitly passed `--slug` that names
an app the server does not know fails with 404 `app_not_found` -- a typo can
no longer silently mint a second app -- and the CLI points you at `--create`
for a deliberate first publish of a new command-line slug:

```bash
cdui project publish . --graph echo --slug echo-svc --create
```

> **Remote / CI deploy is out of scope for v1.** `cdui project validate` runs
> in CI, but publishing requires a local server with the project open. The
> named follow-up is a management-scoped, API-key publish (`--url` / `--key`).

### 8. Invoke

PowerShell:

```powershell
# payload.json: {"inputs": {"x": "hello"}}
curl.exe -s -X POST "http://127.0.0.1:8000/api/apps/echo-svc/invoke" `
  -H "Authorization: Bearer cdui_YOUR_KEY" -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash:

```bash
curl -s -X POST http://127.0.0.1:8000/api/apps/echo-svc/invoke \
  -H "Authorization: Bearer cdui_YOUR_KEY" -H "Content-Type: application/json" \
  --data '{"inputs": {"x": "hello"}}'
```

`# -> {"status": "ok", "outputs": {"y": "hello"}, ...}`

### 9. See "which commit built this"

PowerShell:

```powershell
curl.exe -s "http://127.0.0.1:8000/api/apps/echo-svc/versions" -H "X-CodefyUI-Token: $token"
```

bash:

```bash
curl -s http://127.0.0.1:8000/api/apps/echo-svc/versions \
  -H "X-CodefyUI-Token: $TOKEN"
```

`# -> [{"version": 1, "git_commit": "1a2b...", "git_dirty": false, "active": true, ...}]`

The active version's `GET /api/apps/echo-svc/openapi.json` `info` block also
carries `x-codefyui-git-commit` and `x-codefyui-git-dirty`.

## Migrating an existing flat graphs dir

If you followed the older ["version control your graphs"](./version-control-graphs.md)
recipe (a flat dir of `*.json` behind `CODEFYUI_GRAPHS_DIR`), adopt it in one
command:

```bash
cdui project init my-service --adopt /path/to/old-graphs
```

Every `*.json` is copied into `graphs/` and split into the logic/layout pair.

## Notes and limits (v1)

- One project per server instance (no in-editor project switcher yet).
- `DB_PATH` and custom nodes stay install-global; [plugins](/advanced/plugins)
  are the portable mechanism (pinned by SHA in the manifest).
- Last-write-wins between the editor and hand-edits (a "changed on disk"
  warning is a follow-up). Exclude project dirs from OneDrive/Dropbox sync --
  sync clients corrupt `.git` and race atomic renames; use a real git remote.
- A graph written by a newer CodefyUI opens **read-only** (view/run allowed,
  Save disabled) so an older build can never drop fields it does not know.
  Save As is blocked by the identical guard, by design: the in-memory graph
  already lost those unknown fields the moment it loaded, so Save As would
  just write that lossy copy out under a different name.
