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
  graphs/    <name>.graph.json    logic (nodes/edges/params/presets/subgraphs, optional settings.device / settings.seed)
  layout/    <name>.layout.json   positions, note geometry, segments (reviewable, generated)
  assets/images/   assets/models/   assets/data/    scaffolded empty
  assets/output/                                    created on demand (e.g. ImageWriter)
  assets/media/                                     created on demand (run-produced video, served by /api/media)
  .env.example     committed template of required secret keys
  .env             your secrets (gitignored, never committed)
```

## Why the split?

`graphs/<name>.graph.json` holds only what changes the *behavior* of the graph
(nodes, edges, parameters, embedded presets, subgraph definitions, and an optional `settings` block,
for example `{"device": "cuda"}`, which assigns the device the graph runs on). Node **positions**
(including those of the nodes inside a subgraph definition), note geometry and the
compare segments (`segmentGroups`) live in `layout/<name>.layout.json`. So a drag produces a diff only
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
(empty, `.gitkeep`-tracked), writes `codefyui.project.toml` / `.gitignore` /
`.gitattributes` / `.env.example` / `README.md`, and runs `git init` (no commit
-- it prints the next steps). `--force` writes into a directory that is not
empty; none of those five files is overwritten if it already exists.
`assets/output/` is not created up front; it appears the first time a node (for
example ImageWriter) writes to it.

`.gitattributes` holds two lines. `*.json text eol=lf` keeps graph and layout
files LF on checkout: the server writes them with LF, and a Windows checkout
with `core.autocrlf=true` would otherwise hand them back as CRLF, so every save
would rewrite every line. `layout/*.layout.json linguist-generated=true`
collapses layout files in GitHub diffs. A project created before 2.6.0 keeps
its older `.gitattributes`, which lacks the first line; add
`*.json text eol=lf` to it yourself.

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
  "presets": [],
  "settings": {"device": "cuda"}
}
```

`settings` is optional. A graph without it runs on the device the client
chooses: the Settings device in the editor, or `cpu` for `cdui run` with no
`--device`. The values `settings.device` accepts, and when Save writes the
block, are in [Device Backends](../advanced/device-backends.md#the-graph-settings-object).

### 3. Commit

```bash
git config user.name  "You"
git config user.email "you@example.com"
git add -A
git commit -m "echo service"
```

`.env` is gitignored; `.env.example` is committed. Commit a small fetch script
for large data, never the data or weights themselves. Once the server runs on
the project (step 5), the editor's [Source Control](./source-control) tab does
this same staging and committing from the sidebar.

### 4. Validate (the CI gate)

```bash
cdui project validate .
```

`validate` initializes the FULL registry (builtin + custom + plugin nodes and
presets, exactly like the server) and runs publish's checks on every graph, in
the same order: the secret-in-graph check, contract, entry points, wiring, and
node/preset validity. It is not identical to publish:

- It also checks `settings.device` (an invalid value is an `invalid_settings`
  error); publish does not.
- It does not pass the graph's subgraph definitions to the node check, so a
  graph that uses a subgraph fails with `Unknown subgraph` although publish
  accepts it.
- Its secret check reads top-level nodes only: a SECRET value inside a subgraph
  or preset definition passes `validate`, and publish refuses it with 409
  `secret_in_graph`.

The last two are known issues. `validate` also errors if `.env` is tracked by
git, and warns (errors with `--strict`) on missing plugin pins. In CI, run
**restore then validate**:

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

Freeze rewrites the manifest in place: any custom keys you have added (in
`[project]`, `[publish]`, or your own tables) are preserved, but comments are
not, and the `[plugins]` table is fully regenerated from what is installed.

### 5. Start the server on the project

```bash
cdui start --project .
```

The log prints `Project: <abs> (git <short-sha>)` and warns once, naming
`cdui project restore`, if any pinned plugin is missing. (`cdui dev --project .`
does the same with hot reload.)

Starting with `--project` is also what loads `<project>/.env`: plain
`KEY=VALUE` lines (a leading `export ` and surrounding quotes are tolerated),
applied before node and plugin discovery with `os.environ.setdefault`
semantics, so a variable already set in the shell wins. Only execution-time
secrets belong there -- LLM API keys, anything a node reads from the
environment at run time. `CODEFYUI_*` settings in the file do nothing, because
the server's configuration is fixed before the file is read; set those in the
shell or the systemd unit. The values are never logged, only their count, and
without `--project` no `.env` is read at all.

### 6. Create an API key (invoke needs one)

The session token is `<install dir>/.codefyui_dev/session.token` for a server
started by `cdui start` or `cdui dev` (default install dir `~/CodefyUI`, i.e.
`$HOME\CodefyUI` on Windows), or `<CODEFYUI_USER_DATA_DIR>/session.token` when
that variable was exported before the launcher ran. See
[Graph as a Function](./graph-as-a-function.md#2-getting-the-token-for-external-scripts)
for the other files in that directory, the platform directories a hand-launched
`uvicorn app.main:app` uses, and why the token rotates on every restart.

PowerShell:

```powershell
# payload.json: {"name": "demo"}
$token = Get-Content "$HOME\CodefyUI\.codefyui_dev\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/keys" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash:

```bash
TOKEN=$(cat ~/CodefyUI/.codefyui_dev/session.token)
curl -s -X POST http://127.0.0.1:8000/api/keys \
  -H "X-CodefyUI-Token: $TOKEN" -H "Content-Type: application/json" \
  --data '{"name": "demo"}'
```

`# -> {"id": 1, "name": "demo", "prefix": "cdui_xxxxxxx", "token": "cdui_..."}` (the full key is shown ONCE, in the "token" field)

### 7. Publish (records the git commit)

`cdui project publish` wraps the same [publish](./publish.md) endpoint
(`POST /api/apps/{slug}/publish`) with a project-mode guard and automatic git
provenance. Set the default target once in `codefyui.project.toml`:

```toml
[publish]
graph = "echo"
slug = "echo-svc"
```

`graph` (and `--graph`) is the graph's
[address](./graph-as-a-function.md#the-graphs-address): its file name under
`graphs/` without `.graph.json`, which can differ from the title the **Graphs**
tab shows. The scaffolded manifest also has a commented-out
`record_io = true`. When `record_io` is set, every `cdui project publish` sends
it, so it re-applies that value each time, including over a change made with
`PATCH /api/apps/{slug}`; leave it out to keep the app's current setting (see
[Publish](./publish.md#1-publish-lifecycle)).

Commit it -- an uncommitted manifest change is exactly the kind of dirty tree
the next step warns about -- then publish:

```bash
git add -A && git commit -m "set publish target"
cdui project publish .
# -> Published echo-svc v1 (git 1a2b3c4)
cdui project publish . --note "first cut"   # --note attaches an immutable note to the version
```

Publish is **local-only** in v1: it confirms `GET /api/health` reports THIS
project open (so it can never record the wrong commit against foreign bytes),
computes `git rev-parse HEAD` + `git status --porcelain`, and warns LOUDLY if
the tree is dirty. Every publish from a git repo records `git_dirty` as
`true` or `false` alongside the commit -- a dirty tree additionally prints
the warning banner above. If `git status` itself fails after the commit was
resolved, `git_dirty` is recorded as `null` (= unknown), never a fabricated
`false`.

It connects to `http://127.0.0.1:8000`, or to the port in `CODEFYUI_PORT` when
that variable is set in the shell running the command; unlike `cdui run`, it
does not use the address `cdui start` recorded. For a server started with
`--port 9000`, run `CODEFYUI_PORT=9000 cdui project publish .` (PowerShell: set
`$env:CODEFYUI_PORT = "9000"` first). A server bound to a single LAN address
(`--host 192.168.1.20`) does not listen on 127.0.0.1, so this command cannot
reach it.

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

## The default project and switching projects {/* #the-default-project-and-switching-projects */}

### Which project the server opens

At start-up the server picks, in this order:

1. The project named with `--project` or `CODEFYUI_PROJECT_DIR`.
2. Flat mode, when `CODEFYUI_GRAPHS_DIR` is set.
3. The project that was open last time (`<user data>/projects.json`).
4. Flat mode, when `backend/data/graphs` still holds graphs from before
   projects. Nothing is moved; the project chip offers to copy them into a
   project (see below).
5. Otherwise the **default project** `~/CodefyUI-Projects/workspace`
   (`%USERPROFILE%\CodefyUI-Projects\workspace` on Windows), created with its
   own git repository the first time (see the next section for its branch and
   first commit). `CODEFYUI_PROJECTS_ROOT` moves the
   `CodefyUI-Projects` folder; `CODEFYUI_DEFAULT_PROJECT=0` turns this step
   off. A `workspace` folder that exists, is not empty and is not a project is
   left alone: the server stays in flat mode and logs a warning.

### A new project's repository

The default project, **New project…** and an import into a new project make
the folder a git repository on your `init.defaultBranch`, or `main` when you
have not set one, with one commit, `Initial project`, holding the starter
files. Source Control starts clean, and the first graph you save is the
first change it shows. Without a git identity (`user.name`, `user.email`)
that one commit is made as `CodefyUI <codefyui@localhost>`, given for that
commit only and written to no config. `cdui project init` still makes no
commit.

### The project chip

The chip at the left of the toolbar names the open project, its git branch,
and shows a dot while there are uncommitted changes. On a window narrower than
1600px the chip shows the name alone, with the branch and whether there are
uncommitted changes in its tooltip and in the first row of its menu, and
**Reload Nodes** moves from the toolbar into the **File** menu; together they
keep the toolbar on one line. Its menu opens a recent
project, a folder (**Open folder…**), a new project (**New project…**, created
under the projects root; the name is checked as you type), or a clone (**Clone from a git
URL…**, https only, cloned into the projects root under the repository's
name). In flat mode it reads **No project** and, when there are graphs from
before projects, offers to copy them into a project together with the models,
images and media files they reference; the dialog lists each file with its
size. Everything is copied, never moved.

Switching reloads the page. Each project has its own tabs (see
[Tabs & Persistence](./tabs-persistence)); tabs with unsaved changes are
listed before you switch and stay with the project you leave.

**A switch is refused** (the message says what is busy) while a run is
running, queued or being submitted, a study is running, a plugin or pack is
being installed, a git operation is in progress in Source Control, another
switch is in progress, or a connected worker declares this server's mount
label with a folder of its own (`--mount data=...`): stop that worker,
switch, and start it again with the new project's folder.

**Other windows.** A window that still shows the project you left notices
when you come back to it (or when it tries to save or run anything) and
offers to reload.

**Runs and apps belong to their project.** The Runs panel lists this
project's runs; **All projects** lists every run. A run of another project
keeps its history, but its files (checkpoints, TensorBoard logs, what a
worker uploaded) are available only while that project is open, and it is
not removed by retention until then. An app version published from a
project is served only while that project is open (409
`app_project_not_open` otherwise). Studies, sweeps and apps are not yet
filtered by project.

### The project's `.env`

A project this install created (the default project, **New project**, an
import) and a project named with `--project` load their `.env` as before.
For any other folder you open, and for a clone, the editor first lists the
variable **names** (never the values) and asks. The answer is remembered for
those exact contents in `<user data>/trust.json`; when the file changes, you
are asked again. Saying no opens the project without those variables.

Some names are never loaded when a project is opened at run time, whatever
their letter case (`https_proxy` is blocked as well as `HTTPS_PROXY`):

- how programs start and where code comes from: `PATH`, `PYTHON*`, `LD_*`,
  `DYLD_*`, `CODEFYUI_*`, `NODE_OPTIONS`, `NODE_PATH`, `PERL5*`, `RUBY*`,
  `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS`;
- where packages are installed from: `PIP_*`, `UV_*`, `CONDA_*`;
- how git, ssh and shells run: `GIT_*`, `SSH_*`, `HOME`, `USERPROFILE`,
  `SHELL`, `COMSPEC`, `BASH_ENV`, `ENV`, `IFS`, `XDG_*`, `EDITOR`, `VISUAL`,
  `PAGER`, `BROWSER`, `TMPDIR`, `TEMP`, `TMP`;
- where network traffic goes and which certificates it trusts: `*_PROXY`
  (`HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, …), `SSL_CERT_*`,
  `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`.

Names ending in `_URL`, `_BASE`, `_ENDPOINT`, `_HOST` or `_API_BASE` are
loaded, but the prompt marks them: such a variable can send requests, and
the keys that go with them, to another server (`OPENAI_BASE_URL` pointing
somewhere you do not know, for example). Open the `.env` and check their
values before you agree.

And some libraries read their settings only once, when they
start (`CUDA_VISIBLE_DEVICES`, `HF_HOME` and the other `HF_*`,
`TORCH_HOME`, `OMP_NUM_THREADS` and similar): after a switch the editor says
which of these need a server restart to take effect.

## Migrating an existing flat graphs dir

If you followed the older ["version control your graphs"](./version-control-graphs.md)
recipe (a flat dir of `*.json` behind `CODEFYUI_GRAPHS_DIR`), adopt it in one
command:

```bash
cdui project init my-service --adopt /path/to/old-graphs
```

Every `*.json` is copied into `graphs/` and split into the logic/layout pair.

## Notes and limits (v1)

- One project is open at a time per server; switch with the project chip (see above).
- `DB_PATH`, custom nodes and [exported presets](/advanced/presets#creating-your-own)
  stay install-global; [plugins](/advanced/plugins) are the portable mechanism
  (pinned by SHA in the manifest).
- `assets/data/` is where a relative `Dataset` or `FileReader` path resolves.
  `CSVReader`, `DocumentLoader` and `TextCorpusDataset` resolve a typed
  relative path against the project root instead, so write
  `assets/data/table.csv`, not `table.csv`; a path that leaves the project is
  refused. A bare file name is first looked up among the files uploaded through
  their DATA_FILE dropdown, which are install-global -- `backend/data/files`,
  or `CODEFYUI_DATA_FILES_DIR` -- like `DB_PATH` and custom nodes. The bundled
  samples still resolve against the install: `data/samples/iris.csv` in
  `CSVReader`, and paths under `data/samples/` in `DocumentLoader`.
- `CODEFYUI_MODELS_DIR`, `CODEFYUI_IMAGES_DIR` and `CODEFYUI_MEDIA_DIR` relocate
  the model, image and run-media stores; in project mode they default to
  `<project>/assets/models`, `assets/images` and `assets/media` unless set
  explicitly.
- Last-write-wins between the editor and hand-edits. Only a write made through
  the [Source Control](./source-control) tab (a pull, a checkout, a stash pop,
  a discard) raises the "changed on disk" toast with its Reload button; an edit
  from any other tool is not detected. Exclude project dirs from OneDrive/Dropbox
  sync -- sync clients corrupt `.git` and race atomic renames; use a real git
  remote.
- A graph written by a newer CodefyUI opens **read-only** (view/run allowed,
  Save disabled) so an older build can never drop fields it does not know.
  Save As is blocked by the identical guard, by design: the in-memory graph
  already lost those unknown fields the moment it loaded, so Save As would
  just write that lossy copy out under a different name.
