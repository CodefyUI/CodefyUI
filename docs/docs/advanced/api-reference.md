---
sidebar_position: 5
title: API Reference
description: Backend routes and authentication requirements for nodes, graphs, runs, sweeps, published apps, plugins, optional packs, source control, files, media, and the LLM proxy, plus request limits, error formats, and the execution WebSocket protocol.
---

# API Reference

The backend serves a REST API and an execution WebSocket. All endpoints use the same origin as the app (`http://localhost:8000` by default). The **Auth** column uses the five values defined under [Authentication](#authentication). Each table links to the relevant usage page.

The server also serves FastAPI's generated schema: `GET /openapi.json` (OpenAPI 3.1, every HTTP route on this page; the WebSocket is not in it), `GET /docs` (Swagger UI, with its helper page `/docs/oauth2-redirect`) and `GET /redoc`. All four are open. `/docs` and `/redoc` load their scripts from cdn.jsdelivr.net, so the browser needs internet access to show them; `/openapi.json` does not. When the prebuilt frontend is present, a `GET` for any other path outside `/api/` and `/ws/` returns the file at that path in the frontend build when there is one (the `/assets/...` bundle, `/build-info.json`, and in a release `/LICENSE`, `/NOTICE` and `/THIRD_PARTY_NOTICES.md`), else `index.html`. A missing file under `/assets/` and an unmatched path under `/api/` or `/ws/` are 404s, never the editor page.

## Authentication

| Auth | Meaning |
|------|---------|
| open | No credential. Most `GET` / `HEAD` / `OPTIONS` requests use this; the published-app and API-key rows below name the exceptions. |
| token | The `X-CodefyUI-Token` session header. The generic middleware requires it for every `POST` / `PUT` / `PATCH` / `DELETE` under `/api/` except `/api/apps` and `/api/keys`, whose routes enforce one of the policies in this table. It is also the credential for every app-management route (`GET /api/apps` and `GET /api/apps/{slug}/versions` included) and all of `/api/keys`. Missing or wrong: `403 {"detail": "Missing or invalid X-CodefyUI-Token header"}`. Where the token file lives and why it rotates on every restart: [Graph as a Function, section 2](/usage/graph-as-a-function#2-getting-the-token-for-external-scripts). |
| token+loopback | The token, and the server must be bound to a loopback address (`127.0.0.1`, `localhost`, `::1`) — otherwise 403 — unless `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` (packs) or `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1` (plugins) was exported first. The gate reads `CODEFYUI_HOST`, never the socket. |
| API key | Only `Authorization: Bearer cdui_...` is accepted. A session token returns an error that states that an API key is required. Authentication failures use status 401 inside the run envelope and include `WWW-Authenticate: Bearer`. Keys are created through `/api/keys`; see [Publish](/usage/publish#2-api-keys). |
| key-or-token | Either credential is accepted. These metadata reads are another exception to the general rule that `GET` requests are open. A request with neither credential returns `401 {"detail": "..."}`. |

The Host guard runs before all other checks on every request, including the SPA page and WebSocket. A `Host` header outside the allowlist returns `421 {"detail": "Misdirected Request (Host not allowed)"}`. The allowlist is derived from `CODEFYUI_HOST` and `CODEFYUI_PORT` and extended by `CODEFYUI_EXTRA_ALLOWED_HOSTS`; see [Publish, section 6](/usage/publish#6-serving-on-your-lan) and [Deployment](/usage/deployment). The token middleware exempts `GET /api/auth/bootstrap` so the frontend can obtain the token. Any request with an allowed `Host` can use this endpoint, including a remote client when the server is bound to a LAN address.

## Health and system

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/health` | GET | open | Health probe with `status`, `version`, `boot_id` (a process identifier that lets clients detect a server restart), `nodes_loaded`, `presets_loaded`, `caches` (current bytes in each in-memory store and its budget; see [Training Memory](./training-memory)), and `project` (the absolute project directory) when the server runs with `--project`. |
| `/api/auth/bootstrap` | GET | open | `{"token": "..."}` for any allowed-Host request — how the frontend gets the session token. |
| `/api/system/devices` | GET | open | Compute devices for graph execution: the best-available `default` plus a labelled `devices` list that tells NVIDIA CUDA, AMD ROCm and Apple MPS apart. Backs the editor's device selector. |

## Nodes and presets

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/nodes` | GET | open | List all node definitions. Each node carries `details` (its longer description, empty when `description` says it all), `provider` (`builtin`, `custom`, or `plugin:<id>`) and `requires_pack` (the pack the node declares, or `null`), and each SELECT param `option_packs` (option value to pack id). The editor keeps an unavailable current value selectable with a warning, greys out other unavailable options, and offers the install. Neither field stops a run; a node that must not run without its pack refuses in `execute`, as the built-in nodes do (see [Custom Nodes](/advanced/custom-nodes#anatomy-of-a-node)). |
| `/api/nodes/{node_name}` | GET | open | Get a single node definition. |
| `/api/nodes/reload` | POST | token | Re-discover every node and preset source: custom nodes and plugins are re-imported from disk, built-ins are re-registered (not re-imported), presets are re-scanned. Returns `{builtin, custom, plugins, presets, total}`; identical to `POST /api/plugins/reload`. |
| `/api/nodes/script/validate` | POST | token | Check one PythonScript body (`{"code"}`) against the Tier-0 policy while it is being typed: `{ok, error, line, defines_run, allowed_modules}`. `ok: false` is a normal 200, not an error. |
| `/api/presets` | GET | open | List preset definitions. |
| `/api/presets/{name}` | GET | open | Get a single preset definition. |
| `/api/presets/create` | POST | token | Create a new preset from the request's complete `nodes` and `edges`; the editor submits the whole current canvas. |

## Graphs

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/graph/validate` | POST | token | Validate a graph. |
| `/api/graph/save` | POST | token | Save a graph. The body is the graph plus two optional fields: `file`, the address (filename stem) to write to, and `overwrite`. Omitting `file` means "derive the address from `name`", which is a Save As. Send `overwrite: false` with such a request to opt into the collision guard: one that lands on an address something already occupies then answers `409` with `{"detail": {"error": "graph_exists", "file", "name"}}` instead of writing, where `file` is the stem the server resolved and `name` is the title of the graph already there — re-send with `overwrite: true` to replace it. Leaving `overwrite` out entirely keeps the behaviour this route has always had: the save is written, occupied address or not, so existing clients and scripts are unaffected. A request that carries `file` is saving in place and is never refused this way. The response is `{message, path, file}`: `path` is the file written and `file` is the address the server derived, which later saves, loads and renames use. |
| `/api/graph/load/{name}` | GET | open | Load a saved graph. |
| `/api/graph/list` | GET | open | List saved graphs as `[{name, file, modified}]`: `name` is the title stored in the file, `file` is the address to load, rename or delete it by, and `modified` is the file's modification time in epoch seconds (in a project directory, the `.graph.json` file's, not the layout file's). In a project directory, 409 when a name exists both as `<name>.graph.json` and as a legacy `<name>.json`. |
| `/api/graph/rename` | POST | token | Rename a saved graph: `{"from": "<file>", "to": "<new name>"}`, both non-empty; any other key is 422. Moves the file (and, in a project directory, its `layout/` file) and writes `to` into it as its `name`. Returns `{message, name, file}`, where `file` is the sanitized address to use from then on. 404 when `from` does not exist; 409 when `to` is taken, when a layout file for `to` exists without its graph, or when a name exists in both file forms; 400 when the file is not readable JSON or, in a project directory, when `to` ends in `.graph` or `.layout`. Backs **Rename** in the sidebar's **Graphs** tab ([Tabs & Persistence](/usage/tabs-persistence#saving-and-loading)). |
| `/api/graph/{name}` | DELETE | token | Delete a saved graph, with its layout file in a project directory. Returns `{message, removed}`, the list of deleted paths. 404 when nothing is there; 409 when the name exists in both file forms. Backs **Delete** in the same tab. |
| `/api/graph/export` | POST | token | Export a single-file headless Python runner. It embeds the graph and requires a compatible CodefyUI backend environment, but no running server. |
| `/api/graph/contract/{name}` | GET | open | The saved graph's derived function signature — `inputs`, `outputs`, `problems` — for scripting against it. See [Graph as a Function](/usage/graph-as-a-function#3-inspect-the-contract). |
| `/api/graph/run/{name}` | POST | token | Run the latest saved file as a function: `{inputs, timeout_s, device, record_outputs}` in, the 9-key envelope out on every outcome. See [Graph as a Function](/usage/graph-as-a-function#4-run-the-graph). |
| `/api/examples/list` | GET | open | List the built-in example graphs and those of enabled plugins: `[{name, description, category, path, source, node_count, edge_count, section, family, order}]`. `source` is `builtin` or `plugin:<id>`, `node_count` leaves out notes, and `section`, `family` and `order` place the example in the gallery (see [Examples Gallery, Adding an example](/usage/examples-gallery#adding-an-example)). |
| `/api/examples/load` | GET | open | Load an example graph; `?path=` is a list row's `path`. |

## Runs and sweeps

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/runs` | POST | token | Submit a run to the queue and return immediately: `{run_id, status: "running" \| "queued"}`. |
| `/api/runs` | GET | open | Newest-first list; `?status=` repeats, `?limit=` has a maximum of 500, and `?offset=` selects the starting row. Every row includes `queue_position`, `active`, and `final_metrics`; the response also includes the unpaged `total`. |
| `/api/runs/{run_id}` | GET | open | Return one run and `last_cursor`, which clients use to start event polling. |
| `/api/runs/{run_id}` | DELETE | token | Drop a finished run with its events, metrics, artifact rows and captured outputs; 409 while it is queued or running. Artifact files stay on disk. |
| `/api/runs/{run_id}/cancel` | POST | token | Cooperative stop — `{run_id, status, cancelled}`; `cancelled: false` (still 200) for a run that had already finished. |
| `/api/runs/{run_id}/events` | GET | open | Events after `?cursor=`; `?wait=` long-polls for up to 60 s, and `?limit=` accepts up to 2000. Returns `{run_id, status, active, events[{cursor, type, payload, ts}], cursor}`. A page can contain fewer than `limit` events; use the returned `cursor` for the next request. |
| `/api/runs/{run_id}/metrics` | GET | open | Recorded scalar series in `(name, step)` order; `?name=` filters, `?format=csv` downloads (UTF-8 BOM, formula-safe cells). The JSON form also lists `names`. |
| `/api/runs/{run_id}/artifacts` | GET | open | Files the run recorded (checkpoints, exports, images), oldest first; `?kind=` filters, and an unknown kind is an empty list. |
| `/api/sweeps` | POST | token | Compile a parameter sweep into variant runs and queue them all — `201` with the `sweep_id` and one `run_id` per variant. |
| `/api/sweeps/{sweep_id}` | GET | open | The ranked comparison table — `variants` best first, `best`, `counts`, and `objective_warning` when there is one; `?format=csv` downloads it. |
| `/api/sweeps/{sweep_id}/cancel` | POST | token | Cancel every queued or running variant: `{sweep_id, state, cancelled, already_finished, variants[]}`, one entry per variant in index order. |

**Runs API.** `POST /api/runs` accepts `{"graph": {...}, "options": {...}, "name": "..."}`. The graph uses the saved-graph JSON format (`nodes`, `edges`, and optional `presets`, `subgraphs` and `settings`). The endpoint returns 422 when the body is not JSON or has no `graph` object, 400 for an invalid graph (for example empty `nodes` or a bad `settings.device`) or option, and 503 when the run service is unavailable or an `interactive`-lane submission exceeds the limit. Option keys are a closed set: `device`, `seed`, `deterministic`, `record_outputs`, `lane`, the canvas flags `verbose`, `graph_id`, `weights_persistent`, `backward_mode`, `auto_backward`, and the engine error policy `error_mode`, `max_retries`. The run's device is `options.device` when given, else the graph's `settings.device` (see [the graph settings object](/advanced/device-backends#the-graph-settings-object)), else `cpu`. A run's `status` is one of `queued`, `running`, `succeeded`, `failed`, `cancelled`, or `interrupted`. Two limits apply to `/events`: a payload over `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES` (128 KB) is stored with its outputs replaced by an elision marker, and a response ends once it exceeds `CODEFYUI_RUN_EVENTS_RESPONSE_CAP_BYTES` (4 MB). See [Run Queue](/usage/run-queue) for queue order, lanes, retention, and `cdui run`.

**Sweeps.** `POST /api/sweeps` accepts `base_graph`, a `sweep_spec` (`method` `grid` or `random`, `seed`, `samples`, and `params[{node_id, param, values | range}]`), a required `objective` (`metric` and `direction` `minimize` or `maximize`), the same `options`, a `name`, and `seed_variants`. It creates at most `CODEFYUI_MAX_SWEEP_RUNS` (32) variants. Each variant is an ordinary `/api/runs` row whose `/events` endpoint can be followed separately. See [Run Queue — Sweeps](/usage/run-queue#sweeps) for the specification, validation errors, and cancellation behavior.

## Execution outputs and state

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/execution/outputs/{run_id}` | GET | open | List ports captured for a run. |
| `/api/execution/outputs/{run_id}` | DELETE | token | Clear a captured run. |
| `/api/execution/outputs/{run_id}/{node_id}/{port}` | GET | open | Fetch a captured tensor (supports `?slice=0,:,:` and `?max_elements=`, default 4096, at most 1,000,000); 413 when the slice is still larger. |
| `/api/execution/outputs/{run_id}/{node_id}/{port}/stats` | GET | open | Server-side summary statistics for one captured port: a fixed set of scalars and a 64-bin histogram, or value counts for label tensors. The response is typically one or two kilobytes regardless of tensor size. Tensors above `CODEFYUI_STATS_SAMPLE_THRESHOLD` (4,000,000 elements) are sampled. |
| `/api/execution/outputs/{run_id}/{node_id}/__steps_index` | GET | open | Step-trace metadata for a node (Inspector → Steps tab). |
| `/api/execution/outputs/{run_id}/{node_id}/__grad_index` | GET | open | Captured gradient metadata (Inspector → Backward tab). |
| `/api/execution/state/reset` | POST | token | Reset persisted layer weights (per-node or per-graph). |
| `/api/execution/state/list` | GET | open | List how many modules are persisted (diagnostic). |

## Published apps and API keys

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/apps` | GET | token | List apps — `slug`, `graph_name`, `active_version`, `versions_count`, `record_io`. |
| `/api/apps/{slug}/publish` | POST | token | Freeze the named saved graph as the slug's next version and activate it (`"create": true` for a new slug); runs the `/run` pre-flight first. |
| `/api/apps/{slug}/versions` | GET | token | Every version with its note, provenance (`git_commit`, `git_dirty`) and `active` flag. |
| `/api/apps/{slug}/activate` | POST | token | Point the slug at any existing version (`{"version": n}`) — also the rollback path. |
| `/api/apps/{slug}/unpublish` | POST | token | `active_version = null`; versions and runs are kept. |
| `/api/apps/{slug}` | PATCH | token | Flip `record_io` without republishing. |
| `/api/apps/{slug}` | DELETE | token | Remove the app, all its versions and all its run records — there is no undo. |
| `/api/apps/{slug}/invoke` | POST | API key | Execute the active version: the same body and envelope as `/api/graph/run/{name}`, with `app` and `version` filled in. |
| `/api/apps/{slug}/openapi.json` | GET | key-or-token | A standalone OpenAPI 3.1 document for the active version. |
| `/api/apps/{slug}/runs` | GET | key-or-token | Newest-first run records, metadata only; page with `?before=` and `?before_id=`. |
| `/api/apps/{slug}/runs/{run_id}` | GET | key-or-token | One record with its inputs, outputs and node timings. |
| `/api/keys` | POST | token | Mint a key (`{"name"}`); the full `cdui_...` token appears in this response only. |
| `/api/keys` | GET | token | List keys — `id`, `name`, `prefix`, timestamps, never secrets; revoked rows stay listed. |
| `/api/keys/{key_id}/revoke` | POST | token | Soft revoke; 404 `key_not_found`. |

The lifecycle, run records, pagination and the per-app OpenAPI document are on [Publish](/usage/publish).

## Custom nodes and plugins

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/custom-nodes` | GET | open | List custom nodes. |
| `/api/custom-nodes/upload` | POST | token | Upload a custom node. |
| `/api/custom-nodes/toggle` | POST | token | Enable/disable a custom node. |
| `/api/custom-nodes/{filename}` | DELETE | token | Delete a custom node. |
| `/api/plugins` | GET | open | List installed plugin packs. |
| `/api/plugins/catalog` | GET | open | List catalog and installed plugins. Each row includes its state: installed, disabled, explicitly removed, or missing files. |
| `/api/plugins/generation` | GET | open | Return the reload counter, which every rediscovery of nodes and plugins increments. The editor polls it only when an enabled linked plugin was installed at page load, and re-activates plugin frontends when it changes; see [Plugins](/advanced/plugins#rest-api). |
| `/api/plugins/{id}` | GET | open | Get a plugin's manifest + README. |
| `/api/plugins/reload` | POST | token | Same as `POST /api/nodes/reload`. |
| `/api/plugins/inspect` | POST | token+loopback | Inspect a catalog name, `owner/repo`, or URL at one resolved commit. Returns the installation requirements under an `inspection_id` without installing the plugin. |
| `/api/plugins/install` | POST | token+loopback | Install the manifest identified by `inspection_id`; returns `202` with a `job_id`. The server uses the inspected manifest rather than accepting installation metadata in this request. |
| `/api/plugins/jobs/{job_id}/events` | GET | open | Return an install job's log and progress after `?cursor=`. `?wait=` long-polls for up to 60 s. A job can end with `needs_restart` and include the command to run after stopping the server. |
| `/api/plugins/jobs/{job_id}/cancel` | POST | token+loopback | Cancel a running install and remove partial writes. |
| `/api/plugins/{id}/update` | POST | token+loopback | Check the plugin's GitHub repository for an update. Returns `202 {"job_id"}`, `200 {"status": "up_to_date", "sha"}`, or `200 {"status": "needs_consent", "inspection", "capabilities_added", "allowed_modules_added"}` when the update requires additional permission. Complete a consent-required update with `POST /api/plugins/install {"inspection_id", "accept_capabilities", "trust_author"}` and no `force`. Updates preserve the enabled state. Built-in and linked plugins, and repositories whose manifest now declares another plugin id, return `400 not_updatable`. |
| `/api/plugins/{id}` | DELETE | token+loopback | Uninstall a plugin. A built-in plugin keeps its files and is recorded as removed; a linked directory remains unchanged. Python packages remain installed, and the response includes an uninstall command for them. |
| `/api/plugins/{id}/enable` | POST | token | Turn an installed plugin on and re-discover. |
| `/api/plugins/{id}/disable` | POST | token | Turn it off without uninstalling it. |
| `/plugins/{id}/frontend/{path}` | GET | open | Serve a file from an enabled plugin's `frontend/` directory when its manifest declares `[frontend]`; otherwise return 404. The route reads the lockfile on every request, so install, enable, disable, and uninstall changes apply without a restart. `Cache-Control: no-cache` requires browser revalidation after updates. |
| `/plugins/{id}/assets/{path}` | GET, HEAD | open | Serve a file from an enabled plugin's `assets/` directory with its detected media type, or `application/octet-stream` when unknown. The route uses the same per-request lockfile check and revalidation as the frontend route. Directory requests and paths outside the plugin are rejected. |

:::note Installing a plugin needs a loopback-bound server
`inspect`, `install`, job `cancel`, `update`, and `DELETE` require the loopback gate shown above. These operations can download, install, or remove third-party code, and inspection contacts GitHub using a caller-supplied source. `reload`, `enable`, and `disable` operate on code already present on the server, so they require only the token. Refusals from these routes use a machine-readable `code` such as `busy`, `already_installed`, or `consent_required`; see the table under [Plugin Center](/advanced/plugins#plugin-center). Job events carry an English step `label` and English failure `message` and `hint` text. Clients should key on the step id (`resolve`, `download`, `extract`, `verify`, `deps`, `stage`, `lock`, `reload`): the editor translates those ids and shows the English label only for an id it does not know. Failure messages stay in English.
:::

## Optional packs

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/packs` | GET | open | List all packs, installed items, download sizes, and whether installation is supported on this server, plus `active_job`, `last_restart_job` (how the most recent restart-mode install ended, or `null`), `remote_install_allowed`, `launch_mode`, `restart_available`, and `gpu`. |
| `/api/packs/{id}/install` | POST | token+loopback | Start an install job — `202 {"job_id"}`; one job runs at a time across packs and plugins. Body (optional; an unknown key is 422): `items` (default: the whole pack minus what is already downloaded), `mode` — `live` (default) or `restart`, where a helper installs the packages after the server stops itself (see [Installs that restart the server](/usage/optional-packs#installs-that-restart-the-server)) — and `variant` (GPU pack only: which torch wheel; a name outside the allowlist is 422). Refused before any job exists: 400 for an unknown item, or `{detail, blocked_by}` for a missing prerequisite pack; 409 `{detail, job_id[, reason]}` while another install runs; 409 `{detail, command[, reason]}` when restart mode is unavailable or refused (run `command` yourself); 507 `{detail, needed, free}` when the disk cannot hold the download; 500 when the restart helper could not start. |
| `/api/packs/jobs/{job_id}/cancel` | POST | token+loopback | Cancel the running job. An active download stops without completing the current file. Because graph and pack downloads share one transfer session, cancellation also interrupts any Hugging Face dataset or tokenizer download in progress for a graph. |
| `/api/packs/jobs/{job_id}/events` | GET | open | Return a job's log and progress events after `?cursor=`. `?wait=` long-polls for up to 60 s. |
| `/api/packs/{id}/items/{item_id}` | DELETE | token+loopback | Delete one downloaded model and free its bytes — 404 for an unknown item, 409 `{detail, job_id}` while that pack's own install runs, and `removed: false` when a file another process holds open survived the delete (Windows). A pack's Python packages are not removable from the running server — see `cdui packs remove`. |

The loopback gate protects operations that run a package manager in the server's Python environment. The catalog limits which packages can be requested. See [Optional Packs](/usage/optional-packs) for the catalog, restart mode, and `cdui packs`.

## Source control {/* #source-control */}

These routes back the [Source Control](/usage/source-control) tab. They run the server's own git in the project directory the server was started with (`--project`). Reads are open, every write needs the session token, and there is no loopback gate.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/git/status` | GET | open | `{repo: {state, project_dir, git_version, nested_toplevel}, status}`. `repo.state` is `no_project`, `git_missing`, `git_too_old`, `not_repo` or `ready`; every state other than `ready` is still a 200, with `status: null`. `nested_toplevel` names the repository that a `not_repo` project directory sits inside. |
| `/api/git/log` | GET | open | One page of history, newest first: `{commits[{sha, short, parents, author_name, author_email, authored_at, refs, subject, body}], has_more, unborn}`. `?skip=` is 0 to 2147483647 (default 0) and `?limit=` is 1 to 100 (default 30); a value out of range is 422. |
| `/api/git/commits/{sha}/files` | GET | open | The files one commit changed against its first parent: `[{path, orig_path, kind, xy, score}]`. |
| `/api/git/diff` | GET | open | The patch for one file: `?path=`, `?scope=worktree\|index\|commit`, and `?sha=`, which `scope=commit` requires and the other two scopes refuse (an empty `sha=` counts as absent). `?blobs=1` also returns both whole sides. Returns `{patch, binary, truncated, old_ref, new_ref, old_text, new_text, old_missing, new_missing}`; the patch is cut at 1 MiB. |
| `/api/git/file` | GET | open | One file at `?ref=` (`HEAD`, `index`, `worktree`, or a commit id of 7 to 40 hex characters): `{text, binary, size, truncated}`. `text` holds at most 2 MiB; past that `truncated` is true and `text` can be empty. An ignored file read at `worktree` is 403 `ignored`. |
| `/api/git/config` | GET | open | `{name, email, name_scope, email_scope}`; a scope is `local`, `global`, `system`, or `null` when the value is unset. |
| `/api/git/branches` | GET | open | `{current, detached, local[{name, sha, current, upstream, ahead, behind, gone, subject, committed_at}], remote[{name, remote, sha, subject, committed_at}]}`. |
| `/api/git/remotes` | GET | open | `[{name, fetch_url, push_url}]`. A password or token inside a URL is shown as `***`. |
| `/api/git/stashes` | GET | open | Newest first: `[{index, message, branch, created_at}]`; `index` is git's `stash@{N}`. |
| `/api/git/init` | POST | token | Make the project directory a repository and write the `.gitignore` (first line `.env`) and `.gitattributes` that `cdui project init` writes. An existing `.gitignore` that does not ignore `.env` gets that line appended; other existing files are left alone. No body. |
| `/api/git/stage` | POST | token | `{paths: [...]}` (at most 500) or `{all: true}`, exactly one; both, neither, or an empty list is 422. |
| `/api/git/unstage` | POST | token | Same body; takes the paths back out of the index. |
| `/api/git/discard` | POST | token | Same body; restores tracked files from the index and deletes untracked ones. Ignored files are never touched. |
| `/api/git/commit` | POST | token | `{message, all?: false, amend?: false}`, with `message` up to 10,000 characters; `detail.sha` and `detail.short` name the new commit. |
| `/api/git/config` | PUT | token | `{name?, email?}`, always written to this repository's own config (`--local`); sending neither is 400 `invalid_value`. Returns the identity as it now reads, in the `GET` shape. |
| `/api/git/branches` | POST | token | `{name, checkout?: true, start_point?}`: create a branch, and switch to it unless `checkout` is false. |
| `/api/git/checkout` | POST | token | `{target, kind: "local" \| "remote"}`: switch to a local branch, or create a local branch that tracks a remote one (`target` is then `<remote>/<branch>`). `changed_paths` lists the files that changed. |
| `/api/git/branches/{name}` | PUT | token | `{new_name}`: rename a branch. `{name}` may contain `/`. |
| `/api/git/branches/{name}` | DELETE | token | Delete a merged branch. An unmerged one is 409 `branch_not_merged` until the request is repeated with `?force=1`; the current branch is 400. |
| `/api/git/remotes` | POST | token | `{name, url}`: add a remote. `url` must be `https://`, `ssh://`, `file://` or `user@host:path`, else 400 `invalid_url`; a name already in use is 409 `remote_exists`. |
| `/api/git/remotes/{name}` | PUT | token | `{url}`: point the remote at another URL, under the same URL rule. |
| `/api/git/remotes/{name}` | DELETE | token | Remove the remote. |
| `/api/git/stashes` | POST | token | `{message?, include_untracked?: true}`; `{}` is a valid body (a request with no body is 422). `detail.stash` is always 0. A tree with nothing to stash is 400. |
| `/api/git/stashes/{index}/pop` | POST | token | Apply a stash and drop it. A pop that conflicts is 409 `conflict` and keeps the stash. |
| `/api/git/stashes/{index}/apply` | POST | token | Apply a stash and keep it. |
| `/api/git/stashes/{index}` | DELETE | token | Drop one stash. On this route and the two above, an index that is not in the current list is 404 and a negative index is 422. |
| `/api/git/merge/abort` | POST | token | Undo the merge in progress; 404 when there is none. No body. |
| `/api/git/resolve` | POST | token | `{path, side: "ours" \| "theirs" \| "mark"}` for a file the status lists as conflicted, else 400 `path_not_in_status`. `mark` stages the file as it is on disk. |
| `/api/git/fetch` | POST | token | `{remote?}`; `{}` is the normal body. Fetches and prunes. |
| `/api/git/pull` | POST | token | `{strategy?: "ff-only" \| "merge"}` (default `ff-only`): fetch, then merge the upstream. A diverged branch under `ff-only` is 409 `diverged`; a merge that conflicts is 409 `conflict` and stays in progress. |
| `/api/git/push` | POST | token | `{remote?, set_upstream?: false}`. `set_upstream: true` publishes the branch (`push -u`) to `remote`; when `remote` is omitted, to the branch's upstream remote if it has one, else to the only remote (400 `invalid_value` with several, 409 `no_remote` with none). `remote` without `set_upstream: true` is 400. A plain push with no upstream is 409 `no_upstream`. `detail.published` says which of the two ran. |
| `/api/git/sync` | POST | token | Pull (fast-forward only) and then push, or publish a branch whose upstream is missing or gone. No body. It stops at the first failure, and the steps before it stay done; `detail.steps` lists the steps that ran. |

- **Responses.** Every write except `PUT /api/git/config` answers `{status, changed_paths, head, detail}`: a fresh status, the files the operation changed, HEAD afterwards, and operation-specific extras.
- **Errors.** A failure is `{"detail": {"code", "message", "hint", "stderr"}}`. `code` comes from a fixed list that the editor translates, and `stderr` is the tail of git's own output; `message`, `hint` and `stderr` are redacted of anything that looks like a credential. `busy` (409) also carries `op`, the operation that holds the lock. 503 `git_service_unavailable` means the server's startup did not create the git service.
- **Repository state.** Every route other than `status` and `init` needs `repo.state` to be `ready`, and otherwise fails with that state as its code: `no_project` (409; the server was started without `--project`), `git_missing` (503), `git_too_old` (409) or `not_repo` (409). `init` needs only the project directory and git 2.23 or newer.
- **One operation at a time.** A second write while one runs is 409 `busy`. `fetch`, `push` and the network steps of `pull` and `sync` hold a separate lock, so a commit still works while a push is transferring, and a second network request is the one that gets 409 `busy`.
- **Dotenv files.** `/diff` and `/file` refuse `.env` and `.env.*` (not `.env.example`; matched case-insensitively) in any path segment, at every ref, with 403 `ignored`.

## Files, models, images and media

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/files` | GET | open | List uploaded data files (`.csv`, `.tsv`, `.txt`, `.json`) — the `DATA_FILE` dropdown behind CSVReader and friends. |
| `/api/files/upload` | POST | token | Upload a data file (those four extensions only). |
| `/api/files/download/{filename}` | GET | open | Download a data file. |
| `/api/files/{filename}` | DELETE | token | Delete a data file. |
| `/api/models` | GET | open | List uploaded model files. |
| `/api/models/upload` | POST | token | Upload a model weight file. |
| `/api/models/download/{filename}` | GET | open | Download a model weight file (supports nested paths). |
| `/api/models/{filename}` | DELETE | token | Delete a model file. |
| `/api/images` | GET | open | List uploaded image files. |
| `/api/images/upload` | POST | token | Upload an image file. |
| `/api/images/download/{filename}` | GET | open | Download an image file. |
| `/api/images/{filename}` | DELETE | token | Delete an image file. |
| `/api/media` | GET | open | List run-produced media (`.mp4`, `.webm`, `.gif`, `.png`, `.jpg`, `.jpeg`), recursively. |
| `/api/media/{filename}` | GET | open | Serve one media file inline with its real `Content-Type` and Range support, so a `<video>` element can seek. Read-only: files appear here only because a node (VideoWrite) wrote them — there is no upload and no delete. |

## LLM proxy

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/llm/chat` | POST | token | Stream a unified SSE chat completion from the configured provider (OpenAI / OpenRouter / Anthropic / OpenAI-Codex / custom OpenAI-compatible). |
| `/api/llm/models` | POST | token | List the models available for a provider. |
| `/api/llm/codex/login` | POST | token | Start the OpenAI-Codex (ChatGPT account) OAuth login flow. |
| `/api/llm/codex/status` | GET | open | Report OpenAI-Codex OAuth login status. |
| `/api/llm/codex/logout` | POST | token | Clear stored OpenAI-Codex OAuth tokens. |

## Limits and errors

- **Body size.** `MAX_RUN_BODY_BYTES` (64 MB, `CODEFYUI_MAX_RUN_BODY_BYTES`) limits every request body and counts bytes as they arrive, including chunked requests. Four routes use `MAX_UPLOAD_SIZE` (500 MB, `CODEFYUI_MAX_UPLOAD_SIZE`) instead: `/api/files/upload`, `/api/images/upload`, `/api/models/upload`, and `/api/custom-nodes/upload`. They allow an additional 64 KB for multipart metadata and apply the configured limit to the file itself. Exceeding either limit returns 413.
- **Order of refusals.** The Host guard runs first and can return 421, authentication runs next and can return 403 or 401, and body-size validation runs last. A rejected request body is not read. An unauthenticated request therefore does not receive a 413 response.
- **WebSocket frames.** `WS_MAX_MESSAGE_BYTES` (`CODEFYUI_WS_MAX_MESSAGE_BYTES`, which defaults to the request-body limit) is enforced by the transport. An oversized frame closes the connection with code 1009 rather than returning 413. See [Graph as a Function, section 8](/usage/graph-as-a-function#8-limits-and-gotchas) when launching uvicorn directly.
- **Error shapes.** The API uses four formats by route family:
  1. `{"detail": "<text>"}` — the default format, including 413 responses and the 403 that the loopback gate returns on plugin and pack routes. Pack routes add fields next to `detail`, such as `job_id`, `reason`, `blocked_by`, `command`, `needed`, and `free`.
  2. `{"detail": {"code", "message", "details"}}` — app-management routes and `/api/keys`, for example 404 `app_not_found`, 422 `incomplete_cursor`, and 404 `key_not_found`.
  3. `{"detail": {"code", ...}}` without `message` — Plugin Center routes. Additional fields can include `job_id`, `known`, `missing_capabilities`, `allowed_modules`, `plugin_id`, `inspection_id`, and `id`.
  4. `{"detail": {"code", "message", "hint", "stderr"}}` — every `/api/git` failure; `busy` also carries `op`. See [Source control](#source-control).

  Two shapes cut across the families: the `/api/graph/save` collision guard answers 409 `{"detail": {"error": "graph_exists", "file", "name"}}`, and a request that does not match a route's declared body or query parameters (a missing field, an unknown key where the body forbids one, a value out of range) is FastAPI's 422 `{"detail": [{"type", "loc", "msg", ...}]}`.

  `POST /api/graph/run/{name}` and `POST /api/apps/{slug}/invoke` are the exception to all of these: every response — 413 and 401 included — is the 9-key run envelope, see [the envelope](/usage/graph-as-a-function#the-response-envelope).

## WebSocket protocol

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/ws/execution` | WebSocket | token | Attach/subscribe view over a run: `execute` starts one, `attach` replays its event log from a cursor and then follows it live, `detach` unsubscribes, `cancel` stops it. Closing the socket never cancels. |

**Handshake.** `ws://<host>:<port>/ws/execution?token=<session token>` — the `?token=` query parameter, since browsers cannot set headers on a WebSocket handshake, or the `X-CodefyUI-Token` header for non-browser clients. Close codes before the socket is accepted: **4003** when the `Host` — or, for a browser, the `Origin` — is not whitelisted, **4401** for a missing or invalid token; **1009** later, for a frame over the size cap.

**Client actions** — JSON text frames with an `action` field:

| Action | Fields | Effect |
|--------|--------|--------|
| `execute` | `nodes`, `edges`, `presets`, `subgraphs`, `settings`, `device`, `seed`, `deterministic`, `record_outputs`, `changed_nodes`, and the canvas flags (`verbose_mode`, `graph_id`, `weights_persistent`, `backward_mode`, `auto_backward`, `error_mode`, `max_retries`) | Submit on the interactive lane and attach to the new run. A run this socket already follows is detached, not cancelled. `settings`, the graph's own settings object, is stored with the run's graph; the run itself uses `device`, which is `cpu` when omitted. |
| `attach` | `run_id`, `cursor` (an integer, 0 or more, not past the run's latest cursor) | Replay the event log from `cursor`, then follow live; replaces the previous attachment. |
| `detach` | — | Stop following. Never cancels. |
| `cancel` | `run_id` (optional; defaults to the attached run) | Cooperative stop. `stop` is the pre-v2 alias. |
| `clear_cache` | — | Drop this socket's execution cache. |

**Server frames.** Stored run events use `{type, run_id, cursor, ...payload}` (`execution_start`, `node_status`, `execution_stopped`, and others). Clients use `cursor` when attaching again. Transport-only frames are `attached {run_id, cursor, status}`, `detached {run_id}`, `cancel_ack {run_id, status, cancelled}`, `execution_stopped {reason: "not_running"}` when no run can be cancelled, `cache_cleared`, `error {error}` for a malformed or unknown action, and `execution_error {error}` when the run service rejects a submission. `cancel_ack` acknowledges the request; the run emits its own `execution_stopped` event after it stops. Rejections caused by the interactive limit or the one-run-per-session rule also include `rejected: true` and the currently attached `run_id`. In that case no new run starts, and the existing attachment remains active.
