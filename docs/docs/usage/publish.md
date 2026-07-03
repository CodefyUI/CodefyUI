---
sidebar_position: 7.6
title: Publish (Graphs as Apps)
description: Freeze a saved graph as a versioned app behind a stable, API-key-protected invoke endpoint, with every run recorded to SQLite.
---

# Publish (Graphs as Apps)

[Graph as a Function](./graph-as-a-function) makes any saved graph callable over HTTP — but that endpoint is a moving target: every canvas save changes what it runs, and its session token rotates on every server restart. **Publishing** turns a working graph into a stable product endpoint:

- an immutable **version** snapshot served at `POST /api/apps/{slug}/invoke`,
- protected by durable **API keys** (`Authorization: Bearer cdui_...`) that survive restarts,
- with every resolved invoke recorded to SQLite (`backend/data/codefyui.db`).

Canvas edits touch only the saved graph file; invokes read only the stored snapshot — editing the canvas can never change a published app until you re-publish.

All management calls below use the editor session token (`X-CodefyUI-Token`, obtained exactly as on the [Graph as a Function](./graph-as-a-function) page). On Windows use `curl.exe` and pass bodies as files (`--data "@payload.json"`), never inline JSON.

## 1. Publish lifecycle

```text
POST /api/apps/{slug}/publish        (session token)
body: {"graph": "<saved name>", "record_io": true, "note": "optional", "create": false}
```

- `slug` is the stable public name: `^[a-z][a-z0-9-]{0,63}$`, chosen by you, independent of the graph name — renaming a graph never breaks a published URL.
- Publishing to a slug that does not exist yet requires `"create": true`, otherwise 404 `app_not_found` — a misspelled slug on a re-publish can never silently create a second app.
- Publishing to an existing slug appends the next version — that IS the re-publish path.
- `note` is optional, immutable version metadata, echoed in the versions list.
- Publish runs the exact `/run` pre-flight first: a graph that `POST /api/graph/run/{name}` would refuse (409 `invalid_contract` / `no_entry_points` / `untriggered_input` / `unreachable_output` / `invalid_graph`) cannot be published either.
- Success: `{"slug", "version", "active": true, "created", "graph_name", "note"}`.

**Publish activates immediately.** There is no staging path in v1: canvas Run plus the identical pre-flight IS the verification story. If the graph runs from the Run button and `/run` accepts it, the published version serves that same behavior.

```powershell
# payload.json: {"graph": "my-classifier", "create": true, "note": "first cut"}
$token = Get-Content "$env:LOCALAPPDATA\codefyui\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/apps/classifier/publish" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

Managing versions:

```text
GET    /api/apps                       -> [{slug, graph_name, active_version, versions_count, record_io, ...}]
GET    /api/apps/{slug}/versions       -> [{version, source_graph_name, note, created_at, active}]
POST   /api/apps/{slug}/activate       body {"version": n} — point the slug at ANY existing version
POST   /api/apps/{slug}/unpublish      -> active_version = null; versions and runs are kept
PATCH  /api/apps/{slug}                body {"record_io": bool} — flips run-recording, no republish
DELETE /api/apps/{slug}
```

- `activate` subsumes rollback: activating an older version restores it, and activating from the unpublished state restores service at that version.
- Publishing always sets `record_io` (its body default is `true`). If you disabled recording via `PATCH`, pass `"record_io": false` again when you republish — otherwise recording is silently re-enabled.
- While unpublished, invokes return 409 `app_unpublished` — nothing is lost.
- **`DELETE /api/apps/{slug}` irrevocably removes the app, ALL its versions AND all its run records.** There is no undo. Prefer `unpublish` unless you truly mean delete.

## 2. API keys

```text
POST /api/keys                (session token)  body {"name": "ci-bot"}
GET  /api/keys                (session token)  — id, name, prefix, timestamps; never secrets
POST /api/keys/{id}/revoke    (session token)  — soft revoke; the row stays listed
```

**The full key (`cdui_...`) appears ONCE, in the create response, and is never stored or logged.** Copy it immediately; lists show only the first 12 characters (`prefix`). Keys are stored as sha256 hashes and survive restarts — the deliberate contrast with the rotating session token. Revoked keys fail auth immediately but remain listed with their `revoked_at`, so old run records stay attributable.

## 3. Invoke

```text
POST /api/apps/{slug}/invoke          (auth: Authorization: Bearer cdui_...)
```

The body is the same as [`/api/graph/run`](./graph-as-a-function): optional, with optional `inputs`, `timeout_s`, `device`. Two differences:

- `record_outputs` is **accepted and ignored** — published runs are recorded in SQLite (below), never in the editor's inspector store.
- `timeout_s` covers TOTAL request time INCLUDING queue wait: invokes of one app run one-at-a-time (per-slug lock), and a call that spends its budget waiting behind another invoke fails with the `timeout` envelope noting it expired while queued. Different slugs run in parallel.

The editor session token is NEVER accepted on invoke — pasting it produces a self-diagnosing 401: "this endpoint takes an API key (cdui_...), not the editor session token".

Every response is the same 9-key envelope as `/run` (see [the envelope](./graph-as-a-function)), with the two Stage-2 keys filled in: `graph` and `app` are the slug on every outcome, and `version` is the executed version (`null` on errors before a version was resolved). New error codes on top of the Stage-1 taxonomy:

| `error.code` | HTTP | Trigger |
| --- | --- | --- |
| `invalid_key` | 401 | missing/malformed/unknown/revoked bearer token (response carries `WWW-Authenticate: Bearer`) |
| `app_not_found` | 404 | slug does not exist |
| `app_unpublished` | 409 | app exists but no version is active |

Oversized images are rejected up front on BOTH run routes: 422 `invalid_input` when a single image exceeds `MAX_IMAGE_PIXELS` (default 25,000,000; `CODEFYUI_MAX_IMAGE_PIXELS`). Match on the code, not the message — far above the budget, PIL's own decompression-bomb error text appears instead of ours.

PowerShell:

```powershell
# payload.json: {"inputs": {"x": "hello"}}
curl.exe -s -X POST "http://127.0.0.1:8000/api/apps/classifier/invoke" `
  -H "Authorization: Bearer cdui_YOUR_KEY" `
  -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/apps/classifier/invoke" \
  -H "Authorization: Bearer cdui_YOUR_KEY" \
  -H "Content-Type: application/json" \
  --data "@payload.json"
```

## 4. Run records

Every invoke that resolves to an app version writes one row (status, error code, device, `total_s`, per-node timings, capped inputs/outputs, key id). Pre-resolution rejections (`invalid_key`, `app_not_found`, `app_unpublished`) write nothing. Recording is best-effort: a storage failure after execution is logged and the run result is still returned.

```text
GET /api/apps/{slug}/runs?limit=50&before=<iso>   — newest-first metadata only
GET /api/apps/{slug}/runs/{run_id}                — the full row incl. inputs/outputs/node_timings
```

Reads accept EITHER a valid API key or the editor session token (the editor UI reads runs without ever holding an API key), and reject requests with neither.

Stored inputs/outputs are per-field capped at `RUN_IO_CAP_BYTES` (default 64 KB; base64 images are self-limiting). Fields that are not stored stay parseable via pinned marker objects:

- over the cap: `{"__codefyui__": "truncated", "bytes": N}`
- withheld because the app has `record_io: false`: `{"__codefyui__": "redacted"}`

Retention: `CODEFYUI_RUNS_RETENTION_DAYS` defaults to **0 = keep forever**. When set > 0, older rows are pruned at startup and at most hourly on writes, and every prune logs loudly with the row count.

## 5. Per-app OpenAPI document

```text
GET /api/apps/{slug}/openapi.json      (API key or session token)
```

A complete, standalone OpenAPI 3.1 document for the ACTIVE version — importable into Swagger UI, Postman, or openapi-generator as-is. It carries the typed input schema derived from the graph's contract, the 9-key envelope schema, the bearer security scheme, and an `x-codefyui-curl` object with ready-to-paste `powershell` and `bash` invoke commands. JSON only; there is no generated HTML page.

## 6. Serving on your LAN

```text
cdui start --host 0.0.0.0 --port 8000      # all interfaces
cdui start --host 192.168.1.20             # one concrete interface
```

The Host-header whitelist follows the bind automatically (a concrete LAN IP is whitelisted; a `0.0.0.0` bind whitelists each local interface IP), extra names can be added via `CODEFYUI_EXTRA_ALLOWED_HOSTS="mybox:8000,192.168.1.20:8000"`, and the effective whitelist plus reachable URLs are printed at startup. `cdui status` and `cdui stop` report the real address. `cdui dev` remains loopback-only by design.

Understand what a LAN bind exposes — plainly:

- Binding a LAN address serves the FULL editor, and `GET /api/auth/bootstrap` hands out the session token to any allowed-Host request. **Anyone who can reach the port controls the instance; use only on trusted networks.**
- API keys on the published surface are therefore attribution and off-box script hygiene, NOT LAN access control.
- Transport is plain HTTP — no TLS in v1.
- CORS settings change nothing about this: the exposure is same-origin, and the `Authorization` CORS header exists only so future cross-origin JS callers can be preflighted — it is not a mitigation.
- Editor-scoped LAN hardening (loopback-gated bootstrap; published-surface-only on non-loopback Hosts) is a named follow-up, not in v1.
