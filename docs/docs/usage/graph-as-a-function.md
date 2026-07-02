---
sidebar_position: 7.5
title: Graph as a Function
description: Call any saved graph headlessly over HTTP as a named function — declared inputs in, declared outputs out.
---

# Graph as a Function

Any graph you can run on the canvas can also be called over HTTP as a named function: you declare its inputs and outputs with two nodes, save it, and `POST /api/graph/run/{name}` executes it and returns JSON.

**When to use this vs the [CLI runner](./cli-runner):** the CLI runner executes a `graph.json` file in a fresh Python process with no server — good for batch jobs and CI. The run API calls a *saved* graph on a *running* CodefyUI server — good for scripts, notebooks, and automations that want a function call with typed inputs and outputs instead of a process launch.

## 1. Declare the contract on the canvas

Two nodes in the palette's **IO** group define the graph's function signature:

- **GraphInput** — one per input. Params: `name` (identifier-safe: `^[a-zA-Z_][a-zA-Z0-9_]{0,63}$`), `type` (`string` / `number` / `integer` / `boolean` / `json` / `image`), `required`, `default`, `description`.
- **GraphOutput** — one per output. Params: `name`, `description`. Connect the value you want returned into its `value` port.

**Wire Start into every GraphInput.** GraphInput nodes are data roots; the engine only executes nodes reachable from a Start trigger, so an untriggered GraphInput would be silently skipped. The run endpoint rejects such graphs up front (409 `untriggered_input`) instead of running without your input.

```text
[Start] --trigger--> [GraphInput name="message"] --value--> [Print] --value--> [GraphOutput name="echo"]
```

The `default` param is always the canvas test value, so the same graph still runs unchanged from the Run button. For API calls, `default` applies only when `required` is off. `default` is a string field parsed per `type` (`2.5`, `true`, `{"k": 1}`) — this string parsing is the one place the API's strict typing does not apply.

## 2. Getting the token for external scripts

Mutating requests need the `X-CodefyUI-Token` header. Two ways to get it:

- Read the token file: `<user_data_dir>/codefyui/session.token` — on Windows `%LOCALAPPDATA%\codefyui\session.token`, on macOS `~/Library/Application Support/codefyui/session.token`, on Linux `~/.local/share/codefyui/session.token`. Honors `CODEFYUI_USER_DATA_DIR` for dev-mode installs.
- Or `GET /api/auth/bootstrap` returns `{"token": "..."}` (same-host only).

The token rotates on every server restart — scripts should re-read the file rather than caching the value.

## 3. Inspect the contract

```text
GET /api/graph/contract/{name}     (no auth required, like /api/graph/load)
```

```json
{
  "graph": "my-graph",
  "inputs":  [{"name": "prompt", "type": "string", "required": true, "default": null, "description": ""}],
  "outputs": [{"name": "answer", "type": "SCALAR", "description": ""}],
  "problems": []
}
```

- `default` is `null` for required inputs — the API never applies a required input's default, so it never advertises one. Optional inputs show the parsed default value.
- Output `type` is derived from the port feeding the GraphOutput (`ANY` when it cannot be resolved).
- `problems` lists contract issues (bad names, duplicates, an optional default that does not parse, an optional image input). They are reported here non-fatally so you can inspect a half-built graph — but they block `/run` with 409.

## 4. Run the graph

```text
POST /api/graph/run/{name}         (auth: X-CodefyUI-Token header)
Content-Type: application/json
```

The body is OPTIONAL (absent body means `{}`), and every field is optional:

```json
{
  "inputs": {"prompt": "hello"},   // default {}
  "timeout_s": 300,                // default 300, min 1, max 3600
  "device": "cuda",                // "cpu" / "cuda" / "mps"; falls back to CPU when unavailable
  "record_outputs": false          // default false; see gotchas before enabling
}
```

### The response envelope

Every `/run` response — success or failure — is this one shape, with ALL keys always present (`null` when not applicable):

```json
{
  "status": "ok",                 // "ok" | "error"
  "run_id": "9f2c...",            // assigned at request entry; NEVER null
  "graph": "my-graph",
  "device": "cuda",               // what you actually got; null on early rejections
  "outputs": {"answer": 0.93},    // null unless status == "ok"
  "error": null,                  // null on success, else {"code", "message", "node_id", "details"}
  "timing": {"total_s": 1.234}    // null when execution was never attempted
}
```

The HTTP status mirrors `status`/`error.code`, so `raise_for_status()` works.

Forward compatibility: **Clients MUST ignore unknown envelope fields.** **`error.code` is an open enum — treat unknown codes as generic errors.** A future async mode will return 202 `{"status": "queued", "run_id": ..., "job": {...}}` — sync clients switching on `status` keep working unmodified.

### Error taxonomy

Triage rule: **404 = wrong name; 409 = fix the graph; 413 = shrink your payload; 422 = fix your payload; 500 = run failed (or the graph file was unreadable).**

| `error.code` | HTTP | Trigger |
| --- | --- | --- |
| `graph_not_found` | 404 | no exact-match graph file (strict name matching — `my.graph` never aliases to `my_graph`) |
| `graph_unreadable` | 500 | graph file exists but is corrupt JSON |
| `invalid_contract` | 409 | contract `problems[]` non-empty (`details` lists them) |
| `no_entry_points` | 409 | no Start trigger anywhere in the graph |
| `untriggered_input` | 409 | a GraphInput has no incoming trigger edge (`details`: input names) |
| `unreachable_output` | 409 | a GraphOutput is not reachable from any Start (`details`: output names) |
| `invalid_graph` | 409 | graph validation failed, statically or at runtime (`details`: errors) |
| `invalid_input` | 422 | unknown input name (case-sensitive), missing required input, type mismatch, malformed body — all aggregated in `details` |
| `payload_too_large` | 413 | body exceeds `MAX_RUN_BODY_BYTES` (default 64 MB, `CODEFYUI_MAX_RUN_BODY_BYTES`) |
| `execution_error` | 500 | a node raised; `node_id` names it |
| `timeout` | 500 | `timeout_s` expired; `timing.total_s` = elapsed |
| `output_not_produced` | 500 | declared output missing from the engine result (safety net) |
| `output_too_large` / `unserializable_output` | 500 | see output serialization below |

## 5. Input types

| `type` | Send (JSON) | Rejected |
| --- | --- | --- |
| `string` | a JSON string | numbers, booleans, null (no implicit `str()`) |
| `number` | int or float | strings (`"3"`), booleans, null |
| `integer` | int, or a float with zero fraction (`3.0` -> `3`) | `3.5`, strings, booleans |
| `boolean` | `true` / `false` | `0`/`1`, `"true"` |
| `json` | any JSON value | nothing |
| `image` | a base64 string (optionally `data:image/...;base64,`-prefixed) | non-strings, undecodable data |

Typing is strict on purpose: JSON already carries types, so typos fail loudly instead of silently coercing. The one loosening is `integer` accepting integral floats (JS clients cannot control whether `3` serializes as `3.0`).

An `image` input arrives at the graph as a `(C, H, W)` float32 tensor in `[0, 1]` — exactly what `ImageReader` produces. Image inputs must be `required` (there is no sensible base64 default); the node's `default` is a server-local file path used only for canvas runs.

```python
import base64
from pathlib import Path

img_b64 = base64.b64encode(Path("cat.png").read_bytes()).decode("ascii")
# ... json={"inputs": {"photo": img_b64}}
```

## 6. Output serialization

| Value at a GraphOutput | JSON form |
| --- | --- |
| `None`, bool, int, float, str | as-is (base64-string plots pass through as plain strings) |
| dict / list / tuple | recursively serialized (tuples become lists) |
| numpy scalar | plain number |
| tensor / ndarray | `{"__type__": "tensor", "shape": [...], "dtype": "torch.float32", "values": [...]}` — capped at **65,536 elements**; 0-dim tensors keep `"shape": []` |
| PIL image | `{"__type__": "image", "format": "png", "base64": "..."}` |
| `torch.nn.Module` | error — save it with a ModelSaver node in-graph and return the path string instead |
| anything else | error `unserializable_output` naming the type |

## 7. Examples

Save your graph on the canvas first (the run API executes *saved* graphs by name). Python `requests`:

```python
import os
from pathlib import Path

import requests

# Windows: %LOCALAPPDATA%\codefyui\session.token
# macOS:   ~/Library/Application Support/codefyui/session.token
# Linux:   ~/.local/share/codefyui/session.token
token = (Path(os.environ["LOCALAPPDATA"]) / "codefyui" / "session.token").read_text().strip()

resp = requests.post(
    "http://127.0.0.1:8000/api/graph/run/Api-Function",
    headers={"X-CodefyUI-Token": token},
    json={"inputs": {"message": "hello from Python"}},
    timeout=310,  # slightly above the server-side default timeout_s=300
)
resp.raise_for_status()
envelope = resp.json()
print(envelope["outputs"]["echo"])
```

curl on Windows: use `curl.exe` (PowerShell aliases `curl` to `Invoke-WebRequest`) and ALWAYS pass the body as a file with `--data "@payload.json"` — inline JSON fights cmd's 8191-character limit and PowerShell quoting, and inline base64 images are outright impossible:

```powershell
# payload.json: {"inputs": {"message": "hello from curl"}}
$token = Get-Content "$env:LOCALAPPDATA\codefyui\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/graph/run/Api-Function" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

Inspect the contract first when scripting against an unfamiliar graph:

```powershell
curl.exe -s "http://127.0.0.1:8000/api/graph/contract/Api-Function"
```

A ready-made graph for these exact calls ships in `examples/Usage_Example/Api-Function/` — open it from the Examples gallery, save it, and the commands above work verbatim.

## 8. Limits and gotchas

- 403 (missing/invalid token) and 421 (Host guard) arrive WITHOUT the envelope — they fire before the route.
- This server never emits 504; a 504 always came from an intermediary.
- `record_outputs=true` makes inputs and results readable by anyone on the LAN who learns the `run_id` (the GET outputs endpoint is auth-exempt; transport is plain HTTP) — auth on read endpoints is a hard Stage 2 requirement.
- Do not put secrets in `default` values — `GET /contract` and `/load` are unauthenticated.
- `device: "auto"` (or an unavailable device) silently resolves to CPU; the envelope's `device` field shows what you actually got.
- A single >65,536-element tensor output fails the whole call — remove that GraphOutput or use `record_outputs` + the slicing outputs API (`GET /api/execution/outputs/{run_id}/{node_id}/{port}?slice=...`); an outputs filter is deferred.
- Concurrent runs share the process default thread pool (the per-run parallelism limit of 4 is not a global limit) — heavy runs contend for CPU/GPU.
- After a timeout, no new node starts after a cancel; the in-flight node finishes in the background (nodes may poll `context.cancelled` to stop sooner).
- Client disconnects do not stop a run; only the timeout does. Results of a disconnected run are discarded unless `record_outputs=true`.

## 9. Roadmap

- `cdui call <graph> --input k=v` — CLI wrapper over this API (fast-follow DX item).
- Async job mode (202 + `job.status_url`) reusing the same envelope.
- Stage 2: SQLite run history, publishing, auth on read endpoints.
