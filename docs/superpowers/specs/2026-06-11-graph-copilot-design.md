# Graph Copilot — AI chat assistant plugin + core extension points

Date: 2026-06-11
Status: Approved (design Q&A with maintainer in-session)

## Context and goal

CodefyUI users should be able to build, tune, and improve the node graph (the
directed workflow on the canvas) by chatting with an LLM. Installing a plugin
adds a floating button at the bottom-right of the editor; clicking it opens a
chat window where the user can create a graph from a prompt, adjust node
parameters, or ask for improvements. Past conversations are listed and can be
resumed. A settings view lets the user pick a provider + model and supply API
keys. Providers: OpenAI API (user key), OpenAI Codex (ChatGPT account OAuth),
OpenRouter, Claude API, plus a custom OpenAI-compatible endpoint (base URL).

The plugin lives in its own repository, `CodefyUI-Plugin-Graph-Copilot`
(plugin id `graph-copilot`), built from the `CodefyUI-Plugin-Official`
template.

## Why core changes are required

Today's plugin system is backend/node-centric: plugins ship Python nodes,
examples, presets, and static assets (`/plugins/<id>/assets/`). Nothing loads
plugin JavaScript into the SPA and plugins cannot register API routes. The
floating-button UX therefore requires two new core capabilities, delivered as
two PRs to CodefyUI before the plugin repo can work:

- **PR A — plugin frontend extensions**: a generic mechanism for plugins to
  ship a JS bundle that the SPA loads, plus a small stable API object the
  bundle receives.
- **PR B — LLM provider proxy**: a generic backend route that forwards chat
  requests to LLM providers (solves CORS, normalizes streaming, hosts the
  Codex OAuth flow).

Both are generic core features; Graph Copilot is their first consumer.

## Decisions (maintainer-approved)

| Question | Decision |
| --- | --- |
| How to get UI into the editor | Core gains a plugin frontend-extension mechanism (PR A) |
| What "OpenAI Codex" means | ChatGPT-account OAuth, like `codex login` (subscription quota) |
| How LLM requests leave the app | Backend proxy route in core (PR B); keys live in browser localStorage and ride along per-request |
| Repo name | `CodefyUI-Plugin-Graph-Copilot`, plugin id `graph-copilot` |
| How AI edits land on the canvas | Apply immediately; one undo snapshot per batch (Ctrl+Z reverts a whole AI edit) |
| Extra provider | Yes — "Custom (OpenAI-compatible)" with user-supplied base URL (Ollama / LM Studio / vLLM, also used by e2e tests) |

## Part A — core PR: plugin frontend extensions

### Manifest

New optional table in `cdui.plugin.toml`:

```toml
[frontend]
entry = "frontend/index.js"   # path relative to plugin root; ESM module
```

`validate_manifest` only inspects `[plugin]`, so older CodefyUI versions
install such plugins and simply ignore the frontend; the plugin declares
`requires_codefyui >= <version shipping PR A>` to express the real
constraint. `schema_version` stays 1.

### Backend

- For each enabled plugin whose manifest declares `[frontend].entry` and the
  file exists, mount `<plugin>/frontend/` at `/plugins/<id>/frontend/`
  (same `StaticFiles` pattern as the existing assets mount in
  `backend/app/main.py`).
- `GET /api/plugins` items gain `frontend_entry`: the absolute URL path
  (`/plugins/graph-copilot/frontend/index.js`) or `null`.
- `cdui plugin install` prints a clear notice when the manifest declares
  `[frontend]`: the plugin ships UI code that runs in the browser with full
  access to the CodefyUI UI. (Same trust level as plugin Python, which
  already runs server-side; the AST gate cannot cover JS, so we disclose.)

### Frontend loader

New module `frontend/src/plugins/host.ts(x)`:

- After app bootstrap, fetch `/api/plugins`; for every enabled plugin with a
  `frontend_entry`, `await import(/* @vite-ignore */ url)` and call
  `module.default(api)` where `api` is the `CodefyUIPluginAPI` object below.
- Each plugin activates inside try/catch; a failing plugin logs a console
  warning and raises a toast, and never breaks the app or other plugins.
- A core-owned floating widget stack is rendered at the editor's bottom-right
  (above the React Flow MiniMap, `z-index` between canvas chrome and modals).
  Plugins get container divs inside this stack and mount their own UI (their
  own React copy is bundled in; zero version coupling with core React).

### `CodefyUIPluginAPI` v1

```ts
interface CodefyUIPluginAPI {
  apiVersion: 1;
  pluginId: string;
  ui: {
    addFloatingWidget(opts: { id: string; order?: number }): HTMLElement;
    toast(message: string, type?: 'info' | 'success' | 'error'): void;
  };
  graph: {
    getGraph(): GraphSaveData;                       // active tab, serialized
    getNodeDefinitions(): NodeDefinition[];          // same data as /api/nodes
    applyOperations(ops: GraphOp[]): ApplyResult;    // one undo snapshot per call
    onGraphChanged(cb: () => void): () => void;      // returns unsubscribe
  };
  http: {
    fetch(url: string, init?: RequestInit): Promise<Response>;  // attaches X-CodefyUI-Token
  };
  storage: {
    get(key: string): string | null;                 // localStorage, namespaced
    set(key: string, value: string): void;           // "plugin:<id>:<key>"
    remove(key: string): void;
  };
}
```

### Graph operations

```ts
type GraphOp =
  | { op: 'add_node'; ref?: string; node_type: string;
      params?: Record<string, unknown>; position?: { x: number; y: number } }
  | { op: 'connect'; source: string; source_handle: string;
      target: string; target_handle: string }
  | { op: 'set_params'; node_id: string; params: Record<string, unknown> }
  | { op: 'remove_node'; node_id: string }
  | { op: 'remove_edge'; source: string; target: string;
      source_handle?: string; target_handle?: string }
  | { op: 'clear_graph' }
  | { op: 'auto_layout' };
```

- `add_node.ref` is a caller-chosen label (e.g. `"conv1"`); later ops in the
  same batch may use a ref wherever a node id is expected. The result returns
  the ref → assigned-id map.
- Semantics: one `pushUndoSnapshot()` at batch start; ops apply in order;
  a failing op is skipped and reported (the batch continues) so the agent
  loop can self-correct. `connect` goes through the same validation as the
  store's `onConnect` (port existence, type compatibility); `set_params`
  validates against the node's `ParamDefinition`s; `add_node` rejects
  unknown `node_type`; `auto_layout` reuses `frontend/src/utils/autoLayout.ts`.

```ts
interface ApplyResult {
  results: Array<{ index: number; ok: boolean; error?: string; node_id?: string }>;
  refs: Record<string, string>;          // ref -> created node id
  node_count: number;
  edge_count: number;
}
```

## Part B — core PR: LLM provider proxy

### Chat endpoint

`POST /api/llm/chat` (mutating → covered by the existing `auth_guard`
session-token middleware; no ad-hoc auth). Request:

```jsonc
{
  "provider": "openai" | "openai-codex" | "openrouter" | "anthropic" | "custom",
  "model": "gpt-5.2",
  "messages": [ /* OpenAI-style: role system|user|assistant|tool,
                   content, tool_calls, tool_call_id */ ],
  "tools": [ { "name": "...", "description": "...", "input_schema": { } } ],
  "api_key": "sk-...",        // key-based providers; never logged or persisted
  "base_url": "http://...",   // "custom" provider only
  "max_tokens": 4096,
  "temperature": 0.2
}
```

Response: streamed `text/event-stream`. Text streams incrementally; tool
calls are delivered complete in the terminal event (this avoids normalizing
four different tool-call streaming formats):

```jsonc
{ "type": "text_delta", "text": "..." }
{ "type": "done", "message": { "role": "assistant", "content": "...",
    "tool_calls": [{ "id": "...", "name": "...", "arguments": { } }] },
  "stop_reason": "tool_use" | "end", "usage": { "input_tokens": 0, "output_tokens": 0 } }
{ "type": "error", "message": "..." }
```

The frontend consumes this with `fetch` + `ReadableStream` (not
`EventSource`, which cannot send the token header).

### Provider adapters

| provider | upstream | notes |
| --- | --- | --- |
| `openai` | `api.openai.com/v1/chat/completions` | key from request |
| `openrouter` | `openrouter.ai/api/v1/chat/completions` | key from request |
| `anthropic` | `api.anthropic.com/v1/messages` | request/response transformed; `anthropic-version` header |
| `openai-codex` | `chatgpt.com/backend-api/codex/responses` | OAuth bearer + account id from stored tokens; Responses-API transform |
| `custom` | `<base_url>/chat/completions` | OpenAI adapter against user-supplied base URL |

Upstream hosts are a hard-coded allowlist; `custom` is the only provider that
may leave it, and only with a base URL the user explicitly configured. This
is not an open proxy. API keys are never logged and never persisted
server-side.

`POST /api/llm/models` (`{provider, api_key?, base_url?}`) proxies each
provider's model list (`/v1/models`, OpenRouter `/api/v1/models`, Anthropic
`/v1/models`, a static list for Codex) so the settings UI can offer a picker.

### Codex OAuth

- `POST /api/llm/codex/login` generates a PKCE pair + state, starts a
  temporary listener on `127.0.0.1:1455` (the redirect URI registered for the
  public Codex CLI client id), and returns the authorization URL; the
  frontend opens it in a new tab.
- The callback hits the listener; the backend exchanges the code for tokens
  and stores them at `<USER_DATA>/llm/codex_auth.json` (0600). Access tokens
  auto-refresh via the refresh token.
- `GET /api/llm/codex/status` → `{ status: "pending" | "logged_in" |
  "logged_out", email?, plan? }` (frontend polls during login).
  `POST /api/llm/codex/logout` deletes stored tokens.
- Exact endpoint/header details are verified against the open-source Codex
  CLI at implementation time.
- **Known risk (disclosed and accepted):** reusing the Codex CLI public
  client id is the same approach third-party tools (e.g. opencode) use and
  sits in an OpenAI ToS gray area. The adapter is isolated so breakage or
  policy change degrades only this provider, with a clear error.

## Part C — plugin repo: CodefyUI-Plugin-Graph-Copilot

### Repository layout

```
CodefyUI-Plugin-Graph-Copilot/
  cdui.plugin.toml          # id=graph-copilot, [frontend] entry, no nodes
  frontend/index.js         # committed build output (single-file ESM, CSS injected)
  ui/                       # source: Vite + React + TS
    src/index.ts            # export default activate(api)
    src/components/         # Fab, ChatWindow, MessageList, HistoryView, SettingsView
    src/agent/              # prompt.ts, loop.ts, ops.ts
    src/llm/client.ts       # talks to /api/llm/* via api.http.fetch, parses SSE
    package.json, vite.config.ts, tsconfig.json, vitest
  .github/workflows/ci.yml  # install, test, build, fail if frontend/ is stale
  README.md                 # install + usage, EN with a zh-TW section
  LICENSE                   # MIT (matches template)
```

The build output is committed because end users install via GitHub tarball
with only uv + Python available (no Node). CI rebuilds and fails if
`frontend/` does not match `ui/` source. Vite lib-mode bundles React and
inlines CSS into the single JS file. No Python nodes ship in v1 (the
installer's AST gate has nothing to scan; no `--trust-author` needed).

### UI

Visual language follows the core "Crafted dark" system: `#1e1e1e` surfaces,
teal `#06b6d4` accent, UI strings in English (consistent with core; the
assistant naturally replies in the user's language).

- **FAB**: round teal button in the core floating-widget stack,
  bottom-right above the MiniMap.
- **Chat panel** (~420×600, anchored above the FAB) with three views:
  - *Chat*: streaming messages, code-block rendering, per-turn applied-ops
    chips ("Applied: add Conv2d ×3, connect ×4"), error bubble with retry.
    Input: Enter sends, Shift+Enter newline; disabled with a settings CTA
    when the active provider has no key/login.
  - *History*: conversation list (title = first user message, truncated;
    timestamps), click to resume, delete, new-chat button.
  - *Settings*: provider select, model picker (list from `/api/llm/models` +
    free text), API-key fields per key-based provider, custom base-URL
    field, Codex sign-in button + status. Last-used model remembered per
    provider.

### Agent loop

- Per user message, the plugin rebuilds context: system prompt = role +
  graph-model explanation + compact node catalog (one line per node:
  `Name (Category): in[...] out[...] params[name:type=default{...}]`, from
  `getNodeDefinitions()`) + fresh serialized current graph + rules (connect
  required inputs, respect param ranges, end structural batches with
  `auto_layout`, never `clear_graph` unless the user asked, reply in the
  user's language).
- Two tools exposed to the model: `apply_graph_operations(ops)` and
  `get_current_graph()`.
- Tool loop: stream text → if `stop_reason == "tool_use"`, execute tool
  calls via `api.graph.applyOperations` / `getGraph`, append tool results
  (per-op outcomes + ref→id map + counts), continue. Hard cap 8 iterations
  per user message, then surface a "stopped after 8 tool rounds" notice.
- Each `apply_graph_operations` call is one undo step.

### Persistence

`api.storage` (namespaced localStorage):

- `conversations`: capped at 50, oldest trimmed; each holds id, title,
  timestamps, provider+model used, full message list.
- `settings`: provider, per-provider model, API keys, custom base URL.
- Long conversations: sliding window when sending (system prompt + most
  recent ~20 messages); full history retained locally for display.
- Graph snapshots in prompts are capped (~30k chars) with an explicit
  truncation notice to the model.

## Error handling summary

| Failure | Behavior |
| --- | --- |
| Plugin JS fails to load/activate | try/catch per plugin; console warning + toast; app unaffected |
| No key / not logged in | Input disabled with CTA into Settings view |
| Provider 4xx/5xx or network error | Error bubble in chat with retry; message kept in input history |
| Invalid op (unknown type, bad port, type mismatch) | Op skipped, error fed back to the model in tool result; model retries within loop cap |
| Codex token expired | Auto-refresh; on refresh failure, status flips to logged_out and chat shows sign-in CTA |
| Server restart mid-chat | Plain POST/SSE per message — next message just works (token re-bootstraps on reload) |

## Testing

- **PR A**: pytest — manifest with `[frontend]` exposes `frontend_entry` in
  `/api/plugins`, static mount serves the file, no-frontend plugins
  unaffected, CLI prints the trust notice. vitest — `applyOperations`
  against the real tabStore (add/connect/set_params/remove/refs/undo-batch
  semantics, invalid-op reporting).
- **PR B**: pytest with `httpx.MockTransport` — request/response transforms
  for all five adapters, SSE relay, tool-call normalization, host
  allowlist, key-never-logged, Codex PKCE + refresh logic with mocked
  endpoints.
- **Plugin**: vitest — prompt builder (catalog compaction, graph
  truncation), agent loop against a mocked API + scripted LLM streams
  (tool-loop, self-correction, iteration cap), SSE client parser.
- **E2E (Chrome, before each merge per repo convention)**: dev server +
  `cdui plugin install` from local path; custom provider pointed at a local
  mock OpenAI-compatible server; full flow — FAB appears, chat creates a
  small graph on canvas, params adjust, undo reverts batch, history
  resumes, settings persist across reload.

## Rollout

1. PR A (frontend extensions) → CodefyUI main.
2. PR B (LLM proxy + Codex OAuth) → CodefyUI main (branched from A if
   stacked; retargeted after A merges).
3. Bump CodefyUI version; tag/release so `cdui` distributions pick it up.
4. Plugin repo created from template; v0.1.0 tagged;
   `cdui plugin install treeleaves30760/CodefyUI-Plugin-Graph-Copilot`.
5. Docs site: "Plugin frontend extensions" (API reference for plugin
   authors) + "Graph Copilot" (user guide) pages.

## Out of scope (v1)

- Preview/diff-before-apply for AI edits (undo covers reversibility).
- Server-side storage of chat history or API keys.
- Tool-call *streaming* (calls arrive complete in the terminal event).
- Image/multimodal inputs; per-node inline "ask AI" affordances.
- Frontend extension points beyond the floating widget stack (menus,
  settings popover sections, custom panels) — deliberately deferred until a
  second consumer exists.
