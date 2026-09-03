import type { NodeDefinition, GraphSaveData, PresetDefinition } from '../types';
import { apiFetch } from './_auth';

const BASE_URL = '/api';

export async function fetchNodeDefinitions(): Promise<NodeDefinition[]> {
  const res = await fetch(`${BASE_URL}/nodes`);
  if (!res.ok) throw new Error(`Failed to fetch node definitions: ${res.statusText}`);
  return res.json();
}

/**
 * One node's definition, straight from the registry.
 *
 * The palette already holds every definition, but the Node Detail Modal's Docs
 * tab asks the server again so a node whose plugin was reloaded (or whose
 * device options changed with the hardware) documents what the backend will
 * actually run — not what the page happened to load at boot. Callers seed the
 * view from the local copy first and treat a failure here as "keep the local
 * one", so the tab still works offline.
 */
export async function fetchNodeDefinition(nodeName: string): Promise<NodeDefinition> {
  const res = await fetch(`${BASE_URL}/nodes/${encodeURIComponent(nodeName)}`);
  if (!res.ok) throw new Error(`Failed to fetch node definition: ${res.statusText}`);
  return res.json();
}

export async function fetchPresetDefinitions(): Promise<PresetDefinition[]> {
  const res = await fetch(`${BASE_URL}/presets`);
  if (!res.ok) throw new Error(`Failed to fetch presets: ${res.statusText}`);
  return res.json();
}

/**
 * One cache store's line in the health payload's `caches` block.
 *
 * Deliberately an open map of numbers rather than a named-field interface: the
 * three stores do NOT share a shape. `bytes` is the only key all of them
 * carry; the budget is `max_bytes` for the run-output and node-state stores
 * but `max_bytes_each` for the execution cache (there is one instance per
 * WebSocket, so a single total has no single ceiling), and the item count is
 * `entries` / `runs` / `modules` respectively. A store added backend-side
 * therefore needs no change here (#193 item 2).
 */
export type CacheUsage = Record<string, number>;

export interface HealthInfo {
  status: string;
  /** The running server's version. Unconditional since #135 -- normalized to
   *  `null` only so a pre-#135 server, or a partial test double, renders a
   *  placeholder instead of "undefined". */
  version: string | null;
  nodes_loaded: number;
  presets_loaded: number;
  /** Per-store memory usage, keyed by store name (`execution_cache`,
   *  `run_output_store`, `node_state_store`). Empty when the server reports
   *  none -- see `CacheUsage` for why the inner shape is open. */
  caches: Record<string, CacheUsage>;
  project: string | null;
  /**
   * Identity of the running PROCESS, regenerated on every boot.
   *
   * A restart-mode pack install has to tell "the old server answered again"
   * from "a new server came up", and a reachable /api/health proves only the
   * first -- the old process answers right up until it exits. A CHANGED
   * boot_id is the proof. Optional because a server older than the Package
   * Center omits the key entirely.
   */
  boot_id?: string;
}

/**
 * /api/health, including the additive `project` field (spec ID4).
 *
 * The backend (backend/app/main.py) OMITS the `project` key entirely in
 * non-project mode -- it is never sent as `null`. Normalize that gap here so
 * `HealthInfo.project` is always exactly `string | null`, never `undefined`:
 * `useProjectStore.setProject` / `isProjectMode` key off a strict
 * `!== null` check, which `undefined` would satisfy and misreport project
 * mode as active.
 *
 * `caches` is normalized to `{}` rather than left absent for the same reason:
 * the settings panel maps over it on every render, and a missing key would be
 * a crash rather than an empty list. The backend also omits an individual
 * store that is not running (a test client reaches the endpoint with nothing
 * on `app.state`), so an empty map is a reachable, valid answer.
 */
export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health failed: ${res.statusText}`);
  const data = await res.json();
  return {
    status: data.status,
    version: data.version ?? null,
    nodes_loaded: data.nodes_loaded,
    presets_loaded: data.presets_loaded,
    caches: data.caches ?? {},
    project: data.project ?? null,
    boot_id: data.boot_id ?? undefined,
  };
}

export interface DeviceInfo {
  value: string;
  label: string;
  detail: string;
  available: boolean;
}

export interface DevicesResponse {
  default: string;
  devices: DeviceInfo[];
}

/** Compute devices available for graph execution (CPU + any GPU backend,
 * with NVIDIA-CUDA / AMD-ROCm / Apple-MPS labels). Powers the global device
 * selector. */
export async function fetchDevices(): Promise<DevicesResponse> {
  const res = await fetch(`${BASE_URL}/system/devices`);
  if (!res.ok) throw new Error(`Failed to fetch devices: ${res.statusText}`);
  return res.json();
}

export type RunStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted';

/** Mirrors the backend's `ACTIVE_STATUSES` — a run that still has a future. */
export const ACTIVE_RUN_STATUSES: readonly RunStatus[] = ['queued', 'running'];

/** Mirrors the backend's `TERMINAL_STATUSES` — the rows Delete accepts. */
export const TERMINAL_RUN_STATUSES: readonly RunStatus[] = [
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
];

/**
 * One row of `exec_runs` as the list endpoint returns it (#120/#123).
 *
 * Timestamps are ISO-8601 UTC with a trailing `Z` (`utc_now_iso`), so
 * `new Date(...)` parses them directly.
 */
export interface RunSummary {
  id: string;
  name: string | null;
  status: RunStatus;
  error: string | null;
  /** Submit options verbatim — `device`, `seed`, `record_outputs`, … */
  options: Record<string, unknown>;
  /** The device queue this run belongs to; null until the scheduler starts it. */
  queue_key: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  git_commit: string | null;
  git_dirty: boolean | null;
  plugin_pins: Record<string, unknown> | null;
  /**
   * 1-based place in this run's OWN device queue (#123).
   *
   * `null` is a real answer, not zero: the run is not queued, or the queue
   * is deeper than the server's scan limit and it declined to guess. Render
   * a dash for it — never a 0.
   */
  queue_position: number | null;
  /**
   * Last recorded value of every series, e.g. `{ train_loss: 0.31 }`.
   *
   * Answered for the whole page in one grouped query so the table can print
   * a final loss per row without a metrics request per row. `{}` for a run
   * that recorded nothing; a series whose last point diverged (NaN) is
   * omitted rather than reported as 0.
   */
  final_metrics: Record<string, number>;
  /** Whether THIS server process is currently driving the run. */
  active: boolean;
}

/** A single run, as `GET /api/runs/{id}` returns it — the row plus a cursor. */
export interface RunInfo extends RunSummary {
  /** Highest event cursor issued so far — where a follower should resume. */
  last_cursor: number;
}

export interface RunListPage {
  runs: RunSummary[];
  /** Unpaged count for the active filter, so a table can size itself. */
  total: number;
  limit: number;
  offset: number;
}

/**
 * Fetch one run, or `null` when the server has never heard of it.
 *
 * A 404 is an ordinary answer here rather than an error: the caller is a
 * page-load re-attach check (#121) asking "is the run this tab was watching
 * still going?", and a run pruned by retention, or one from a previous
 * install, is simply a "no".
 *
 * This and the other two run readers go through `apiFetch` rather than bare
 * `fetch`. It is a pass-through for GET today (reads are gated by the
 * Host-header middleware, not the token), but `api.runs` — the plugin facade
 * (#132) — is built on them, and its contract is that the HOST attaches
 * whatever authentication a request needs. Routing through the one auth-aware
 * client is what keeps that true if reads ever become token-gated, without a
 * plugin ever seeing the token.
 */
export async function getRun(runId: string): Promise<RunInfo | null> {
  const res = await apiFetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to fetch run: ${res.statusText}`);
  return res.json();
}

/** Newest-first page of runs, optionally narrowed to a set of statuses. */
export async function listRuns(
  opts: { status?: readonly RunStatus[]; limit?: number; offset?: number } = {},
): Promise<RunListPage> {
  const params = new URLSearchParams();
  // `?status=` repeats rather than joining with commas — that is the shape
  // FastAPI's `list[str] = Query(...)` binds.
  for (const status of opts.status ?? []) params.append('status', status);
  if (opts.limit !== undefined) params.set('limit', String(opts.limit));
  if (opts.offset !== undefined) params.set('offset', String(opts.offset));
  const query = params.toString();
  const res = await apiFetch(`${BASE_URL}/runs${query ? `?${query}` : ''}`);
  if (!res.ok) throw new Error(`Failed to list runs: ${res.statusText}`);
  return res.json();
}

/**
 * Ask a run to stop. Cooperative, so the returned `status` may still say
 * `running` — `cancelled` reports whether the request did anything.
 */
export async function cancelRun(
  runId: string,
): Promise<{ run_id: string; status: RunStatus; cancelled: boolean }> {
  const res = await apiFetch(
    `${BASE_URL}/runs/${encodeURIComponent(runId)}/cancel`,
    { method: 'POST' },
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Cancel failed: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Delete a finished run and everything hanging off it.
 *
 * The server refuses a queued or running row with a 409 — history is
 * deletable, live state is not.
 */
export async function deleteRun(
  runId: string,
): Promise<{ run_id: string; deleted: boolean }> {
  const res = await apiFetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Delete failed: ${res.statusText}`);
  }
  return res.json();
}

/** One durable row of `exec_run_events`, as `/events` serialises it. */
export interface RunEvent {
  cursor: number;
  type: string;
  payload: any;
  ts: string;
}

export interface RunEventsPage {
  run_id: string;
  status: RunStatus;
  active: boolean;
  events: RunEvent[];
  /** Where to resume; never moves backwards on an empty page. */
  cursor: number;
}

/**
 * Events strictly after `cursor`, oldest first.
 *
 * `wait` turns this into a long poll: the request parks server-side and
 * returns the moment an event lands, the run ends, or the deadline passes.
 * Pass a `signal` so a closing panel can abandon a parked request instead of
 * holding a connection open until it times out.
 */
export async function getRunEvents(
  runId: string,
  opts: { cursor?: number; wait?: number; limit?: number; signal?: AbortSignal } = {},
): Promise<RunEventsPage> {
  const params = new URLSearchParams();
  if (opts.cursor !== undefined) params.set('cursor', String(opts.cursor));
  if (opts.wait !== undefined) params.set('wait', String(opts.wait));
  if (opts.limit !== undefined) params.set('limit', String(opts.limit));
  const query = params.toString();
  const res = await fetch(
    `${BASE_URL}/runs/${encodeURIComponent(runId)}/events${query ? `?${query}` : ''}`,
    { signal: opts.signal },
  );
  if (!res.ok) throw new Error(`Failed to fetch run events: ${res.statusText}`);
  return res.json();
}

export interface RunMetricPoint {
  node_id: string | null;
  name: string;
  step: number;
  /** null for a non-finite value (a diverged loss) — a gap, not a zero. */
  value: number | null;
  /**
   * When the point was recorded, ISO-8601 UTC.
   *
   * `/metrics` has always returned it; it was simply undeclared until the
   * plugin contract started publishing this type (#132), and an undeclared
   * field reaching plugin authors through a typed facade is how it becomes
   * load-bearing by accident. Optional because the live `metric` event shares
   * this type and carries only what a chart plots against `step`.
   */
  ts?: string;
}

export interface RunMetrics {
  run_id: string;
  /** Every series name in the run, so a legend needs no scan of the points. */
  names: string[];
  metrics: RunMetricPoint[];
}

/** Recorded scalar series, ordered `(name, step)` — chart order. */
export async function getRunMetrics(
  runId: string,
  name?: string,
): Promise<RunMetrics> {
  const query = name ? `?name=${encodeURIComponent(name)}` : '';
  const res = await apiFetch(
    `${BASE_URL}/runs/${encodeURIComponent(runId)}/metrics${query}`,
  );
  if (!res.ok) throw new Error(`Failed to fetch run metrics: ${res.statusText}`);
  return res.json();
}

/** One `exec_run_artifacts` row — a file the run wrote and wants remembered. */
export interface RunArtifact {
  /**
   * The `exec_run_artifacts` row id. A client that learned of an artifact
   * from the event log rather than this endpoint may substitute a prefixed
   * synthetic key, which is why this is not `number`.
   */
  id: number | string;
  /** Open vocabulary: `checkpoint`, `export`, `image`, or a node pack's own. */
  kind: string;
  /** Path as the node wrote it, relative to the data root. */
  path: string;
  meta: Record<string, unknown> | null;
  created_at: string;
}

/**
 * Files the run recorded, oldest first.
 *
 * Artifacts also ride the event log, but mining that only works for a client
 * that watched the whole run — the panel opens on runs it never saw.
 */
export async function getRunArtifacts(
  runId: string,
  kind?: string,
): Promise<{ run_id: string; artifacts: RunArtifact[] }> {
  const query = kind ? `?kind=${encodeURIComponent(kind)}` : '';
  const res = await fetch(
    `${BASE_URL}/runs/${encodeURIComponent(runId)}/artifacts${query}`,
  );
  if (!res.ok) throw new Error(`Failed to fetch run artifacts: ${res.statusText}`);
  return res.json();
}

/**
 * Download a run's metrics as CSV.
 *
 * Goes through fetch + a blob rather than pointing the browser at the URL:
 * an error then surfaces as a rejected promise the panel can show, instead
 * of a blank tab containing a JSON `detail`.
 */
export async function downloadRunMetricsCsv(runId: string): Promise<void> {
  const res = await fetch(
    `${BASE_URL}/runs/${encodeURIComponent(runId)}/metrics?format=csv`,
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Download failed: ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `run-${runId}-metrics.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Verdict on one in-canvas script, from `POST /api/nodes/script/validate`. */
export interface ScriptValidation {
  ok: boolean;
  error: string | null;
  /** 1-based line the rejection points at, when the server knows it. */
  line: number | null;
  /** Whether a module-level `def run(...)` exists yet. */
  defines_run: boolean;
  /** The server's Tier-0 import allowlist, so the UI never hard-codes it. */
  allowed_modules: string[];
}

/**
 * Check a PythonScript body against the server's Tier-0 policy (core#131).
 *
 * The gate is an AST walk, so it can only live on the server; the editor
 * calls this while the user types so a rejected import is a banner rather
 * than a failed run. A policy REJECTION comes back as `ok: false` with 200 —
 * only a transport failure throws.
 */
export async function validateScript(code: string): Promise<ScriptValidation> {
  const res = await apiFetch(`${BASE_URL}/nodes/script/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });
  if (!res.ok) throw new Error(`Script validation failed: ${res.statusText}`);
  return res.json();
}

export async function validateGraph(
  nodes: any[],
  edges: any[],
  presets: any[] = [],
  subgraphs: any[] = [],
) {
  const res = await apiFetch(`${BASE_URL}/graph/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // Subgraph definitions are graph-local (core#137): there is no registry
    // to fall back on, so validation only sees inside a block if we send it.
    body: JSON.stringify({ nodes, edges, presets, subgraphs }),
  });
  if (!res.ok) throw new Error(`Validation failed: ${res.statusText}`);
  return res.json();
}

export async function saveGraph(data: GraphSaveData) {
  const res = await apiFetch(`${BASE_URL}/graph/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Save failed: ${res.statusText}`);
  return res.json();
}

export async function loadGraph(name: string) {
  const res = await fetch(`${BASE_URL}/graph/load/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Load failed: ${res.statusText}`);
  return res.json();
}

export async function listGraphs() {
  const res = await fetch(`${BASE_URL}/graph/list`);
  if (!res.ok) throw new Error(`List failed: ${res.statusText}`);
  return res.json();
}

export async function exportGraph(
  nodes: any[],
  edges: any[],
  name?: string,
  presets?: PresetDefinition[],
  // core#136: the canvas's reproducibility settings travel with the export
  // and become the exported script's --seed / --deterministic defaults.
  // Without them an exported augmenting graph drew fresh entropy on every
  // invocation, while the docs promised the same crops every time.
  run?: { seed?: number | null; deterministic?: boolean },
  // core#137: subgraph definitions are graph-local -- the instance node only
  // carries `subgraph:<id>`, and there is no server-side registry to look the
  // id up in. Omit these and `prepare_executable_graph` rejects the whole
  // export with `Unknown subgraph: <id>`, so every graph containing a
  // collapsed block was un-exportable from the UI.
  subgraphs?: any[],
) {
  const body: {
    nodes: any[];
    edges: any[];
    name?: string;
    presets?: PresetDefinition[];
    seed?: number | null;
    deterministic?: boolean;
    subgraphs?: any[];
  } = { nodes, edges };
  if (name) body.name = name;
  if (presets && presets.length > 0) body.presets = presets;
  // Same only-when-present rule as `presets`: the backend defaults the field
  // to `[]`, so a graph with no blocks keeps posting the body it always did.
  if (subgraphs && subgraphs.length > 0) body.subgraphs = subgraphs;
  if (run?.seed !== undefined && run.seed !== null) body.seed = run.seed;
  if (run?.deterministic) body.deterministic = true;
  const res = await apiFetch(`${BASE_URL}/graph/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errorBody = await res.json();
      if (typeof errorBody.detail === 'string') detail = errorBody.detail;
      else if (Array.isArray(errorBody.detail)) detail = errorBody.detail.join('; ');
    } catch {
      // Preserve the HTTP status text when the response is not JSON.
    }
    throw new Error(`Export failed: ${detail}`);
  }
  return res.json() as Promise<{ script: string }>;
}

/** A2: clear persisted layer weights kept by the backend NodeStateStore.
 * Pass `node_ids` to scope the reset to specific nodes, omit to reset
 * the entire graph. Returns `{ graph_id, scope, evicted }`.
 */
export async function resetWeights(graphId: string, nodeIds?: string[]) {
  const res = await apiFetch(`${BASE_URL}/execution/state/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph_id: graphId,
      ...(nodeIds && nodeIds.length > 0 ? { node_ids: nodeIds } : {}),
    }),
  });
  if (!res.ok) throw new Error(`Reset weights failed: ${res.statusText}`);
  return res.json() as Promise<{ graph_id: string; scope: string; evicted: number }>;
}

export async function createPreset(data: {
  name: string;
  description?: string;
  category?: string;
  tags?: string[];
  nodes: any[];
  edges: any[];
}): Promise<PresetDefinition> {
  const res = await apiFetch(`${BASE_URL}/presets/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Export failed: ${res.statusText}`);
  }
  return res.json();
}

// ── Examples ──

export interface ExampleSummary {
  name: string;
  description: string;
  category: string;
  path: string;
  node_count: number;
  edge_count: number;
  /**
   * Where the example ships from: `builtin`, or `plugin:<id>` for one a
   * plugin pack contributed. The gallery (core#128) shows it so an example
   * can be traced back to the pack that has to stay installed for it to work.
   *
   * Optional because this type describes what arrives over the wire, not a
   * guarantee: a frontend built from source can meet an older prebuilt
   * backend. Every consumer already treats a missing value as "built-in".
   */
  source?: string;
}

export async function listExamples(): Promise<ExampleSummary[]> {
  const res = await fetch(`${BASE_URL}/examples/list`);
  if (!res.ok) throw new Error(`Failed to list examples: ${res.statusText}`);
  return res.json();
}

export async function loadExample(path: string) {
  const res = await fetch(`${BASE_URL}/examples/load?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new Error(`Failed to load example: ${res.statusText}`);
  return res.json();
}

// ── Plugins ──

/** One installed plugin pack, as listed by `GET /api/plugins`.
 *
 * The endpoint returns disabled packs too (with `enabled: false` and an empty
 * `nodes` list, since a disabled pack is not in the registry), so the sidebar
 * can show them greyed out instead of pretending they are not installed. Only
 * the fields the UI renders are typed here; the endpoint returns more.
 */
export interface PluginSummary {
  id: string;
  name: string;
  version: string;
  description: string;
  enabled: boolean;
  nodes: string[];
  source_kind?: string;
}

export async function listPlugins(): Promise<PluginSummary[]> {
  const res = await fetch(`${BASE_URL}/plugins`);
  if (!res.ok) throw new Error(`Failed to list plugins: ${res.statusText}`);
  return res.json();
}

export async function reloadNodes() {
  const res = await apiFetch(`${BASE_URL}/nodes/reload`, { method: 'POST' });
  if (!res.ok) throw new Error(`Reload failed: ${res.statusText}`);
  return res.json();
}


// LLM / Codex auth

export type CodexAuthStatus =
  | { status: 'logged_out' }
  | { status: 'pending' }
  | { status: 'logged_in'; email?: string };

export async function fetchCodexStatus(): Promise<CodexAuthStatus> {
  const res = await fetch(`${BASE_URL}/llm/codex/status`);
  if (!res.ok) throw new Error(`Codex status failed: ${res.statusText}`);
  return res.json();
}

export async function startCodexLogin(): Promise<{ auth_url: string }> {
  const res = await apiFetch(`${BASE_URL}/llm/codex/login`, { method: 'POST' });
  if (!res.ok) throw new Error(`Codex login failed: ${res.statusText}`);
  return res.json();
}

export async function logoutCodex(): Promise<{ status: 'logged_out' }> {
  const res = await apiFetch(`${BASE_URL}/llm/codex/logout`, { method: 'POST' });
  if (!res.ok) throw new Error(`Codex logout failed: ${res.statusText}`);
  return res.json();
}
// ── Custom Node Manager ──

export interface CustomNodeInfo {
  filename: string;
  enabled: boolean;
  nodes: string[];
}

export async function listCustomNodes(): Promise<CustomNodeInfo[]> {
  const res = await fetch(`${BASE_URL}/custom-nodes`);
  if (!res.ok) throw new Error(`Failed to list custom nodes: ${res.statusText}`);
  return res.json();
}

export async function toggleCustomNode(filename: string) {
  const res = await apiFetch(`${BASE_URL}/custom-nodes/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename }),
  });
  if (!res.ok) throw new Error(`Toggle failed: ${res.statusText}`);
  return res.json();
}

export async function uploadCustomNode(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await apiFetch(`${BASE_URL}/custom-nodes/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export async function deleteCustomNode(filename: string) {
  const res = await apiFetch(`${BASE_URL}/custom-nodes/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

// ── Model Files ──

export interface ModelFileInfo {
  filename: string;
  size: number;
}

export async function listModelFiles(): Promise<ModelFileInfo[]> {
  const res = await fetch(`${BASE_URL}/models`);
  if (!res.ok) throw new Error(`Failed to list model files: ${res.statusText}`);
  return res.json();
}

export async function uploadModelFile(file: File): Promise<ModelFileInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await apiFetch(`${BASE_URL}/models/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteModelFile(filename: string) {
  const res = await apiFetch(`${BASE_URL}/models/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

export async function downloadModelFile(filename: string) {
  // Preserve slashes for nested paths (e.g. runs/exp1/model.pt)
  const urlPath = filename.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(`${BASE_URL}/models/download/${urlPath}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Download failed: ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  // Only the basename — the browser shouldn't recreate sub-directories locally
  // .pop() on a split result is always a string (possibly ''), never undefined
  /* v8 ignore start */
  a.download = filename.split('/').pop() ?? filename;
  /* v8 ignore stop */
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ── Image Files ──

export interface ImageFileInfo {
  filename: string;
  size: number;
}

// ── Data files (CSV / TSV / TXT / JSON) — backs the `data_file` param type ──
export interface DataFileInfo {
  filename: string;
  size: number;
}

export async function listDataFiles(): Promise<DataFileInfo[]> {
  const res = await fetch(`${BASE_URL}/files`);
  if (!res.ok) throw new Error(`Failed to list data files: ${res.statusText}`);
  return res.json();
}

export async function uploadDataFile(file: File): Promise<DataFileInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await apiFetch(`${BASE_URL}/files/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteDataFile(filename: string) {
  const res = await apiFetch(`${BASE_URL}/files/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

export async function downloadDataFile(filename: string) {
  const urlPath = filename.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(`${BASE_URL}/files/download/${urlPath}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Download failed: ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.split('/').pop() as string;
  a.click();
  URL.revokeObjectURL(url);
}

export async function listImageFiles(): Promise<ImageFileInfo[]> {
  const res = await fetch(`${BASE_URL}/images`);
  if (!res.ok) throw new Error(`Failed to list image files: ${res.statusText}`);
  return res.json();
}

export async function uploadImageFile(file: File): Promise<ImageFileInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await apiFetch(`${BASE_URL}/images/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteImageFile(filename: string) {
  const res = await apiFetch(`${BASE_URL}/images/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

export async function downloadImageFile(filename: string) {
  const urlPath = filename.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(`${BASE_URL}/images/download/${urlPath}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Download failed: ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  // .pop() on a split result is always a string (possibly ''), never undefined
  /* v8 ignore start */
  a.download = filename.split('/').pop() ?? filename;
  /* v8 ignore stop */
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ── Shared: refused requests, and jobs to follow ─────────────────────────

/**
 * An `/api` call the server refused, with the status AND the parsed body kept.
 *
 * The Package Center and the Plugin Center both answer with a status-coded
 * error vocabulary -- 403 remote, 404 gone, 409 busy, 507 out of disk -- and
 * each status carries extra keys the panel shows (`job_id`, `command`,
 * `blocked_by`, `missing_capabilities`). `body` is the parsed JSON, so none of
 * that has to be recovered from the message.
 */
export class ApiError extends Error {
  /**
   * The parsed error body, or null when it was not JSON.
   *
   * Assignable rather than `readonly`: a caller (a test building a refusal to
   * hand a store, say) constructs the error first and attaches the body it
   * should carry second.
   */
  body: Record<string, unknown> | null;

  constructor(
    public readonly status: number,
    message: string,
    body: Record<string, unknown> | null = null,
  ) {
    super(message);
    // `new.target` is the class actually being constructed, so `PackApiError`
    // reports its own name -- unchanged from before it had a base class --
    // without restating this line.
    this.name = new.target.name;
    this.body = body;
  }
}

/**
 * Read a refused response once, for every error factory below.
 *
 * The two bodies the backend sends are `{detail: "text"}` and `{detail:
 * {code, ...}}`. `code` is what a panel switches on, so a coded detail with
 * no `message` still yields something worth showing rather than a bare
 * "Conflict".
 */
async function readApiError(
  res: Response,
): Promise<{ message: string; body: Record<string, unknown> | null }> {
  const raw = await res.json().catch(() => null);
  const body =
    raw !== null && typeof raw === 'object' ? (raw as Record<string, unknown>) : null;
  const detail = body?.detail;
  if (typeof detail === 'string') return { message: detail, body };
  if (detail !== null && typeof detail === 'object') {
    const coded = detail as Record<string, unknown>;
    const message = coded.message ?? coded.code;
    if (typeof message === 'string') return { message, body };
  }
  return { message: res.statusText, body };
}

/** Build the error for a refused request, body and all. */
export async function apiError(res: Response): Promise<ApiError> {
  const { message, body } = await readApiError(res);
  return new ApiError(res.status, message, body);
}

/**
 * Where a background job -- a pack install, a plugin install or update -- has
 * got to. `needs_restart` is a job that finished everything it could do from
 * inside the running server.
 */
export type JobStatus =
  | 'running'
  | 'done'
  | 'failed'
  | 'cancelled'
  | 'needs_restart';

/**
 * One line of a job's log.
 *
 * Deliberately open rather than a union: the payload keys differ per `type`
 * (`log{line}`, `progress{item, bytes_done, ...}`, `job_done{sha, nodes}`),
 * and a closed union would have to be edited every time the backend grows a
 * step. Consumers narrow on `type`.
 */
export interface JobEvent {
  type: string;
  /** 1-based and monotonic. Resume from the PAGE's cursor, not from this. */
  cursor?: number;
  ts?: string;
  [k: string]: unknown;
}

/** One page of a job's events, oldest first. */
export interface JobEventsPage {
  job_id: string;
  status: JobStatus;
  events: JobEvent[];
  /** Where to resume; never moves backwards on an empty page. */
  cursor: number;
}

// ── Optional packs / Package Center ──────────────────────────────────────

/**
 * How a pack installs. `live` finishes inside the running server; `restart`
 * is a wheel swap (the GPU PyTorch pack) that no process can do to the
 * interpreter it is running on, so the server hands back a command instead.
 */
export type PackInstallMode = 'live' | 'restart';

export type PackStatus =
  | 'not_installed'
  | 'partial'
  | 'installed'
  | 'installing'
  | 'needs_restart';

export type PackItemStatus = 'missing' | 'present' | 'downloading';

/**
 * The pack panel's name for `JobStatus`. One alias rather than a second copy
 * of the union, so the two centers cannot drift apart.
 */
export type PackJobStatus = JobStatus;

/**
 * How the server was launched (`CODEFYUI_MANAGED`). `unknown` is a bare
 * `uvicorn app.main:app`: nothing supervises it, so nothing can restart it
 * for the user.
 */
export type LaunchMode = 'start' | 'dev' | 'unknown';

/** One downloadable file inside a pack -- a Hugging Face repo or a plain URL. */
export interface PackItem {
  id: string;
  kind: 'hf' | 'asset';
  /**
   * Both keys are always present with one of them null: the backend
   * deliberately serialises ONE item shape rather than one per `kind`.
   */
  repo_id: string | null;
  url: string | null;
  size_bytes: number;
  license: string | null;
  status: PackItemStatus;
}

export interface PackSummary {
  id: string;
  title: string;
  description: string;
  install_mode: PackInstallMode;
  status: PackStatus;
  pip_ready: boolean;
  usable: boolean;
  depends_on: string[];
  /** Packs that must be installed first. Non-empty means an install is
   *  refused with a 400 naming exactly these. */
  blocked_by: string[];
  pip: { spec: string }[];
  items: PackItem[];
  /** Bytes still to FETCH -- an already-downloaded item contributes nothing,
   *  which makes this 0 for an installed pack without a special case. */
  size_bytes_total: number;
  install_command: string | null;
}

/** What this machine can offer for the GPU PyTorch pack. */
export interface PackGpuInfo {
  detected_label: string | null;
  recommended_variant: string | null;
  /** null when the installed wheel cannot be told, which is not "none". */
  installed_variant: string | null;
  variants: string[];
  install_command: string | null;
}

/** The install running right now. A FINISHED job keeps its events but is not
 *  reported here. */
export interface PackJobRef {
  job_id: string;
  pack_id: string;
}

export interface PackCatalog {
  packs: PackSummary[];
  active_job: PackJobRef | null;
  /**
   * What the last restart-mode job left behind, so the panel can report an
   * install that finished while it was not running. Left as an open map: it
   * is a record this client only echoes, and the backend already treats a
   * corrupt one as no record at all.
   */
  last_restart_job: Record<string, unknown> | null;
  remote_install_allowed: boolean;
  launch_mode: LaunchMode;
  /**
   * Whether the server can install a restart-mode pack and relaunch itself.
   *
   * Narrower than `launch_mode === 'start'`, and the only thing worth asking:
   * the server also wants its launcher still on disk and its kill switch off
   * before it will promise to come back. So this is what the panel gates the
   * "Install and restart" button on — the server is the authority on whether
   * it can restart, not the mode it happens to have been started in.
   */
  restart_available: boolean;
  gpu: PackGpuInfo | null;
}

/**
 * One line of a pack install job's log: a `JobEvent` plus the two keys the
 * restart contract adds.
 */
export interface PackJobEvent extends JobEvent {
  /**
   * `needs_restart` only: which helper finishes the install (`torch`, `pip`).
   *
   * Named here rather than left to the index signature because it is part of
   * the restart contract a reader of this file is trying to find, and `string`
   * rather than a union for the same reason `type` is: a newer backend may
   * grow a kind, and the consumer narrows on the ones it knows.
   */
  kind?: string;
  /**
   * `needs_restart` only, and only on a LIVE install the resolver stopped:
   * the mode that CAN finish it (`restart` today). ABSENT — not null — when
   * the server cannot restart itself, so its presence is the whole check.
   */
  retry_mode?: string;
}

/** A page of pack events -- `JobEventsPage` narrowed to `PackJobEvent`. */
export interface PackJobEventsPage extends JobEventsPage {
  events: PackJobEvent[];
}

/**
 * A `/api/packs` call the server refused.
 *
 * Its own class rather than a bare `ApiError` so that the pack store's
 * `err instanceof PackApiError` keeps meaning "a PACK call refused" now that
 * the Plugin Center throws the same base error, and so that `err.name` still
 * reads `PackApiError` in a console.
 */
export class PackApiError extends ApiError {}

/** Build the error for a refused pack request, body and all. */
async function packApiError(res: Response): Promise<PackApiError> {
  const { message, body } = await readApiError(res);
  return new PackApiError(res.status, message, body);
}

/**
 * The whole catalog: every pack, what is installed, and what this machine can
 * install. The one route the panel polls.
 *
 * Normalized field by field rather than passed through, because the panel
 * maps over `packs`, `items`, `pip` and `blocked_by` on every repaint: an
 * absent key has to arrive as an empty list, not as a crash mid-render.
 * `remote_install_allowed` defaults to ALLOWED for the same reason the UI
 * never enforces it -- the server refuses a remote install itself, with a 403
 * this then reports, so a missing key must not hide a button that works.
 */
export async function listPacks(): Promise<PackCatalog> {
  const res = await fetch(`${BASE_URL}/packs`);
  if (!res.ok) throw await packApiError(res);
  const data = await res.json();
  const launchMode = data.launch_mode;
  return {
    packs: (data.packs ?? []).map(
      (pack: any): PackSummary => ({
        id: pack.id,
        title: pack.title,
        description: pack.description,
        install_mode: pack.install_mode,
        status: pack.status ?? 'not_installed',
        pip_ready: pack.pip_ready ?? false,
        usable: pack.usable ?? false,
        depends_on: pack.depends_on ?? [],
        blocked_by: pack.blocked_by ?? [],
        pip: pack.pip ?? [],
        items: pack.items ?? [],
        size_bytes_total: pack.size_bytes_total ?? 0,
        install_command: pack.install_command ?? null,
      }),
    ),
    active_job: data.active_job ?? null,
    last_restart_job: data.last_restart_job ?? null,
    remote_install_allowed: data.remote_install_allowed ?? true,
    launch_mode:
      launchMode === 'start' || launchMode === 'dev' ? launchMode : 'unknown',
    // The opposite default to `remote_install_allowed` above, for the same
    // reason: a server too old to answer this cannot restart itself, and a
    // button that promises one would hand the user a 409 instead of a
    // command. Absent — or any non-boolean — is NO.
    restart_available: data.restart_available === true,
    gpu: data.gpu ?? null,
  };
}

/**
 * Start installing a pack. 202 and a `job_id` to follow with
 * `getPackJobEvents`.
 *
 * `items` omitted means the whole pack, minus what is already downloaded.
 * `JSON.stringify` drops undefined values, so a caller spreading a partly
 * filled options object sends only the keys it actually set -- which matters
 * because the backend's request model forbids anything it did not declare.
 */
export async function installPack(
  packId: string,
  body: { items?: string[]; mode?: PackInstallMode; variant?: string } = {},
): Promise<{ job_id: string }> {
  const res = await apiFetch(
    `${BASE_URL}/packs/${encodeURIComponent(packId)}/install`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) throw await packApiError(res);
  return res.json();
}

/**
 * Ask the running install to stop. `cancelled` reports whether the request
 * did anything -- false for a job that had already finished, which is a
 * normal answer rather than an error.
 */
export async function cancelPackJob(
  jobId: string,
): Promise<{ job_id: string; cancelled: boolean }> {
  const res = await apiFetch(
    `${BASE_URL}/packs/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST' },
  );
  if (!res.ok) throw await packApiError(res);
  return res.json();
}

/**
 * A job's events strictly after `cursor`, oldest first.
 *
 * `wait` turns this into a long poll, exactly like `getRunEvents`: the
 * request parks server-side and returns the moment an event lands, the job
 * ends, or the deadline passes. Pass a `signal` so a closing panel can
 * abandon a parked request instead of holding a connection open.
 */
export async function getPackJobEvents(
  jobId: string,
  opts: { cursor?: number; wait?: number; limit?: number; signal?: AbortSignal } = {},
): Promise<PackJobEventsPage> {
  const params = new URLSearchParams();
  if (opts.cursor !== undefined) params.set('cursor', String(opts.cursor));
  if (opts.wait !== undefined) params.set('wait', String(opts.wait));
  if (opts.limit !== undefined) params.set('limit', String(opts.limit));
  const query = params.toString();
  const res = await fetch(
    `${BASE_URL}/packs/jobs/${encodeURIComponent(jobId)}/events${query ? `?${query}` : ''}`,
    { signal: opts.signal },
  );
  if (!res.ok) throw await packApiError(res);
  return res.json();
}

/**
 * Delete one downloaded model. `removed` false means the sentinel was
 * cleared but bytes are still on disk (Windows keeps an open file), which
 * the caller is entitled to report rather than promise the space back.
 *
 * Items only: nothing removes a pack's pip packages, here or anywhere.
 */
export async function removePackItem(
  packId: string,
  itemId: string,
): Promise<{ pack_id: string; item_id: string; removed: boolean }> {
  const res = await apiFetch(
    `${BASE_URL}/packs/${encodeURIComponent(packId)}/items/${encodeURIComponent(itemId)}`,
    { method: 'DELETE' },
  );
  if (!res.ok) throw await packApiError(res);
  return res.json();
}
