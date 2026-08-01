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

export interface HealthInfo {
  status: string;
  nodes_loaded: number;
  presets_loaded: number;
  project: string | null;
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
 */
export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health failed: ${res.statusText}`);
  const data = await res.json();
  return {
    status: data.status,
    nodes_loaded: data.nodes_loaded,
    presets_loaded: data.presets_loaded,
    project: data.project ?? null,
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
 */
export async function getRun(runId: string): Promise<RunInfo | null> {
  const res = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}`);
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
  const res = await fetch(`${BASE_URL}/runs${query ? `?${query}` : ''}`);
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
  const res = await fetch(
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

export async function validateGraph(nodes: any[], edges: any[], presets: any[] = []) {
  const res = await apiFetch(`${BASE_URL}/graph/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes, edges, presets }),
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
) {
  const body: {
    nodes: any[];
    edges: any[];
    name?: string;
    presets?: PresetDefinition[];
  } = { nodes, edges };
  if (name) body.name = name;
  if (presets && presets.length > 0) body.presets = presets;
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
   */
  source: string;
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
