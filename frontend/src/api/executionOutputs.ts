import type { OutputData, RunOutputRef } from '../types';
import { apiFetch } from './_auth';

const BASE_URL = '/api/execution/outputs';

export class RunDataExpiredError extends Error {
  constructor(runId: string) {
    super(`Run '${runId}' data no longer available on server`);
    this.name = 'RunDataExpiredError';
  }
}

/**
 * 204 from a port read: the node ran and this port produced no value (`None`).
 * Nothing expired and nothing was left unrecorded, so it is not a 404.
 */
export class NoValueError extends Error {
  constructor(runId: string, nodeId: string, port: string) {
    super(`'${nodeId}.${port}' produced no value in run '${runId}'`);
    this.name = 'NoValueError';
  }
}

export class PayloadTooLargeError extends Error {
  constructor(detail: string) {
    super(detail);
    this.name = 'PayloadTooLargeError';
  }
}

export class InvalidSliceError extends Error {
  constructor(detail: string) {
    super(detail);
    this.name = 'InvalidSliceError';
  }
}

async function readDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === 'string') return body.detail;
  } catch {
    // ignore, fall through
  }
  return res.statusText;
}

type CaptureRead = 'value' | 'stats' | 'steps' | 'grads';

/**
 * The URL of one read of a run's captures for a node.
 *
 * A node inside a block runs, and is captured, as `<instance>/<inner>`
 * (#621). The server decodes `%2F` before it routes, so such an id in a path
 * segment reaches no route at all; it goes in the query instead,
 * `/{run}/{read}?node_id=&port=`. Every id without a slash keeps the path
 * form, byte for byte.
 */
function captureUrl(
  runId: string,
  read: CaptureRead,
  nodeId: string,
  port: string | null,
  params: URLSearchParams = new URLSearchParams(),
): string {
  const run = `${BASE_URL}/${encodeURIComponent(runId)}`;
  if (nodeId.includes('/') || port?.includes('/')) {
    const query = new URLSearchParams({ node_id: nodeId });
    if (port !== null) query.set('port', port);
    params.forEach((value, key) => query.set(key, value));
    return `${run}/${read}?${query.toString()}`;
  }
  const node = encodeURIComponent(nodeId);
  const portPart = port === null ? '' : encodeURIComponent(port);
  const path =
    read === 'steps'
      ? `${node}/__steps_index`
      : read === 'grads'
        ? `${node}/__grad_index`
        : read === 'stats'
          ? `${node}/${portPart}/stats`
          : `${node}/${portPart}`;
  const qs = params.toString();
  return `${run}/${path}` + (qs ? `?${qs}` : '');
}

export async function fetchOutput(
  runId: string,
  nodeId: string,
  port: string,
  opts: { slice?: string; maxElements?: number } = {},
): Promise<OutputData> {
  const params = new URLSearchParams();
  if (opts.slice) params.set('slice', opts.slice);
  if (opts.maxElements != null) params.set('max_elements', String(opts.maxElements));

  const url = captureUrl(runId, 'value', nodeId, port, params);

  const res = await fetch(url);
  // Before `res.ok`: a 204 is ok and has no body to parse.
  if (res.status === 204) throw new NoValueError(runId, nodeId, port);
  if (res.status === 404) throw new RunDataExpiredError(runId);
  if (res.status === 400) throw new InvalidSliceError(await readDetail(res));
  if (res.status === 413) throw new PayloadTooLargeError(await readDetail(res));
  if (!res.ok) throw new Error(`fetchOutput failed: ${await readDetail(res)}`);
  return res.json();
}

export async function listRunOutputs(runId: string): Promise<RunOutputRef[]> {
  const res = await fetch(`${BASE_URL}/${encodeURIComponent(runId)}`);
  if (res.status === 404) throw new RunDataExpiredError(runId);
  if (!res.ok) throw new Error(`listRunOutputs failed: ${await readDetail(res)}`);
  return res.json();
}

export async function deleteRun(runId: string): Promise<void> {
  const res = await apiFetch(`${BASE_URL}/${encodeURIComponent(runId)}`, { method: 'DELETE' });
  if (res.status === 404) throw new RunDataExpiredError(runId);
  if (!res.ok) throw new Error(`deleteRun failed: ${await readDetail(res)}`);
}

/** Metadata about one algorithmic step recorded by an instrumented node. */
export interface StepIndexEntry {
  index: number;
  name: string;
  description: string;
  scalars: Record<string, number>;
  tensor_keys: string[];
}

export async function fetchStepIndex(
  runId: string,
  nodeId: string,
): Promise<StepIndexEntry[]> {
  const url = captureUrl(runId, 'steps', nodeId, null);
  const res = await fetch(url);
  if (res.status === 404) {
    // No steps recorded for this node — treat as empty list rather than error.
    return [];
  }
  if (!res.ok) throw new Error(`fetchStepIndex failed: ${await readDetail(res)}`);
  return res.json();
}

/** A single gradient entry returned by ``__grad_index``. */
export interface GradIndexEntry {
  port: string;
  kind: 'port' | 'weight';
  has_grad: boolean;
  health: {
    status: 'healthy' | 'vanishing' | 'exploding';
    norm: number;
    mean: number;
    max: number;
  } | null;
}

export async function fetchGradIndex(
  runId: string,
  nodeId: string,
): Promise<GradIndexEntry[]> {
  const url = captureUrl(runId, 'grads', nodeId, null);
  const res = await fetch(url);
  // A node without gradients in a run the server holds is a 200 with [];
  // 404 is the run itself gone, which the Backward tab reports as expired.
  if (res.status === 404) throw new RunDataExpiredError(runId);
  if (!res.ok) throw new Error(`fetchGradIndex failed: ${await readDetail(res)}`);
  return res.json();
}

// ── Port statistics (#129) ───────────────────────────────────────────────────

/** A fixed-bin histogram: `bins` counts between `bins + 1` edges. */
export interface StatsHistogram {
  bins: number;
  /** Nullable like every other float here — a non-finite edge reports `null`. */
  edges: (number | null)[];
  counts: number[];
}

/** One class in an integer tensor's class balance. */
export interface StatsValueCount {
  value: number | boolean;
  count: number;
}

/** Per-column summary for a tabular value (columnar dict, records, or list). */
export interface StatsColumn {
  name: string;
  dtype: 'int' | 'float' | 'str' | 'bool' | 'mixed' | 'empty';
  count: number;
  missing: number;
  unique: number;
  top: { value: string | number | boolean; count: number }[];
  mean: number | null;
  min: number | null;
  max: number | null;
}

/**
 * What `GET /api/execution/outputs/{run}/{node}/{port}/stats` returns.
 *
 * Every numeric field is `null` rather than NaN when the statistic is
 * undefined — an all-NaN tensor has no mean, and the server says so with
 * `null` instead of a token no JSON parser accepts.
 *
 * `sampled` covers the DISTRIBUTION only. `count`, `min`, `max`, `nan_count`,
 * `inf_count`, `zero_frac` and `value_counts` are exact at any size.
 */
export interface PortStats {
  run_id: string;
  node_id: string;
  port: string;
  kind: 'tensor' | 'tabular' | 'unsupported';
  sampled?: boolean;
  sample_size?: number | null;

  // kind === 'tensor'
  shape?: number[];
  dtype?: string;
  device?: string;
  count?: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  quantiles?: Record<string, number | null>;
  nan_count?: number;
  inf_count?: number;
  zero_frac?: number | null;
  histogram?: StatsHistogram | null;
  value_counts?: StatsValueCount[] | null;

  // kind === 'tabular'
  rows?: number;
  column_count?: number;
  columns_truncated?: boolean;
  columns?: StatsColumn[];

  // kind === 'unsupported'
  type?: string;
}

/** 404 from the stats endpoint — carries the server's "turn on Rec" hint. */
export class StatsNotCapturedError extends Error {
  constructor(detail: string) {
    super(detail);
    this.name = 'StatsNotCapturedError';
  }
}

export async function fetchPortStats(
  runId: string,
  nodeId: string,
  port: string,
  opts: { signal?: AbortSignal } = {},
): Promise<PortStats> {
  const url = captureUrl(runId, 'stats', nodeId, port);
  const res = await fetch(url, { signal: opts.signal });
  if (res.status === 204) throw new NoValueError(runId, nodeId, port);
  if (res.status === 404) throw new StatsNotCapturedError(await readDetail(res));
  if (!res.ok) throw new Error(`fetchPortStats failed: ${await readDetail(res)}`);
  return res.json();
}
