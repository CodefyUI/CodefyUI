import type { OutputData, RunOutputRef } from '../types';

const BASE_URL = '/api/execution/outputs';

export class RunDataExpiredError extends Error {
  constructor(runId: string) {
    super(`Run '${runId}' data no longer available on server`);
    this.name = 'RunDataExpiredError';
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

export async function fetchOutput(
  runId: string,
  nodeId: string,
  port: string,
  opts: { slice?: string; maxElements?: number } = {},
): Promise<OutputData> {
  const params = new URLSearchParams();
  if (opts.slice) params.set('slice', opts.slice);
  if (opts.maxElements != null) params.set('max_elements', String(opts.maxElements));

  const qs = params.toString();
  const url =
    `${BASE_URL}/${encodeURIComponent(runId)}/${encodeURIComponent(nodeId)}/${encodeURIComponent(port)}` +
    (qs ? `?${qs}` : '');

  const res = await fetch(url);
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
  const res = await fetch(`${BASE_URL}/${encodeURIComponent(runId)}`, { method: 'DELETE' });
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
  const url = `${BASE_URL}/${encodeURIComponent(runId)}/${encodeURIComponent(nodeId)}/__steps_index`;
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
  const url = `${BASE_URL}/${encodeURIComponent(runId)}/${encodeURIComponent(nodeId)}/__grad_index`;
  const res = await fetch(url);
  if (res.status === 404) return [];
  if (!res.ok) throw new Error(`fetchGradIndex failed: ${await readDetail(res)}`);
  return res.json();
}
