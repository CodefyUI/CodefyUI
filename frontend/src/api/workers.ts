import { apiFetch } from './_auth';

/**
 * Worker agents as the control plane sees them (#772): `GET /api/workers`
 * and the drain / undrain actions, for the Workers panel (#776).
 *
 * A module of its own rather than more of `rest.ts`, so the panel's helpers
 * and the run-target selector's (#775) can land independently and be folded
 * together later.
 */

const BASE_URL = '/api/workers';

/**
 * `draining` is reported for a connected worker that has the flag set;
 * `offline` arrives with persisted workers (#780).
 */
export type WorkerStatus = 'online' | 'degraded' | 'disconnected' | 'draining' | 'offline';

/** Memory facts from the worker's heartbeat; `null` counts mean unknown. */
export interface WorkerDeviceMemory {
  total_bytes: number | null;
  free_bytes: number | null;
  source: string;
}

/** Per-device admission slots (#773). */
export interface WorkerDeviceSlots {
  limit: number;
  taken: number;
}

export interface WorkerDevice {
  value: string;
  kind: string;
  index: number | null;
  name: string;
  memory: WorkerDeviceMemory | null;
  /** Absent until the server reports per-device slots (#773). */
  slots?: WorkerDeviceSlots;
  /** Run ids on this device, when the server reports them per device. */
  running?: string[];
}

export interface WorkerInfo {
  id: string;
  name: string;
  status: WorkerStatus;
  draining: boolean;
  version: string;
  /** The worker's environment digest; `null` until it reports one (#781). */
  fingerprint_digest: string | null;
  platform: Record<string, string>;
  /** Storage labels to paths on the worker (#782). */
  mounts: Record<string, string>;
  devices: WorkerDevice[];
  /** Run ids the worker is executing. */
  running: string[];
  /** Epoch seconds. */
  connected_at: number;
  /** Epoch seconds. */
  last_heartbeat_at: number;
}

async function failure(res: Response, fallback: string): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  return new Error(typeof body.detail === 'string' ? body.detail : `${fallback}: ${res.statusText}`);
}

/** Every worker the server knows, connected or recently disconnected. */
export async function fetchWorkers(): Promise<WorkerInfo[]> {
  const res = await fetch(BASE_URL);
  if (!res.ok) throw await failure(res, 'Workers failed');
  const body = await res.json();
  return Array.isArray(body.workers) ? body.workers : [];
}

async function setDraining(workerId: string, action: 'drain' | 'undrain'): Promise<WorkerInfo> {
  const res = await apiFetch(`${BASE_URL}/${encodeURIComponent(workerId)}/${action}`, {
    method: 'POST',
  });
  if (!res.ok) throw await failure(res, action === 'drain' ? 'Drain failed' : 'Undrain failed');
  return res.json();
}

/** Stop assigning new runs to the worker; its running runs carry on. */
export function drainWorker(workerId: string): Promise<WorkerInfo> {
  return setDraining(workerId, 'drain');
}

/** Let the worker take new runs again. */
export function undrainWorker(workerId: string): Promise<WorkerInfo> {
  return setDraining(workerId, 'undrain');
}
