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
 * `offline` is an enrolled worker that has not connected since the server
 * started (#780).
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
  /** Epoch seconds; `null` for an `offline` worker. */
  connected_at: number | null;
  /** Epoch seconds; `null` for an enrolled worker never seen. */
  last_heartbeat_at: number | null;
  /** It holds a worker credential the server can revoke (#780). */
  enrolled?: boolean;
  /** How its connection authenticated: `worker_token` or `session_token`. */
  credential?: string | null;
  /**
   * Its connection is plain `ws://` from another machine: its token and the
   * secrets in its runs cross the network unencrypted (#786).
   */
  plain_http?: boolean;
}

/** `pending` can still enroll; `expired` and `used` are listed until deleted. */
export type EnrollmentState = 'pending' | 'expired' | 'used';

/** An enrollment token as listed: never the secret itself. */
export interface Enrollment {
  /** The token's public half (the six characters before the dot). */
  id: string;
  label: string | null;
  /** The worker a re-key token is bound to, or `null`. */
  worker: string | null;
  /** ISO-8601 UTC. */
  created_at: string;
  /** ISO-8601 UTC. */
  expires_at: string;
  uses_left: number;
  state: EnrollmentState;
}

/** A new enrollment token: the only response that carries the secret. */
export interface CreatedEnrollment extends Enrollment {
  token: string;
  control_url: string;
  command: string;
}

export interface EnrollmentList {
  enrollments: Enrollment[];
  /** This server as the browser reached it (scheme and Host). */
  control_url: string;
  /** Whitelisted addresses another machine could dial. */
  other_machine_urls: string[];
  /** The server is bound to a loopback address: no other machine reaches it. */
  listening_on_loopback: boolean;
  /**
   * `sha256:<hex>` of the TLS certificate's public key when the server can
   * read it (`CODEFYUI_TLS_CERT_FILE`), else `null` (#786).
   */
  tls_fingerprint?: string | null;
  /** The `openssl` command that computes the fingerprint on the proxy host. */
  fingerprint_command?: string;
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

/** Revoke the worker's credential, disconnect it and forget it (#780). */
export async function removeWorker(workerId: string): Promise<void> {
  const res = await apiFetch(`${BASE_URL}/${encodeURIComponent(workerId)}`, { method: 'DELETE' });
  if (!res.ok) throw await failure(res, 'Remove failed');
}

/** The enrollment tokens and the addresses the dialog offers (#780). */
export async function fetchEnrollments(): Promise<EnrollmentList> {
  // A GET the server guards with the session token.
  const res = await apiFetch(`${BASE_URL}/enrollments`, {}, { authenticate: true });
  if (!res.ok) throw await failure(res, 'Enrollment tokens failed');
  return res.json();
}

/**
 * A new enrollment token, valid for `ttl_s` seconds and `uses` agents.
 * With `worker`, a re-key token: it can enroll that worker only. Its
 * `command` dials `control_url` (default: the address the page uses).
 */
export async function createEnrollment(
  options: { ttl_s: number; uses: number; worker?: string; control_url?: string },
): Promise<CreatedEnrollment> {
  const res = await apiFetch(`${BASE_URL}/enrollments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!res.ok) throw await failure(res, 'Generate token failed');
  return res.json();
}

/** Revoke (or clear away) an enrollment token. */
export async function deleteEnrollment(id: string): Promise<void> {
  const res = await apiFetch(`${BASE_URL}/enrollments/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!res.ok) throw await failure(res, 'Revoke failed');
}

/**
 * The agent command for an enrollment *token* at *controlUrl* (#786): the
 * server builds it, with the same function as the `command` of
 * {@link createEnrollment}, so the flags (`--name`, `--ca-fingerprint`)
 * are decided in one place. The token goes in the body, never a URL.
 */
export async function agentCommand(
  options: { control_url: string; token: string; worker?: string | null },
): Promise<string> {
  const res = await apiFetch(`${BASE_URL}/agent-command`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!res.ok) throw await failure(res, 'Command failed');
  const body = await res.json();
  return body.command;
}
