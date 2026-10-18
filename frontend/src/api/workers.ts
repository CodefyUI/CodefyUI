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
  /**
   * The offerer whose limit a card several workers share takes (the
   * smallest among the online ones, #842); absent when it is this slot's
   * own. `worker` is `local` for this server.
   */
  limited_by?: { worker: string; device: string };
}

/** How a worker's card admits runs (#788): by count, or by memory budget. */
export type DevicePolicy = 'exclusive' | 'shared';

/**
 * A shared card's ledger (#788): the budgets promised on it (reserved:
 * assigned, not started; running: started with the cap in place), what a
 * new budget may take (`headroom_bytes`), and the runs holding budgets.
 * `by_memory` is false when the worker's agent cannot enforce budgets: the
 * card then admits by count.
 */
export interface WorkerDeviceAdmission {
  by_memory: boolean;
  total_bytes: number | null;
  free_bytes: number | null;
  reserved_bytes: number;
  running_bytes: number;
  headroom_bytes: number | null;
  /** The CUDA context each run holds beyond its budget; 0 off CUDA. */
  context_bytes?: number;
  runs: string[];
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
  /** `exclusive` (the default) or `shared` (#788); absent from an older server. */
  policy?: DevicePolicy;
  /** A shared card's ledger; `null` for an exclusive one (#788). */
  admission?: WorkerDeviceAdmission | null;
  /**
   * The other offerers of this physical card, matched by UUID (#842):
   * `worker` is a worker id, or `local` for this server. Empty, or absent
   * from an older server, for a card nobody else offers.
   */
  same_card?: WorkerCardPeer[];
}

/** One other offerer of a physical card (#842). */
export interface WorkerCardPeer {
  worker: string;
  device: string;
  /** How that offerer admits runs; `null` for this server (by count). */
  policy?: DevicePolicy | null;
  /** That offerer can take runs now (this server always). */
  online?: boolean;
}

/**
 * The worker's node catalog against this server's (#781): `match`, or
 * `mismatch` with the node types this server has and the worker lacks
 * (`missing`) and the other way round (`extra`), or `unknown` when the
 * worker reported none.
 */
export interface WorkerNodeCatalog {
  status: 'match' | 'mismatch' | 'unknown';
  hash: string | null;
  missing: string[];
  extra: string[];
}

/** A worker name's last refused handshake (#781), kept in server memory. */
export interface WorkerRejection {
  name: string;
  reason: string;
  /** Epoch seconds. */
  at: number;
  /** The refused client's address, when known. */
  peer?: string | null;
  /**
   * A connected worker holds the name: the refusal was another agent's
   * attempt to take it, not that worker's own.
   */
  live_owner?: boolean;
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
  /** Its node catalog against this server's (#781). */
  node_catalog?: WorkerNodeCatalog;
  /** Why `status` reads `degraded`: `node_catalog`, `heartbeat` (#781). */
  degraded_reasons?: string[];
  /** Its agent enforces memory budgets, so its cards can be shared (#788). */
  memory_budget?: boolean;
  /**
   * A `cdui agent start --one-shot` worker (#790): `run_id` is the one run
   * it takes (`--run-id`), `took` the run it took. `null` (or absent) for a
   * regular worker.
   */
  one_shot?: WorkerOneShot | null;
}

/** {@link WorkerInfo.one_shot}. */
export interface WorkerOneShot {
  run_id: string | null;
  took: string | null;
}

/** `GET /api/workers`: the workers and each name's last refused handshake. */
export interface WorkerList {
  workers: WorkerInfo[];
  rejections: WorkerRejection[];
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

/** An API failure: the server's `detail` (or *fallback*), and the HTTP status. */
export class WorkersApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function failure(res: Response, fallback: string): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  return new WorkersApiError(
    typeof body.detail === 'string' ? body.detail : `${fallback}: ${res.statusText}`,
    res.status,
  );
}

/** The worker list with the handshakes the server refused (#781). */
export async function fetchWorkerList(): Promise<WorkerList> {
  const res = await fetch(BASE_URL);
  if (!res.ok) throw await failure(res, 'Workers failed');
  const body = await res.json();
  return {
    workers: Array.isArray(body.workers) ? body.workers : [],
    rejections: Array.isArray(body.rejections) ? body.rejections : [],
  };
}

/** Every worker the server knows, connected or recently disconnected. */
export async function fetchWorkers(): Promise<WorkerInfo[]> {
  return (await fetchWorkerList()).workers;
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

/** Set how the worker's *device* admits runs (#788); answers the worker. */
export async function setDevicePolicy(
  workerId: string,
  device: string,
  policy: DevicePolicy,
): Promise<WorkerInfo> {
  const path = `${encodeURIComponent(workerId)}/devices/${encodeURIComponent(device)}/policy`;
  const res = await apiFetch(`${BASE_URL}/${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ policy }),
  });
  if (!res.ok) throw await failure(res, 'Policy change failed');
  return res.json();
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
 * An SSH worker's tunnel (#784): `connecting` while ssh runs and the agent
 * has not connected, `up` once it has, `failed` while waiting to retry
 * after ssh exited, `stopped` when disabled.
 */
export type TunnelState = 'connecting' | 'up' | 'failed' | 'stopped';

/** A worker this server starts itself over `ssh -R` (#784). */
export interface SshWorker {
  /** The worker name. */
  id: string;
  /** `user@host` or a Host alias from the server account's ssh config. */
  target: string;
  port: number;
  /** A key file on the server's machine, or `null` for ssh's defaults. */
  identity: string | null;
  /** The cdui launcher on the remote host. */
  remote_cdui: string;
  /** The remote loopback port of the reverse forward. */
  remote_port: number | null;
  enabled: boolean;
  state: TunnelState;
  /** The last lines ssh (or the remote agent) wrote before it exited. */
  last_error: string | null;
  exit_code: number | null;
  tail: string[];
  attempts: number;
  /** Epoch seconds of the next attempt, while waiting to retry. */
  retry_at: number | null;
  /** Epoch seconds the worker came online through the tunnel. */
  up_since: number | null;
  /** It stopped retrying (the remote sshd forbids forwarding); Enable retries. */
  gave_up?: boolean;
  /** The agent's pid on the remote host, once the remote script printed it. */
  remote_pid?: number | null;
  /** That agent survived a dropped session and this session re-attached to it. */
  reattached?: boolean;
}

export interface SshWorkerList {
  ssh_workers: SshWorker[];
  /** The server's machine has an `ssh` client on PATH. */
  ssh_available: boolean;
  port_range: [number, number];
  default_remote_cdui: string;
}

export interface NewSshWorker {
  target: string;
  port?: number;
  identity?: string;
  remote_cdui?: string;
  name?: string;
}

/** The SSH workers and their tunnel state (#784). */
export async function fetchSshWorkers(): Promise<SshWorkerList> {
  // A GET the server guards with the session token.
  const res = await apiFetch(`${BASE_URL}/ssh`, {}, { authenticate: true });
  if (!res.ok) throw await failure(res, 'SSH workers failed');
  return res.json();
}

/** Store an SSH worker; the server starts its ssh session at once. */
export async function addSshWorker(body: NewSshWorker): Promise<SshWorker> {
  const res = await apiFetch(`${BASE_URL}/ssh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await failure(res, 'Add SSH worker failed');
  return res.json();
}

/** Enable (start ssh) or disable (drain, then end ssh) an SSH worker. */
export async function setSshWorkerEnabled(id: string, enabled: boolean): Promise<SshWorker> {
  const action = enabled ? 'enable' : 'disable';
  const res = await apiFetch(`${BASE_URL}/ssh/${encodeURIComponent(id)}/${action}`, { method: 'POST' });
  if (!res.ok) throw await failure(res, enabled ? 'Enable failed' : 'Disable failed');
  return res.json();
}

/** Delete an SSH worker: end its session, revoke it, and clean up on its host. */
export async function deleteSshWorker(id: string): Promise<void> {
  const res = await apiFetch(`${BASE_URL}/ssh/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!res.ok) throw await failure(res, 'Delete failed');
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
