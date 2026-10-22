/**
 * The choices the "Queue on..." dialog (#775) offers, worked out from what
 * the server reports: this server's devices (`GET /api/system/devices`) and
 * its workers (`GET /api/workers`). Pure, so the rules are tested without a
 * React tree.
 */
import type { DeviceInfo } from '../../api/rest';
import type { WorkerDevice, WorkerInfo } from '../../api/workers';

/** `target.worker` for this server, and for "wherever there is room". */
export const TARGET_LOCAL = 'local';
export const TARGET_ANY = 'any';
/** The picker's value for the one-shot pool (#790): `target.pool`, not a
 * worker. The `:` keeps it apart from every worker name. */
export const TARGET_ONE_SHOT = 'pool:one-shot';
/** The device kinds a one-shot pool run can ask for: any, since the worker
 * that will take it may not exist yet. */
export const ONE_SHOT_KINDS = ['cpu', 'cuda', 'mps'];

/** One device a target offers: a concrete id (`cpu`, `cuda:1`, `mps`). */
export interface DeviceChoice {
  value: string;
  kind: string;
  /** What the server says the device is (`NVIDIA CUDA #1 · RTX 4090`). */
  detail: string;
}

/** `cuda:1` -> `cuda`; `cpu` -> `cpu`. */
export function deviceKind(value: string): string {
  return value.split(':')[0];
}

/**
 * This server's devices as a target's device set may list them: concrete
 * ids only. The device list names a single CUDA card `cuda` (there is no
 * choice to make) and adds `cuda:N` rows only when there are several; a set
 * needs the card, so a lone `cuda` becomes `cuda:0` and a `cuda` beside
 * per-card rows is left out. `auto` names no device.
 */
export function localDeviceChoices(devices: readonly DeviceInfo[]): DeviceChoice[] {
  const perCard = devices.some((d) => d.value.startsWith('cuda:'));
  const choices: DeviceChoice[] = [];
  for (const d of devices) {
    if (d.value === 'auto' || (d.value === 'cuda' && perCard)) continue;
    const value = d.value === 'cuda' ? 'cuda:0' : d.value;
    choices.push({
      value,
      kind: deviceKind(value),
      detail: d.detail ? `${d.label} · ${d.detail}` : d.label,
    });
  }
  return choices;
}

/** A worker's devices as its agent reported them. */
export function workerDeviceChoices(worker: WorkerInfo): DeviceChoice[] {
  return worker.devices.map((d: WorkerDevice) => ({
    value: d.value,
    kind: d.kind || deviceKind(d.value),
    detail: d.name,
  }));
}

/** The workers a run can be queued on: connected ones. A disconnected
 * worker is listed by the server until it restarts, and an `offline` one
 * (persisted, #780) until it dials in; neither can take a run. */
export function connectedWorkers(workers: readonly WorkerInfo[]): WorkerInfo[] {
  return workers.filter((w) => w.status !== 'disconnected' && w.status !== 'offline');
}

/** The workers a run can name in the dialog: every worker the server lists
 * (#840: a run for one that is `offline` or `disconnected` waits until it
 * connects), except an instance of a shared credential (#845) that left,
 * which never comes back under its id. */
export function targetableWorkers(workers: readonly WorkerInfo[]): WorkerInfo[] {
  return workers.filter((w) => !(w.instance_of && w.status === 'offline'));
}

/**
 * Whether *worker* is an SSH worker that is disabled (#840). The worker's
 * own `ssh_enabled` decides when the server sends it: it comes with the
 * worker list, while the SSH list may be an older read kept after a failed
 * one. The SSH list is the fallback for an older server that sends no
 * `ssh_enabled`. The Workers panel and the run target list both read it
 * from here.
 */
export function sshDisabled(worker: WorkerInfo, ssh?: { enabled: boolean }): boolean {
  if (worker.ssh_enabled != null) return !worker.ssh_enabled;
  return ssh !== undefined && !ssh.enabled;
}

/** Whether *worker* is started over ssh by this server (#784). */
export function sshManaged(worker: WorkerInfo, ssh?: { enabled: boolean }): boolean {
  if (worker.ssh_enabled !== undefined) return worker.ssh_enabled !== null;
  return ssh !== undefined;
}

/** The status to show for *worker*: `disabled` for a disabled SSH worker,
 * whatever its connection reads, else the server's. */
export function shownStatus(worker: WorkerInfo, ssh?: { enabled: boolean }): string {
  return sshDisabled(worker, ssh) ? 'disabled' : worker.status;
}

/** Whether a run cannot be queued on *worker* now: it does not take new
 * runs (drained, or a disabled SSH worker, which is drained too). */
export function refusesRuns(worker: WorkerInfo): boolean {
  return worker.draining || worker.status === 'draining' || sshDisabled(worker);
}

const ACCELERATORS = ['cuda', 'mps'];

/**
 * The device set a target starts with: the graph's device resolved against
 * the target's devices, else the first accelerator, else the first device.
 * A concrete device the target has is itself; a bare kind (`cuda`), or a
 * card the target lacks (`cuda:3`), is the target's first device of that
 * kind; `auto` is the first accelerator.
 */
export function defaultDevices(choices: readonly DeviceChoice[], graphDevice: string): string[] {
  const wanted = graphDevice.trim().toLowerCase();
  const exact = choices.find((c) => c.value === wanted || (wanted === 'mps' && c.value === 'mps:0'));
  if (exact) return [exact.value];
  if (wanted !== 'auto') {
    // `cuda:3` on a target without it still means a CUDA card.
    const sameKind = choices.find((c) => c.kind === deviceKind(wanted));
    if (sameKind) return [sameKind.value];
  }
  const accelerator = choices.find((c) => ACCELERATORS.includes(c.kind));
  const first = accelerator ?? choices[0];
  return first ? [first.value] : [];
}

/**
 * Add *value* to a device set, or take it out. A set is one kind (the server
 * refuses `cuda:0` with `cpu`), so a device of another kind replaces it.
 * The result is in *order* (the target's devices as it reported them), so
 * the set, its first device (the run's device) and every card's in-run name
 * (#774) depend on which cards are in it, never on the order of the clicks.
 */
export function toggleDevice(selected: readonly string[], value: string, order: readonly string[]): string[] {
  if (selected.includes(value)) return selected.filter((v) => v !== value);
  const kind = deviceKind(value);
  const next = selected.every((v) => deviceKind(v) === kind) ? [...selected, value] : [value];
  return next.sort((a, b) => order.indexOf(a) - order.indexOf(b));
}

/**
 * The device kinds an `any` run can ask for: CPU always (this server has
 * one), then each accelerator kind this server or a connected worker has.
 */
export function anyKinds(local: readonly DeviceChoice[], workers: readonly WorkerInfo[]): string[] {
  const offered = new Set<string>([
    ...local.map((c) => c.kind),
    // `any` never places a run on a one-shot worker (#790).
    ...connectedWorkers(workers)
      .filter((w) => !w.one_shot)
      .flatMap((w) => workerDeviceChoices(w).map((c) => c.kind)),
  ]);
  return ['cpu', ...ACCELERATORS.filter((kind) => offered.has(kind))];
}

/** A device shared by memory on a worker whose agent enforces budgets (#788). */
function sharedByMemory(worker: WorkerInfo, device: WorkerDevice): boolean {
  return device.policy === 'shared' && worker.memory_budget === true;
}

/**
 * The devices shared by memory (#788) a run sent where the picker points
 * could land on: the chosen worker's chosen devices; for `any`, the devices
 * of the chosen kind on connected workers that are not one-shot; for the
 * one-shot pool (#790), those on connected one-shot workers.
 */
function sharedCandidates(
  target: string,
  worker: WorkerInfo | null,
  selected: readonly string[],
  online: readonly WorkerInfo[],
  kind: string,
): WorkerDevice[] {
  if (target === TARGET_ANY || target === TARGET_ONE_SHOT) {
    const pool = target === TARGET_ONE_SHOT;
    return online
      .filter((w) => Boolean(w.one_shot) === pool)
      .flatMap((w) => w.devices.filter((d) => deviceKind(d.value) === kind && sharedByMemory(w, d)));
  }
  if (worker === null) return [];
  return worker.devices.filter((d) => selected.includes(d.value) && sharedByMemory(worker, d));
}

/**
 * Whether a run sent where the picker points could land on a device shared
 * by memory (#788), so its memory budget matters.
 */
export function usesSharedDevice(
  target: string,
  worker: WorkerInfo | null,
  selected: readonly string[],
  online: readonly WorkerInfo[],
  kind: string,
): boolean {
  return sharedCandidates(target, worker, selected, online, kind).length > 0;
}

/** A device's memory total as the server last reported it, or null. */
function memoryTotal(device: WorkerDevice): number | null {
  return device.admission?.total_bytes ?? device.memory?.total_bytes ?? null;
}

/**
 * The shared MPS devices the run could land on when none of the devices it
 * could land on reports a memory total: an MPS card reports none, so
 * admission by budget can never judge a budget against it and the server
 * refuses such a run whatever its budget; the dialog says so instead of
 * offering the field. Empty when one of them reports a total, or when one
 * is not MPS ({@link sharedAwaitingReport}).
 */
export function sharedWithoutTotal(
  target: string,
  worker: WorkerInfo | null,
  selected: readonly string[],
  online: readonly WorkerInfo[],
  kind: string,
): string[] {
  const devices = sharedCandidates(target, worker, selected, online, kind);
  if (devices.some((d) => memoryTotal(d) !== null)) return [];
  return devices.length > 0 && devices.every((d) => deviceKind(d.value) === 'mps') ? devices.map((d) => d.value) : [];
}

/**
 * The shared devices other than MPS the run could land on that have not
 * reported a memory total yet (a CUDA card before its worker's first
 * memory heartbeat), when none of the candidates has: the field stays, and
 * the dialog says the budget is judged once the report arrives.
 */
export function sharedAwaitingReport(
  target: string,
  worker: WorkerInfo | null,
  selected: readonly string[],
  online: readonly WorkerInfo[],
  kind: string,
): string[] {
  const devices = sharedCandidates(target, worker, selected, online, kind);
  if (devices.some((d) => memoryTotal(d) !== null)) return [];
  return devices.filter((d) => deviceKind(d.value) !== 'mps').map((d) => d.value);
}

/**
 * A memory budget field's text as `target.memory_gib` (#788): `null` for a
 * blank field (the server's default applies), `undefined` for text that is
 * not a number above 0 and at most {@link MAX_MEMORY_GIB}.
 */
export const MAX_MEMORY_GIB = 4096;

export function parseMemoryGib(raw: string): number | null | undefined {
  const text = raw.trim();
  if (text === '') return null;
  if (!/^\d*\.?\d+$/.test(text)) return undefined;
  const value = Number(text);
  return value > 0 && value <= MAX_MEMORY_GIB ? value : undefined;
}

/**
 * What a run's memory history is kept by (#788): its `device` and how many
 * devices it uses (`target.devices`), as `POST /api/runs/estimate` takes
 * them. A worker's chosen set; one device of the kind for `any` and the
 * one-shot pool.
 */
export function estimateOptions(
  target: string,
  selected: readonly string[],
  kind: string,
): { device: string; target: { devices: string[] | null } } {
  if (target === TARGET_ANY || target === TARGET_ONE_SHOT || selected.length === 0) {
    return { device: kind, target: { devices: null } };
  }
  return { device: selected[0], target: { devices: [...selected] } };
}

/** The kind an `any` run starts with: the graph device's kind when it is
 * offered, else the first accelerator, else CPU. */
export function defaultKind(kinds: readonly string[], graphDevice: string): string {
  const wanted = deviceKind(graphDevice.trim().toLowerCase());
  if (kinds.includes(wanted)) return wanted;
  return kinds.find((kind) => ACCELERATORS.includes(kind)) ?? 'cpu';
}
