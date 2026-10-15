/**
 * The choices the "Queue on..." dialog (#775) offers, worked out from what
 * the server reports: this server's devices (`GET /api/system/devices`) and
 * its workers (`GET /api/workers`). Pure, so the rules are tested without a
 * React tree.
 */
import type { DeviceInfo, WorkerDevice, WorkerInfo } from '../../api/rest';

/** `target.worker` for this server, and for "wherever there is room". */
export const TARGET_LOCAL = 'local';
export const TARGET_ANY = 'any';

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

/** The workers a run can be queued on: connected ones (a disconnected
 * worker is listed by the server until it restarts, but cannot take one). */
export function connectedWorkers(workers: readonly WorkerInfo[]): WorkerInfo[] {
  return workers.filter((w) => w.status !== 'disconnected');
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
    ...connectedWorkers(workers).flatMap((w) => workerDeviceChoices(w).map((c) => c.kind)),
  ]);
  return ['cpu', ...ACCELERATORS.filter((kind) => offered.has(kind))];
}

/** The kind an `any` run starts with: the graph device's kind when it is
 * offered, else the first accelerator, else CPU. */
export function defaultKind(kinds: readonly string[], graphDevice: string): string {
  const wanted = deviceKind(graphDevice.trim().toLowerCase());
  if (kinds.includes(wanted)) return wanted;
  return kinds.find((kind) => ACCELERATORS.includes(kind)) ?? 'cpu';
}
