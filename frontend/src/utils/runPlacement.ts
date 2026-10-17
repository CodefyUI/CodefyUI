/**
 * Where a run executes (#769, #772, #775), read off the fields a run row
 * carries: the Runs list's rows and a sweep's variants (#777) both have
 * them, so both cells are worded by these functions.
 */
import type { useI18n } from '../i18n';
import { aliasForDevice } from './deviceAliases';

/** The placement fields of a run row or a sweep variant. */
export interface RunPlacement {
  worker_id?: string | null;
  devices?: string[] | null;
  queue_key?: string | null;
  options?: Record<string, unknown> | null;
}

type Translate = ReturnType<typeof useI18n.getState>['t'];

/** The device a run is on: what it actually got, else what it asked for. */
export function runDevice(run: RunPlacement): string {
  if (run.queue_key) return run.queue_key;
  const requested = run.options?.device;
  return typeof requested === 'string' && requested ? requested : '-';
}

/** Where a run executes (#769): `local` for this server and for a row from
 * before #769, a worker's name, or `any` while an `any` run waits. */
export function runWorker(run: RunPlacement): string {
  return run.worker_id || 'local';
}

/** {@link runWorker} as the Worker cell shows it: `local` and `any` in the
 * active language, a worker by its name. */
export function runWorkerLabel(run: RunPlacement, t: Translate): string {
  const worker = runWorker(run);
  if (worker === 'local') return t('runs.worker.local');
  return worker === 'any' ? t('runs.worker.any') : worker;
}

/**
 * The Devices column (#775): the run's device set when it named one, else
 * its device. A worker run's queue key is `<worker>/<device>`; the worker
 * has its own column, so only the device is shown here.
 */
export function runDevices(run: RunPlacement): string {
  if (run.devices && run.devices.length > 0) return run.devices.join(', ');
  const device = runDevice(run);
  return device.slice(device.lastIndexOf('/') + 1);
}

/**
 * The in-run names of a worker run's CUDA cards (#774), one line each
 * (`cuda:2 -> cuda:1 in this run`), or null when none is renamed. A run on
 * this server is not renumbered: it executes on its first device as named.
 */
export function deviceAliasLines(run: RunPlacement, t: Translate): string | null {
  if (runWorker(run) === 'local' || !run.devices) return null;
  const devices = run.devices;
  const lines = devices.flatMap((device) => {
    const alias = aliasForDevice(devices, device);
    return alias !== null && alias !== device
      ? [t('queueRun.alias', { device, alias })]
      : [];
  });
  return lines.length > 0 ? lines.join('\n') : null;
}
