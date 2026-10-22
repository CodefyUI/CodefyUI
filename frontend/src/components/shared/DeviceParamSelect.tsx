import { useEffect, useId } from 'react';
import type { ParamDefinition } from '../../types';
import type { WorkerInfo } from '../../api/workers';
import { useI18n } from '../../i18n';
import { useWorkerStore } from '../../store/workerStore';
import styles from './ParamField.module.css';

/**
 * A node's `device` SELECT (#839).
 *
 * The served options are this server's devices (`auto`, `cpu`, the backends
 * present). Two more kinds of value are real, and the plain select hid both:
 *
 *  - a value the graph already holds and this server does not list
 *    (`cuda:0` imported on a Mac). A controlled `<select>` whose value
 *    matches no option shows its first option, `auto`, which misstates the
 *    node. It is listed as stored, selectable so the browser keeps it, with
 *    a hint saying a run here falls back (and records that in its run log).
 *  - the names a node can use inside a run on a worker (U1). A node's
 *    device is a name INSIDE the run: a worker run sees the cards of its
 *    device set as `cuda:0..N-1` in set order (#774), whichever physical
 *    cards they are. So the select offers those in-run names, up to the
 *    most cards any known worker has, and never a worker's physical card
 *    (`cuda:1 · gpu-box` would promise a card the run may not get).
 */

/** Whether *param* is a node `device` select (a device vocabulary). */
export function isDeviceParam(param: ParamDefinition): boolean {
  return param.param_type === 'select' && param.name === 'device' && param.options.includes('cpu');
}

const DEVICE_SYNTAX = /^(cpu|cuda|mps)(?::(\d+))?$/;

/**
 * One spelling per device for comparing a stored value with an option, as
 * the server's `canonical_device` reads it: trimmed and lower-cased, the
 * index without leading zeros (`CUDA:01` is `cuda:1`), `mps` and `mps:0`
 * one device, and `cuda:0` the bare `cuda` when *local* lists only that (a
 * one-card server, which lists no `cuda:N`). A value that is not a device
 * spelling is compared as written, trimmed and lower-cased.
 */
export function canonical(value: string, local: readonly string[]): string {
  const probe = value.trim().toLowerCase();
  const match = DEVICE_SYNTAX.exec(probe);
  if (match === null) return probe;
  const [, kind, digits] = match;
  const index = digits === undefined ? null : Number.parseInt(digits, 10);
  if (kind === 'mps' && (index === null || index === 0)) return 'mps';
  if (kind === 'cuda' && index === 0 && local.includes('cuda') && !local.some((v) => v.startsWith('cuda:'))) {
    return 'cuda';
  }
  return index === null ? kind : `${kind}:${index}`;
}

/**
 * In-run device names the known workers make possible and the local list
 * lacks: `cuda:0..cuda:(N-1)` for the most CUDA cards any worker has, and
 * `mps` when a worker offers it. A string, so a component can select it
 * from the store and re-render only when it changes (the Workers poll
 * replaces the list every 5 s).
 */
export function inRunChoicesKey(workers: readonly WorkerInfo[], local: readonly string[]): string {
  let cards = 0;
  let mps = false;
  for (const worker of workers) {
    cards = Math.max(cards, worker.devices.filter((d) => d.kind === 'cuda').length);
    mps = mps || worker.devices.some((d) => d.kind === 'mps');
  }
  const names = Array.from({ length: cards }, (_, i) => `cuda:${i}`);
  if (mps) names.push('mps');
  const have = new Set(local.map((v) => canonical(v, local)));
  return names.filter((name) => !have.has(canonical(name, local))).join(' ');
}

/** A focused select reads the list again only when it is older than this. */
export const WORKER_LIST_MAX_AGE_MS = 5000;

/** The read in flight, shared by every select, so they never stack up. */
let inFlight: Promise<void> | null = null;
/** When a read last answered without an error; 0 before one has. */
let lastGoodRead = 0;

/**
 * Read the worker list for the device selects. A read already in flight is
 * joined; a list read successfully within *maxAgeMs* is kept (`Infinity`:
 * once one read succeeded, never again). A failed read leaves `lastGoodRead`
 * alone, so the next mount or focus tries again.
 */
export function requestWorkerList(maxAgeMs: number): Promise<void> {
  if (inFlight !== null) return inFlight;
  if (lastGoodRead > 0 && Date.now() - lastGoodRead < maxAgeMs) return Promise.resolve();
  const store = useWorkerStore.getState();
  inFlight = store
    .refresh()
    .then(() => {
      if (useWorkerStore.getState().error === null) lastGoodRead = Date.now();
    })
    .finally(() => {
      inFlight = null;
    });
  return inFlight;
}

/** Test seam: forget the last read and any read in flight. */
export function _resetWorkerListRequestForTesting(): void {
  inFlight = null;
  lastGoodRead = 0;
}

export function DeviceParamSelect({
  param,
  value,
  onChange,
  displayLabel,
}: {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  displayLabel: string;
}) {
  const { t } = useI18n();
  const hintId = useId();
  const local = param.options;
  const choicesKey = useWorkerStore((s) => (s.loaded ? inRunChoicesKey(s.workers, local) : ''));

  // Read when nothing has read the list successfully yet (a failed read is
  // retried by the next mount), and again when the select is focused and
  // the list is older than WORKER_LIST_MAX_AGE_MS, so a worker connected
  // since shows up. The Workers panel's poll writes the same store.
  useEffect(() => {
    const { loaded, error } = useWorkerStore.getState();
    if (!loaded || error !== null) void requestWorkerList(Infinity);
  }, []);

  const selected = value ?? param.default;
  const current = selected == null ? '' : String(selected);
  const inRun = choicesKey === '' ? [] : choicesKey.split(' ');
  const key = canonical(current, local);
  // The option that names the stored device, spelled as the option is
  // (`mps` for a stored `mps:0`), so the select shows it; the stored value
  // is not rewritten until someone picks another. An exact match first, so
  // `cuda:0` on a multi-card server is its own option.
  const localMatch = local.find((v) => v === current) ?? local.find((v) => canonical(v, local) === key);
  const inRunMatch = inRun.find((v) => v === current || canonical(v, local) === key);
  const onLocal = current === '' || localMatch !== undefined;
  const notListed = !onLocal && inRunMatch === undefined;
  const shown = localMatch ?? inRunMatch ?? current;
  const describedBy = [onLocal ? null : hintId, inRun.length > 0 ? `${hintId}-in-run` : null]
    .filter(Boolean)
    .join(' ');

  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <select
        value={shown}
        onChange={(e) => onChange(param.name, e.target.value)}
        className={`${styles.input} ${styles.select}`}
        aria-describedby={describedBy || undefined}
        onFocus={() => void requestWorkerList(WORKER_LIST_MAX_AGE_MS)}
      >
        {local.map((option) => (
          <option key={option} value={option} style={{ background: 'var(--surface-input)' }}>
            {option}
          </option>
        ))}
        {notListed && (
          <option value={current} style={{ background: 'var(--surface-input)' }}>
            {t('paramField.device.stored', { device: current })}
          </option>
        )}
        {inRun.length > 0 && (
          <optgroup label={t('paramField.device.inRunGroup')}>
            {inRun.map((name) => (
              <option key={name} value={name} style={{ background: 'var(--surface-input)' }}>
                {name}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      {!onLocal && (
        <span id={hintId} className={`${styles.hint} ${styles.hintWarning}`}>
          {t('paramField.device.notLocal', { device: current })}
        </span>
      )}
      {inRun.length > 0 && (
        <span id={`${hintId}-in-run`} className={styles.hint}>
          {t('paramField.device.inRunHint')}
        </span>
      )}
    </div>
  );
}
