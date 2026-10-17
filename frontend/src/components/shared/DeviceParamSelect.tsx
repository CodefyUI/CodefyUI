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
 *  - the cards of the workers this server knows, by the name a run on that
 *    worker uses (`cuda:1`), so a node can be pinned to a worker's card from
 *    a control plane without one (U1).
 */

/** Whether *param* is a node `device` select (a device vocabulary). */
export function isDeviceParam(param: ParamDefinition): boolean {
  return param.param_type === 'select' && param.name === 'device' && param.options.includes('cpu');
}

/** `mps` and `mps:0` are one device; everything else compares as spelled. */
function sameDevice(a: string, b: string): boolean {
  const norm = (v: string) => (v === 'mps:0' ? 'mps' : v);
  return norm(a) === norm(b);
}

/** Accelerator values the workers offer that the local list lacks, with the
 * workers offering each, in first-seen order. */
export function workerDeviceChoices(
  workers: readonly WorkerInfo[],
  local: readonly string[],
): { value: string; workers: string[] }[] {
  const found = new Map<string, string[]>();
  for (const worker of workers) {
    for (const device of worker.devices) {
      if (device.kind === 'cpu' || device.value === 'cpu') continue;
      if (local.some((v) => sameDevice(v, device.value))) continue;
      const key = [...found.keys()].find((v) => sameDevice(v, device.value)) ?? device.value;
      const names = found.get(key) ?? [];
      if (!names.includes(worker.name)) names.push(worker.name);
      found.set(key, names);
    }
  }
  return [...found.entries()].map(([value, names]) => ({ value, workers: names }));
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
  const workers = useWorkerStore((s) => s.workers);
  const loaded = useWorkerStore((s) => s.loaded);

  // One read when nothing has asked for the list yet; the Workers panel's
  // poll keeps it fresh while that is open.
  useEffect(() => {
    if (!useWorkerStore.getState().loaded) void useWorkerStore.getState().refresh();
  }, []);

  const selected = value ?? param.default;
  const current = selected == null ? '' : String(selected);
  const local = param.options;
  const remote = loaded ? workerDeviceChoices(workers, local) : [];
  // The option that names the stored device, spelled as the option is
  // (`mps` for a stored `mps:0`), so the select shows it; the stored value
  // is not rewritten until someone picks another.
  const localMatch = local.find((v) => sameDevice(v, current));
  const remoteMatch = remote.find((choice) => sameDevice(choice.value, current))?.value;
  const onLocal = current === '' || localMatch !== undefined;
  const notListed = !onLocal && remoteMatch === undefined;
  const shown = localMatch ?? remoteMatch ?? current;

  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <select
        value={shown}
        onChange={(e) => onChange(param.name, e.target.value)}
        className={`${styles.input} ${styles.select}`}
        aria-describedby={onLocal ? undefined : hintId}
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
        {remote.length > 0 && (
          <optgroup label={t('paramField.device.workerGroup')}>
            {remote.map((choice) => (
              <option
                key={choice.value}
                value={choice.value}
                style={{ background: 'var(--surface-input)' }}
              >
                {t('paramField.device.onWorkers', {
                  device: choice.value,
                  workers: choice.workers.join(', '),
                })}
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
    </div>
  );
}
