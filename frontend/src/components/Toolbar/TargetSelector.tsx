import { useEffect, useId, useMemo, useState, type Ref } from 'react';
import type { RunTarget } from '../../api/rest';
import { fetchWorkers, type WorkerInfo } from '../../api/workers';
import { useDeviceOptions } from '../../hooks/useDeviceOptions';
import { useI18n, type TranslationKey } from '../../i18n';
import { aliasForDevice } from '../../utils/deviceAliases';
import {
  TARGET_ANY,
  TARGET_LOCAL,
  anyKinds,
  connectedWorkers,
  defaultDevices,
  targetableWorkers,
  defaultKind,
  deviceKind,
  localDeviceChoices,
  refusesRuns,
  toggleDevice,
  workerDeviceChoices,
  type DeviceChoice,
} from './queueTargets';
import styles from './TargetSelector.module.css';

/** Where a submission goes, as `options.target` and `options.device` say it. */
export interface ResolvedTarget {
  target: RunTarget;
  device: string;
}

/** The state behind a {@link TargetSelector}: who can take the run, and
 * what the user picked. One hook so every dialog that queues work (#775
 * Queue on..., #777 New sweep) offers the same choices by the same rules. */
export interface RunTargetChoice {
  workers: WorkerInfo[];
  /** False until the worker list answered (or failed). */
  workersLoaded: boolean;
  workersError: string | null;
  online: WorkerInfo[];
  /** The workers a run can name: the connected ones and the enrolled
   * offline ones, for which it waits (#780). */
  listed: WorkerInfo[];
  /** The chosen worker, when the target is one. */
  worker: WorkerInfo | null;
  target: string;
  isWorker: boolean;
  choices: DeviceChoice[];
  kinds: string[];
  selected: string[];
  kind: string;
  /**
   * Whether the choice differs from the one the picker opened with (This
   * server, the graph's device). Computed, so picking the first device
   * again is not a change.
   */
  touched: boolean;
  /** Set when {@link resolve} found no device chosen; any change clears it. */
  deviceError: boolean;
  changeTarget: (next: string) => void;
  toggle: (value: string) => void;
  chooseKind: (kind: string) => void;
  /** The submission's target and device; null (and `deviceError`) when no
   * device is chosen. */
  resolve: () => ResolvedTarget | null;
}

export interface RunTargetOptions {
  /** Called when the target changes (not the devices). */
  onTargetChange?: () => void;
  /**
   * List the graph's own device on This server when it is not one of the
   * concrete devices there (`auto`, a bare `cuda`, a card this server
   * lacks), and start on it, so the untouched picker shows the device the
   * caller sends without a target (#777).
   */
  keepGraphDevice?: boolean;
  /**
   * What a worker's default device is resolved against, when it differs
   * from *graphDevice* (#840): the Settings device describes this server,
   * not a worker, so a caller following it passes `auto` here and a GPU
   * worker starts on its accelerator.
   */
  workerDevice?: string;
}

/**
 * The target picker's state. The worker list is read once, when the dialog
 * opens: who is connected now is what a submission can name. *graphDevice*
 * is what the defaults are resolved against.
 */
export function useRunTargetChoice(graphDevice: string, options: RunTargetOptions = {}): RunTargetChoice {
  const { onTargetChange, keepGraphDevice = false, workerDevice = graphDevice } = options;
  const { t } = useI18n();
  const { devices: localDevices } = useDeviceOptions();
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const [workersLoaded, setWorkersLoaded] = useState(false);
  const [workersError, setWorkersError] = useState<string | null>(null);
  const [target, setTarget] = useState(TARGET_LOCAL);
  // null: not chosen yet, so the target's default applies.
  const [chosenDevices, setChosenDevices] = useState<string[] | null>(null);
  const [chosenKind, setChosenKind] = useState<string | null>(null);
  const [deviceError, setDeviceError] = useState(false);

  useEffect(() => {
    let live = true;
    fetchWorkers().then(
      (listed) => {
        if (!live) return;
        setWorkers(listed);
        setWorkersLoaded(true);
      },
      (err: unknown) => {
        if (!live) return;
        setWorkersError(err instanceof Error ? err.message : String(err));
        setWorkersLoaded(true);
      },
    );
    return () => { live = false; };
  }, []);

  const online = useMemo(() => connectedWorkers(workers), [workers]);
  const listed = useMemo(() => targetableWorkers(workers), [workers]);
  const worker = listed.find((candidate) => candidate.id === target) ?? null;
  const isWorker = target !== TARGET_LOCAL;
  const localChoices = useMemo(() => localDeviceChoices(localDevices), [localDevices]);
  const herePicks = useMemo(() => {
    if (!keepGraphDevice || localChoices.some((c) => c.value === graphDevice)) return localChoices;
    const detail = graphDevice === 'auto' ? t('queueRun.serverPicks') : t('queueRun.tabDevice');
    return [{ value: graphDevice, kind: deviceKind(graphDevice), detail }, ...localChoices];
  }, [keepGraphDevice, localChoices, graphDevice, t]);
  const initialHere = useMemo(() => defaultDevices(herePicks, graphDevice), [herePicks, graphDevice]);
  const choices: DeviceChoice[] = useMemo(
    () => (worker ? workerDeviceChoices(worker) : herePicks),
    [worker, herePicks],
  );
  const kinds = useMemo(() => anyKinds(localChoices, online), [localChoices, online]);
  const selected = chosenDevices ?? defaultDevices(choices, worker ? workerDevice : graphDevice);
  // A set is kept in the worker's own (physical) order, so the in-run names
  // depend on which cards are ticked, never on the order they were ticked.
  const choiceOrder = useMemo(() => choices.map((c) => c.value), [choices]);
  const kind = chosenKind ?? defaultKind(kinds, graphDevice);

  return {
    workers,
    workersLoaded,
    workersError,
    online,
    listed,
    worker,
    target,
    isWorker,
    choices,
    kinds,
    selected,
    kind,
    touched: target !== TARGET_LOCAL || selected.join() !== initialHere.join(),
    deviceError,
    changeTarget: (next) => {
      setTarget(next);
      setChosenDevices(null);
      setChosenKind(null);
      setDeviceError(false);
      onTargetChange?.();
    },
    toggle: (value) => {
      setChosenDevices(isWorker ? toggleDevice(selected, value, choiceOrder) : [value]);
      setDeviceError(false);
    },
    chooseKind: setChosenKind,
    resolve: () => {
      if (target === TARGET_ANY) return { target: { worker: TARGET_ANY }, device: kind };
      if (selected.length === 0) {
        setDeviceError(true);
        return null;
      }
      return { target: { worker: target, devices: selected }, device: selected[0] };
    },
  };
}

interface TargetSelectorProps {
  choice: RunTargetChoice;
  /** The target `<select>`, for a dialog that focuses it first. */
  selectRef?: Ref<HTMLSelectElement>;
  /** The hints under the device list; the run dialog's by default. */
  anyHint?: TranslationKey;
  localHint?: TranslationKey;
  /** Open the Workers panel; offered beside the no-worker hint (#840). */
  onOpenWorkers?: () => void;
}

/** The target `<select>` and the device list under it. */
export function TargetSelector({
  choice,
  selectRef,
  anyHint = 'queueRun.anyHint',
  localHint = 'queueRun.localHint',
  onOpenWorkers,
}: TargetSelectorProps) {
  const { t } = useI18n();
  const baseId = useId();
  const { online, listed, worker, target, isWorker, choices, kinds, selected, kind } = choice;

  const workerLabel = (candidate: WorkerInfo): string => {
    const devices = candidate.devices.map((d) => d.value).join(', ') || t('queueRun.noDevices');
    return t('queueRun.target.worker', {
      name: candidate.name,
      status: t(`workers.status.${candidate.status}`),
      devices,
    });
  };

  return (
    <>
      <label className={styles.field}>
        {t('queueRun.target')}
        <select ref={selectRef} value={target} onChange={(event) => choice.changeTarget(event.target.value)}>
          <option value={TARGET_LOCAL}>{t('queueRun.target.local')}</option>
          {listed.map((candidate) => (
            <option key={candidate.id} value={candidate.id} disabled={refusesRuns(candidate)}>
              {workerLabel(candidate)}
            </option>
          ))}
          {online.length > 0 && <option value={TARGET_ANY}>{t('queueRun.target.any')}</option>}
        </select>
      </label>
      {worker?.status === 'offline' && (
        <p className={styles.hint}>{t('queueRun.offlineHint', { name: worker.name })}</p>
      )}
      {worker?.status === 'disconnected' && (
        <p className={styles.hint}>{t('queueRun.disconnectedHint', { name: worker.name })}</p>
      )}
      {choice.workersLoaded && choice.workersError === null && listed.length === 0 && (
        <p className={styles.hint}>
          {t('queueRun.noWorkers')}
          {onOpenWorkers && (
            <button type="button" className={styles.linkBtn} onClick={onOpenWorkers}>
              {t('queueRun.openWorkers')}
            </button>
          )}
        </p>
      )}
      {choice.workersError !== null && (
        <p className={styles.hint}>{t('queueRun.workersError', { error: choice.workersError })}</p>
      )}

      <fieldset className={styles.devices}>
        <legend>{t('queueRun.devices')}</legend>
        {target === TARGET_ANY ? (
          <>
            {kinds.map((option) => (
              <label key={option} className={styles.check}>
                <input
                  type="radio"
                  name={`${baseId}-kind`}
                  value={option}
                  checked={kind === option}
                  onChange={() => choice.chooseKind(option)}
                />
                {t(`queueRun.kind.${option}` as 'queueRun.kind.cpu')}
              </label>
            ))}
            <p className={styles.hint}>{t(anyHint)}</p>
          </>
        ) : choices.length === 0 ? (
          <p className={styles.muted}>{t('queueRun.noDevices')}</p>
        ) : (
          choices.map((option) => {
            const checked = selected.includes(option.value);
            // In-run name of a card (#774): a worker run sees its set as
            // CUDA_VISIBLE_DEVICES, renumbered from cuda:0 in the listed order.
            const alias = isWorker && checked ? aliasForDevice(selected, option.value) : null;
            return (
              <label key={option.value} className={styles.check}>
                <input
                  type={isWorker ? 'checkbox' : 'radio'}
                  name={`${baseId}-device`}
                  value={option.value}
                  checked={checked}
                  onChange={() => choice.toggle(option.value)}
                />
                <span className={styles.deviceId}>{option.value}</span>
                {option.detail && <span>{option.detail}</span>}
                {alias !== null && (
                  <span className={styles.alias}>
                    {t('queueRun.alias', { device: option.value, alias })}
                  </span>
                )}
              </label>
            );
          })
        )}
        {!isWorker && <p className={styles.hint}>{t(localHint)}</p>}
      </fieldset>
      {choice.deviceError && (
        <div className={styles.error} role="alert">{t('queueRun.error.noDevice')}</div>
      )}
    </>
  );
}
