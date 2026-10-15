import { useEffect, useId, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { listWorkers, submitRun, type RunTarget, type WorkerInfo } from '../../api/rest';
import { useDeviceOptions } from '../../hooks/useDeviceOptions';
import { useI18n } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { useSweepStore } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { aliasForDevice } from '../../utils/deviceAliases';
import { isRunSeed } from '../../utils/graphSettings';
import { useToastStore } from '../../store/toastStore';
import { MAX_RUN_NAME_LENGTH, storableRunName, tabDefaultName } from '../../utils/runName';
import {
  TARGET_ANY,
  TARGET_LOCAL,
  anyKinds,
  connectedWorkers,
  defaultDevices,
  defaultKind,
  localDeviceChoices,
  toggleDevice,
  workerDeviceChoices,
  type DeviceChoice,
} from './queueTargets';
import styles from './QueueRunDialog.module.css';

const FOCUSABLE = 'button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])';

interface QueueRunDialogProps {
  onClose: () => void;
}

/**
 * A seed field's text as the option: `null` for a blank field (an unseeded
 * run), `undefined` for text that is not a run seed (`12a`, `-1`, `1.5`, or
 * past `MAX_RUN_SEED`). Digits only: `Number` would read `1e3` or `0x10`.
 */
function parseSeed(raw: string): number | null | undefined {
  const text = raw.trim();
  if (text === '') return null;
  if (!/^\d+$/.test(text)) return undefined;
  const value = Number(text);
  return isRunSeed(value) ? value : undefined;
}

/**
 * "Queue on..." (#775): submit the active tab's graph on the queued lane
 * (`POST /api/runs`) to this server, a connected worker or whichever has
 * room, with a device set. The canvas's own Run stays interactive and local.
 */
/**
 * Show a queued run where queued runs live: the Runs tab of the Results
 * dock (expanded if collapsed), with the run open. The list refresh is
 * best effort: a failure is logged, and the next poll catches up.
 */
function showQueuedRun(runId: string): void {
  useSweepStore.getState().closeSweep();
  useUIStore.getState().requestRunsTab();
  const runs = useRunStore.getState();
  void runs.refresh()
    .catch((err: unknown) => console.warn('Runs list refresh after a queued run failed', err))
    .then(() => useRunStore.getState().select(runId))
    .catch((err: unknown) => console.warn('Opening the queued run failed', err));
}

export function QueueRunDialog({ onClose }: QueueRunDialogProps) {
  const { t } = useI18n();
  const baseId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const firstRef = useRef<HTMLSelectElement>(null);
  const { devices: localDevices } = useDeviceOptions();
  const globalDevice = useUIStore((state) => state.globalDevice);
  // Mounted by the Toolbar, which exists only while a tab is active.
  const tab = useTabStore((state) => state.tabs.find((candidate) => candidate.id === state.activeTabId)!);
  const graphDevice = tab.graphDevice ?? globalDevice;

  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const [workersError, setWorkersError] = useState<string | null>(null);
  const [target, setTarget] = useState(TARGET_LOCAL);
  // null: not chosen yet, so the target's default applies.
  const [chosenDevices, setChosenDevices] = useState<string[] | null>(null);
  const [chosenKind, setChosenKind] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [seed, setSeed] = useState(() => (tab.seed == null ? '' : String(tab.seed)));
  const [deterministic, setDeterministic] = useState(tab.deterministic);
  const [recordOutputs, setRecordOutputs] = useState(tab.recordOutputs);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The worker list is read when the dialog opens: who is connected now is
  // what a submission can name.
  useEffect(() => {
    let live = true;
    listWorkers().then(
      (listed) => { if (live) setWorkers(listed); },
      (err: unknown) => { if (live) setWorkersError(err instanceof Error ? err.message : String(err)); },
    );
    return () => { live = false; };
  }, []);

  // The open flag lives in uiStore (so the canvas shortcuts stand down);
  // however the dialog goes away, the flag goes with it.
  useEffect(() => () => useUIStore.getState().setQueueRunOpen(false), []);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    firstRef.current!.focus();
    return () => {
      if (previous.isConnected) previous.focus();
    };
  }, []);

  const online = useMemo(() => connectedWorkers(workers), [workers]);
  const worker = online.find((candidate) => candidate.id === target) ?? null;
  const isWorker = target !== TARGET_LOCAL;
  const localChoices = useMemo(() => localDeviceChoices(localDevices), [localDevices]);
  const choices: DeviceChoice[] = useMemo(
    () => (worker ? workerDeviceChoices(worker) : localChoices),
    [worker, localChoices],
  );
  const kinds = useMemo(() => anyKinds(localChoices, online), [localChoices, online]);
  const selected = chosenDevices ?? defaultDevices(choices, graphDevice);
  // A set is kept in the worker's own (physical) order, so the in-run names
  // depend on which cards are ticked, never on the order they were ticked.
  const choiceOrder = useMemo(() => choices.map((c) => c.value), [choices]);
  const kind = chosenKind ?? defaultKind(kinds, graphDevice);

  const changeTarget = (next: string) => {
    setTarget(next);
    setChosenDevices(null);
    setChosenKind(null);
    setError(null);
  };

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = Array.from(dialogRef.current!.querySelectorAll<HTMLElement>(FOCUSABLE));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const atStart = document.activeElement === first || document.activeElement === dialogRef.current;
    if (event.shiftKey && atStart) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    const seedValue = parseSeed(seed);
    if (seedValue === undefined) {
      setError(t('queueRun.error.seed'));
      return;
    }
    if (target !== TARGET_ANY && selected.length === 0) {
      setError(t('queueRun.error.noDevice'));
      return;
    }
    const store = useTabStore.getState();
    // The run needs the session key in memory, as a sweep's children do:
    // RunService removes it before persisting the snapshot.
    const graph = store.getSerializedGraph({ keepSecrets: true });
    const active = store.getActiveTab();
    const runTarget: RunTarget = target === TARGET_ANY
      ? { worker: TARGET_ANY }
      : { worker: target, devices: selected };
    const options: Record<string, unknown> & { target: RunTarget } = {
      target: runTarget,
      device: target === TARGET_ANY ? kind : selected[0],
      record_outputs: !isWorker && recordOutputs,
      verbose: active.verboseMode,
      graph_id: active.graphId,
      weights_persistent: active.weightsPersistent,
      backward_mode: active.backwardMode,
      auto_backward: active.autoBackward,
      // Sent even when blank: `seed: null` is an unseeded run, said outright.
      seed: seedValue,
      ...(deterministic ? { deterministic: true } : {}),
    };
    const runName = storableRunName(name) || tabDefaultName(active) || null;
    setSubmitting(true);
    let runId: string;
    try {
      const result = await submitRun({
        graph,
        options,
        name: runName,
      });
      runId = result.run_id;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
      return;
    }
    // Queued: nothing after this point may reopen the form, or a second
    // click would queue the graph twice.
    showQueuedRun(runId);
    useToastStore.getState().addToast(
      t('queueRun.queued', { name: runName ?? runId.slice(0, 8) }),
      'success',
      { action: { label: t('queueRun.show'), onClick: () => showQueuedRun(runId) } },
    );
    onClose();
  };

  const workerLabel = (candidate: WorkerInfo): string => {
    const devices = candidate.devices.map((d) => d.value).join(', ') || t('queueRun.noDevices');
    return t('queueRun.target.worker', {
      name: candidate.name,
      status: t(`queueRun.status.${candidate.status}`),
      devices,
    });
  };

  const deviceList = (
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
                onChange={() => setChosenKind(option)}
              />
              {t(`queueRun.kind.${option}` as 'queueRun.kind.cpu')}
            </label>
          ))}
          <p className={styles.hint}>{t('queueRun.anyHint')}</p>
        </>
      ) : choices.length === 0 ? (
        <p className={styles.muted}>{t('queueRun.noDevices')}</p>
      ) : (
        choices.map((choice) => {
          const checked = selected.includes(choice.value);
          // In-run name of a card (#774): a worker run sees its set as
          // CUDA_VISIBLE_DEVICES, renumbered from cuda:0 in the listed order.
          const alias = isWorker && checked ? aliasForDevice(selected, choice.value) : null;
          return (
            <label key={choice.value} className={styles.check}>
              <input
                type={isWorker ? 'checkbox' : 'radio'}
                name={`${baseId}-device`}
                value={choice.value}
                checked={checked}
                onChange={() => setChosenDevices(isWorker ? toggleDevice(selected, choice.value, choiceOrder) : [choice.value])}
              />
              <span className={styles.deviceId}>{choice.value}</span>
              {choice.detail && <span>{choice.detail}</span>}
              {alias !== null && (
                <span className={styles.alias}>
                  {t('queueRun.alias', { device: choice.value, alias })}
                </span>
              )}
            </label>
          );
        })
      )}
      {!isWorker && <p className={styles.hint}>{t('queueRun.localHint')}</p>}
    </fieldset>
  );

  return createPortal(
    <div className={styles.backdrop} onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${baseId}-title`}
        tabIndex={-1}
        onKeyDown={handleKeys}
      >
        {/* noValidate: a bad seed is explained in the dialog, beside the
            server's own refusals, not in a browser tooltip. */}
        <form noValidate onSubmit={(event) => void submit(event)}>
          <header className={styles.header}>
            <h2 id={`${baseId}-title`}>{t('queueRun.title')}</h2>
            <button type="button" className={styles.close} onClick={onClose} aria-label={t('queueRun.close')}>&#215;</button>
          </header>

          <div className={styles.body}>
            <label className={styles.field}>
              {t('queueRun.target')}
              <select ref={firstRef} value={target} onChange={(event) => changeTarget(event.target.value)}>
                <option value={TARGET_LOCAL}>{t('queueRun.target.local')}</option>
                {online.map((candidate) => (
                  <option key={candidate.id} value={candidate.id} disabled={candidate.status === 'draining'}>
                    {workerLabel(candidate)}
                  </option>
                ))}
                {online.length > 0 && <option value={TARGET_ANY}>{t('queueRun.target.any')}</option>}
              </select>
            </label>
            {workersError !== null && (
              <p className={styles.hint}>{t('queueRun.workersError', { error: workersError })}</p>
            )}

            {deviceList}

            <div className={styles.pair}>
              <label className={styles.field}>
                {t('queueRun.name')}
                <input
                  value={name}
                  placeholder={tabDefaultName(tab)}
                  maxLength={MAX_RUN_NAME_LENGTH}
                  onChange={(event) => setName(event.target.value)}
                />
              </label>
              <label className={styles.field}>
                {t('queueRun.seed')}
                <input
                  inputMode="numeric"
                  value={seed}
                  placeholder={t('queueRun.seedNone')}
                  onChange={(event) => setSeed(event.target.value)}
                />
              </label>
            </div>
            <label className={styles.check}>
              <input type="checkbox" checked={deterministic} onChange={(event) => setDeterministic(event.target.checked)} />
              {t('queueRun.deterministic')}
            </label>
            <label className={styles.check}>
              <input
                type="checkbox"
                checked={!isWorker && recordOutputs}
                disabled={isWorker}
                onChange={(event) => setRecordOutputs(event.target.checked)}
              />
              {t('queueRun.recordOutputs')}
            </label>
            {isWorker && <p className={styles.hint}>{t('queueRun.recordOutputsWorker')}</p>}
            {error !== null && <div className={styles.error} role="alert">{error}</div>}
          </div>

          <footer className={styles.footer}>
            <button type="button" onClick={onClose}>{t('dialog.cancel')}</button>
            <button type="submit" className={styles.primary} disabled={submitting}>
              {submitting ? t('queueRun.submitting') : t('queueRun.submit')}
            </button>
          </footer>
        </form>
      </div>
    </div>,
    document.body,
  );
}
