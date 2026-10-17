import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { estimateMemory, submitRun, type MemoryEstimate } from '../../api/rest';
import { useI18n } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { useSweepStore } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { isRunSeed } from '../../utils/graphSettings';
import { useToastStore } from '../../store/toastStore';
import { formatGiB } from '../../utils/formatGiB';
import { MAX_RUN_NAME_LENGTH, storableRunName, tabDefaultName } from '../../utils/runName';
import { TARGET_ANY, estimateOptions, parseMemoryGib, usesSharedDevice } from './queueTargets';
import { TargetSelector, useRunTargetChoice } from './TargetSelector';
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
  const globalDevice = useUIStore((state) => state.globalDevice);
  // Mounted by the Toolbar, which exists only while a tab is active.
  const tab = useTabStore((state) => state.tabs.find((candidate) => candidate.id === state.activeTabId)!);
  const graphDevice = tab.graphDevice ?? globalDevice;

  const [name, setName] = useState('');
  const [seed, setSeed] = useState(() => (tab.seed == null ? '' : String(tab.seed)));
  const [deterministic, setDeterministic] = useState(tab.deterministic);
  const [recordOutputs, setRecordOutputs] = useState(tab.recordOutputs);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const choice = useRunTargetChoice(graphDevice);
  const isWorker = choice.isWorker;

  // Memory admission (#788): a budget matters only where the run may land
  // on a device shared by memory. The suggestion is looked up when that is
  // so, again whenever the device kind or count changes (the history is
  // kept per both), and fills the field until the user types in it.
  const shared = usesSharedDevice(choice.target, choice.worker, choice.selected, choice.online, choice.kind);
  const [memoryText, setMemoryText] = useState<string | null>(null);
  const [wait, setWait] = useState(false);
  const [estimate, setEstimate] = useState<MemoryEstimate | 'loading' | null>(null);
  const [estimateError, setEstimateError] = useState<string | null>(null);
  const shape = estimateOptions(choice.target, choice.selected, choice.kind);
  const shapeKey = `${shape.device.split(':')[0]} x${shape.target.devices?.length ?? 1}`;
  const shapeJson = JSON.stringify(shape);
  const [estimatedFor, setEstimatedFor] = useState<string | null>(null);
  // The answer for the shape asked last wins; an earlier one landing late
  // is dropped.
  const askedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!shared || estimatedFor === shapeKey) return;
    setEstimatedFor(shapeKey);
    setEstimate('loading');
    setEstimateError(null);
    askedFor.current = shapeKey;
    // Without secrets: the estimate needs what the graph computes, never a
    // session key (the server scrubs them anyway).
    const graph = useTabStore.getState().getSerializedGraph();
    estimateMemory(graph, JSON.parse(shapeJson)).then(
      (result) => {
        if (askedFor.current === shapeKey) setEstimate(result);
      },
      (err: unknown) => {
        if (askedFor.current !== shapeKey) return;
        setEstimate({ suggested_gib: null, source: null, peak_bytes: null, from_run: null });
        setEstimateError(err instanceof Error ? err.message : String(err));
      },
    );
  }, [shared, shapeKey, shapeJson, estimatedFor]);
  const suggested = estimate !== null && estimate !== 'loading' ? estimate : null;
  const memoryValue = memoryText ?? (suggested?.suggested_gib == null ? '' : String(suggested.suggested_gib));

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
    const memoryGib = shared ? parseMemoryGib(memoryValue) : null;
    if (memoryGib === undefined) {
      setError(t('queueRun.error.memory'));
      return;
    }
    // No device: the picker says so beside the devices.
    const resolved = choice.resolve();
    if (resolved === null) return;
    // The suggestion left as it is goes unsaid: the server derives the same
    // default and keeps where it came from (the run it was measured on).
    const declared = memoryText !== null ? memoryGib : null;
    const target = {
      ...resolved.target,
      ...(declared !== null ? { memory_gib: declared } : {}),
      ...(shared && wait && choice.target !== TARGET_ANY ? { wait: true } : {}),
    };
    const store = useTabStore.getState();
    // The run needs the session key in memory, as a sweep's children do:
    // RunService removes it before persisting the snapshot.
    const graph = store.getSerializedGraph({ keepSecrets: true });
    const active = store.getActiveTab();
    const options = {
      target,
      device: resolved.device,
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
            <TargetSelector choice={choice} selectRef={firstRef} />

            {shared && (
              <div className={styles.memory}>
                <label className={styles.field}>
                  {t('queueRun.memory')}
                  <input
                    inputMode="decimal"
                    value={memoryValue}
                    onChange={(event) => setMemoryText(event.target.value)}
                  />
                </label>
                <p className={styles.hint} aria-live="polite">
                  {estimate === 'loading'
                    ? t('queueRun.memory.loading')
                    : estimateError !== null
                      ? t('queueRun.memory.estimateFailed', { error: estimateError })
                      : suggested?.peak_bytes != null
                        ? t('queueRun.memory.suggested', { peak: formatGiB(suggested.peak_bytes) })
                        : t('queueRun.memory.none')}
                </p>
                <p className={styles.hint}>{t('queueRun.memory.hint')}</p>
                {choice.target !== TARGET_ANY && (
                  <label className={styles.check}>
                    <input type="checkbox" checked={wait} onChange={(event) => setWait(event.target.checked)} />
                    {t('queueRun.memory.wait')}
                  </label>
                )}
              </div>
            )}

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
