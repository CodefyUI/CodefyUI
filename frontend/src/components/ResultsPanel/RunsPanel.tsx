import { Suspense, useCallback, useEffect, useMemo, useRef, useState, type Ref } from 'react';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { friendlyError } from '../../utils/errorMessages';
import { formatBytes } from '../../utils/formatBytes';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import {
  isActiveRun,
  seriesNames,
  toChartSeries,
  useRunStore,
  type RunLogLine,
  type RunStatusFilter,
} from '../../store/runStore';
import { useStudyStore } from '../../store/studyStore';
import { useSweepStore } from '../../store/sweepStore';
import { useWorkerStore } from '../../store/workerStore';
import type { RunSummary } from '../../api/rest';
import { deviceAliasLines, runBudget, runDevice, runDevices, runWorkerLabel } from '../../utils/runPlacement';
import { LossChart } from './LossChart';
import { RunProvenance } from './RunProvenance';
import { SweepDetail } from './SweepDetail';
import { lazyRetry } from '../shared/lazyRetry';
import styles from './RunsPanel.module.css';

/**
 * The New sweep dialog, in a chunk of its own: it is opened from a button,
 * and with the target picker (#777) keeping it out of the first paint keeps
 * the entry chunk under the 500 kB build limit.
 */
const NewSweepDialog = lazyRetry(
  () => import('./NewSweepDialog').then((module) => module.NewSweepDialog),
  (props) => props.onClose(),
);

const FILTERS: RunStatusFilter[] = [
  'all',
  'running',
  'disconnected',
  'queued',
  'succeeded',
  'failed',
  'cancelled',
  'interrupted',
];

/**
 * Which series to print in the "final loss" column, in preference order.
 *
 * Series names are whatever a node chose to log, so this is a lookup with a
 * fallback rather than a fixed key: the built-in training loop writes
 * `train_loss`, a node pack may write `loss`, and anything else ending in
 * `loss` is still a loss.
 */
const LOSS_PREFERENCE = ['train_loss', 'loss', 'val_loss'];

function finalLoss(run: RunSummary): number | null {
  const metrics = run.final_metrics ?? {};
  for (const name of LOSS_PREFERENCE) {
    if (typeof metrics[name] === 'number') return metrics[name];
  }
  const fallback = Object.keys(metrics)
    .sort()
    .find((name) => name.toLowerCase().includes('loss'));
  return fallback === undefined ? null : metrics[fallback];
}

/**
 * Seconds as `mm:ss`, or `h:mm:ss` from an hour on: how long a run's
 * worker has been away (#783). Negative (clock skew) reads as zero.
 */
export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const pad = (n: number) => String(n).padStart(2, '0');
  const hours = Math.floor(total / 3600);
  const rest = `${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
  return hours > 0 ? `${hours}:${rest}` : rest;
}

/** "worker disconnected 00:42 ago" for a `disconnected` run, else null. */
export function disconnectedNote(
  run: RunSummary, now: number, t: ReturnType<typeof useI18n.getState>['t'],
): string | null {
  const since = run.worker_status?.disconnected_at;
  if (run.status !== 'disconnected' || typeof since !== 'number') return null;
  return t('runs.workerDisconnected', { elapsed: formatElapsed(now / 1000 - since) });
}

function formatClock(iso: string | null): string {
  if (!iso) return '-';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '-';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/**
 * Start time for the table: a clock for today, a date for anything older.
 *
 * Runs are retained across restarts, so a column of bare clock times reads
 * as "all of these ran this morning" when half of them are from last week.
 */
export function formatStarted(iso: string | null, now: number = Date.now()): string {
  if (!iso) return '-';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '-';
  const pad = (n: number) => String(n).padStart(2, '0');
  const today = new Date(now);
  const sameDay = date.getFullYear() === today.getFullYear()
    && date.getMonth() === today.getMonth()
    && date.getDate() === today.getDate();
  return sameDay
    ? `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
    : `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/**
 * Wall time a run has occupied, as `1m 04s`.
 *
 * A run still going is measured against NOW, so the column ticks up on
 * every poll instead of freezing at the moment it started.
 */
export function formatDuration(
  startedAt: string | null,
  finishedAt: string | null,
  now: number = Date.now(),
): string {
  if (!startedAt) return '-';
  const start = new Date(startedAt).getTime();
  const end = finishedAt ? new Date(finishedAt).getTime() : now;
  if (Number.isNaN(start) || Number.isNaN(end)) return '-';
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

// The placement cells (#769, #775) are worded by utils/runPlacement, which
// the sweep view (#777) shares; re-exported for this panel's callers.
export { deviceAliasLines, runDevice, runDevices, runWorker } from '../../utils/runPlacement';

/**
 * The device the run ASKED for, when the scheduler gave it a different
 * one (`auto` became `mps:0`, `cuda` on a box without CUDA became `cpu`).
 * `null` when there is nothing to explain: not started yet, or the same
 * string, or no run at all.
 */
export function requestedDeviceNote(run: RunSummary | null): string | null {
  if (!run?.queue_key) return null;
  const requested = run.options?.device;
  if (typeof requested !== 'string' || !requested) return null;
  return requested === run.queue_key ? null : requested;
}

/**
 * One device of a run's recorded memory (#770). `peak` is a CUDA card's
 * high-water mark over the run; `held` is what the MPS driver still held
 * when the run ended (cache included), which is not a peak.
 */
export interface DeviceMemoryEntry {
  device: string;
  kind: 'peak' | 'held';
  bytes: number;
  reserved: number | null;
  exact: boolean;
  shared: boolean;
}

/**
 * The run's recorded device memory, one entry per device, or `null` when
 * there is nothing to show: a CPU run (`peak_memory: null`), a run that has
 * not finished, or one from before the server recorded it. Entries the
 * server could not have written are skipped rather than shown as zero.
 */
export function deviceMemoryEntries(run: RunSummary | null): DeviceMemoryEntry[] | null {
  const recorded = run?.execution_facts?.peak_memory;
  if (!recorded || typeof recorded !== 'object') return null;
  const entries: DeviceMemoryEntry[] = [];
  for (const [device, raw] of Object.entries(recorded as Record<string, unknown>)) {
    if (!raw || typeof raw !== 'object') continue;
    const fields = raw as Record<string, unknown>;
    const exact = fields.exact === true;
    if (typeof fields.allocated_bytes === 'number') {
      entries.push({
        device,
        kind: 'peak',
        bytes: fields.allocated_bytes,
        reserved: typeof fields.reserved_bytes === 'number' ? fields.reserved_bytes : null,
        exact,
        shared: fields.shared === true,
      });
    } else if (typeof fields.held_bytes === 'number') {
      entries.push({
        device, kind: 'held', bytes: fields.held_bytes, reserved: null, exact, shared: false,
      });
    }
  }
  return entries.length ? entries : null;
}

/** One hover line per device, for the header chip's title. */
function deviceMemoryLine(entry: DeviceMemoryEntry, t: ReturnType<typeof useI18n.getState>['t']): string {
  if (entry.kind === 'held') {
    return t('runs.detail.memoryHeldDevice', {
      device: entry.device, held: formatBytes(entry.bytes),
    });
  }
  const line = entry.reserved === null
    ? t('runs.detail.peakMemoryDevice', {
        device: entry.device, allocated: formatBytes(entry.bytes),
      })
    : t('runs.detail.peakMemoryDeviceReserved', {
        device: entry.device,
        allocated: formatBytes(entry.bytes),
        reserved: formatBytes(entry.reserved),
      });
  return entry.shared ? `${line} ${t('runs.detail.peakMemoryShared')}` : line;
}

/** How many hex characters of an environment digest the header shows. */
export const DIGEST_SHORT_LENGTH = 12;

function StatusChip({ run }: { run: RunSummary }) {
  const { t } = useI18n();
  return (
    <span className={`${styles.chip} ${styles[`chip_${run.status}`]}`}>
      {t(`runs.status.${run.status}`)}
    </span>
  );
}

/**
 * `execution_stopped` carries `cancelled` / `interrupted`, which are also
 * status names — so the log line and the chip cannot word the same fact two
 * different ways. An unrecognised reason falls through to the raw string
 * rather than being dropped.
 */
const STOP_REASON_KEY = {
  cancelled: 'runs.status.cancelled',
  interrupted: 'runs.status.interrupted',
} as const;

/** The engine's node statuses, as they arrive on `node_status` events. */
const NODE_STATUS_KEY = {
  running: 'runs.nodeStatus.running',
  completed: 'runs.nodeStatus.completed',
  cached: 'runs.nodeStatus.cached',
  skipped: 'runs.nodeStatus.skipped',
  error: 'runs.nodeStatus.error',
  bypassed: 'runs.nodeStatus.bypassed',
  unselected: 'runs.nodeStatus.unselected',
} as const;

function LogLine({ line }: { line: RunLogLine }) {
  const { t } = useI18n();
  const text = (() => {
    switch (line.kind) {
      case 'started':
        return t('runs.log.started');
      case 'completed':
        return t('runs.log.completed');
      case 'failed':
        return t('runs.log.failed', { error: line.detail ?? '' });
      case 'stopped': {
        const reasonKey =
          STOP_REASON_KEY[line.detail as keyof typeof STOP_REASON_KEY];
        return t('runs.log.stopped', {
          reason: reasonKey ? t(reasonKey) : (line.detail ?? ''),
        });
      }
      case 'warning':
        return t('runs.log.warning', { detail: line.detail ?? '' });
      default: {
        // The backend's node statuses are a small closed vocabulary; an
        // unknown one from a newer backend falls through as its raw token
        // rather than being dropped or rendered as a missing key.
        const key = NODE_STATUS_KEY[line.status as keyof typeof NODE_STATUS_KEY];
        return (
          t('runs.log.node', {
            node: (line.nodeId ?? '').slice(0, 8),
            status: key ? t(key) : (line.status ?? ''),
          }) + (line.detail ? `: ${friendlyError(line.detail)}` : '')
        );
      }
    }
  })();
  return (
    <div className={`${styles.logLine} ${styles[`tone_${line.tone}`]}`}>
      <span className={styles.logTime}>{formatClock(line.ts)}</span>
      <span className={styles.logText}>{text}</span>
    </div>
  );
}

function RunDetailView({ chartHeight, parentRef, closeRef }: {
  chartHeight: number;
  /** Where focus goes when a sweep row opens this run (see RunsPanel). */
  parentRef: Ref<HTMLButtonElement>;
  closeRef: Ref<HTMLButtonElement>;
}) {
  const { t } = useI18n();
  const detail = useRunStore((s) => s.detail);
  const select = useRunStore((s) => s.select);
  const exportCsv = useRunStore((s) => s.exportCsv);
  const busy = useRunStore((s) => s.busy);
  // The LIST row first (a poll keeps it fresh), then the detail's own copy.
  // A filter change or a status transition can drop the run out of `runs`
  // while its detail is open, and a header that blanks out then would look
  // like the run had been deleted.
  const listRow = useRunStore((s) =>
    s.runs.find((candidate) => candidate.id === s.selectedRunId) ?? null,
  );
  const run = listRow ?? detail?.row ?? null;

  const chart = useMemo(() => {
    if (!detail) return [];
    return toChartSeries(detail.series, seriesNames(detail.series));
  }, [detail]);

  const copyPath = useCallback(async (path: string) => {
    try {
      await navigator.clipboard.writeText(path);
      useToastStore.getState().addToast(t('runs.detail.copied'), 'success');
    } catch {
      useToastStore.getState().addToast(t('runs.detail.copyFailed'), 'error');
    }
  }, [t]);

  if (!detail) return null;

  const seed = run?.options?.seed;
  // Shown next to the seed because the two together are what "reproducible"
  // means: the seed fixes what was drawn, this fixes how it was combined.
  const deterministic = run?.options?.deterministic === true;
  // #344: the environment that executed the run. Short in the header, the
  // whole digest on hover -- it is what a reader compares across runs.
  const digest = run?.fingerprint_digest ?? null;
  const requestedDevice = requestedDeviceNote(run);
  // #770: the header shows the largest CUDA peak ("Peak memory"), or, for
  // an MPS run, what the driver held at the end -- a different quantity,
  // so a different label. The hover names every device and says when the
  // number is approximate.
  const memoryEntries = deviceMemoryEntries(run);
  const peaks = memoryEntries?.filter((entry) => entry.kind === 'peak') ?? [];
  const memoryLabel = !memoryEntries
    ? ''
    : peaks.length
      ? t('runs.detail.peakMemory', {
          amount: formatBytes(Math.max(...peaks.map((entry) => entry.bytes))),
        })
      : t('runs.detail.memoryHeld', {
          amount: formatBytes(Math.max(...memoryEntries.map((entry) => entry.bytes))),
        });
  const memoryTitle = memoryEntries
    ? [
        ...memoryEntries.map((entry) => deviceMemoryLine(entry, t)),
        ...(memoryEntries.every((entry) => entry.exact)
          ? []
          : [t('runs.detail.peakMemoryApproximate')]),
      ].join('\n')
    : '';

  return (
    <div className={styles.detail} data-testid="run-detail">
      <div className={styles.detailHeader}>
        <span className={styles.detailTitle}>
          {run?.name || t('runs.unnamed')}
        </span>
        {run && <StatusChip run={run} />}
        {run && (
          <span
            className={styles.detailMeta}
            title={
              requestedDevice
                ? t('runs.detail.requestedDevice', { device: requestedDevice })
                : undefined
            }
          >
            {runDevice(run)}
          </span>
        )}
        {typeof seed === 'number' && (
          <span className={styles.detailMeta}>
            {t('runs.detail.seed')} {seed}
          </span>
        )}
        {deterministic && (
          <span className={styles.detailMeta}>
            {t('runs.detail.deterministic')}
          </span>
        )}
        {digest && (
          <span
            className={styles.detailMeta}
            data-testid="run-env-digest"
            title={t('runs.detail.environmentTitle', { digest })}
          >
            {t('runs.detail.environment')} {digest.slice(0, DIGEST_SHORT_LENGTH)}
          </span>
        )}
        {memoryEntries && (
          <span
            className={styles.detailMeta}
            data-testid="run-peak-memory"
            title={memoryTitle}
          >
            {memoryLabel}
          </span>
        )}
        <button
          ref={closeRef}
          type="button"
          className={styles.detailClose}
          onClick={() => void select(null)}
          aria-label={t('runs.detail.close')}
          title={t('runs.detail.close')}
        >
          ×
        </button>
      </div>

      {/* A line of its own: in the header, beside the seed and Deterministic,
          it squeezed the run's name to nothing on an 860 px window. */}
      {run?.sweep_id && (
        <div className={styles.detailParent}>
          <button
            ref={parentRef}
            type="button"
            className={styles.rowBtn}
            onClick={() => {
              void select(null);
              void useSweepStore.getState().openSweep(run.sweep_id!);
            }}
          >
            {t('sweeps.detail.openParent')}
          </button>
        </div>
      )}

      {detail.error && (
        <div className={styles.detailError}>{friendlyError(detail.error)}</div>
      )}
      {run?.error && (
        <div className={styles.detailError}>
          {t('runs.detail.error')}: {friendlyError(run.error)}
        </div>
      )}

      <div className={styles.detailSection}>
        <div className={styles.sectionHeaderRow}>
          <div className={styles.sectionHeader}>{t('runs.detail.metrics')}</div>
          {/* The same export the run row offers, put where someone reading
              the chart is actually looking — and gated exactly the same
              way, on that run's in-flight action alone.

              The CHART is deliberately not the test. `runStore` drops every
              non-finite reading before it becomes a series, while the server
              still writes a row for it (an empty value cell), so a diverged
              run has a complete CSV and a blank curve. A `/metrics` request
              that failed reads the same way. Either would grey this button
              out while the SAME run's row button downloads the file. */}
          <button
            type="button"
            className={styles.rowBtn}
            data-testid="detail-csv"
            disabled={busy[detail.runId] === true}
            onClick={() => void exportCsv(detail.runId)}
            title={t('runs.action.csvTitle')}
          >
            {t('runs.detail.downloadCsv')}
          </button>
        </div>
        {chart.length === 0 ? (
          <div className={styles.detailEmpty}>
            {detail.loading ? t('runs.loading') : t('runs.detail.noMetrics')}
          </div>
        ) : (
          <LossChart series={chart} height={chartHeight} xLabel={t('runs.detail.step')} />
        )}
      </div>

      <div className={styles.detailSection}>
        <div className={styles.sectionHeader}>{t('runs.detail.artifacts')}</div>
        {detail.artifacts.length === 0 ? (
          <div className={styles.detailEmpty}>{t('runs.detail.noArtifacts')}</div>
        ) : (
          <div className={styles.artifactList}>
            {detail.artifacts.map((artifact) => {
              const { path } = artifact;
              if (path === null) {
                // On a shared mount this server does not hold: name it by
                // label, never as a local path, and offer nothing to copy.
                const where = `${artifact.mount_label ?? '?'}:${artifact.relative_path ?? ''}`;
                return (
                  <div key={artifact.id} className={styles.artifactRow}>
                    <span className={styles.artifactKind}>{artifact.kind}</span>
                    <span
                      className={styles.artifactPath}
                      title={t('runs.detail.notOnThisServer', { label: artifact.mount_label ?? '?' })}
                    >
                      {where}
                    </span>
                  </div>
                );
              }
              return (
                <div key={artifact.id} className={styles.artifactRow}>
                  <span className={styles.artifactKind}>{artifact.kind}</span>
                  <span className={styles.artifactPath} title={path}>
                    {path}
                  </span>
                  <button
                    type="button"
                    className={styles.rowBtn}
                    onClick={() => void copyPath(path)}
                    title={t('runs.detail.copyPath')}
                  >
                    {t('runs.detail.copyPath')}
                  </button>
                  {(artifact.backend === 'transfer' || artifact.backend === 's3')
                    && typeof artifact.id === 'number' && (
                    // Uploaded by a worker without the shared mount, here
                    // (#785) or to the S3 bucket (#791): the server serves
                    // the file, by row id (an event-log key like `cursor:7`
                    // names no row).
                    <a
                      className={styles.rowBtn}
                      href={`/api/runs/${encodeURIComponent(detail.runId)}/artifacts/${artifact.id}/download`}
                      download
                      title={t('runs.detail.download')}
                    >
                      {t('runs.detail.download')}
                    </a>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* The detail's own copy: list rows do not carry the records. */}
      <RunProvenance records={detail.row?.provenance} />

      <div className={styles.detailSection}>
        <div className={styles.sectionHeader}>{t('runs.detail.log')}</div>
        {detail.log.length === 0 ? (
          <div className={styles.detailEmpty}>{t('runs.detail.noLog')}</div>
        ) : (
          <div className={styles.logList}>
            {detail.log.map((line) => (
              <LogLine key={line.cursor} line={line} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface RunsPanelProps {
  panelHeight: number;
  /** Called after a successful Watch, so the dock can reveal the log. */
  onWatchRun?: () => void;
}

export function RunsPanel({ panelHeight, onWatchRun }: RunsPanelProps) {
  const { t } = useI18n();
  const runs = useRunStore((s) => s.runs);
  const total = useRunStore((s) => s.total);
  const loading = useRunStore((s) => s.loading);
  const error = useRunStore((s) => s.error);
  const filter = useRunStore((s) => s.filter);
  const selectedRunId = useRunStore((s) => s.selectedRunId);
  const busy = useRunStore((s) => s.busy);
  const setFilter = useRunStore((s) => s.setFilter);
  const refresh = useRunStore((s) => s.refresh);
  const select = useRunStore((s) => s.select);
  const cancel = useRunStore((s) => s.cancel);
  const remove = useRunStore((s) => s.remove);
  const exportCsv = useRunStore((s) => s.exportCsv);
  const selectedSweepId = useSweepStore((s) => s.selectedSweepId);
  const closeSweep = useSweepStore((s) => s.closeSweep);
  // In the sweep store, set on the click, so the canvas shortcuts stand
  // down before the dialog's chunk has loaded; cleared when this panel
  // goes, whether or not the dialog ever mounted.
  const newSweepOpen = useSweepStore((s) => s.newSweepOpen);
  const setNewSweepOpen = useSweepStore((s) => s.setNewSweepOpen);
  useEffect(() => () => useSweepStore.getState().setNewSweepOpen(false), []);
  const rootRef = useRef<HTMLDivElement>(null);
  const newSweepRef = useRef<HTMLButtonElement>(null);
  const sweepBackRef = useRef<HTMLButtonElement>(null);
  const runParentRef = useRef<HTMLButtonElement>(null);
  const runCloseRef = useRef<HTMLButtonElement>(null);

  // Re-render on a cadence so the duration column of a live run ticks;
  // `runs` itself only changes when the poll brings something new.
  const [clock, setClock] = useState(() => Date.now());
  useEffect(() => {
    const anyActive = runs.some((run) => isActiveRun(run.status));
    if (!anyActive) return;
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [runs]);

  // One list poller for however many ResultsPanels are mounted (one per open
  // canvas tab), started only while this tab is actually on screen.
  useEffect(() => useRunStore.getState().watch(), []);
  // The sweep poller follows the panel the same way. The dock unmounts this
  // panel for another tab and when collapsed, so on the way back the shown
  // sweep is read again and polled again rather than frozen as it was left.
  useEffect(() => {
    void useSweepStore.getState().resumePolling();
    return () => useSweepStore.getState().stopPolling();
  }, []);

  // Every switch between the list and a sweep unmounts the control that was
  // pressed (Start, Open parent sweep, Back, Open run), and focus would fall
  // to the page. It goes to the new view's way back instead -- unless it is
  // somewhere else on purpose.
  const view = selectedSweepId !== null ? 'sweep' : 'list';
  const shownView = useRef(view);
  useEffect(() => {
    if (shownView.current === view) return;
    shownView.current = view;
    const active = document.activeElement;
    if (active && active !== document.body && !rootRef.current?.contains(active)) return;
    const target = view === 'sweep'
      ? sweepBackRef.current
      : runParentRef.current ?? runCloseRef.current ?? newSweepRef.current;
    target?.focus();
  }, [view]);

  // A second click while the confirm or the socket handshake is pending
  // would send a second `attach`, and the server REPLACES an attachment
  // rather than ignoring the duplicate — the run's whole log would replay
  // twice into a panel that was just cleared for it.
  const attaching = useRef(false);

  // No status guard: the button only renders for an active run, and one that
  // finished between the poll and the click still attaches correctly — the
  // server replays its whole log and ends the stream, which is exactly what
  // "show me this run" should do.
  const attachToRun = useCallback(async (run: RunSummary) => {
    const tab = useTabStore.getState().getActiveTab();
    // One socket holds exactly one attachment (#121), so pointing this tab
    // at another run stops the events it is currently showing. The other run
    // is NOT cancelled — but the user should still be the one to decide.
    if (tab.status === 'running' && tab.lastRunId && tab.lastRunId !== run.id) {
      const ok = await confirm({
        title: t('runs.reattach.title'),
        message: t('runs.reattach.message', { current: tab.lastRunId.slice(0, 8) }),
        confirmText: t('runs.reattach.confirm'),
      });
      if (!ok) return;
    }
    if (!tab.ws.connected) {
      try {
        await tab.ws.connect();
      } catch {
        useToastStore.getState().addToast(t('runs.reattach.offline'), 'error');
        return;
      }
    }
    // The awaits above can outlast the user's attention. `clearLogs` and
    // `clearExecutionStatus` act on whichever canvas tab is active NOW, so a
    // tab switch in that window would empty the wrong tab's panel while the
    // socket we are about to attach belongs to the old one.
    const store = useTabStore.getState();
    if (store.activeTabId !== tab.id) return;
    // Clear first and replay from cursor 0: the panel is about to show one
    // run's whole story, and interleaving it with whatever was already there
    // would read as a single confused execution.
    store.clearLogs();
    store.clearExecutionStatus();
    store.setLastRunId(tab.id, run.id);
    // `running` is set when the server ACKNOWLEDGES the attach, exactly as
    // the page-load re-attach does — an attach the server refuses must not
    // leave the tab disabled with nothing streaming to it.
    tab.ws.send({ action: 'attach', run_id: run.id, cursor: 0 });
    // Show the user where the events are about to appear. Without this,
    // "Watch" looks like it did nothing: the stream lands in the Execution
    // Log, which is a tab away from the button they just pressed.
    onWatchRun?.();
  }, [t, onWatchRun]);

  const reattach = useCallback(async (run: RunSummary) => {
    if (attaching.current) return;
    attaching.current = true;
    try {
      await attachToRun(run);
    } finally {
      attaching.current = false;
    }
  }, [attachToRun]);

  const askDelete = useCallback(async (run: RunSummary) => {
    const ok = await confirm({
      title: t('runs.delete.title'),
      message: t('runs.delete.message', { name: run.name || t('runs.unnamed') }),
      confirmText: t('runs.delete.confirm'),
      variant: 'danger',
    });
    if (!ok) return;
    await remove(run.id);
  }, [remove, t]);

  const chartHeight = Math.max(90, Math.min(220, panelHeight - 190));

  const openChildRun = useCallback((runId: string) => {
    closeSweep();
    void select(runId);
  }, [closeSweep, select]);

  if (selectedSweepId !== null) {
    return (
      <div ref={rootRef} className={styles.runsBody}>
        <SweepDetail
          chartHeight={chartHeight}
          onBack={closeSweep}
          onOpenRun={openChildRun}
          backRef={sweepBackRef}
        />
      </div>
    );
  }

  return (
    <div ref={rootRef} className={styles.runsBody}>
      <div className={styles.listCol}>
        <div className={styles.toolbar}>
          <div className={styles.filters}>
            {FILTERS.map((value) => (
              <button
                key={value}
                type="button"
                className={`${styles.filterBtn} ${filter === value ? styles.filterActive : ''}`}
                onClick={() => setFilter(value)}
              >
                {value === 'all'
                  ? t('runs.filter.all')
                  : t(`runs.status.${value}`)}
              </button>
            ))}
          </div>
          <div className={styles.toolbarRight}>
            <button
              ref={newSweepRef}
              type="button"
              className={`${styles.rowBtn} ${styles.newSweepBtn}`}
              onClick={() => setNewSweepOpen(true)}
            >
              {t('sweeps.new.title')}
            </button>
            <button
              type="button"
              className={styles.rowBtn}
              title={t('workers.openTitle')}
              onClick={() => useWorkerStore.getState().openPanel()}
            >
              {t('workers.open')}
            </button>
            <button
              type="button"
              className={styles.rowBtn}
              title={t('studies.openTitle')}
              onClick={() => useStudyStore.getState().openPanel()}
            >
              {t('studies.open')}
            </button>
            <span className={styles.countText}>
              {t('runs.showing', { shown: runs.length, total })}
            </span>
            <button
              type="button"
              className={styles.rowBtn}
              onClick={() => void refresh()}
              title={t('runs.refresh')}
            >
              {t('runs.refresh')}
            </button>
          </div>
        </div>

        {error && <div className={styles.listError}>{friendlyError(error)}</div>}

        <div className={styles.table} role="table">
          <div className={`${styles.row} ${styles.rowHeader}`} role="row">
            <span role="columnheader">{t('runs.col.name')}</span>
            <span role="columnheader">{t('runs.col.status')}</span>
            <span role="columnheader" className={styles.colWorker}>
              {t('runs.col.worker')}
            </span>
            <span role="columnheader" className={styles.colDevice}>
              {t('runs.col.devices')}
            </span>
            <span role="columnheader" className={styles.colBudget}>
              {t('runs.col.budget')}
            </span>
            <span role="columnheader" className={styles.colStarted}>
              {t('runs.col.started')}
            </span>
            <span role="columnheader">{t('runs.col.duration')}</span>
            <span role="columnheader">{t('runs.col.loss')}</span>
            <span role="columnheader" aria-hidden="true" />
          </div>

          {runs.length === 0 ? (
            <div className={styles.empty}>
              {loading
                ? t('runs.loading')
                : filter === 'all'
                  ? t('runs.empty')
                  : t('runs.emptyFiltered')}
            </div>
          ) : (
            runs.map((run) => {
              const loss = finalLoss(run);
              const isBusy = busy[run.id] === true;
              const active = isActiveRun(run.status);
              const expanded = selectedRunId === run.id;
              const toggle = () => void select(expanded ? null : run.id);
              const workerName = runWorkerLabel(run, t);
              const budget = runBudget(run, t);
              // Why a queued run waits beyond its place in line (#772), or
              // how long a running one has waited for its worker (#783).
              const queueReason = run.status === 'queued'
                ? run.queue_reason || null
                : disconnectedNote(run, clock, t);
              return (
                // Opening a run is this panel's primary interaction, so the
                // row is a real control: focusable, Enter/Space activated,
                // and announced with the run's name and status rather than
                // as an unlabelled group of cells.
                <div
                  key={run.id}
                  role="button"
                  tabIndex={0}
                  aria-expanded={expanded}
                  aria-label={`${run.name || t('runs.unnamed')} — ${t(`runs.status.${run.status}`)}`}
                  className={`${styles.row} ${expanded ? styles.rowSelected : ''}`}
                  onClick={toggle}
                  onKeyDown={(e) => {
                    if (e.key !== 'Enter' && e.key !== ' ') return;
                    // Space scrolls the table otherwise, and Enter would also
                    // reach a parent handler.
                    e.preventDefault();
                    toggle();
                  }}
                  data-run-id={run.id}
                >
                  <span className={styles.nameCell} title={run.name ?? run.id}>
                    {run.name || t('runs.unnamed')}
                  </span>
                  <StatusChip run={run} />
                  <span
                    className={`${styles.workerCell} ${styles.colWorker}`}
                    title={queueReason ?? undefined}
                  >
                    {workerName}
                    {queueReason && (
                      <span className={styles.queueReason}>{queueReason}</span>
                    )}
                  </span>
                  <span
                    className={`${styles.mono} ${styles.colDevice}`}
                    title={deviceAliasLines(run, t) ?? undefined}
                  >
                    {runDevices(run)}
                    {run.status === 'queued' && (
                      <span className={styles.queuePos}>
                        {/* A null position is "we cannot say", not zero — the
                            queue was deeper than the server's scan limit. */}
                        {run.queue_position === null
                          ? ' -'
                          : ` ${t('runs.queuePosition', { position: run.queue_position })}`}
                      </span>
                    )}
                  </span>
                  <span className={`${styles.mono} ${styles.colBudget}`} title={budget?.title}>
                    {budget?.text ?? '-'}
                  </span>
                  <span className={`${styles.mono} ${styles.colStarted}`}>
                    {formatStarted(run.started_at, clock)}
                  </span>
                  <span className={styles.mono}>
                    {formatDuration(run.started_at, run.finished_at, clock)}
                  </span>
                  <span className={styles.mono}>
                    {loss === null ? '-' : loss.toFixed(4)}
                  </span>
                  <span
                    className={styles.actions}
                    onClick={(e) => e.stopPropagation()}
                  >
                    {active && (
                      <>
                        <button
                          type="button"
                          className={styles.rowBtn}
                          disabled={isBusy}
                          onClick={() => void cancel(run.id)}
                          title={t('runs.action.cancelTitle')}
                        >
                          {t('runs.action.cancel')}
                        </button>
                        <button
                          type="button"
                          className={styles.rowBtn}
                          disabled={isBusy}
                          onClick={() => void reattach(run)}
                          title={t('runs.action.reattachTitle')}
                        >
                          {t('runs.action.reattach')}
                        </button>
                      </>
                    )}
                    <button
                      type="button"
                      className={styles.rowBtn}
                      disabled={isBusy}
                      onClick={() => void exportCsv(run.id)}
                      title={t('runs.action.csvTitle')}
                    >
                      {t('runs.action.csv')}
                    </button>
                    {!active && (
                      <button
                        type="button"
                        className={`${styles.rowBtn} ${styles.rowBtnDanger}`}
                        disabled={isBusy}
                        // No `title`: a run in this panel IS its metrics, its
                        // events and its artifacts -- those are the sections
                        // the row expands into -- so listing them was the
                        // button's own label at length. The full scope, and
                        // the checkpoints on disk that survive, are in the
                        // confirm dialog this always raises.
                        onClick={() => void askDelete(run)}
                      >
                        {t('runs.action.delete')}
                      </button>
                    )}
                  </span>
                </div>
              );
            })
          )}
        </div>
        {newSweepOpen && (
          <Suspense fallback={null}>
            <NewSweepDialog onClose={() => setNewSweepOpen(false)} />
          </Suspense>
        )}
      </div>

      {selectedRunId !== null && (
        <RunDetailView chartHeight={chartHeight} parentRef={runParentRef} closeRef={runCloseRef} />
      )}
    </div>
  );
}
