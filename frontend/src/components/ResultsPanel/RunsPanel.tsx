import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { friendlyError } from '../../utils/errorMessages';
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
import type { RunSummary } from '../../api/rest';
import { LossChart } from './LossChart';
import styles from './RunsPanel.module.css';

const FILTERS: RunStatusFilter[] = [
  'all',
  'running',
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

/** The device a run is on: what it actually got, else what it asked for. */
export function runDevice(run: RunSummary): string {
  if (run.queue_key) return run.queue_key;
  const requested = run.options?.device;
  return typeof requested === 'string' && requested ? requested : '-';
}

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

function RunDetailView({ chartHeight }: { chartHeight: number }) {
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

  return (
    <div className={styles.detail} data-testid="run-detail">
      <div className={styles.detailHeader}>
        <span className={styles.detailTitle}>
          {run?.name || t('runs.unnamed')}
        </span>
        {run && <StatusChip run={run} />}
        {run && <span className={styles.detailMeta}>{runDevice(run)}</span>}
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
        <button
          type="button"
          className={styles.detailClose}
          onClick={() => void select(null)}
          aria-label={t('runs.detail.close')}
          title={t('runs.detail.close')}
        >
          ×
        </button>
      </div>

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
            {detail.artifacts.map((artifact) => (
              <div key={artifact.id} className={styles.artifactRow}>
                <span className={styles.artifactKind}>{artifact.kind}</span>
                <span className={styles.artifactPath} title={artifact.path}>
                  {artifact.path}
                </span>
                <button
                  type="button"
                  className={styles.rowBtn}
                  onClick={() => void copyPath(artifact.path)}
                  title={t('runs.detail.copyPath')}
                >
                  {t('runs.detail.copyPath')}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

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

  return (
    <div className={styles.runsBody}>
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
            <span role="columnheader" className={styles.colDevice}>
              {t('runs.col.device')}
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
                  <span className={`${styles.mono} ${styles.colDevice}`}>
                    {runDevice(run)}
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
                        onClick={() => void askDelete(run)}
                        title={t('runs.action.deleteTitle')}
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
      </div>

      {selectedRunId !== null && <RunDetailView chartHeight={chartHeight} />}
    </div>
  );
}
