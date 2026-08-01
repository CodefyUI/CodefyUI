import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchPortStats,
  StatsNotCapturedError,
  type PortStats,
  type StatsColumn,
} from '../../api/executionOutputs';
import { useI18n } from '../../i18n';
import { keyOf, type PortTarget } from '../InspectorPanel/PortGroup';
import { resolveSingleNodePorts } from '../InspectorPanel/portCaptures';
import { HistogramPlot, type HistogramBar } from '../shared/HistogramPlot';
import { getPortColor } from '../../utils';
import type { NodeDetailTabContext } from './tabs';
import modal from './NodeDetailModal.module.css';
import styles from './StatsTab.module.css';

interface StatsFetchState {
  loading: boolean;
  error: string | null;
  data: PortStats | null;
}

type StatsMap = Record<string, StatsFetchState>;

/**
 * Load `/stats` for every port of one node.
 *
 * Mirrors `usePortFetches`, with one difference that matters: the in-flight
 * requests are tied to an `AbortController` and cancelled on unmount or port
 * change (#124). Stats are the one capture read that can cost the server a
 * second of CPU per port, so a user arrowing through ten nodes must not leave
 * ten abandoned computations queued behind the one they are looking at.
 */
export function usePortStats(
  runId: string | null,
  ports: readonly PortTarget[],
): StatsMap {
  const [stats, setStats] = useState<StatsMap>({});

  const portsRef = useRef(ports);
  portsRef.current = ports;
  const portsKey = ports.map((p) => keyOf(p.nodeId, p.port)).join('|');

  useEffect(() => {
    if (!runId) return;
    const all = portsRef.current;
    if (all.length === 0) return;
    const controller = new AbortController();

    const pending: StatsMap = {};
    for (const p of all) {
      pending[keyOf(p.nodeId, p.port)] = { loading: true, error: null, data: null };
    }
    setStats((prev) => ({ ...prev, ...pending }));

    void Promise.all(
      all.map(async (p) => {
        const key = keyOf(p.nodeId, p.port);
        try {
          const data = await fetchPortStats(runId, p.nodeId, p.port, {
            signal: controller.signal,
          });
          if (controller.signal.aborted) return;
          setStats((prev) => ({ ...prev, [key]: { loading: false, error: null, data } }));
        } catch (e) {
          // An abort is this component going away, not a failure to report.
          if (controller.signal.aborted) return;
          const message =
            e instanceof StatsNotCapturedError ? e.message : (e as Error).message;
          setStats((prev) => ({
            ...prev,
            [key]: { loading: false, error: message, data: null },
          }));
        }
      }),
    );

    return () => controller.abort();
  }, [runId, portsKey]);

  return stats;
}

/** Drop trailing zeros only from something that actually has a decimal point. */
function trimZeros(text: string): string {
  return text.includes('.') && !text.includes('e')
    ? text.replace(/0+$/, '').replace(/\.$/, '')
    : text;
}

/** Compact display for one statistic. Undefined statistics read as an em dash. */
export function formatStat(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  if (!Number.isFinite(value)) return '—';
  if (Number.isInteger(value) && Math.abs(value) < 1e9) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e9)) {
    return value.toExponential(3);
  }
  return trimZeros(value.toPrecision(6));
}

function formatCount(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toLocaleString();
}

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.statCell}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue}>{value}</span>
    </div>
  );
}

/**
 * NaN / Inf / zero counts, on their own line above everything else.
 *
 * Position is the feature: "are there NaNs in this tensor" is the question
 * that brings people to a stats panel in the first place, and an answer
 * buried in row seven of a grid is an answer nobody reads. A non-zero NaN or
 * Inf count also flips the badge to the alert style, so the panel looks
 * different from across the room.
 */
function HealthRow({ stats }: { stats: PortStats }) {
  const { t } = useI18n();
  const nan = stats.nan_count ?? 0;
  const inf = stats.inf_count ?? 0;
  const zeroFrac = stats.zero_frac;
  return (
    <div className={styles.healthRow}>
      <span
        className={`${styles.healthBadge} ${nan > 0 ? styles.healthAlert : ''}`}
        data-testid="stat-nan"
      >
        {t('nodeDetail.stats.nan')} <b>{formatCount(nan)}</b>
      </span>
      <span
        className={`${styles.healthBadge} ${inf > 0 ? styles.healthAlert : ''}`}
        data-testid="stat-inf"
      >
        {t('nodeDetail.stats.inf')} <b>{formatCount(inf)}</b>
      </span>
      {zeroFrac !== null && zeroFrac !== undefined && (
        <span className={styles.healthBadge}>
          {t('nodeDetail.stats.zeros')} <b>{(zeroFrac * 100).toFixed(1)}%</b>
        </span>
      )}
    </div>
  );
}

function TensorStats({ stats }: { stats: PortStats }) {
  const { t } = useI18n();
  const quantiles = stats.quantiles ?? {};
  const quantileNames = ['p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99'];

  const bars: HistogramBar[] = useMemo(() => {
    if (stats.value_counts?.length) {
      return stats.value_counts.map((entry) => ({
        label: `${String(entry.value)} · ${entry.count.toLocaleString()}`,
        count: entry.count,
      }));
    }
    const hist = stats.histogram;
    if (!hist) return [];
    return hist.counts.map((count, i) => ({
      label: `[${formatStat(hist.edges[i])}, ${formatStat(hist.edges[i + 1])})`,
      count,
    }));
  }, [stats.value_counts, stats.histogram]);

  const isClassBalance = Boolean(stats.value_counts?.length);

  return (
    <>
      <HealthRow stats={stats} />

      <div className={styles.statGrid}>
        <StatCell
          label={t('nodeDetail.stats.shape')}
          value={stats.shape ? `[${stats.shape.join(', ')}]` : '—'}
        />
        <StatCell label={t('nodeDetail.stats.dtype')} value={stats.dtype ?? '—'} />
        <StatCell label={t('nodeDetail.stats.device')} value={stats.device ?? '—'} />
        <StatCell label={t('nodeDetail.stats.count')} value={formatCount(stats.count)} />
        <StatCell label={t('nodeDetail.stats.mean')} value={formatStat(stats.mean)} />
        <StatCell label={t('nodeDetail.stats.std')} value={formatStat(stats.std)} />
        <StatCell label={t('nodeDetail.stats.min')} value={formatStat(stats.min)} />
        <StatCell label={t('nodeDetail.stats.max')} value={formatStat(stats.max)} />
      </div>

      {quantileNames.some((name) => quantiles[name] !== undefined) && (
        <div className={styles.quantileBlock}>
          <div className={styles.blockTitle}>{t('nodeDetail.stats.quantiles')}</div>
          <div className={styles.quantileRow}>
            {quantileNames.map((name) => (
              <div key={name} className={styles.quantileCell}>
                <span className={styles.statLabel}>{name}</span>
                <span className={styles.statValue}>{formatStat(quantiles[name])}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {bars.length > 0 && (
        <div className={styles.chartBlock}>
          <div className={styles.blockTitle}>
            {isClassBalance
              ? t('nodeDetail.stats.classBalance')
              : t('nodeDetail.stats.distribution')}
          </div>
          <HistogramPlot
            bars={bars}
            variant={isClassBalance ? 'categorical' : 'continuous'}
            axisMin={isClassBalance ? undefined : formatStat(stats.min)}
            axisMax={isClassBalance ? undefined : formatStat(stats.max)}
            ariaLabel={
              isClassBalance
                ? t('nodeDetail.stats.classBalance')
                : t('nodeDetail.stats.distribution')
            }
          />
        </div>
      )}
    </>
  );
}

function ColumnRow({ column }: { column: StatsColumn }) {
  const top = column.top
    .slice(0, 3)
    .map((entry) => `${String(entry.value)} (${entry.count})`)
    .join(', ');
  return (
    <tr>
      <td className={styles.colName}>{column.name}</td>
      <td className={styles.colType}>{column.dtype}</td>
      <td>{formatCount(column.count)}</td>
      <td className={column.missing > 0 ? styles.colMissing : undefined}>
        {formatCount(column.missing)}
      </td>
      <td>{formatCount(column.unique)}</td>
      <td className={styles.colTop} title={top}>
        {top || '—'}
      </td>
      <td>
        {column.mean == null
          ? '—'
          : `${formatStat(column.mean)} (${formatStat(column.min)} … ${formatStat(column.max)})`}
      </td>
    </tr>
  );
}

function TabularStats({ stats }: { stats: PortStats }) {
  const { t } = useI18n();
  const columns = stats.columns ?? [];
  return (
    <>
      <div className={styles.healthRow}>
        <span className={styles.healthBadge}>
          {t('nodeDetail.stats.rows')} <b>{formatCount(stats.rows)}</b>
        </span>
        <span className={styles.healthBadge}>
          {t('nodeDetail.stats.columns')} <b>{formatCount(stats.column_count)}</b>
        </span>
      </div>
      {stats.columns_truncated && (
        <div className={styles.note}>
          {t('nodeDetail.stats.columnsTruncated', {
            shown: columns.length,
            total: stats.column_count ?? columns.length,
          })}
        </div>
      )}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>{t('nodeDetail.stats.column')}</th>
              <th>{t('nodeDetail.stats.dtype')}</th>
              <th>{t('nodeDetail.stats.count')}</th>
              <th>{t('nodeDetail.stats.missing')}</th>
              <th>{t('nodeDetail.stats.unique')}</th>
              <th>{t('nodeDetail.stats.top')}</th>
              <th>{t('nodeDetail.stats.mean')}</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((column, index) => (
              <ColumnRow key={`${column.name}::${index}`} column={column} />
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function PortStatsBlock({
  port,
  state,
  focused,
}: {
  port: PortTarget;
  state: StatsFetchState | undefined;
  focused: boolean;
}) {
  const { t } = useI18n();
  const ref = useRef<HTMLElement | null>(null);
  const stats = state?.data;

  // The edge tooltip's "View stats" lands here — bring the port it named into
  // view rather than leaving the user to find it in the list.
  // `state?.data` is in the deps because the block is a one-line header until
  // the fetch lands and a full stat panel afterwards — scrolling only on mount
  // aims at the wrong height and leaves the port off screen.
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ block: 'nearest' });
  }, [focused, state?.data]);

  return (
    <section
      ref={ref}
      className={`${styles.portBlock} ${focused ? styles.portFocused : ''}`}
      data-testid={`stats-port-${port.nodeId}-${port.port}`}
    >
      <header className={styles.portHeader}>
        {port.dataType && (
          <span
            className={styles.portDot}
            style={{ background: getPortColor(port.dataType) }}
          />
        )}
        <span className={styles.portName}>{port.displayName ?? port.port}</span>
        {stats?.sampled && (
          <span className={styles.sampledChip} title={t('nodeDetail.stats.sampledNote')}>
            {t('nodeDetail.stats.sampled', {
              size: (stats.sample_size ?? 0).toLocaleString(),
            })}
          </span>
        )}
      </header>

      {state?.loading && <div className={styles.muted}>{t('nodeDetail.stats.loading')}</div>}
      {state?.error && <div className={styles.error}>{state.error}</div>}
      {stats?.kind === 'tensor' && <TensorStats stats={stats} />}
      {stats?.kind === 'tabular' && <TabularStats stats={stats} />}
      {stats?.kind === 'unsupported' && (
        <div className={styles.muted}>
          {t('nodeDetail.stats.unsupported', { type: stats.type ?? '?' })}
        </div>
      )}
    </section>
  );
}

function PortStatsGroup({
  title,
  ports,
  stats,
  focusPort,
  emptyText,
}: {
  title: string;
  ports: PortTarget[];
  stats: StatsMap;
  focusPort: string | null;
  emptyText: string;
}) {
  return (
    <div className={styles.group}>
      <div className={styles.groupTitle}>{title}</div>
      {ports.length === 0 ? (
        <div className={styles.muted}>{emptyText}</div>
      ) : (
        ports.map((port) => (
          <PortStatsBlock
            key={keyOf(port.nodeId, port.port)}
            port={port}
            state={stats[keyOf(port.nodeId, port.port)]}
            focused={focusPort === keyOf(port.nodeId, port.port)}
          />
        ))
      )}
    </div>
  );
}

/**
 * Server-computed statistics for every port of one node (#129).
 *
 * Nothing here downloads a tensor. Each port is summarised by
 * `/api/execution/outputs/{run}/{node}/{port}/stats`, which returns a couple
 * of kilobytes whatever the value weighs — so this tab costs the same for a
 * [2, 3] tensor and for a 2 GB activation.
 *
 * Ports come from the Inspector's own `resolveSingleNodePorts`, the same
 * resolution the Inputs and Outputs tabs use, which is why a port named on an
 * edge tooltip is findable here under the node that consumes it.
 */
export function StatsTab({ ctx }: { ctx: NodeDetailTabContext }) {
  const { t } = useI18n();

  const { inputs, outputs } = useMemo(
    () => resolveSingleNodePorts(ctx.nodeId, ctx.nodes, ctx.edges),
    [ctx.nodeId, ctx.nodes, ctx.edges],
  );
  const allPorts = useMemo(() => [...inputs, ...outputs], [inputs, outputs]);
  const stats = usePortStats(ctx.runId, allPorts);

  if (ctx.runId === null) {
    return (
      <div className={modal.tabBody}>
        <div className={modal.emptyState}>
          <div className={modal.emptyIcon}>~</div>
          <div>{t('nodeDetail.stats.title')}</div>
          <div className={modal.emptyHint}>{t('nodeDetail.captures.notRunHint')}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={modal.tabBody}>
      {!ctx.recordOutputs && (
        <div className={modal.warnHint}>{t('nodeDetail.captures.recordingOff')}</div>
      )}
      <PortStatsGroup
        title={t('nodeDetail.inputs.title', { count: inputs.length })}
        ports={inputs}
        stats={stats}
        focusPort={ctx.focusPort}
        emptyText={t('nodeDetail.inputs.empty')}
      />
      <PortStatsGroup
        title={t('nodeDetail.outputs.title', { count: outputs.length })}
        ports={outputs}
        stats={stats}
        focusPort={ctx.focusPort}
        emptyText={t('nodeDetail.outputs.empty')}
      />
    </div>
  );
}
