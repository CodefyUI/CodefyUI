import { useMemo, useRef, useState, type Ref } from 'react';
import { downloadSweepCsv, type SweepDetail as SweepDetailData, type SweepVariant } from '../../api/rest';
import { useI18n, type TranslationKey } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useSweepStore } from '../../store/sweepStore';
import { flushSubgraphEditing, useTabStore } from '../../store/tabStore';
import { friendlyError } from '../../utils/errorMessages';
import { LossChart } from './LossChart';
import { producerLabel } from './sweepObjective';
import { sweepParamTitle } from './sweepParams';
import styles from './SweepDetail.module.css';

type SortKey = 'rank' | 'status' | 'objective' | 'index';
type SortDirection = 'asc' | 'desc';

interface SweepDetailProps {
  onBack: () => void;
  onOpenRun: (runId: string) => void;
  chartHeight: number;
  /** Back, which the Runs panel focuses when it switches to this view. */
  backRef?: Ref<HTMLButtonElement>;
}

/** A variant's status as the Runs list words it; `missing` is the sweep's own. */
const STATUS_KEY: Record<string, TranslationKey> = {
  running: 'runs.status.running',
  queued: 'runs.status.queued',
  succeeded: 'runs.status.succeeded',
  failed: 'runs.status.failed',
  cancelled: 'runs.status.cancelled',
  interrupted: 'runs.status.interrupted',
  missing: 'sweeps.status.missing',
};

/** Live work first, then outcomes, so a status sort reads the same in every language. */
const STATUS_ORDER = ['running', 'queued', 'succeeded', 'failed', 'cancelled', 'interrupted', 'missing'];

/** The summary's counts, in that order; one with no variants is left out. */
const COUNT_KEYS: Array<[keyof SweepDetailData['counts'], TranslationKey]> = [
  ['succeeded', 'sweeps.detail.completed'],
  ['failed', 'sweeps.detail.failed'],
  ['cancelled', 'sweeps.detail.cancelled'],
  ['interrupted', 'sweeps.detail.interrupted'],
  ['missing', 'sweeps.detail.missing'],
];

export function SweepDetail({ onBack, onOpenRun, chartHeight, backRef }: SweepDetailProps) {
  const { t } = useI18n();
  const detail = useSweepStore((state) => state.detail);
  const error = useSweepStore((state) => state.error);
  const notFound = useSweepStore((state) => state.notFound);
  const curves = useSweepStore((state) => state.curves);
  const cancelledRequested = useSweepStore((state) => state.cancelledRequested);
  const cancelPending = useSweepStore((state) => state.cancelPending);
  const cancelError = useSweepStore((state) => state.cancelError);
  const cancelSweep = useSweepStore((state) => state.cancelSweep);
  const definitions = useNodeDefStore((state) => state.definitions);
  // The graph a sweep came from, when this session created it; that tab may
  // since be closed, and then the columns keep their ids. A sweep from before
  // falls back to the active tab, which the id-and-param guard below checks.
  const originTabId = useSweepStore((state) =>
    state.detail ? state.origins[state.detail.sweep_id] : undefined,
  );
  const tab = useTabStore((state) =>
    state.tabs.find((candidate) => candidate.id === (originTabId ?? state.activeTabId)) ?? null,
  );
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [sortKey, setSortKey] = useState<SortKey>('index');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [exportError, setExportError] = useState<string | null>(null);

  // That graph's ROOT nodes, the graph a sweep runs; an address it cannot
  // vouch for keeps its id.
  const nodes = useMemo(() => (tab ? flushSubgraphEditing(tab).nodes : []), [tab]);
  const columns = useMemo(() => {
    if (!detail) return [];
    return detail.params.map((param) => {
      const address = `${param.node_id}.${param.param}`;
      return {
        address,
        title: sweepParamTitle(param, definitions, nodes) ?? `${param.node_id.slice(0, 8)} · ${param.param}`,
      };
    });
  }, [detail, definitions, nodes]);

  const rows = useMemo(() => {
    if (!detail) return [];
    return detail.variants
      .map((variant, serverIndex) => ({ variant, serverIndex }))
      .sort((left, right) => {
        const a = sortValue(left.variant, sortKey);
        const b = sortValue(right.variant, sortKey);
        // A row with no value sorts last whichever way the column runs.
        if (a === null || b === null) {
          if (a === b) return left.serverIndex - right.serverIndex;
          return a === null ? 1 : -1;
        }
        // Ties keep the server's order, which is its ranking.
        if (a === b) return left.serverIndex - right.serverIndex;
        return sortDirection === 'asc' ? a - b : b - a;
      });
  }, [detail, sortKey, sortDirection]);

  const chart = useMemo(() => curves.map((curve) => ({
    name: t('sweeps.detail.variantName', { index: curve.variantIndex + 1 }),
    // The colour follows the variant, not the words of the current language.
    colorKey: `variant-${curve.variantIndex}`,
    points: curve.points,
  })), [curves, t]);

  if (!detail) return (
    <section className={styles.detail} data-testid="sweep-detail">
      <header className={styles.header}><button ref={backRef} type="button" className={styles.back} onClick={onBack}>{t('sweeps.detail.back')}</button></header>
      <p className={styles.empty}>{notFound
        ? t('sweeps.detail.notFound')
        : error ? friendlyError(error) : t('sweeps.detail.loading')}</p>
    </section>
  );

  const activeOrQueued = (detail.counts.running ?? 0) + (detail.counts.queued ?? 0);
  const canCancel = detail.state === 'running'
    || (detail.state === 'failed' && activeOrQueued > 0);
  // Finished is not success: a stopped or partly failed sweep finishes too.
  const allSucceeded = (detail.counts.succeeded ?? 0) === detail.variants.length;
  const stateClass = detail.state === 'finished' && !allSucceeded
    ? styles.state_settled
    : styles[`state_${detail.state}`];
  const setSort = (key: SortKey) => {
    if (sortKey === key) setSortDirection((value) => value === 'asc' ? 'desc' : 'asc');
    else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  const exportCsv = async () => {
    setExportError(null);
    try {
      await downloadSweepCsv(detail.sweep_id);
    } catch (caught) {
      setExportError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const stop = () => {
    // Stop goes away once the sweep is stopping; the heading keeps the focus
    // in this view rather than letting it fall to the page.
    headingRef.current?.focus();
    void cancelSweep();
  };

  // The server's `objective_warning` comes whenever nothing is ranked yet --
  // every sweep until its first variant ends -- and in English. Warned here
  // only once it is true and worth acting on: no run has logged the objective,
  // ranked or live, and either a variant succeeded without it (the name is
  // likely wrong) or the sweep ended.
  const metric = detail.objective.metric;
  const objectiveNode = detail.objective.node_id ?? null;
  const nodeName = (node: string | null) => node === null ? t('sweeps.detail.runLevel') : producerLabel(node, nodes);
  // A name-only objective several nodes logged (#641): those variants are
  // unranked on purpose, and the fix is a sweep with a node chosen.
  const ambiguousNodes = [...new Set(detail.variants.flatMap((variant) => variant.ambiguous_producers ?? []))]
    .sort((left, right) => left === null ? -1 : right === null ? 1 : left.localeCompare(right));
  // `final_metrics` collapses producers, so for a chosen node only a
  // harvested value says that node logged the objective.
  const recorded = detail.variants.some((variant) => variant.rank != null
    || (objectiveNode !== null
      ? variant.objective !== null
      : variant.final_metrics !== undefined
        && Object.prototype.hasOwnProperty.call(variant.final_metrics, metric)));
  const ended = detail.state === 'finished' || (detail.state === 'failed' && activeOrQueued === 0);
  const objectiveMissing = ambiguousNodes.length === 0 && !recorded
    && ((detail.counts.succeeded ?? 0) > 0 || ended);
  const recordedSeries = objectiveMissing
    ? [...new Set(detail.variants.flatMap((variant) => Object.keys(variant.final_metrics ?? {})))].sort()
    : [];

  const stopping = cancelledRequested === null || cancelledRequested === 0
    ? t('sweeps.detail.stoppingUnknown')
    : cancelledRequested === 1
      ? t('sweeps.detail.stoppingOne')
      : t('sweeps.detail.stopping', { count: cancelledRequested });

  return (
    <section className={styles.detail} data-testid="sweep-detail">
      <header className={styles.header}>
        <button ref={backRef} type="button" className={styles.back} onClick={onBack}>{t('sweeps.detail.back')}</button>
        <div className={styles.heading}>
          <h2 ref={headingRef} tabIndex={-1}>{detail.name || t('sweeps.detail.unnamed')}</h2>
          <span className={`${styles.state} ${stateClass}`}>{t(`sweeps.state.${detail.state}`)}</span>
        </div>
        <button type="button" onClick={() => void exportCsv()}>{t('sweeps.detail.downloadCsv')}</button>
        {canCancel && <button type="button" className={styles.stop} disabled={cancelPending} onClick={stop}>{t('sweeps.detail.stop')}</button>}
      </header>

      <div className={styles.summary}>
        <span>{objectiveNode !== null
          ? t('sweeps.detail.objectiveNode', { metric, node: nodeName(objectiveNode), direction: t(`sweeps.direction.${detail.objective.direction}`) })
          : t('sweeps.detail.objective', { metric, direction: t(`sweeps.direction.${detail.objective.direction}`) })}</span>
        {activeOrQueued > 0 && <span>{t('sweeps.detail.activeQueued', { count: activeOrQueued })}</span>}
        {COUNT_KEYS.map(([status, key]) => (detail.counts[status] ?? 0) > 0
          && <span key={status}>{t(key, { count: detail.counts[status] })}</span>)}
      </div>

      {detail.state === 'cancelling' && <div className={styles.cancelling}>
        <span>{stopping}</span>
        <span>{t('sweeps.detail.cooperative')}</span>
      </div>}
      {notFound && <div className={styles.error}>{t('sweeps.detail.notFound')}</div>}
      {error && !notFound && <div className={styles.error}>{friendlyError(error)}</div>}
      {cancelError && <div className={styles.error}>{t('sweeps.detail.stopFailed', { error: friendlyError(cancelError) })}</div>}
      {detail.error && <div className={styles.error}>{friendlyError(detail.error)}</div>}
      {ambiguousNodes.length > 0 && <div className={styles.warning}>{t('sweeps.detail.ambiguousObjective', {
        metric, nodes: ambiguousNodes.map(nodeName).join(', '),
      })}</div>}
      {objectiveMissing && <div className={styles.warning}>{objectiveNode !== null
        ? t('sweeps.detail.noObjectiveNode', { metric, node: nodeName(objectiveNode) })
        : recordedSeries.length > 0
          ? t('sweeps.detail.noObjectiveSeries', { metric, names: recordedSeries.join(', ') })
          : t('sweeps.detail.noObjective', { metric })}</div>}
      {exportError && <div className={styles.error}>{friendlyError(exportError)}</div>}

      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead><tr>
            <SortableHeader label={t('sweeps.col.rank')} active={sortKey === 'rank'} direction={sortDirection} onSort={() => setSort('rank')} />
            <th>{t('sweeps.col.variant')}</th>
            <SortableHeader label={t('sweeps.col.status')} active={sortKey === 'status'} direction={sortDirection} onSort={() => setSort('status')} />
            {columns.map((column) => <th key={column.address} title={column.address}>{column.title}</th>)}
            <SortableHeader label={t('sweeps.col.objective')} active={sortKey === 'objective'} direction={sortDirection} onSort={() => setSort('objective')} />
            <th>{t('sweeps.col.metrics')}</th>
            <th>{t('sweeps.col.run')}</th>
          </tr></thead>
          <tbody>{rows.map(({ variant }) => {
            const best = detail.best?.index === variant.index;
            const values = new Map(variant.params.map((param) => [`${param.node_id}.${param.param}`, param.value]));
            const statusKey = variant.status ? STATUS_KEY[variant.status] : undefined;
            return <tr key={variant.index} className={best ? styles.best : undefined} data-testid={`sweep-variant-${variant.index}`}>
              <td>{variant.rank ?? '-'}</td>
              <td>{variant.index + 1}{best && <span className={styles.bestMark}>{t('sweeps.detail.best')}</span>}</td>
              {/* A status from a newer server falls through as its own token. */}
              <td>{statusKey ? t(statusKey) : (variant.status ?? '-')}</td>
              {columns.map((column) => <td key={column.address}>{formatValue(values.get(column.address))}</td>)}
              <td>{variant.ambiguous_producers?.length
                ? <span title={variant.ambiguous_producers.map(nodeName).join(', ')}>{t('sweeps.detail.ambiguous')}</span>
                : formatValue(variant.objective)}</td>
              <td>{formatMetrics(variant.final_metrics)}</td>
              <td>{variant.run_id && variant.run_exists !== false
                ? <button type="button" onClick={() => onOpenRun(variant.run_id!)}>{t('sweeps.detail.openRun')}</button>
                : <button type="button" disabled>{t('sweeps.detail.unavailable')}</button>}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>

      <section className={styles.chartSection} aria-label={t('sweeps.detail.curves')}>
        <h3>{t('sweeps.detail.curves')}</h3>
        {chart.length > 0
          ? <LossChart series={chart} height={chartHeight} xLabel={t('runs.detail.step')} />
          : <p className={styles.empty}>{t('sweeps.detail.noCurves')}</p>}
      </section>
    </section>
  );
}

function SortableHeader({ label, active, direction, onSort }: {
  label: string; active: boolean; direction: SortDirection; onSort: () => void;
}) {
  return <th aria-sort={active ? (direction === 'asc' ? 'ascending' : 'descending') : 'none'}>
    <button type="button" className={styles.sort} onClick={onSort} onKeyDown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSort();
      }
    }}>{label}{active ? (direction === 'asc' ? ' ↑' : ' ↓') : ''}</button>
  </th>;
}

/** What a column sorts on; null is "no value", which always sorts last. */
function sortValue(variant: SweepVariant, key: SortKey): number | null {
  switch (key) {
    case 'index':
      return variant.index;
    case 'rank':
      return variant.rank ?? null;
    case 'objective':
      return variant.objective;
    case 'status': {
      if (!variant.status) return null;
      const order = STATUS_ORDER.indexOf(variant.status);
      return order === -1 ? STATUS_ORDER.length : order;
    }
  }
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
  return String(value);
}

function formatMetrics(metrics: Record<string, number> | undefined): string {
  if (!metrics || Object.keys(metrics).length === 0) return '-';
  return Object.entries(metrics).sort(([left], [right]) => left.localeCompare(right))
    .map(([name, value]) => `${name} ${formatValue(value)}`).join(', ');
}
