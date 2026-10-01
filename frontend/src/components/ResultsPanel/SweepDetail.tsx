import { useEffect, useMemo, useState } from 'react';
import { downloadSweepCsv, type SweepVariant } from '../../api/rest';
import { useI18n } from '../../i18n';
import { useSweepStore } from '../../store/sweepStore';
import { friendlyError } from '../../utils/errorMessages';
import { LossChart } from './LossChart';
import styles from './SweepDetail.module.css';

type SortKey = 'rank' | 'status' | 'objective' | 'index';
type SortDirection = 'asc' | 'desc';

interface SweepDetailProps {
  onBack: () => void;
  onOpenRun: (runId: string) => void;
  chartHeight: number;
}

export function SweepDetail({ onBack, onOpenRun, chartHeight }: SweepDetailProps) {
  const { t } = useI18n();
  const detail = useSweepStore((state) => state.detail);
  const error = useSweepStore((state) => state.error);
  const curves = useSweepStore((state) => state.curves);
  const cancelledRequested = useSweepStore((state) => state.cancelledRequested);
  const cancelSweep = useSweepStore((state) => state.cancelSweep);
  const loadCurves = useSweepStore((state) => state.loadCurves);
  const [sortKey, setSortKey] = useState<SortKey>('index');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    void loadCurves();
  }, [loadCurves, detail?.sweep_id, detail?.state]);

  const rows = useMemo(() => {
    if (!detail) return [];
    return detail.variants
      .map((variant, serverIndex) => ({ variant, serverIndex }))
      .sort((left, right) => {
        const compared = compareVariants(left.variant, right.variant, sortKey);
        if (compared === 0) return left.serverIndex - right.serverIndex;
        return sortDirection === 'asc' ? compared : -compared;
      });
  }, [detail, sortKey, sortDirection]);

  if (!detail) return (
    <section className={styles.detail} data-testid="sweep-detail">
      <header className={styles.header}><button type="button" onClick={onBack}>{t('sweeps.detail.back')}</button></header>
      <p className={styles.empty}>{error ?? t('sweeps.detail.loading')}</p>
    </section>
  );

  const activeOrQueued = (detail.counts.running ?? 0) + (detail.counts.queued ?? 0);
  const canCancel = detail.state === 'running'
    || (detail.state === 'failed' && activeOrQueued > 0);
  const statusLabel = t(`sweeps.state.${detail.state}`);
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

  return (
    <section className={styles.detail} data-testid="sweep-detail">
      <header className={styles.header}>
        <button type="button" className={styles.back} onClick={onBack}>{t('sweeps.detail.back')}</button>
        <div className={styles.heading}>
          <h2>{detail.name || t('sweeps.detail.unnamed')}</h2>
          <span className={`${styles.state} ${styles[`state_${detail.state}`]}`}>{statusLabel}</span>
        </div>
        <button type="button" onClick={() => void exportCsv()}>{t('sweeps.detail.downloadCsv')}</button>
        {canCancel && <button type="button" className={styles.stop} onClick={() => void cancelSweep()}>{t('sweeps.detail.stop')}</button>}
      </header>

      <div className={styles.summary}>
        <span>{t('sweeps.detail.objective', { metric: detail.objective.metric, direction: t(`sweeps.direction.${detail.objective.direction}`) })}</span>
        <span>{t('sweeps.detail.activeQueued', { count: activeOrQueued })}</span>
        <span>{t('sweeps.detail.completed', { count: detail.counts.succeeded ?? 0 })}</span>
        <span>{t('sweeps.detail.failed', { count: detail.counts.failed ?? 0 })}</span>
      </div>

      {detail.state === 'cancelling' && <div className={styles.cancelling}>
        <strong>{cancelledRequested === null
          ? t('sweeps.detail.stoppingUnknown')
          : t('sweeps.detail.stopping', { count: cancelledRequested })}</strong>
        <span>{t('sweeps.detail.cooperative')}</span>
      </div>}
      {error && <div className={styles.error}>{friendlyError(error)}</div>}
      {detail.error && <div className={styles.error}>{friendlyError(detail.error)}</div>}
      {detail.objective_warning && <div className={styles.warning}>{detail.objective_warning}</div>}
      {exportError && <div className={styles.error}>{friendlyError(exportError)}</div>}

      <section className={styles.chartSection} aria-label={t('sweeps.detail.curves')}>
        <h3>{t('sweeps.detail.curves')}</h3>
        {curves.length > 0
          ? <LossChart series={curves} height={chartHeight} xLabel={t('runs.detail.step')} />
          : <p className={styles.empty}>{t('sweeps.detail.noCurves')}</p>}
      </section>

      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead><tr>
            <SortableHeader label={t('sweeps.col.rank')} active={sortKey === 'rank'} direction={sortDirection} onSort={() => setSort('rank')} />
            <th>{t('sweeps.col.variant')}</th>
            <SortableHeader label={t('sweeps.col.status')} active={sortKey === 'status'} direction={sortDirection} onSort={() => setSort('status')} />
            {detail.params.map((param) => <th key={`${param.node_id}.${param.param}`}>{param.node_id}.{param.param}</th>)}
            <SortableHeader label={t('sweeps.col.objective')} active={sortKey === 'objective'} direction={sortDirection} onSort={() => setSort('objective')} />
            <th>{t('sweeps.col.metrics')}</th>
            <th>{t('sweeps.col.run')}</th>
          </tr></thead>
          <tbody>{rows.map(({ variant }) => {
            const best = detail.best?.index === variant.index;
            const values = new Map(variant.params.map((param) => [`${param.node_id}.${param.param}`, param.value]));
            return <tr key={variant.index} className={best ? styles.best : undefined} data-testid={`sweep-variant-${variant.index}`}>
              <td>{variant.rank ?? '-'}</td>
              <td>{variant.index + 1}{best && <span className={styles.bestMark}>{t('sweeps.detail.best')}</span>}</td>
              <td>{variant.status ?? '-'}</td>
              {detail.params.map((param) => <td key={`${param.node_id}.${param.param}`}>{formatValue(values.get(`${param.node_id}.${param.param}`))}</td>)}
              <td>{formatValue(variant.objective)}</td>
              <td>{formatMetrics(variant.final_metrics)}</td>
              <td>{variant.run_id && variant.run_exists !== false
                ? <button type="button" onClick={() => onOpenRun(variant.run_id!)}>{t('sweeps.detail.openRun')}</button>
                : <button type="button" disabled>{t('sweeps.detail.unavailable')}</button>}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
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

function compareVariants(left: SweepVariant, right: SweepVariant, key: SortKey): number {
  if (key === 'index') return left.index - right.index;
  if (key === 'status') return String(left.status ?? '').localeCompare(String(right.status ?? ''));
  const a = key === 'rank' ? left.rank : left.objective;
  const b = key === 'rank' ? right.rank : right.objective;
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return a - b;
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
