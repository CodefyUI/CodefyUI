import { useMemo } from 'react';
import { useI18n } from '../../i18n';
import type { OutputSummary } from '../../types';
import { PortGroup } from '../InspectorPanel/PortGroup';
import { resolveSingleNodePorts, usePortFetches } from '../InspectorPanel/portCaptures';
import { getPortColor } from '../../utils';
import { useTabStore, type LogEntry } from '../../store/tabStore';
import { ChartView } from '../shared/ChartView';
import type { NodeDetailTabContext } from './tabs';
import styles from './NodeDetailModal.module.css';

/** Stable empty reference — a fresh `[]` from a zustand selector re-renders forever. */
const NO_LOGS: LogEntry[] = [];

/**
 * Condense one streamed port summary into a single readable line —
 * `tensor · [2, 3] · float32`. Used only for the pre-run fallback, where the
 * shape is all we know.
 */
function summaryLine(summary: OutputSummary): string {
  const parts: string[] = [summary.type];
  if (summary.shape) parts.push(`[${summary.shape.join(', ')}]`);
  if (summary.dtype) parts.push(summary.dtype);
  if (summary.shape === undefined && summary.length !== undefined) {
    parts.push(`length ${summary.length}`);
  }
  return parts.join(' · ');
}

/**
 * One side of a node's data: the captured values for its inputs, or for its
 * outputs.
 *
 * Ports and captures both come from `InspectorPanel/portCaptures`, and the
 * rows are the Inspector's own `PortGroup`, so a node inspected here and a
 * node inspected in the side panel show byte-identical tensors — that shared
 * path is the whole point, not an implementation detail.
 *
 * Before anything has run there is nothing to fetch, so the tab falls back to
 * the shapes and dtypes streamed during the last run's validation, plus a hint
 * saying how to get the real values.
 */
export function CapturesTab({
  kind,
  ctx,
}: {
  kind: 'input' | 'output';
  ctx: NodeDetailTabContext;
}) {
  const { t } = useI18n();

  const ports = useMemo(() => {
    const resolved = resolveSingleNodePorts(ctx.nodeId, ctx.nodes, ctx.edges);
    return kind === 'input' ? resolved.inputs : resolved.outputs;
  }, [kind, ctx.nodeId, ctx.nodes, ctx.edges]);

  const fetches = usePortFetches(ctx.runId, ports);

  // Charts this node emitted during the run (#130). They arrive on the
  // node_status stream rather than through the captures API, so they are read
  // from the tab's log rather than fetched — which also means they survive
  // with Record outputs off, when the port fetches below have nothing to show.
  const logs = useTabStore(
    (s) => s.tabs.find((t) => t.id === s.activeTabId)?.logs ?? NO_LOGS,
  );
  const charts = useMemo(
    () =>
      kind === 'output'
        ? logs.filter((e) => e.kind === 'chart' && e.nodeId === ctx.nodeId && e.chart?.kind)
        : [],
    [kind, logs, ctx.nodeId],
  );

  const title =
    kind === 'input'
      ? t('nodeDetail.inputs.title', { count: ports.length })
      : t('nodeDetail.outputs.title', { count: ports.length });
  const emptyText =
    kind === 'input' ? t('nodeDetail.inputs.empty') : t('nodeDetail.outputs.empty');

  const chartBlock = charts.length > 0 && (
    <div className={styles.chartBlock}>
      <div className={styles.chartBlockTitle}>
        {t('nodeDetail.captures.charts', { count: charts.length })}
      </div>
      {charts.map((entry, i) => (
        <ChartView key={`${entry.timestamp}-${i}`} chart={entry.chart!} width={380} />
      ))}
    </div>
  );

  if (ctx.runId === null) {
    return (
      <div className={styles.tabBody}>
        {chartBlock}
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>▶</div>
          <div>{t('nodeDetail.captures.notRun')}</div>
          <div className={styles.emptyHint}>{t('nodeDetail.captures.notRunHint')}</div>
        </div>
        <div className={styles.summaryBlock}>
          <div className={styles.summaryTitle}>{t('nodeDetail.captures.summaryTitle')}</div>
          {ports.length === 0 ? (
            <div className={styles.summaryEmpty}>{emptyText}</div>
          ) : (
            ports.map((p) => {
              const summary = ctx.outputSummaries[p.nodeId]?.[p.port];
              return (
                <div key={`${p.nodeId}::${p.port}`} className={styles.summaryRow}>
                  {p.dataType && (
                    <span
                      className={styles.summaryDot}
                      style={{ background: getPortColor(p.dataType) }}
                    />
                  )}
                  <span className={styles.summaryName}>{p.displayName ?? p.port}</span>
                  <span className={styles.summaryValue}>
                    {summary ? summaryLine(summary) : t('nodeDetail.captures.noSummary')}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.tabBody}>
      {chartBlock}
      {!ctx.recordOutputs && (
        <div className={styles.warnHint}>{t('nodeDetail.captures.recordingOff')}</div>
      )}
      <PortGroup
        kind={kind}
        title={title}
        ports={ports}
        fetches={fetches}
        emptyText={emptyText}
      />
    </div>
  );
}
