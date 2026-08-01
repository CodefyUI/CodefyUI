import { useMemo } from 'react';
import { useI18n } from '../../i18n';
import type { OutputSummary } from '../../types';
import { PortGroup } from '../InspectorPanel/PortGroup';
import { resolveSingleNodePorts, usePortFetches } from '../InspectorPanel/portCaptures';
import { getPortColor } from '../../utils';
import type { NodeDetailTabContext } from './tabs';
import styles from './NodeDetailModal.module.css';

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

  const title =
    kind === 'input'
      ? t('nodeDetail.inputs.title', { count: ports.length })
      : t('nodeDetail.outputs.title', { count: ports.length });
  const emptyText =
    kind === 'input' ? t('nodeDetail.inputs.empty') : t('nodeDetail.outputs.empty');

  if (ctx.runId === null) {
    return (
      <div className={styles.tabBody}>
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
