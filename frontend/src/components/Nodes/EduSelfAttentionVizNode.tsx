import { memo, useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { HeatmapModal } from '../shared/HeatmapModal';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

/**
 * Inline heatmap for EduSelfAttention.weights ([seq, seq]).
 *
 * The backend's `_summarize_single` only embeds tensor values when
 * numel ≤ 256. For longer sequences we still get the shape summary, just
 * without the cell values — the modal then REST-fetches the full tensor
 * (max_elements=4096) so users can still see it at a larger size.
 */
function EduSelfAttentionVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const runId = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.lastRunId ?? null;
  });

  const [expanded, setExpanded] = useState(false);

  const matrix = useMemo<number[][] | null>(() => {
    const v = summaries?.weights?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
    if (Array.isArray(v[0]) && Array.isArray((v[0] as unknown[])[0])) {
      return (v as unknown as number[][][])[0];
    }
    return v as number[][];
  }, [summaries]);

  const labels = useMemo<string[] | undefined>(() => {
    const lv = summaries?.labels?.values;
    if (!Array.isArray(lv) || lv.length === 0) return undefined;
    return lv.map((s) => String(s));
  }, [summaries]);

  const causal = String(data.params?.causal) === 'true';
  const hasShape = !!summaries?.weights;

  const bodyExtra = (
    <div className={styles.vizArea}>
      {matrix === null && !hasShape && (
        <div className={styles.emptyHint}>{t('attention.runHint')}</div>
      )}
      {matrix === null && hasShape && (
        <div className={styles.tooBigHint}>
          <div>{t('attention.tooLargeInline')}</div>
          <button
            type="button"
            className={styles.expandLink}
            onClick={() => setExpanded(true)}
          >
            {t('attention.viewFull')} →
          </button>
        </div>
      )}
      {matrix !== null && (
        <>
          <HeatmapPlot
            data={matrix}
            rowLabels={labels}
            panelWidth={220}
            panelHeight={220}
            onExpand={() => setExpanded(true)}
            normalizePerRow
          />
          {causal && (
            <div className={styles.metaRow}>
              <span>{t('attention.causalMasked')}</span>
              <span>causal=true</span>
            </div>
          )}
        </>
      )}
      <HeatmapModal
        isOpen={expanded}
        onClose={() => setExpanded(false)}
        title={`EduSelfAttention · ${data.label ?? id}`}
        inlineData={matrix}
        rowLabels={labels}
        runId={runId}
        nodeId={id}
        port="weights"
        detectCausalMask
        normalizePerRow
      />
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EduSelfAttentionVizNode);
