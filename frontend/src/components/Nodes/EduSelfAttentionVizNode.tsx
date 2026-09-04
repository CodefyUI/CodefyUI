import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import { labelList, selfAttentionWeights } from './vizViewers';
import styles from './AttentionVizNode.module.css';

/**
 * Inline heatmap for EduSelfAttention.weights ([seq, seq]).
 *
 * The backend's `_summarize_single` only embeds tensor values when
 * numel ≤ 256. For longer sequences we still get the shape summary, just
 * without the cell values — the viewer (`VizViewerModal`, core#324) then
 * REST-fetches the full tensor (max_elements=4096) so users can still see
 * it at a larger size.
 */
function EduSelfAttentionVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const openVizModal = useTabStore((s) => s.openVizModal);

  const matrix = useMemo(() => selfAttentionWeights(summaries?.weights?.values), [summaries]);
  const labels = useMemo(() => labelList(summaries?.labels?.values), [summaries]);

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
            onClick={() => openVizModal(id)}
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
            onExpand={() => openVizModal(id)}
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
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EduSelfAttentionVizNode);
