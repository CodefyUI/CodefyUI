import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot, type HeatmapColormap } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import { attentionWeights, labelList } from './vizViewers';
import styles from './AttentionVizNode.module.css';

/**
 * Pure-viz pass-through for any `weights:TENSOR` upstream — works with the
 * Edu* nodes as well as the production Transformer/MultiHeadAttention.
 *
 * The "view full" path REST-fetches the tensor when WS values weren't
 * embedded inline (numel > 256), so this works for production-sized
 * attention matrices too. The viewer itself is `VizViewerModal` at the app
 * root (core#324); the card only asks the store to open it.
 */
function AttentionHeatmapVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const openVizModal = useTabStore((s) => s.openVizModal);

  const matrix = useMemo(() => attentionWeights(summaries?.weights?.values), [summaries]);
  const labels = useMemo(() => labelList(summaries?.labels?.values), [summaries]);

  const colormap = (data.params?.colormap as HeatmapColormap | undefined) ?? 'viridis';
  const hasShape = !!summaries?.weights;
  const is3D = matrix !== null && Array.isArray((matrix as number[][][])[0]?.[0]);

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
        <HeatmapPlot
          data={matrix}
          rowLabels={labels}
          colormap={colormap}
          panelWidth={is3D ? 140 : 220}
          panelHeight={is3D ? 140 : 220}
          onExpand={() => openVizModal(id)}
          normalizePerRow
        />
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(AttentionHeatmapVizNode);
