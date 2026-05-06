import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot, type HeatmapColormap } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

/**
 * Pure-viz pass-through for any `weights:TENSOR` upstream — works with the
 * Edu* nodes as well as the production Transformer/MultiHeadAttention.
 *
 * Auto-detects 2D vs 3D shape; 4D ([B, H, seq, seq]) is collapsed to its
 * first batch like the other Edu wrappers.
 */
function AttentionHeatmapVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });

  const matrix = useMemo<number[][] | number[][][] | null>(() => {
    const v = summaries?.weights?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
    // 4D collapse to 3D
    if (Array.isArray(v[0]) && Array.isArray(v[0][0]) && Array.isArray((v[0][0] as unknown[])[0])) {
      return (v as unknown as number[][][][])[0];
    }
    return v as number[][] | number[][][];
  }, [summaries]);

  const labels = useMemo<string[] | undefined>(() => {
    const lv = summaries?.labels?.values;
    if (!Array.isArray(lv) || lv.length === 0) return undefined;
    return lv.map((s) => String(s));
  }, [summaries]);

  const colormap = (data.params?.colormap as HeatmapColormap | undefined) ?? 'viridis';

  const bodyExtra = (
    <div className={styles.vizArea}>
      {matrix === null ? (
        <div className={styles.emptyHint}>{t('attention.runHint')}</div>
      ) : (
        <HeatmapPlot
          data={matrix}
          rowLabels={labels}
          colormap={colormap}
          panelWidth={Array.isArray((matrix as number[][][])[0]?.[0]) ? 140 : 220}
          panelHeight={Array.isArray((matrix as number[][][])[0]?.[0]) ? 140 : 220}
        />
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(AttentionHeatmapVizNode);
