import { memo, useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot, type HeatmapColormap } from '../shared/HeatmapPlot';
import { HeatmapModal } from '../shared/HeatmapModal';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

/**
 * Pure-viz pass-through for any `weights:TENSOR` upstream — works with the
 * Edu* nodes as well as the production Transformer/MultiHeadAttention.
 *
 * The "view full" path REST-fetches the tensor when WS values weren't
 * embedded inline (numel > 256), so this works for production-sized
 * attention matrices too.
 */
function AttentionHeatmapVizNode(props: NodeProps<AppNode>) {
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

  const matrix = useMemo<number[][] | number[][][] | null>(() => {
    const v = summaries?.weights?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
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
            onClick={() => setExpanded(true)}
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
          onExpand={() => setExpanded(true)}
        />
      )}
      <HeatmapModal
        isOpen={expanded}
        onClose={() => setExpanded(false)}
        title={`AttentionHeatmap · ${data.label ?? id}`}
        inlineData={matrix}
        rowLabels={labels}
        colormap={colormap}
        runId={runId}
        nodeId={id}
        port="weights"
        detectCausalMask
      />
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(AttentionHeatmapVizNode);
