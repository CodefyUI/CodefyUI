import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

/**
 * Side-by-side per-head heatmap grid for EduMultiHeadAttention.weights.
 *
 * Backend shape is [H, seq, seq] for 2D inputs and [batch, H, seq, seq]
 * for 3D inputs. We render H panels in a flex-wrap grid (HeatmapPlot
 * handles the per-panel layout itself).
 */
function EduMultiHeadAttentionVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });

  const heads = useMemo<number[][][] | null>(() => {
    const v = summaries?.weights?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
    // [H, seq, seq] (number[][][]) or [B, H, seq, seq] (number[][][][])
    if (Array.isArray(v[0]) && Array.isArray(v[0][0]) && Array.isArray((v[0][0] as unknown[])[0])) {
      // 4D — pick batch=0.
      return (v as unknown as number[][][][])[0];
    }
    return v as unknown as number[][][];
  }, [summaries]);

  const labels = useMemo<string[] | undefined>(() => {
    const lv = summaries?.labels?.values;
    if (!Array.isArray(lv) || lv.length === 0) return undefined;
    return lv.map((s) => String(s));
  }, [summaries]);

  const numHeads = heads?.length ?? Number(data.params?.num_heads ?? 0);
  const causal = String(data.params?.causal) === 'true';

  // Smaller panels when there are many heads — keeps the node card a
  // sensible size even at num_heads=4 or 8.
  const panelSize = numHeads >= 4 ? 140 : 180;

  const bodyExtra = (
    <div className={styles.vizArea}>
      {heads === null ? (
        <div className={styles.emptyHint}>{t('attention.runHint')}</div>
      ) : (
        <>
          <HeatmapPlot
            data={heads}
            rowLabels={labels}
            panelWidth={panelSize}
            panelHeight={panelSize}
          />
          <div className={styles.metaRow}>
            <span>
              {t('attention.heads', { count: numHeads })}
              {causal ? ' · causal' : ''}
            </span>
            <span>weights [H, seq, seq]</span>
          </div>
        </>
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EduMultiHeadAttentionVizNode);
