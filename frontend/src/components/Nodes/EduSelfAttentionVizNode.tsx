import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

/**
 * Inline heatmap for EduSelfAttention.weights ([seq, seq]).
 *
 * The backend's `_summarize_single` embeds tensor values when numel ≤ 256,
 * so a 16×16 attention matrix fits inline. Larger sequences fall back to
 * the empty hint — students would normally use the Inspector for those.
 *
 * Reads the `labels:LIST` output (pass-through of the optional `labels`
 * input) so token names appear on the heatmap axes when the upstream
 * Tokenizer is wired through.
 */
function EduSelfAttentionVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });

  const matrix = useMemo<number[][] | null>(() => {
    const v = summaries?.weights?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
    // Either [seq, seq] (number[][]) or [batch, seq, seq] (number[][][])
    // — we render a single panel either way (take batch=0 for 3D).
    if (Array.isArray(v[0]) && Array.isArray(v[0][0])) {
      const first = v[0] as number[][];
      return first;
    }
    return v as number[][];
  }, [summaries]);

  const labels = useMemo<string[] | undefined>(() => {
    const lv = summaries?.labels?.values;
    if (!Array.isArray(lv) || lv.length === 0) return undefined;
    return lv.map((s) => String(s));
  }, [summaries]);

  const causal = String(data.params?.causal) === 'true';

  const bodyExtra = (
    <div className={styles.vizArea}>
      {matrix === null ? (
        <div className={styles.emptyHint}>{t('attention.runHint')}</div>
      ) : (
        <>
          <HeatmapPlot
            data={matrix}
            rowLabels={labels}
            panelWidth={220}
            panelHeight={220}
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
