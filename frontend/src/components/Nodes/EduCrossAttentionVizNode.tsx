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
 * Inline cross-attention heatmap. Shape is [H, Q_seq, K_seq] (or
 * [batch, H, Q_seq, K_seq] when batched). Q and K may have different
 * lengths so the heatmap is rectangular — q_labels go on the row
 * axis, k_labels on the column axis.
 */
function EduCrossAttentionVizNode(props: NodeProps<AppNode>) {
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

  const heads = useMemo<number[][][] | null>(() => {
    const v = summaries?.weights?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
    if (Array.isArray(v[0]) && Array.isArray(v[0][0]) && Array.isArray((v[0][0] as unknown[])[0])) {
      // 4D — [B, H, Q_seq, K_seq] → take batch=0.
      return (v as unknown as number[][][][])[0];
    }
    return v as unknown as number[][][];
  }, [summaries]);

  const qLabels = useMemo<string[] | undefined>(() => {
    const lv = summaries?.q_labels?.values;
    if (!Array.isArray(lv) || lv.length === 0) return undefined;
    return lv.map((s) => String(s));
  }, [summaries]);

  const kLabels = useMemo<string[] | undefined>(() => {
    const lv = summaries?.k_labels?.values;
    if (!Array.isArray(lv) || lv.length === 0) return undefined;
    return lv.map((s) => String(s));
  }, [summaries]);

  const numHeads = heads?.length ?? Number(data.params?.num_heads ?? 0);
  const hasShape = !!summaries?.weights;
  const panelSize = numHeads >= 4 ? 140 : 180;

  const bodyExtra = (
    <div className={styles.vizArea}>
      {heads === null && !hasShape && (
        <div className={styles.emptyHint}>{t('attention.runHint')}</div>
      )}
      {heads === null && hasShape && (
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
      {heads !== null && (
        <>
          <HeatmapPlot
            data={heads}
            rowLabels={qLabels}
            colLabels={kLabels}
            panelWidth={panelSize}
            panelHeight={panelSize}
            onExpand={() => setExpanded(true)}
            normalizePerRow
            // Cross-attention is always rectangular and lives outside the
            // causal regime — don't try to detect a causal pattern.
            detectCausalMask={false}
          />
          <div className={styles.metaRow}>
            <span>{t('attention.heads', { count: numHeads })}</span>
            <span>cross-attn [Q × K]</span>
          </div>
        </>
      )}
      <HeatmapModal
        isOpen={expanded}
        onClose={() => setExpanded(false)}
        title={`EduCrossAttention · ${data.label ?? id}`}
        inlineData={heads}
        rowLabels={qLabels}
        colLabels={kLabels}
        runId={runId}
        nodeId={id}
        port="weights"
        detectCausalMask={false}
        normalizePerRow
      />
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EduCrossAttentionVizNode);
