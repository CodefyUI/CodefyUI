import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import { attentionHeads, labelList } from './vizViewers';
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
  const openVizModal = useTabStore((s) => s.openVizModal);

  const heads = useMemo(() => attentionHeads(summaries?.weights?.values), [summaries]);
  const qLabels = useMemo(() => labelList(summaries?.q_labels?.values), [summaries]);
  const kLabels = useMemo(() => labelList(summaries?.k_labels?.values), [summaries]);

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
            onClick={() => openVizModal(id)}
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
            onExpand={() => openVizModal(id)}
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
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EduCrossAttentionVizNode);
