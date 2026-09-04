import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import { attentionHeads, labelList } from './vizViewers';
import styles from './AttentionVizNode.module.css';

function EduMultiHeadAttentionVizNode(props: NodeProps<AppNode>) {
  const { id, data } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const openVizModal = useTabStore((s) => s.openVizModal);

  const heads = useMemo(() => attentionHeads(summaries?.weights?.values), [summaries]);
  const labels = useMemo(() => labelList(summaries?.labels?.values), [summaries]);

  const numHeads = heads?.length ?? Number(data.params?.num_heads ?? 0);
  const causal = String(data.params?.causal) === 'true';
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
            rowLabels={labels}
            panelWidth={panelSize}
            panelHeight={panelSize}
            onExpand={() => openVizModal(id)}
            normalizePerRow
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
