import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import { maskMatrix } from './vizViewers';
import styles from './AttentionVizNode.module.css';

function AttentionMaskVizNode(props: NodeProps<AppNode>) {
  const { id } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const openVizModal = useTabStore((s) => s.openVizModal);

  const matrix = useMemo(() => maskMatrix(summaries?.mask?.values), [summaries]);

  const hasShape = !!summaries?.mask;

  const bodyExtra = (
    <div className={styles.vizArea}>
      {matrix === null && !hasShape && (
        <div className={styles.emptyHint}>{t('attention.maskRunHint')}</div>
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
          colormap="RdBu"
          panelWidth={180}
          panelHeight={180}
          detectCausalMask={false}
          onExpand={() => openVizModal(id)}
        />
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(AttentionMaskVizNode);
