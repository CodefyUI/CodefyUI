import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { ScatterPlot } from '../shared/ScatterPlot';
import { BaseNodeBody } from './BaseNode';
import { scatterPoints } from './vizViewers';
import styles from './EmbeddingScatterVizNode.module.css';

function EmbeddingScatterVizNode(props: NodeProps<AppNode>) {
  const { id } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });
  const openVizModal = useTabStore((s) => s.openVizModal);

  const points = useMemo(
    () => scatterPoints(summaries?.points_2d?.values, summaries?.labels?.values),
    [summaries],
  );

  // When N is large the backend skips the inline values (numel > 256), so
  // `points` is null even though the node ran. The shape still tells us how
  // many points there are — offer the viewer, which REST-fetches the full set.
  const shape = summaries?.points_2d?.shape;
  const nPoints = Array.isArray(shape) && typeof shape[0] === 'number' ? shape[0] : 0;
  const tooLarge = points === null && nPoints > 0;

  const bodyExtra = (
    <div className={styles.vizArea}>
      {points === null && !tooLarge && (
        <div className={styles.emptyHint}>{t('scatter.runHint')}</div>
      )}
      {tooLarge && (
        <div className={styles.tooBigHint}>
          <div>{t('scatter.tooLargeInline')}</div>
          <button
            type="button"
            className={styles.expandLink}
            onClick={() => openVizModal(id)}
          >
            {t('scatter.openDetail')} →
          </button>
        </div>
      )}
      {points !== null && (
        <ScatterPlot
          points={points}
          width={300}
          height={207}
          showLabels
          onExpand={() => openVizModal(id)}
        />
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EmbeddingScatterVizNode);
