import { memo, useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { ScatterPlot, type ScatterPoint } from '../shared/ScatterPlot';
import { ScatterModal } from '../shared/ScatterModal';
import { BaseNodeBody } from './BaseNode';
import styles from './EmbeddingScatterVizNode.module.css';

function EmbeddingScatterVizNode(props: NodeProps<AppNode>) {
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

  const points = useMemo<ScatterPoint[] | null>(() => {
    const tensorVals = summaries?.points_2d?.values;
    if (!Array.isArray(tensorVals) || tensorVals.length === 0) return null;
    const labels = summaries?.labels?.values;
    return tensorVals.map((row, i) => {
      const r = Array.isArray(row) ? row : [];
      const x = typeof r[0] === 'number' ? r[0] : 0;
      const y = typeof r[1] === 'number' ? r[1] : 0;
      const label = Array.isArray(labels) && typeof labels[i] === 'string' ? (labels[i] as string) : undefined;
      return { x, y, label, cluster: i };
    });
  }, [summaries]);

  // When N is large the backend skips the inline values (numel > 256), so
  // `points` is null even though the node ran. The shape still tells us how
  // many points there are — offer the modal, which REST-fetches the full set.
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
            onClick={() => setExpanded(true)}
          >
            {t('scatter.openDetail')} →
          </button>
        </div>
      )}
      {points !== null && (
        <ScatterPlot
          points={points}
          width={320}
          height={220}
          showLabels
          onExpand={() => setExpanded(true)}
        />
      )}
      <ScatterModal
        isOpen={expanded}
        onClose={() => setExpanded(false)}
        title={`EmbeddingScatter · ${data.label ?? id}`}
        inlinePoints={points}
        runId={runId}
        nodeId={id}
        pointsPort="points_2d"
        labelsPort="labels"
      />
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EmbeddingScatterVizNode);
