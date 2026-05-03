import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { ScatterPlot, type ScatterPoint } from '../shared/ScatterPlot';
import { BaseNodeBody } from './BaseNode';
import styles from './EmbeddingScatterVizNode.module.css';

function EmbeddingScatterVizNode(props: NodeProps<AppNode>) {
  const { id } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });

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

  const bodyExtra = (
    <div className={styles.vizArea}>
      {points === null ? (
        <div className={styles.emptyHint}>{t('scatter.runHint')}</div>
      ) : (
        <ScatterPlot points={points} width={320} height={220} showLabels />
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EmbeddingScatterVizNode);
