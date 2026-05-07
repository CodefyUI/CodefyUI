import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { ScatterPlot, type ScatterPoint } from '../shared/ScatterPlot';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

/**
 * Inline scatter for EduKNN. Shows training points coloured by class
 * plus query points highlighted in white. The first two feature
 * dimensions are used for the scatter axes — students dialing in a
 * 4-dim Iris dataset typically pre-select two features (e.g. petal
 * length × petal width) before this node, so the projection is
 * meaningful.
 *
 * The tensor-summary cap (numel ≤ 256) limits inline rendering to
 * around 128 training points — larger datasets fall back to a hint.
 */
function EduKNNVizNode(props: NodeProps<AppNode>) {
  const { id } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });

  const data = useMemo(() => {
    const trainV = summaries?.train_coords?.values;
    const queryV = summaries?.query_coords?.values;
    const labels = summaries?.train_labels?.values;
    if (!Array.isArray(trainV) || !Array.isArray(queryV)) return null;
    if (trainV.length === 0 || queryV.length === 0) return null;
    // Class index map for colour cycling (string label → cluster id).
    const labelStrings = Array.isArray(labels) ? labels.map((l) => String(l)) : [];
    const classOrder: string[] = [];
    const points: ScatterPoint[] = [];
    (trainV as unknown[]).forEach((row, i) => {
      if (!Array.isArray(row) || row.length < 2) return;
      const x = typeof row[0] === 'number' ? row[0] : 0;
      const y = typeof row[1] === 'number' ? row[1] : 0;
      const label = labelStrings[i] ?? '';
      let cluster = classOrder.indexOf(label);
      if (cluster === -1) {
        cluster = classOrder.length;
        classOrder.push(label);
      }
      points.push({ x, y, label, cluster });
    });
    (queryV as unknown[]).forEach((row, i) => {
      if (!Array.isArray(row) || row.length < 2) return;
      const x = typeof row[0] === 'number' ? row[0] : 0;
      const y = typeof row[1] === 'number' ? row[1] : 0;
      // Mark query points with a "?<i>" label and a distinctive cluster
      // index that won't collide with class colours.
      points.push({ x, y, label: `?${i}`, cluster: 99 + i });
    });
    return { points, classCount: classOrder.length };
  }, [summaries]);

  const hasShape = !!summaries?.train_coords;

  const bodyExtra = (
    <div className={styles.vizArea}>
      {data === null && !hasShape && (
        <div className={styles.emptyHint}>{t('attention.runHint')}</div>
      )}
      {data === null && hasShape && (
        <div className={styles.tooBigHint}>
          <div>{t('attention.tooLargeInline')}</div>
          <div className={styles.metaRow}>
            <span>view in inspector</span>
          </div>
        </div>
      )}
      {data !== null && (
        <>
          <ScatterPlot points={data.points} width={240} height={200} showLabels />
          <div className={styles.metaRow}>
            <span>{data.classCount} classes</span>
            <span>? = query</span>
          </div>
        </>
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EduKNNVizNode);
