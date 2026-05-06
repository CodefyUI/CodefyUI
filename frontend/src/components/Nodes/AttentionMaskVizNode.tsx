import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

/**
 * Visualises the [seq, seq] boolean mask emitted by AttentionMask.
 *
 * Backend serialises booleans as 0.0 / 1.0 in the embedded `values`. We
 * coerce to numbers so HeatmapPlot can render: blocked cells (1.0) show
 * up dark with the diagonal-stripe overlay; allowed cells (0.0) stay
 * light. Use the diverging RdBu colormap so 0/1 contrast is loud.
 */
function AttentionMaskVizNode(props: NodeProps<AppNode>) {
  const { id } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });

  const matrix = useMemo<number[][] | null>(() => {
    const v = summaries?.mask?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
    // Coerce booleans (or 0/1) to 0/1 numbers.
    const rows = v as unknown[];
    const numeric: number[][] = rows.map((row) =>
      Array.isArray(row) ? row.map((x) => (x ? 1 : 0)) : [],
    );
    return numeric;
  }, [summaries]);

  const bodyExtra = (
    <div className={styles.vizArea}>
      {matrix === null ? (
        <div className={styles.emptyHint}>{t('attention.maskRunHint')}</div>
      ) : (
        <HeatmapPlot
          data={matrix}
          colormap="RdBu"
          panelWidth={180}
          panelHeight={180}
          // Mask matrices are themselves the "block" signal — don't try to
          // detect a causal pattern (the whole upper triangle is "1.0", not "0").
          detectCausalMask={false}
        />
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(AttentionMaskVizNode);
