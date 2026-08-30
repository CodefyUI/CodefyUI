import { memo, useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { HeatmapModal } from '../shared/HeatmapModal';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

function AttentionMaskVizNode(props: NodeProps<AppNode>) {
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

  const matrix = useMemo<number[][] | null>(() => {
    const v = summaries?.mask?.values;
    if (!Array.isArray(v) || v.length === 0) return null;
    const rows = v as unknown[];
    return rows.map((row) =>
      Array.isArray(row) ? row.map((x) => (x ? 1 : 0)) : [],
    );
  }, [summaries]);

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
            onClick={() => setExpanded(true)}
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
          onExpand={() => setExpanded(true)}
        />
      )}
      <HeatmapModal
        isOpen={expanded}
        onClose={() => setExpanded(false)}
        title={`AttentionMask · ${data.label ?? id}`}
        inlineData={matrix}
        colormap="RdBu"
        detectCausalMask={false}
        runId={runId}
        nodeId={id}
        port="mask"
        variant="mask"
      />
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(AttentionMaskVizNode);
