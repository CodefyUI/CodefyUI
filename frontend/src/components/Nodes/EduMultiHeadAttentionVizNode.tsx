import { memo, useMemo, useState } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { HeatmapPlot } from '../shared/HeatmapPlot';
import { HeatmapModal } from '../shared/HeatmapModal';
import { BaseNodeBody } from './BaseNode';
import styles from './AttentionVizNode.module.css';

function EduMultiHeadAttentionVizNode(props: NodeProps<AppNode>) {
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
      return (v as unknown as number[][][][])[0];
    }
    return v as unknown as number[][][];
  }, [summaries]);

  const labels = useMemo<string[] | undefined>(() => {
    const lv = summaries?.labels?.values;
    if (!Array.isArray(lv) || lv.length === 0) return undefined;
    return lv.map((s) => String(s));
  }, [summaries]);

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
            rowLabels={labels}
            panelWidth={panelSize}
            panelHeight={panelSize}
            onExpand={() => setExpanded(true)}
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
      <HeatmapModal
        isOpen={expanded}
        onClose={() => setExpanded(false)}
        title={`EduMultiHeadAttention · ${data.label ?? id}`}
        inlineData={heads}
        rowLabels={labels}
        runId={runId}
        nodeId={id}
        port="weights"
        detectCausalMask
        normalizePerRow
      />
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(EduMultiHeadAttentionVizNode);
