import { memo, useMemo } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { TokenChip } from '../shared/TokenChip';
import { BaseNodeBody } from './BaseNode';
import styles from './TokenizerVizNode.module.css';

const MAX_CHIPS_INLINE = 64;

function TokenizerVizNode(props: NodeProps<AppNode>) {
  const { id } = props;
  const { t } = useI18n();
  const summaries = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.outputSummaries?.[id];
  });

  const chips = useMemo(() => {
    const tokens = summaries?.tokens?.values;
    if (!Array.isArray(tokens) || tokens.length === 0) return null;
    const ids = summaries?.token_ids?.values;
    const offsets = summaries?.offsets?.values;
    return tokens.slice(0, MAX_CHIPS_INLINE).map((tok, i) => {
      const idVal = Array.isArray(ids) ? ids[i] : undefined;
      const off = Array.isArray(offsets) ? offsets[i] : undefined;
      const offset =
        Array.isArray(off) && off.length === 2 &&
        typeof off[0] === 'number' && typeof off[1] === 'number'
          ? ([off[0], off[1]] as [number, number])
          : undefined;
      return {
        token: String(tok),
        id: typeof idVal === 'number' ? idVal : undefined,
        offset,
      };
    });
  }, [summaries]);

  const totalLen = summaries?.tokens?.length ?? chips?.length ?? 0;
  const truncated = chips !== null && totalLen > chips.length;

  const bodyExtra = (
    <div className={styles.vizArea}>
      {chips === null ? (
        <div className={styles.emptyHint}>{t('tokenizer.runHint')}</div>
      ) : (
        <>
          <div className={styles.chipRow}>
            {chips.map((c, i) => (
              <TokenChip
                key={`${i}-${c.id ?? c.token}`}
                token={c.token}
                id={c.id}
                index={i}
                offset={c.offset}
                animated
              />
            ))}
          </div>
          {truncated && (
            <div className={styles.truncatedHint}>
              {t('tokenizer.truncatedInline', {
                shown: chips.length,
                total: totalLen,
              })}
            </div>
          )}
        </>
      )}
    </div>
  );

  return <BaseNodeBody {...props} bodyExtra={bodyExtra} />;
}

export default memo(TokenizerVizNode);
