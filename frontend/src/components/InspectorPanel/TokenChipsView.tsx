import { useI18n } from '../../i18n';
import type { OutputData, ListOutput } from '../../types';
import { TokenChip } from '../shared/TokenChip';
import styles from './TokenChipsView.module.css';

interface Props {
  tokens: OutputData | null;
  tokenIds: OutputData | null;
  offsets: OutputData | null;
}

function asListValues(data: OutputData | null): unknown[] | null {
  if (data === null) return null;
  if (data.type !== 'list') return null;
  const list = data as ListOutput;
  return list.values ?? null;
}

/**
 * Friendly view for the Tokenizer node — colored chips with hover details,
 * pulled from the `tokens`, `token_ids`, and `offsets` output ports.
 */
export function TokenChipsView({ tokens, tokenIds, offsets }: Props) {
  const { t } = useI18n();
  const tokenStrings = asListValues(tokens);
  const idValues = asListValues(tokenIds);
  const offsetValues = asListValues(offsets);

  if (tokenStrings === null) return null;

  if (tokenStrings.length === 0) {
    return <div className={styles.empty}>{t('tokenizer.emptyOutput')}</div>;
  }

  const total = tokenStrings.length;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.headerLabel}>
          {t('tokenizer.tokenCount', { count: total })}
        </span>
      </div>
      <div className={styles.chipRow}>
        {tokenStrings.map((tok, i) => {
          const idVal = idValues?.[i];
          const id = typeof idVal === 'number' ? idVal : undefined;
          const off = offsetValues?.[i];
          const offset =
            Array.isArray(off) && off.length === 2 &&
            typeof off[0] === 'number' && typeof off[1] === 'number'
              ? ([off[0], off[1]] as [number, number])
              : undefined;
          return (
            <TokenChip
              key={i}
              token={String(tok)}
              id={id}
              index={i}
              offset={offset}
              animated={false}
            />
          );
        })}
      </div>
    </div>
  );
}
