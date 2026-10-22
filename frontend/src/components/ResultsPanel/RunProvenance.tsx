import type { NodeProvenance } from '../../api/rest';
import { useI18n } from '../../i18n';
import styles from './RunsPanel.module.css';

/** Characters of a digest or revision shown before the hover gives the rest. */
const SHORT = 12;

function sourceLabel(record: NodeProvenance): string {
  // Optional chaining throughout: a plugin's provider or an older server
  // may send a partial record, which must not take the run detail down.
  const { source } = record;
  const name = source?.name ?? '';
  const path = source?.path ?? null;
  const where = path !== null ? `${source?.location ?? '?'}:${path}` : '';
  const subset = source?.subset ? `/${source.subset}` : '';
  const split = record.split ? ` [${record.split}]` : '';
  return [name + subset, where].filter(Boolean).join(' ') + split;
}

/**
 * The data and weights a run read (#344): one line per node that has a
 * provenance record, for a run submitted with the `provenance` option.
 * Nothing at all for a run without records, so the detail of an ordinary
 * run is unchanged.
 *
 * A stat identity and a content hash are labelled apart on purpose: the
 * first is path, size and modified time hashed together and can be fooled
 * by an in-place edit, the second is the bytes.
 */
export function RunProvenance({ records }: { records: NodeProvenance[] | undefined }) {
  const { t } = useI18n();
  if (!records || records.length === 0) return null;
  return (
    <div className={styles.detailSection} data-testid="run-provenance">
      <div className={styles.sectionHeader}>{t('runs.detail.provenance')}</div>
      <div className={styles.artifactList}>
        {records.map((record) => {
          const { content } = record;
          const resolved = record.revision?.resolved ?? null;
          const digest = content?.digest ?? null;
          const isContent = content?.method === 'sha256';
          return (
            <div key={record.node_id} className={styles.artifactRow}>
              <span className={styles.artifactKind}>{record.node_type}</span>
              <span className={styles.artifactPath} title={[sourceLabel(record), ...(record.notes ?? [])].join('\n')}>
                {sourceLabel(record) || record.node_id}
              </span>
              <span title={resolved ?? undefined}>
                {resolved
                  ? t('runs.detail.provenance.revision', { revision: resolved.slice(0, SHORT) })
                  : t('runs.detail.provenance.noRevision')}
              </span>
              <span
                title={
                  digest
                    ? t(isContent ? 'runs.detail.provenance.sha256Title' : 'runs.detail.provenance.statTitle', { digest })
                    : undefined
                }
              >
                {digest
                  ? t(isContent ? 'runs.detail.provenance.sha256' : 'runs.detail.provenance.stat', {
                      digest: digest.slice(0, SHORT),
                    })
                  : t('runs.detail.provenance.noContent')}
              </span>
              <span>
                {typeof record.rows === 'number'
                  ? t('runs.detail.provenance.rows', { rows: record.rows })
                  : t('runs.detail.provenance.rowsUnknown')}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
