import { useI18n } from '../../i18n';
import styles from './NodeDetailModal.module.css';

/**
 * Placeholder for the per-node statistics tab.
 *
 * The tab exists now so the modal ships its full shape and the slot is
 * reserved; P3-14 replaces this body by registering a `stats` spec over the
 * built-in one (see `registerNodeDetailTab`). Nothing else in the modal needs
 * to change when it does.
 */
export function StatsTab() {
  const { t } = useI18n();
  return (
    <div className={styles.tabBody}>
      <div className={styles.emptyState}>
        <div className={styles.emptyIcon}>~</div>
        <div>{t('nodeDetail.stats.title')}</div>
        <div className={styles.emptyHint}>{t('nodeDetail.stats.placeholder')}</div>
      </div>
    </div>
  );
}
