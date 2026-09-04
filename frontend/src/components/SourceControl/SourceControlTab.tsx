import { useI18n } from '../../i18n';
import styles from '../Sidebar/NodePalette.module.css';

/**
 * The sidebar's fifth tab.
 *
 * PLACEHOLDER. This is the shell wiring only — the rail entry, the panel slot
 * and the title — so the composition can be tested before the panel exists.
 * The real tab (status, stage / unstage / discard, commit, init, the commit
 * identity, and the poll this component's own effect owns) replaces the body
 * below; the header row it shares with the other four tabs stays.
 *
 * Deliberately fetches nothing yet: the sidebar mounts only the open tab, so
 * whatever this component does on mount is what opening the tab costs.
 */
export function SourceControlTab() {
  const { t } = useI18n();

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerRow}>
          <div className={styles.headerTitle}>{t('sidebar.tab.git')}</div>
        </div>
      </div>
      <div className={styles.panelBody} />
    </>
  );
}
