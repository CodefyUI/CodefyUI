/**
 * One toolbar toggle per side panel (`dock: 'side'`, apiVersion 6).
 *
 * The host adds these, not the plugin: every side panel gets exactly one,
 * built from the panel's own `icon` and `title`, with `aria-pressed` saying
 * whether it is the panel showing. They are not part of the plugin-button
 * overflow group: a panel the user cannot reopen is worse than a crowded
 * toolbar, and there is rarely more than one or two.
 *
 * A panel's `badge` (attention, working, done) draws a small dot on its
 * toggle, static on purpose, and is read out in the button's name as well,
 * so the state never rests on colour alone.
 *
 * Shown on the welcome bar too. Unlike a plugin's toolbar button, which acts
 * on the graph in front of the user, a toggle only opens a panel, which is
 * as useful with no tab open as with one.
 */
import { useI18n } from '../../i18n';
import { usePluginPanels } from '../PluginPanels/PluginPanels';
import { toggleSidePanel, useSideDockStore } from '../../plugins/sideDock';
import styles from './Toolbar.module.css';

export function SidePanelToggles() {
  const panels = usePluginPanels('side');
  const activeKey = useSideDockStore((s) => s.activeKey);
  const { t } = useI18n();
  if (panels.length === 0) return null;
  return (
    <div className={styles.cluster} data-testid="side-panel-toggles">
      {panels.map((panel) => {
        const open = panel.key === activeKey;
        const label = panel.badge
          ? t('plugins.side.toggleWithState', {
            title: panel.title, state: t(`plugins.side.badge.${panel.badge}`),
          })
          : t('plugins.side.toggle', { title: panel.title });
        return (
          <button
            key={panel.key}
            type="button"
            className={`${styles.iconBtn} ${styles.sideToggle} ${open ? styles.active : ''}`}
            aria-pressed={open}
            aria-label={label}
            title={`${label} (${t('plugins.side.shortcut')})`}
            onClick={() => toggleSidePanel(panel.key)}
            data-testid={`side-panel-toggle-${panel.key}`}
          >
            {panel.icon || panel.title.slice(0, 1)}
            {panel.badge && (
              <span
                className={`${styles.sideBadge} ${styles[`sideBadge_${panel.badge}`]}`}
                aria-hidden="true"
                data-testid={`side-panel-badge-${panel.key}`}
                data-badge={panel.badge}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}
