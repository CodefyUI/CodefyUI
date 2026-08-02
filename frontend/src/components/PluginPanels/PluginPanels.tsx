/**
 * Host chrome for plugin panels (`api.ui.addPanel`, #132).
 *
 * These components are the "placement" half of the split described in
 * `plugins/panels.ts`: the panel ELEMENT is created and owned by that
 * registry, outside React, and lives as long as the panel is registered. All
 * that happens here is attaching it where it belongs and detaching it again.
 *
 * That is why the mounting effects `appendChild` and `remove()` rather than
 * rendering children: React must be free to unmount this adapter — which the
 * dock does on every tab switch — without the plugin's DOM going with it.
 */
import { useEffect, useMemo, useRef, useSyncExternalStore } from 'react';
import { useI18n } from '../../i18n';
import {
  getPluginPanel, getPluginPanels, notifyPanelVisibility, subscribePluginPanels,
  type PluginPanel,
} from '../../plugins/panels';
import styles from './PluginPanels.module.css';

/** Live list of registered panels, filtered to one dock. */
export function usePluginPanels(dock: PluginPanel['dock']): PluginPanel[] {
  const all = useSyncExternalStore(subscribePluginPanels, getPluginPanels);
  // `all` is a cached snapshot whose identity only changes when the registry
  // does, so the result is stable between registrations — which matters
  // because callers put it in effect dependencies.
  return useMemo(() => all.filter((p) => p.dock === dock), [all, dock]);
}

/**
 * Attach one panel's element to `host` for as long as this stays mounted.
 *
 * The panel is looked up by key on every (re)mount instead of being captured:
 * a plugin can unregister while its tab is open, and holding a stale panel
 * would re-attach a detached element on the next render.
 */
function useAttachedPanel(
  panelKey: string,
  host: React.RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    const panel = getPluginPanel(panelKey);
    const container = host.current;
    if (!panel || !container) return;
    container.appendChild(panel.element);
    notifyPanelVisibility(panel, true);
    return () => {
      notifyPanelVisibility(panel, false);
      // Detach, never destroy: the plugin's React root (or whatever it
      // mounted) is inside and has to survive the switch.
      panel.element.remove();
    };
  }, [panelKey, host]);
}

/** The body of the bottom dock's tab for one plugin panel. */
export function PluginDockPanel({ panelKey }: { panelKey: string }) {
  const host = useRef<HTMLDivElement>(null);
  useAttachedPanel(panelKey, host);
  return (
    <div
      ref={host}
      className={styles.dockBody}
      data-testid={`plugin-dock-panel-${panelKey}`}
    />
  );
}

function RightSection({ panel }: { panel: PluginPanel }) {
  const host = useRef<HTMLDivElement>(null);
  useAttachedPanel(panel.key, host);
  return (
    <section className={styles.section}>
      <header className={styles.sectionHeader}>
        {panel.icon && <span className={styles.icon} aria-hidden="true">{panel.icon}</span>}
        <span className={styles.sectionTitle}>{panel.title}</span>
      </header>
      <div
        ref={host}
        className={styles.sectionBody}
        data-testid={`plugin-right-panel-${panel.key}`}
      />
    </section>
  );
}

/**
 * Every right-docked panel, stacked in one column.
 *
 * Unlike the dock, all right sections are mounted at once — that is what a
 * right rail is — but each is still only attached while this column is
 * mounted, so `onHide` fires when the editor tears the column down.
 */
export function PluginRightPanels() {
  const panels = usePluginPanels('right');
  const { t } = useI18n();
  if (panels.length === 0) return null;
  return (
    <aside
      className={styles.rightColumn}
      aria-label={t('plugins.panels')}
      data-testid="plugin-right-panels"
    >
      {panels.map((panel) => <RightSection key={panel.key} panel={panel} />)}
    </aside>
  );
}
