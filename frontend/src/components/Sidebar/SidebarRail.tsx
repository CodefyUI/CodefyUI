import { useCallback, useRef, type ComponentType } from 'react';
import { useUIStore, SIDEBAR_TABS, type SidebarTab } from '../../store/uiStore';
import { useI18n, type TranslationKey } from '../../i18n';
import { MOD_LABEL } from '../../utils/platform';
import {
  BookIcon,
  GitBranchIcon,
  LayersIcon,
  LibraryIcon,
  PackageIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
} from '../shared/Icons';
import styles from './SidebarRail.module.css';

const TAB_META: Record<
  SidebarTab,
  { Icon: ComponentType<{ size?: number }>; labelKey: TranslationKey }
> = {
  nodes: { Icon: LibraryIcon, labelKey: 'sidebar.tab.nodes' },
  presets: { Icon: LayersIcon, labelKey: 'sidebar.tab.presets' },
  templates: { Icon: BookIcon, labelKey: 'sidebar.tab.templates' },
  custom: { Icon: PackageIcon, labelKey: 'sidebar.tab.custom' },
  git: { Icon: GitBranchIcon, labelKey: 'sidebar.tab.git' },
};

/**
 * The always-visible vertical icon rail (#126).
 *
 * Selection semantics follow ComfyUI's and VS Code's activity bars: clicking a
 * DIFFERENT tab opens it (expanding the sidebar if it was collapsed), clicking
 * the tab that is already open collapses the sidebar back to this rail. That
 * makes one icon both "show me this" and "give the canvas its width back".
 *
 * Keyboard: the four tabs form a vertical tablist with a roving tabindex, so
 * they are a single Tab stop and Up/Down/Home/End move between the icons.
 * Movement activates the tab it lands on (automatic activation) — the panels
 * are cheap and it keeps arrow-key browsing and clicking behaving the same way.
 *
 * The collapse toggle sits in the rail but OUTSIDE the tablist element: it is
 * not a tab, and a `tablist` may only own `tab` children. Keeping it out also
 * stops arrow navigation from landing on a control that would then have no
 * panel to describe.
 */
export function SidebarRail() {
  const sidebarTab = useUIStore((s) => s.sidebarTab);
  const collapsed = useUIStore((s) => s.sidebarCollapsed);
  const setSidebarTab = useUIStore((s) => s.setSidebarTab);
  const setSidebarCollapsed = useUIStore((s) => s.setSidebarCollapsed);
  const toggleSidebarCollapsed = useUIStore((s) => s.toggleSidebarCollapsed);
  const { t } = useI18n();

  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const openTab = useCallback(
    (tab: SidebarTab) => {
      setSidebarTab(tab);
      setSidebarCollapsed(false);
    },
    [setSidebarTab, setSidebarCollapsed],
  );

  const handleClick = useCallback(
    (tab: SidebarTab) => {
      if (tab === sidebarTab && !collapsed) setSidebarCollapsed(true);
      else openTab(tab);
    },
    [sidebarTab, collapsed, setSidebarCollapsed, openTab],
  );

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent, index: number) => {
      const count = SIDEBAR_TABS.length;
      let next: number;
      if (event.key === 'ArrowDown') next = (index + 1) % count;
      else if (event.key === 'ArrowUp') next = (index - 1 + count) % count;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = count - 1;
      else return;

      event.preventDefault();
      openTab(SIDEBAR_TABS[next]);
      buttonRefs.current[next]?.focus();
    },
    [openTab],
  );

  const toggleLabel = collapsed ? t('sidebar.expand') : t('sidebar.collapse');

  return (
    <div className={styles.rail}>
      <div
        className={styles.railTabs}
        role="tablist"
        aria-orientation="vertical"
        aria-label={t('sidebar.rail.aria')}
      >
        {SIDEBAR_TABS.map((tab, index) => {
          const { Icon, labelKey } = TAB_META[tab];
          const label = t(labelKey);
          const selected = tab === sidebarTab;
          return (
            <button
              key={tab}
              ref={(el) => {
                buttonRefs.current[index] = el;
              }}
              type="button"
              role="tab"
              id={`sidebar-tab-${tab}`}
              // Exactly one panel exists at a time, and only while expanded —
              // so only the tab that owns it may claim it. Every other tab
              // would be pointing at an id that is not in the document.
              aria-controls={
                selected && !collapsed ? `sidebar-panel-${tab}` : undefined
              }
              aria-selected={selected}
              aria-label={label}
              title={label}
              tabIndex={selected ? 0 : -1}
              data-tab={tab}
              className={
                selected && !collapsed ? `${styles.railButton} ${styles.railButtonActive}` : styles.railButton
              }
              onClick={() => handleClick(tab)}
              onKeyDown={(e) => handleKeyDown(e, index)}
            >
              <Icon size={18} />
            </button>
          );
        })}
      </div>

      <div className={styles.railSpacer} />

      <button
        type="button"
        className={styles.railButton}
        onClick={toggleSidebarCollapsed}
        aria-label={toggleLabel}
        aria-expanded={!collapsed}
        title={`${toggleLabel} (${MOD_LABEL}+B)`}
      >
        {collapsed ? <PanelLeftOpenIcon size={18} /> : <PanelLeftCloseIcon size={18} />}
      </button>
    </div>
  );
}
