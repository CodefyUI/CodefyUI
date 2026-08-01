import { useCallback, type ComponentType } from 'react';
import { useUIStore, type SidebarTab } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { SidebarRail } from './SidebarRail';
import { NodesTab } from './NodesTab';
import { PresetsTab } from './PresetsTab';
import { TemplatesTab } from './TemplatesTab';
import { CustomTab } from './CustomTab';
import styles from './NodePalette.module.css';

const TAB_PANELS: Record<SidebarTab, ComponentType> = {
  nodes: NodesTab,
  presets: PresetsTab,
  templates: TemplatesTab,
  custom: CustomTab,
};

/**
 * The left sidebar: a fixed icon rail plus the content panel for whichever tab
 * the rail has open (#126).
 *
 * Only the ACTIVE tab is mounted. That is what lets the Templates and Custom
 * tabs fetch on mount without every page load paying for three requests it may
 * never show, and it means a tab's scroll position and search box reset when
 * you come back to it — the same trade the editor already makes per canvas tab.
 *
 * Collapsing removes the panel from the DOM entirely rather than hiding it, so
 * the freed width goes to the canvas host (`.canvasHost { flex: 1 }`) and React
 * Flow's own resize observer picks the new size up. The rail stays put, so the
 * sidebar never disappears with no way back.
 *
 * This component is mounted once for the whole workspace (#125), so its state
 * — including everything the rail persists — is deliberately workspace-global
 * rather than per canvas tab.
 */
export function NodePalette() {
  const sidebarTab = useUIStore((s) => s.sidebarTab);
  const collapsed = useUIStore((s) => s.sidebarCollapsed);
  const width = useUIStore((s) => s.sidebarWidth);
  const setSidebarWidth = useUIStore((s) => s.setSidebarWidth);
  const { t } = useI18n();

  // Width is committed to the store (and localStorage) on every move rather
  // than on mouseup: the panel has to follow the pointer anyway, and the store
  // setter is what clamps the value, so a drag can never park the panel outside
  // its usable range.
  const handleResizeStart = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      const startX = event.clientX;
      const startWidth = width;

      const onMouseMove = (moveEvent: MouseEvent) => {
        setSidebarWidth(startWidth + (moveEvent.clientX - startX));
      };

      const onMouseUp = () => {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };

      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    },
    [width, setSidebarWidth],
  );

  const ActivePanel = TAB_PANELS[sidebarTab];

  return (
    <div className={styles.sidebar} data-collapsed={collapsed}>
      <SidebarRail />

      {!collapsed && (
        <>
          <div
            className={styles.panel}
            style={{ width }}
            role="tabpanel"
            id={`sidebar-panel-${sidebarTab}`}
            aria-labelledby={`sidebar-tab-${sidebarTab}`}
          >
            <ActivePanel />
          </div>
          {/* A layout sibling rather than an overlay: it can neither be clipped
              by the panel's overflow nor swallow clicks meant for the jump
              index hugging the panel's inner edge. */}
          <div
            className={styles.resizeHandle}
            onMouseDown={handleResizeStart}
            role="separator"
            aria-orientation="vertical"
            aria-label={t('sidebar.resize')}
          />
        </>
      )}
    </div>
  );
}
