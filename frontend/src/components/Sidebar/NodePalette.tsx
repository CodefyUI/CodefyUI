import { useCallback, useEffect, useRef, type ComponentType } from 'react';
import {
  useUIStore,
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_MIN_WIDTH,
  type SidebarTab,
} from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { useNodeDefinitionsBootstrap } from '../../hooks/useNodeDefinitionsBootstrap';
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

/** Pixels a single arrow-key press moves the splitter. */
const RESIZE_STEP = 16;

/**
 * The left sidebar: a fixed icon rail plus the content panel for whichever tab
 * the rail has open (#126).
 *
 * Only the ACTIVE tab is mounted. That is what lets the Templates and Custom
 * tabs fetch on mount without every page load paying for three requests it may
 * never show, and it means a tab's scroll position and search box reset when
 * you come back to it — the same trade the editor already makes per canvas tab.
 *
 * Because of that, the shell — not the Nodes tab — is what starts the node and
 * preset catalog load: this component is mounted whatever tab is open and
 * whether or not the sidebar is collapsed, and the rest of the app (canvas,
 * quick search, plugin host) needs that catalog regardless of what the sidebar
 * happens to be showing.
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

  useNodeDefinitionsBootstrap();

  // Ends an in-flight resize drag: set while dragging, null otherwise.
  const endDragRef = useRef<(() => void) | null>(null);

  // A drag parks listeners on `document` and a cursor on `body`; unmounting
  // mid-drag (a tab close, a hot reload) would otherwise leak both and leave
  // the whole page stuck showing a col-resize cursor.
  useEffect(() => () => endDragRef.current?.(), []);

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

      const endDrag = () => {
        endDragRef.current = null;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', endDrag);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };

      endDragRef.current = endDrag;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', endDrag);
    },
    [width, setSidebarWidth],
  );

  // A `separator` that is focusable has to be operable from the keyboard too,
  // per the window-splitter pattern: arrows nudge, Home/End go to the bounds.
  const handleResizeKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === 'ArrowLeft') setSidebarWidth(width - RESIZE_STEP);
      else if (event.key === 'ArrowRight') setSidebarWidth(width + RESIZE_STEP);
      else if (event.key === 'Home') setSidebarWidth(SIDEBAR_MIN_WIDTH);
      else if (event.key === 'End') setSidebarWidth(SIDEBAR_MAX_WIDTH);
      else return;
      event.preventDefault();
    },
    [width, setSidebarWidth],
  );

  // Collapsing unmounts the panel. If focus was inside it, the browser drops
  // focus to <body> and the keyboard is stranded halfway down the document, so
  // hand it to the rail's open tab — the control that can put the panel back.
  // Containment is tracked as focus moves rather than read when collapsing,
  // because by the time this effect runs the panel is already gone.
  const focusWasInPanel = useRef(false);

  useEffect(() => {
    if (!collapsed || !focusWasInPanel.current) return;
    focusWasInPanel.current = false;
    document.getElementById(`sidebar-tab-${sidebarTab}`)?.focus();
  }, [collapsed, sidebarTab]);

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
            onFocusCapture={() => {
              focusWasInPanel.current = true;
            }}
            onBlurCapture={() => {
              focusWasInPanel.current = false;
            }}
          >
            <ActivePanel />
          </div>
          {/* A layout sibling rather than an overlay: it can neither be clipped
              by the panel's overflow nor swallow clicks meant for the jump
              index hugging the panel's inner edge. */}
          <div
            className={styles.resizeHandle}
            onMouseDown={handleResizeStart}
            onKeyDown={handleResizeKeyDown}
            role="separator"
            tabIndex={0}
            aria-orientation="vertical"
            aria-label={t('sidebar.resize')}
            aria-valuenow={width}
            aria-valuemin={SIDEBAR_MIN_WIDTH}
            aria-valuemax={SIDEBAR_MAX_WIDTH}
          />
        </>
      )}
    </div>
  );
}
