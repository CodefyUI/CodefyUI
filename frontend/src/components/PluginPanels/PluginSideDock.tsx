/**
 * Host chrome for side panels (`dock: 'side'`, apiVersion 6).
 *
 * One column at the far right of the editor, from under the tab bar to the
 * bottom of the window. It spans the Results panel on purpose: an agent's
 * conversation needs height more than anything else on screen does.
 *
 * What the host draws is the frame and nothing inside it: the left border,
 * a 4px resize handle on that border, and a tab strip when more than one
 * plugin registered a side panel (or a close button when the column floats).
 * The title bar is the plugin's, the same rule `PluginPanels.module.css`
 * states for every dock: the host does not style what a plugin renders.
 *
 * The element is attached only while its panel is open and selected, through
 * the same `useAttachedPanel` the other docks use, so `onShow`/`onHide` fire
 * on open, close and tab switch, and the element itself survives all three.
 *
 * Below `SIDE_OVERLAY_BELOW` (1100px) the column stops taking width from the
 * canvas and floats over it instead, with a close button, because a 400px
 * column on a 1000px window would leave the canvas too narrow to use.
 */
import { useCallback, useEffect, useRef } from 'react';
import { useI18n } from '../../i18n';
import { useWindowWidth } from '../../hooks/useWindowWidth';
import {
  SIDE_OVERLAY_BELOW, resetSidePanelWidth, setSidePanelOpen, setSidePanelWidth,
  sidePanelWidth, useSideDockStore,
} from '../../plugins/sideDock';
import type { PluginPanel } from '../../plugins/panels';
import { useAttachedPanel, usePluginPanels } from './PluginPanels';
import styles from './PluginSideDock.module.css';

/** Keyboard step of the resize handle, in px. */
const RESIZE_STEP = 16;
/** Room kept free on the left when the column floats over a narrow window. */
const OVERLAY_MARGIN = 48;

function SideBody({ panelKey }: { panelKey: string }) {
  const host = useRef<HTMLDivElement>(null);
  useAttachedPanel(panelKey, host);
  return (
    <div
      ref={host}
      className={styles.body}
      role="tabpanel"
      id={`plugin-side-panel-${panelKey}`}
      data-testid={`plugin-side-panel-${panelKey}`}
    />
  );
}

function ResizeHandle({ panel, width }: { panel: PluginPanel; width: number }) {
  const { t } = useI18n();
  const endDragRef = useRef<(() => void) | null>(null);
  // A drag parks listeners on `document`; unmounting mid-drag (the panel
  // closing under the pointer) must not leave them, or the col-resize cursor,
  // behind.
  useEffect(() => () => endDragRef.current?.(), []);

  const onMouseDown = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = width;
    const onMove = (move: MouseEvent) => {
      // The handle is on the LEFT edge: dragging left widens the column.
      setSidePanelWidth(panel.key, startWidth - (move.clientX - startX));
    };
    const end = () => {
      endDragRef.current = null;
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', end);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    endDragRef.current = end;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', end);
  }, [panel.key, width]);

  const onKeyDown = useCallback((event: React.KeyboardEvent) => {
    const side = panel.side!;
    if (event.key === 'ArrowLeft') setSidePanelWidth(panel.key, width + RESIZE_STEP);
    else if (event.key === 'ArrowRight') setSidePanelWidth(panel.key, width - RESIZE_STEP);
    else if (event.key === 'Home') setSidePanelWidth(panel.key, side.minWidth);
    else if (event.key === 'End') setSidePanelWidth(panel.key, side.maxWidth);
    else return;
    event.preventDefault();
  }, [panel, width]);

  return (
    <div
      className={styles.handle}
      role="separator"
      tabIndex={0}
      aria-orientation="vertical"
      aria-label={t('plugins.side.resize')}
      aria-valuenow={width}
      aria-valuemin={panel.side!.minWidth}
      aria-valuemax={panel.side!.maxWidth}
      title={t('plugins.side.resizeHint')}
      onMouseDown={onMouseDown}
      onKeyDown={onKeyDown}
      onDoubleClick={() => resetSidePanelWidth(panel.key)}
      data-testid="plugin-side-resize"
    />
  );
}

function TabStrip({ panels, activeKey }: { panels: PluginPanel[]; activeKey: string }) {
  const { t } = useI18n();
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const index = panels.findIndex((p) => p.key === activeKey);
    const step = event.key === 'ArrowRight' ? 1 : -1;
    const next = panels[(index + step + panels.length) % panels.length];
    setSidePanelOpen(next.key, true);
    document.getElementById(`plugin-side-tab-${next.key}`)?.focus();
    event.preventDefault();
  };
  return (
    <div className={styles.tabs} role="tablist" aria-label={t('plugins.side.tabs')} onKeyDown={onKeyDown}>
      {panels.map((panel) => {
        const selected = panel.key === activeKey;
        return (
          <button
            key={panel.key}
            id={`plugin-side-tab-${panel.key}`}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-controls={`plugin-side-panel-${panel.key}`}
            tabIndex={selected ? 0 : -1}
            className={`${styles.tab} ${selected ? styles.tabActive : ''}`}
            onClick={() => setSidePanelOpen(panel.key, true)}
            data-testid={`plugin-side-tab-${panel.key}`}
          >
            {panel.icon && <span className={styles.tabIcon} aria-hidden="true">{panel.icon}</span>}
            <span className={styles.tabTitle}>{panel.title}</span>
          </button>
        );
      })}
    </div>
  );
}

export function PluginSideDock() {
  const panels = usePluginPanels('side');
  const activeKey = useSideDockStore((s) => s.activeKey);
  // Subscribed so a width change re-renders; the value is read through
  // `sidePanelWidth`, which also knows the stored and default widths.
  useSideDockStore((s) => s.widths);
  const windowWidth = useWindowWidth();
  const { t } = useI18n();

  const active = panels.find((p) => p.key === activeKey);
  if (!active) return null;

  const overlay = windowWidth < SIDE_OVERLAY_BELOW;
  const width = sidePanelWidth(active);
  const shownWidth = overlay ? Math.min(width, Math.max(240, windowWidth - OVERLAY_MARGIN)) : width;
  const showTabs = panels.length > 1;

  return (
    <aside
      className={`${styles.dock} ${overlay ? styles.overlay : ''}`}
      style={{ width: shownWidth }}
      aria-label={active.title}
      data-testid="plugin-side-dock"
      data-overlay={overlay ? 'true' : 'false'}
    >
      <ResizeHandle panel={active} width={width} />
      <div className={styles.inner}>
        {(showTabs || overlay) && (
          <div className={styles.strip}>
            {showTabs ? <TabStrip panels={panels} activeKey={active.key} /> : <div className={styles.spacer} />}
            {overlay && (
              <button
                type="button"
                className={styles.close}
                onClick={() => setSidePanelOpen(active.key, false)}
                aria-label={t('plugins.side.close', { title: active.title })}
                title={t('plugins.side.close', { title: active.title })}
                data-testid="plugin-side-close"
              >
                ×
              </button>
            )}
          </div>
        )}
        <SideBody key={active.key} panelKey={active.key} />
      </div>
    </aside>
  );
}
