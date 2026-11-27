/**
 * State of the side dock (`dock: 'side'`, apiVersion 6): which side panel is
 * open, which was used last, and how wide each one is.
 *
 * The side dock is one column at the far right of the editor. At most one
 * side panel shows in it at a time, so the state is a single `activeKey`
 * rather than a flag per panel: opening one closes the other by construction.
 *
 * ── What is remembered, and where ────────────────────────────────────────
 *
 * Per panel, in localStorage, named after `codefyui-sidebar-width`:
 *
 *   codefyui-side-panel-width:<plugin>:<id>   the width the user dragged to
 *   codefyui-side-panel-open:<plugin>:<id>    '1' open, '0' closed
 *   codefyui-side-panel-last                  the key Ctrl/Cmd+Shift+X toggles
 *
 * The open flag is written only when the USER opens or closes a panel (the
 * toolbar toggle, the shortcut, the close button, or the plugin's own
 * `setPanelOpen`), never when a panel merely unregisters: a plugin reloading
 * must not forget that the user had it open.
 *
 * Panels register whenever their plugin activates, which can be after this
 * module loads, so the first-open decision is made as each side panel first
 * appears in the registry: open it when its flag says '1', or when it has no
 * flag at all and the plugin asked for `defaultOpen`. Every storage access is
 * guarded: a browser that refuses storage still gets a working dock.
 */
import { create } from 'zustand';
import {
  getPluginPanel, getPluginPanels, subscribePluginPanels, type PluginPanel,
} from './panels';

const WIDTH_PREFIX = 'codefyui-side-panel-width:';
const OPEN_PREFIX = 'codefyui-side-panel-open:';
const LAST_KEY = 'codefyui-side-panel-last';

/** Below this window width the side dock floats over the canvas. */
export const SIDE_OVERLAY_BELOW = 1100;
/** Below this window width an open side dock starts the Inspector collapsed. */
export const SIDE_COLLAPSES_INSPECTOR_BELOW = 1680;
/** Below this window width an open side dock, with a node selected (so the
 * node config column is showing), collapses the left node panel to its rail. */
export const SIDE_COLLAPSES_PALETTE_BELOW = 1440;

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage refused: the dock still works for this page */
  }
}

function clamp(panel: PluginPanel, width: number): number {
  const side = panel.side!;
  return Math.min(side.maxWidth, Math.max(side.minWidth, Math.round(width)));
}

function storedWidth(key: string): number | null {
  const value = Number(read(WIDTH_PREFIX + key));
  return Number.isFinite(value) && value > 0 ? value : null;
}

interface SideDockState {
  /** The open side panel's key, or null when the column is closed. */
  activeKey: string | null;
  /** The panel Ctrl/Cmd+Shift+X toggles: the last one opened. */
  lastKey: string | null;
  /** Widths the user set this session, by key (persisted as they change). */
  widths: Record<string, number>;
}

export const useSideDockStore = create<SideDockState>(() => ({
  activeKey: null,
  lastKey: read(LAST_KEY),
  widths: {},
}));

function sidePanel(key: string): PluginPanel | undefined {
  const panel = getPluginPanel(key);
  return panel && panel.dock === 'side' ? panel : undefined;
}

/** Every registered side panel, in registration order. */
export function getSidePanels(): PluginPanel[] {
  return getPluginPanels().filter((p) => p.dock === 'side');
}

/** Open (or close) a side panel as the user's choice. Returns whether it is open after. */
export function setSidePanelOpen(key: string, open: boolean): boolean {
  if (!sidePanel(key)) return false;
  const { activeKey } = useSideDockStore.getState();
  if (open) {
    if (activeKey && activeKey !== key) write(OPEN_PREFIX + activeKey, '0');
    write(OPEN_PREFIX + key, '1');
    write(LAST_KEY, key);
    useSideDockStore.setState({ activeKey: key, lastKey: key });
    return true;
  }
  write(OPEN_PREFIX + key, '0');
  if (activeKey === key) useSideDockStore.setState({ activeKey: null });
  return false;
}

export function isSidePanelOpen(key: string): boolean {
  return useSideDockStore.getState().activeKey === key && !!sidePanel(key);
}

export function toggleSidePanel(key: string): boolean {
  return setSidePanelOpen(key, !isSidePanelOpen(key));
}

/**
 * Ctrl/Cmd+Shift+X: toggle the side panel used last, or the first registered
 * one when none was used yet (or the last one is gone). False when there is
 * no side panel at all, so the caller can leave the key alone.
 */
export function toggleLastSidePanel(): boolean {
  const { activeKey, lastKey } = useSideDockStore.getState();
  if (activeKey && sidePanel(activeKey)) {
    setSidePanelOpen(activeKey, false);
    return true;
  }
  const target = (lastKey && sidePanel(lastKey)) ? lastKey : getSidePanels()[0]?.key;
  if (!target) return false;
  setSidePanelOpen(target, true);
  return true;
}

/** The width a side panel shows at: the user's, else its default, clamped. */
export function sidePanelWidth(panel: PluginPanel): number {
  const own = useSideDockStore.getState().widths[panel.key] ?? storedWidth(panel.key);
  return clamp(panel, own ?? panel.side!.defaultWidth);
}

export function setSidePanelWidth(key: string, width: number): void {
  const panel = sidePanel(key);
  if (!panel) return;
  const next = clamp(panel, width);
  write(WIDTH_PREFIX + key, String(next));
  useSideDockStore.setState((s) => ({ widths: { ...s.widths, [key]: next } }));
}

/** Double-click on the handle: back to the plugin's default width. */
export function resetSidePanelWidth(key: string): void {
  const panel = sidePanel(key);
  if (!panel) return;
  try {
    window.localStorage.removeItem(WIDTH_PREFIX + key);
  } catch {
    /* ignore */
  }
  useSideDockStore.setState((s) => {
    const widths = { ...s.widths };
    delete widths[key];
    return { widths };
  });
}

// ── following the registry ─────────────────────────────────────────────────

const known = new Set<string>();

function reconcile(): void {
  const present = new Set<string>();
  for (const panel of getSidePanels()) {
    present.add(panel.key);
    if (known.has(panel.key)) continue;
    known.add(panel.key);
    if (useSideDockStore.getState().activeKey) continue;
    const flag = read(OPEN_PREFIX + panel.key);
    if (flag === '1' || (flag === null && panel.side?.defaultOpen)) {
      useSideDockStore.setState({ activeKey: panel.key });
    }
  }
  for (const key of Array.from(known)) {
    if (present.has(key)) continue;
    known.delete(key);
    // Gone, not closed: the stored flag stays so a reload reopens it.
    if (useSideDockStore.getState().activeKey === key) {
      useSideDockStore.setState({ activeKey: null });
    }
  }
}

subscribePluginPanels(reconcile);
reconcile();

/** Test helper: forget everything this module learned. */
export function _resetSideDock(): void {
  known.clear();
  useSideDockStore.setState({ activeKey: null, lastKey: read(LAST_KEY), widths: {} });
  reconcile();
}
