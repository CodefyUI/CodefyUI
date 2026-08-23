import { create } from 'zustand';

export type FontSize = 'small' | 'default' | 'large';
export type EdgeStyle = 'circuit' | 'curve';

/** Which section the left sidebar's icon rail has open (#126). */
export type SidebarTab = 'nodes' | 'presets' | 'templates' | 'custom';

/** Rail order — also the arrow-key navigation order. */
export const SIDEBAR_TABS = ['nodes', 'presets', 'templates', 'custom'] as const;

/** Content-panel width bounds. The rail's own ~44px sits outside these. */
export const SIDEBAR_MIN_WIDTH = 180;
export const SIDEBAR_MAX_WIDTH = 520;
/** Matches the pre-#126 fixed palette width, so an existing install's sidebar
 * is exactly where it was before the rail landed. */
export const SIDEBAR_DEFAULT_WIDTH = 250;

interface UIState {
  tooltipsEnabled: boolean;
  toggleTooltips: () => void;
  gridSnapEnabled: boolean;
  toggleGridSnap: () => void;
  isCanvasPanning: boolean;
  setCanvasPanning: (panning: boolean) => void;
  shortcutsModalOpen: boolean;
  toggleShortcutsModal: () => void;
  /**
   * Template gallery modal (core#128). Workspace-global like the rest of the
   * panel state (#125) and deliberately NOT persisted — reopening the app on
   * top of a modal nobody remembers opening is never what you want.
   */
  templateGalleryOpen: boolean;
  openTemplateGallery: () => void;
  closeTemplateGallery: () => void;
  draggingSourceType: string | null;
  setDraggingSourceType: (type: string | null) => void;
  /** Endpoint of the edge currently being detached during an edge-reconnect
   * drag (transient, never persisted). While set, the matching handle renders
   * a red "detaching" ring warning that dropping on empty space deletes the
   * edge. Cleared unconditionally when the reconnect drag ends. */
  reconnectingHandle: { nodeId: string; handleId: string; type: 'source' | 'target' } | null;
  setReconnectingHandle: (
    handle: { nodeId: string; handleId: string; type: 'source' | 'target' } | null,
  ) => void;
  beginnerMode: boolean;
  toggleBeginnerMode: () => void;
  lastLayoutMode: 'experiments' | 'all' | 'selected';
  setLastLayoutMode: (mode: 'experiments' | 'all' | 'selected') => void;
  /** Set after auto-layout so the visible canvas re-fits the viewport to the
   * laid-out nodes' bounding box; the consumer clears it once handled
   * (one-shot). Carrying the bounds (not node ids) lets the canvas fit from
   * store data without racing React Flow's internal position sync. */
  layoutFitRequest: { bounds: { x: number; y: number; width: number; height: number } } | null;
  requestLayoutFit: (bounds: { x: number; y: number; width: number; height: number }) => void;
  clearLayoutFit: () => void;
  fontSize: FontSize;
  setFontSize: (size: FontSize) => void;
  /** Global compute device sent with every graph run ('cpu' | 'cuda' | 'mps').
   * Nodes whose own device param is 'auto' follow this.
   *
   * CPU is the baseline and nothing switches away from it on the user's
   * behalf: an accelerator is something you opt into in Settings, where the
   * dropdown lists every device the backend can see. Startup used to adopt
   * the best one automatically, which made the device a property of the
   * hardware rather than of the user's choice -- and a run that silently
   * moved to a GPU is a run whose failure modes the user never asked for. */
  globalDevice: string;
  setGlobalDevice: (device: string) => void;
  /** How value edges are drawn on the canvas: orthogonal circuit-board
   * traces ('circuit') or the classic curved beziers ('curve'). */
  edgeStyle: EdgeStyle;
  setEdgeStyle: (style: EdgeStyle) => void;
  /** Left sidebar (#126). The rail is always visible; `sidebarCollapsed` hides
   * only the content panel, handing its width back to the canvas. All three
   * are persisted so a reload restores the exact sidebar the user left. */
  sidebarTab: SidebarTab;
  setSidebarTab: (tab: SidebarTab) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebarCollapsed: () => void;
  sidebarWidth: number;
  setSidebarWidth: (width: number) => void;
}

const TOOLTIPS_KEY = 'codefyui-tooltips';
const GRIDSNAP_KEY = 'codefyui-gridsnap';
const BEGINNER_KEY = 'codefyui-beginner-mode';
const LAYOUT_MODE_KEY = 'codefyui-last-layout-mode';
const FONT_SIZE_KEY = 'codefyui-font-size';
const GLOBAL_DEVICE_KEY = 'codefyui-global-device';
const EDGE_STYLE_KEY = 'codefyui-edge-style';
const SIDEBAR_TAB_KEY = 'codefyui-sidebar-tab';
const SIDEBAR_COLLAPSED_KEY = 'codefyui-sidebar-collapsed';
const SIDEBAR_WIDTH_KEY = 'codefyui-sidebar-width';

const loadGlobalDevice = (): string => localStorage.getItem(GLOBAL_DEVICE_KEY) || 'cpu';

const loadEdgeStyle = (): EdgeStyle => {
  const saved = localStorage.getItem(EDGE_STYLE_KEY);
  if (saved === 'circuit' || saved === 'curve') return saved;
  return 'circuit';
};

const loadLayoutMode = (): 'experiments' | 'all' | 'selected' => {
  const saved = localStorage.getItem(LAYOUT_MODE_KEY);
  if (saved === 'experiments' || saved === 'all' || saved === 'selected') return saved;
  return 'experiments';
};

/** UI state persisted before #126 has no sidebar-tab entry at all, and a value
 * written by some other build could be anything. Both cases land on the Nodes
 * tab rather than leaving the rail with nothing selected. */
const loadSidebarTab = (): SidebarTab => {
  const saved = localStorage.getItem(SIDEBAR_TAB_KEY);
  return SIDEBAR_TABS.includes(saved as SidebarTab) ? (saved as SidebarTab) : 'nodes';
};

const clampSidebarWidth = (width: number): number =>
  Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, Math.round(width)));

/** A missing, non-numeric or out-of-range persisted width falls back to (or is
 * clamped into) the usable range — a 0px or 4000px panel is unrecoverable
 * without clearing storage. */
const loadSidebarWidth = (): number => {
  const saved = Number(localStorage.getItem(SIDEBAR_WIDTH_KEY));
  return Number.isFinite(saved) && saved > 0 ? clampSidebarWidth(saved) : SIDEBAR_DEFAULT_WIDTH;
};

const loadFontSize = (): FontSize => {
  const saved = localStorage.getItem(FONT_SIZE_KEY);
  if (saved === 'small' || saved === 'default' || saved === 'large') return saved;
  return 'default';
};

export const useUIStore = create<UIState>((set) => ({
  tooltipsEnabled: localStorage.getItem(TOOLTIPS_KEY) !== 'false',
  toggleTooltips: () =>
    set((state) => {
      const next = !state.tooltipsEnabled;
      localStorage.setItem(TOOLTIPS_KEY, String(next));
      return { tooltipsEnabled: next };
    }),
  gridSnapEnabled: localStorage.getItem(GRIDSNAP_KEY) === 'true',
  toggleGridSnap: () =>
    set((state) => {
      const next = !state.gridSnapEnabled;
      localStorage.setItem(GRIDSNAP_KEY, String(next));
      return { gridSnapEnabled: next };
    }),
  isCanvasPanning: false,
  setCanvasPanning: (panning) => set({ isCanvasPanning: panning }),
  shortcutsModalOpen: false,
  toggleShortcutsModal: () => set((state) => ({ shortcutsModalOpen: !state.shortcutsModalOpen })),
  templateGalleryOpen: false,
  openTemplateGallery: () => set({ templateGalleryOpen: true }),
  closeTemplateGallery: () => set({ templateGalleryOpen: false }),
  draggingSourceType: null,
  setDraggingSourceType: (type) => set({ draggingSourceType: type }),
  reconnectingHandle: null,
  setReconnectingHandle: (handle) => set({ reconnectingHandle: handle }),
  beginnerMode: localStorage.getItem(BEGINNER_KEY) === 'true',
  toggleBeginnerMode: () =>
    set((state) => {
      const next = !state.beginnerMode;
      localStorage.setItem(BEGINNER_KEY, String(next));
      return { beginnerMode: next };
    }),
  lastLayoutMode: loadLayoutMode(),
  setLastLayoutMode: (mode) => {
    localStorage.setItem(LAYOUT_MODE_KEY, mode);
    set({ lastLayoutMode: mode });
  },
  layoutFitRequest: null,
  requestLayoutFit: (bounds) => set({ layoutFitRequest: { bounds } }),
  clearLayoutFit: () => set({ layoutFitRequest: null }),
  fontSize: loadFontSize(),
  setFontSize: (size) => {
    localStorage.setItem(FONT_SIZE_KEY, size);
    set({ fontSize: size });
  },
  globalDevice: loadGlobalDevice(),
  setGlobalDevice: (device) => {
    localStorage.setItem(GLOBAL_DEVICE_KEY, device);
    set({ globalDevice: device });
  },
  edgeStyle: loadEdgeStyle(),
  setEdgeStyle: (style) => {
    localStorage.setItem(EDGE_STYLE_KEY, style);
    set({ edgeStyle: style });
  },
  sidebarTab: loadSidebarTab(),
  setSidebarTab: (tab) => {
    localStorage.setItem(SIDEBAR_TAB_KEY, tab);
    set({ sidebarTab: tab });
  },
  sidebarCollapsed: localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === 'true',
  setSidebarCollapsed: (collapsed) => {
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(collapsed));
    set({ sidebarCollapsed: collapsed });
  },
  toggleSidebarCollapsed: () =>
    set((state) => {
      const next = !state.sidebarCollapsed;
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      return { sidebarCollapsed: next };
    }),
  sidebarWidth: loadSidebarWidth(),
  setSidebarWidth: (width) => {
    const next = clampSidebarWidth(width);
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(next));
    set({ sidebarWidth: next });
  },
}));
