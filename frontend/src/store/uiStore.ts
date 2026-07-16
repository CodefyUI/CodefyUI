import { create } from 'zustand';

export type FontSize = 'small' | 'default' | 'large';

interface UIState {
  tooltipsEnabled: boolean;
  toggleTooltips: () => void;
  gridSnapEnabled: boolean;
  toggleGridSnap: () => void;
  isCanvasPanning: boolean;
  setCanvasPanning: (panning: boolean) => void;
  shortcutsModalOpen: boolean;
  toggleShortcutsModal: () => void;
  draggingSourceType: string | null;
  setDraggingSourceType: (type: string | null) => void;
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
   * Nodes whose own device param is 'auto' follow this. */
  globalDevice: string;
  setGlobalDevice: (device: string) => void;
}

const TOOLTIPS_KEY = 'codefyui-tooltips';
const GRIDSNAP_KEY = 'codefyui-gridsnap';
const BEGINNER_KEY = 'codefyui-beginner-mode';
const LAYOUT_MODE_KEY = 'codefyui-last-layout-mode';
const FONT_SIZE_KEY = 'codefyui-font-size';
const GLOBAL_DEVICE_KEY = 'codefyui-global-device';

const loadGlobalDevice = (): string => localStorage.getItem(GLOBAL_DEVICE_KEY) || 'cpu';

const loadLayoutMode = (): 'experiments' | 'all' | 'selected' => {
  const saved = localStorage.getItem(LAYOUT_MODE_KEY);
  if (saved === 'experiments' || saved === 'all' || saved === 'selected') return saved;
  return 'experiments';
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
  draggingSourceType: null,
  setDraggingSourceType: (type) => set({ draggingSourceType: type }),
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
}));
