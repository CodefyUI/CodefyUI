import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  useUIStore,
  SIDEBAR_DEFAULT_WIDTH,
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_MIN_WIDTH,
} from './uiStore';

const KEYS = {
  TOOLTIPS: 'codefyui-tooltips',
  GRIDSNAP: 'codefyui-gridsnap',
  BEGINNER: 'codefyui-beginner-mode',
  LAYOUT_MODE: 'codefyui-last-layout-mode',
  FONT_SIZE: 'codefyui-font-size',
  GLOBAL_DEVICE: 'codefyui-global-device',
  EDGE_STYLE: 'codefyui-edge-style',
  SIDEBAR_TAB: 'codefyui-sidebar-tab',
  SIDEBAR_COLLAPSED: 'codefyui-sidebar-collapsed',
  SIDEBAR_WIDTH: 'codefyui-sidebar-width',
};

describe('useUIStore', () => {
  beforeEach(() => {
    localStorage.clear();
    // Reset to a deterministic baseline; the module-load initial values depend
    // on localStorage which we've just cleared.
    useUIStore.setState({
      tooltipsEnabled: true,
      gridSnapEnabled: false,
      isCanvasPanning: false,
      shortcutsModalOpen: false,
      packCenterOpen: false,
      packCenterFocusPackId: null,
      pluginCenterOpen: false,
      pluginCenterFocusPluginId: null,
      draggingSourceType: null,
      reconnectingHandle: null,
      beginnerMode: false,
      lastLayoutMode: 'experiments',
      fontSize: 'default',
      globalDevice: 'cpu',
      edgeStyle: 'circuit',
      sidebarTab: 'nodes',
      sidebarCollapsed: false,
      sidebarWidth: SIDEBAR_DEFAULT_WIDTH,
    });
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe('toggleTooltips', () => {
    it('flips the flag and persists String(false)', () => {
      useUIStore.getState().toggleTooltips();
      expect(useUIStore.getState().tooltipsEnabled).toBe(false);
      expect(localStorage.getItem(KEYS.TOOLTIPS)).toBe('false');
    });

    it('flips back to true and persists String(true)', () => {
      useUIStore.getState().toggleTooltips();
      useUIStore.getState().toggleTooltips();
      expect(useUIStore.getState().tooltipsEnabled).toBe(true);
      expect(localStorage.getItem(KEYS.TOOLTIPS)).toBe('true');
    });
  });

  describe('toggleGridSnap', () => {
    it('flips the flag and persists String(true)', () => {
      useUIStore.getState().toggleGridSnap();
      expect(useUIStore.getState().gridSnapEnabled).toBe(true);
      expect(localStorage.getItem(KEYS.GRIDSNAP)).toBe('true');
    });

    it('flips back to false', () => {
      useUIStore.getState().toggleGridSnap();
      useUIStore.getState().toggleGridSnap();
      expect(useUIStore.getState().gridSnapEnabled).toBe(false);
      expect(localStorage.getItem(KEYS.GRIDSNAP)).toBe('false');
    });
  });

  describe('setCanvasPanning', () => {
    it('sets isCanvasPanning to true and back to false', () => {
      useUIStore.getState().setCanvasPanning(true);
      expect(useUIStore.getState().isCanvasPanning).toBe(true);
      useUIStore.getState().setCanvasPanning(false);
      expect(useUIStore.getState().isCanvasPanning).toBe(false);
    });
  });

  describe('toggleShortcutsModal', () => {
    it('flips shortcutsModalOpen', () => {
      expect(useUIStore.getState().shortcutsModalOpen).toBe(false);
      useUIStore.getState().toggleShortcutsModal();
      expect(useUIStore.getState().shortcutsModalOpen).toBe(true);
      useUIStore.getState().toggleShortcutsModal();
      expect(useUIStore.getState().shortcutsModalOpen).toBe(false);
    });
  });

  describe('setDraggingSourceType', () => {
    it('sets a type then clears it back to null', () => {
      useUIStore.getState().setDraggingSourceType('Dataset');
      expect(useUIStore.getState().draggingSourceType).toBe('Dataset');
      useUIStore.getState().setDraggingSourceType(null);
      expect(useUIStore.getState().draggingSourceType).toBeNull();
    });
  });

  describe('setReconnectingHandle', () => {
    it('sets the detaching endpoint then clears it back to null', () => {
      const endpoint = { nodeId: 'n1', handleId: 'in', type: 'target' as const };
      useUIStore.getState().setReconnectingHandle(endpoint);
      expect(useUIStore.getState().reconnectingHandle).toEqual(endpoint);
      useUIStore.getState().setReconnectingHandle(null);
      expect(useUIStore.getState().reconnectingHandle).toBeNull();
    });

    it('is transient — writes nothing to localStorage', () => {
      useUIStore
        .getState()
        .setReconnectingHandle({ nodeId: 'n1', handleId: 'out', type: 'source' });
      expect(localStorage.length).toBe(0);
      useUIStore.getState().setReconnectingHandle(null);
      expect(localStorage.length).toBe(0);
    });
  });

  describe('openPackCenter / closePackCenter', () => {
    it('records the focus pack and closePackCenter clears it', () => {
      expect(useUIStore.getState().packCenterOpen).toBe(false);
      expect(useUIStore.getState().packCenterFocusPackId).toBeNull();

      // Opened from a node that needs a pack: the panel scrolls to it.
      useUIStore.getState().openPackCenter('word-vectors');
      expect(useUIStore.getState().packCenterOpen).toBe(true);
      expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');

      useUIStore.getState().closePackCenter();
      expect(useUIStore.getState().packCenterOpen).toBe(false);
      // Cleared with the modal, so the next open starts at the top of the
      // list instead of on whatever the last node happened to need.
      expect(useUIStore.getState().packCenterFocusPackId).toBeNull();
    });

    it('opens with no focus when reached from the menu', () => {
      useUIStore.getState().openPackCenter('word-vectors');
      useUIStore.getState().closePackCenter();
      useUIStore.getState().openPackCenter();
      expect(useUIStore.getState().packCenterOpen).toBe(true);
      expect(useUIStore.getState().packCenterFocusPackId).toBeNull();
    });

    it('is transient — writes nothing to localStorage', () => {
      useUIStore.getState().openPackCenter('gpu-torch');
      expect(localStorage.length).toBe(0);
      useUIStore.getState().closePackCenter();
      expect(localStorage.length).toBe(0);
    });

    it('setPackCenterFocus re-points the open panel without closing it', () => {
      useUIStore.getState().openPackCenter();

      // A card's "Install X first" link jumps to the pack that is blocking it.
      useUIStore.getState().setPackCenterFocus('sentence-embeddings');
      expect(useUIStore.getState().packCenterFocusPackId).toBe('sentence-embeddings');
      expect(useUIStore.getState().packCenterOpen).toBe(true);

      // And the panel clears the request once it has honoured it, so a later
      // render does not scroll the list out from under the reader again.
      useUIStore.getState().setPackCenterFocus(null);
      expect(useUIStore.getState().packCenterFocusPackId).toBeNull();
      expect(useUIStore.getState().packCenterOpen).toBe(true);
    });
  });

  describe('openPluginCenter / closePluginCenter', () => {
    it('records the focus plugin and closePluginCenter clears it', () => {
      expect(useUIStore.getState().pluginCenterOpen).toBe(false);
      expect(useUIStore.getState().pluginCenterFocusPluginId).toBeNull();

      // Opened from a toast about an install that failed: the panel scrolls
      // to that plugin, where its log is.
      useUIStore.getState().openPluginCenter('demo');
      expect(useUIStore.getState().pluginCenterOpen).toBe(true);
      expect(useUIStore.getState().pluginCenterFocusPluginId).toBe('demo');

      useUIStore.getState().closePluginCenter();
      expect(useUIStore.getState().pluginCenterOpen).toBe(false);
      expect(useUIStore.getState().pluginCenterFocusPluginId).toBeNull();
    });

    it('opens with no focus when reached from the menu', () => {
      useUIStore.getState().openPluginCenter('demo');
      useUIStore.getState().closePluginCenter();
      useUIStore.getState().openPluginCenter();
      expect(useUIStore.getState().pluginCenterOpen).toBe(true);
      expect(useUIStore.getState().pluginCenterFocusPluginId).toBeNull();
    });

    it('is transient — writes nothing to localStorage', () => {
      useUIStore.getState().openPluginCenter('demo');
      expect(localStorage.length).toBe(0);
      useUIStore.getState().closePluginCenter();
      expect(localStorage.length).toBe(0);
    });

    it('setPluginCenterFocus re-points the open panel without closing it', () => {
      useUIStore.getState().openPluginCenter();

      useUIStore.getState().setPluginCenterFocus('c1');
      expect(useUIStore.getState().pluginCenterFocusPluginId).toBe('c1');
      expect(useUIStore.getState().pluginCenterOpen).toBe(true);

      useUIStore.getState().setPluginCenterFocus(null);
      expect(useUIStore.getState().pluginCenterFocusPluginId).toBeNull();
      expect(useUIStore.getState().pluginCenterOpen).toBe(true);
    });

    it('is a separate modal from the Package Center', () => {
      // Two panels, two flags: opening one must not close or focus the other.
      useUIStore.getState().openPackCenter('word-vectors');
      useUIStore.getState().openPluginCenter('demo');

      expect(useUIStore.getState().packCenterOpen).toBe(true);
      expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');

      useUIStore.getState().closePluginCenter();
      expect(useUIStore.getState().packCenterOpen).toBe(true);
    });
  });

  describe('toggleBeginnerMode', () => {
    it('flips the flag and persists String(true)', () => {
      useUIStore.getState().toggleBeginnerMode();
      expect(useUIStore.getState().beginnerMode).toBe(true);
      expect(localStorage.getItem(KEYS.BEGINNER)).toBe('true');
    });

    it('flips back to false', () => {
      useUIStore.getState().toggleBeginnerMode();
      useUIStore.getState().toggleBeginnerMode();
      expect(useUIStore.getState().beginnerMode).toBe(false);
      expect(localStorage.getItem(KEYS.BEGINNER)).toBe('false');
    });
  });

  describe('setLastLayoutMode', () => {
    it('persists and updates each valid mode', () => {
      useUIStore.getState().setLastLayoutMode('all');
      expect(useUIStore.getState().lastLayoutMode).toBe('all');
      expect(localStorage.getItem(KEYS.LAYOUT_MODE)).toBe('all');

      useUIStore.getState().setLastLayoutMode('selected');
      expect(useUIStore.getState().lastLayoutMode).toBe('selected');
      expect(localStorage.getItem(KEYS.LAYOUT_MODE)).toBe('selected');

      useUIStore.getState().setLastLayoutMode('experiments');
      expect(useUIStore.getState().lastLayoutMode).toBe('experiments');
      expect(localStorage.getItem(KEYS.LAYOUT_MODE)).toBe('experiments');
    });
  });

  describe('setFontSize', () => {
    it('persists and updates each valid size', () => {
      useUIStore.getState().setFontSize('small');
      expect(useUIStore.getState().fontSize).toBe('small');
      expect(localStorage.getItem(KEYS.FONT_SIZE)).toBe('small');

      useUIStore.getState().setFontSize('large');
      expect(useUIStore.getState().fontSize).toBe('large');
      expect(localStorage.getItem(KEYS.FONT_SIZE)).toBe('large');

      useUIStore.getState().setFontSize('default');
      expect(useUIStore.getState().fontSize).toBe('default');
      expect(localStorage.getItem(KEYS.FONT_SIZE)).toBe('default');
    });
  });

  describe('setGlobalDevice', () => {
    it('updates the device and persists it', () => {
      useUIStore.getState().setGlobalDevice('mps');
      expect(useUIStore.getState().globalDevice).toBe('mps');
      expect(localStorage.getItem(KEYS.GLOBAL_DEVICE)).toBe('mps');

      useUIStore.getState().setGlobalDevice('cuda');
      expect(useUIStore.getState().globalDevice).toBe('cuda');
      expect(localStorage.getItem(KEYS.GLOBAL_DEVICE)).toBe('cuda');
    });
  });

  describe('the CPU baseline', () => {
    // These re-import the module rather than reading the store the
    // `beforeEach` just seeded. Reading the seeded value would only restate
    // the fixture: the initial `globalDevice` is computed at module load
    // from localStorage, so that is the code a test of "what does a fresh
    // profile start on" has to actually run.
    const freshStore = async () => {
      vi.resetModules();
      return (await import('./uiStore')).useUIStore;
    };

    it('starts on the CPU when nothing has been chosen', async () => {
      // The whole device contract rests on this: CPU always works, so it is
      // where a fresh profile starts and where a profile with no stored
      // choice stays. Startup used to adopt the best device the backend
      // reported, which made the run device a property of the hardware
      // rather than of the user's choice.
      localStorage.clear();
      const store = await freshStore();
      expect(store.getState().globalDevice).toBe('cpu');
      expect(localStorage.getItem(KEYS.GLOBAL_DEVICE)).toBeNull();
    });

    it('starts on the accelerator the user chose last time', async () => {
      // The other half: an opt-in is a decision, so it survives a reload.
      // The old adopted value deliberately did not persist, which is what
      // made "is this a choice or a guess?" unanswerable from the value.
      localStorage.setItem(KEYS.GLOBAL_DEVICE, 'cuda');
      const store = await freshStore();
      expect(store.getState().globalDevice).toBe('cuda');
    });

    it('offers no way to adopt a device on the app\'s behalf', () => {
      // The removed behaviour lived in `main.tsx`, which nothing imports and
      // coverage excludes. This is the guard that notices if it comes back
      // through the store rather than through a reviewer.
      expect(useUIStore.getState()).not.toHaveProperty('adoptDefaultDevice');
    });
  });

  describe('setEdgeStyle', () => {
    it('updates the style and persists each valid value', () => {
      useUIStore.getState().setEdgeStyle('curve');
      expect(useUIStore.getState().edgeStyle).toBe('curve');
      expect(localStorage.getItem(KEYS.EDGE_STYLE)).toBe('curve');

      useUIStore.getState().setEdgeStyle('circuit');
      expect(useUIStore.getState().edgeStyle).toBe('circuit');
      expect(localStorage.getItem(KEYS.EDGE_STYLE)).toBe('circuit');
    });
  });

  // ── Sidebar (#126) ──────────────────────────────────────────────────────────

  describe('setSidebarTab', () => {
    it('persists and updates each rail tab', () => {
      for (const tab of ['presets', 'templates', 'custom', 'git', 'nodes'] as const) {
        useUIStore.getState().setSidebarTab(tab);
        expect(useUIStore.getState().sidebarTab).toBe(tab);
        expect(localStorage.getItem(KEYS.SIDEBAR_TAB)).toBe(tab);
      }
    });
  });

  describe('setSidebarCollapsed / toggleSidebarCollapsed', () => {
    it('sets the flag explicitly and persists it', () => {
      useUIStore.getState().setSidebarCollapsed(true);
      expect(useUIStore.getState().sidebarCollapsed).toBe(true);
      expect(localStorage.getItem(KEYS.SIDEBAR_COLLAPSED)).toBe('true');

      useUIStore.getState().setSidebarCollapsed(false);
      expect(useUIStore.getState().sidebarCollapsed).toBe(false);
      expect(localStorage.getItem(KEYS.SIDEBAR_COLLAPSED)).toBe('false');
    });

    it('toggles the flag and persists both directions', () => {
      useUIStore.getState().toggleSidebarCollapsed();
      expect(useUIStore.getState().sidebarCollapsed).toBe(true);
      expect(localStorage.getItem(KEYS.SIDEBAR_COLLAPSED)).toBe('true');

      useUIStore.getState().toggleSidebarCollapsed();
      expect(useUIStore.getState().sidebarCollapsed).toBe(false);
      expect(localStorage.getItem(KEYS.SIDEBAR_COLLAPSED)).toBe('false');
    });
  });

  describe('setSidebarWidth', () => {
    it('stores an in-range width and persists it', () => {
      useUIStore.getState().setSidebarWidth(320);
      expect(useUIStore.getState().sidebarWidth).toBe(320);
      expect(localStorage.getItem(KEYS.SIDEBAR_WIDTH)).toBe('320');
    });

    it('clamps below the minimum and above the maximum', () => {
      useUIStore.getState().setSidebarWidth(10);
      expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_MIN_WIDTH);

      useUIStore.getState().setSidebarWidth(9999);
      expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_MAX_WIDTH);
      expect(localStorage.getItem(KEYS.SIDEBAR_WIDTH)).toBe(String(SIDEBAR_MAX_WIDTH));
    });

    it('rounds a fractional drag width to whole pixels', () => {
      useUIStore.getState().setSidebarWidth(287.6);
      expect(useUIStore.getState().sidebarWidth).toBe(288);
    });
  });

  // ── module-load loaders (loadLayoutMode / loadFontSize) ──────────────────────
  // These run once at import time. To exercise every branch we reset the module
  // registry with localStorage pre-seeded and re-import, observing the initial
  // state the factory computed.
  describe('initial value loaders', () => {
    afterEach(() => {
      vi.resetModules();
      localStorage.clear();
    });

    it('loadLayoutMode reads a persisted valid mode', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.LAYOUT_MODE, 'all');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().lastLayoutMode).toBe('all');
    });

    it('loadLayoutMode falls back to experiments for an unknown value', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.LAYOUT_MODE, 'garbage');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().lastLayoutMode).toBe('experiments');
    });

    it('loadFontSize reads a persisted valid size', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.FONT_SIZE, 'large');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().fontSize).toBe('large');
    });

    it('loadFontSize falls back to default for an unknown value', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.FONT_SIZE, 'huge');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().fontSize).toBe('default');
    });

    it('tooltipsEnabled is false when persisted as the string "false"', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.TOOLTIPS, 'false');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().tooltipsEnabled).toBe(false);
    });

    it('gridSnapEnabled and beginnerMode are true when persisted as "true"', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.GRIDSNAP, 'true');
      localStorage.setItem(KEYS.BEGINNER, 'true');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().gridSnapEnabled).toBe(true);
      expect(mod.useUIStore.getState().beginnerMode).toBe(true);
    });

    it('globalDevice defaults to cpu when nothing is persisted', async () => {
      vi.resetModules();
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().globalDevice).toBe('cpu');
    });

    it('globalDevice loads the persisted value', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.GLOBAL_DEVICE, 'mps');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().globalDevice).toBe('mps');
    });

    it('edgeStyle defaults to circuit when nothing is persisted', async () => {
      vi.resetModules();
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().edgeStyle).toBe('circuit');
    });

    it('edgeStyle loads a persisted valid value', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.EDGE_STYLE, 'curve');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().edgeStyle).toBe('curve');
    });

    it('edgeStyle falls back to circuit for an unknown persisted value', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.EDGE_STYLE, 'zigzag');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().edgeStyle).toBe('circuit');
    });

    // ── Sidebar migration (#126) ──
    // Storage written by a pre-#126 build has none of the three sidebar keys.
    // The whole point of these is that such an install boots into a sane,
    // fully-usable sidebar rather than a blank rail or a 0px panel.

    it('migrates pre-#126 UI state (no sidebar keys) to the Nodes tab, expanded, at the old width', async () => {
      vi.resetModules();
      // A realistic pre-#126 storage snapshot: other UI keys present, no
      // sidebar ones.
      localStorage.setItem(KEYS.TOOLTIPS, 'false');
      localStorage.setItem(KEYS.FONT_SIZE, 'large');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarTab).toBe('nodes');
      expect(mod.useUIStore.getState().sidebarCollapsed).toBe(false);
      expect(mod.useUIStore.getState().sidebarWidth).toBe(mod.SIDEBAR_DEFAULT_WIDTH);
      // The untouched pre-existing keys still load as before.
      expect(mod.useUIStore.getState().tooltipsEnabled).toBe(false);
      expect(mod.useUIStore.getState().fontSize).toBe('large');
    });

    it('sidebarTab loads each persisted valid tab', async () => {
      for (const tab of ['presets', 'templates', 'custom', 'git', 'nodes']) {
        vi.resetModules();
        localStorage.setItem(KEYS.SIDEBAR_TAB, tab);
        const mod = await import('./uiStore');
        expect(mod.useUIStore.getState().sidebarTab).toBe(tab);
      }
    });

    it('sidebarTab falls back to nodes for an unknown persisted value', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.SIDEBAR_TAB, 'queue');
      const mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarTab).toBe('nodes');
    });

    it('sidebarCollapsed loads true only for the exact string "true"', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.SIDEBAR_COLLAPSED, 'true');
      let mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarCollapsed).toBe(true);

      vi.resetModules();
      localStorage.setItem(KEYS.SIDEBAR_COLLAPSED, 'yes');
      mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarCollapsed).toBe(false);
    });

    it('sidebarWidth loads a persisted value and clamps a stored out-of-range one', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.SIDEBAR_WIDTH, '300');
      let mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarWidth).toBe(300);

      vi.resetModules();
      localStorage.setItem(KEYS.SIDEBAR_WIDTH, '5000');
      mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarWidth).toBe(mod.SIDEBAR_MAX_WIDTH);
    });

    it('sidebarWidth falls back to the default for a non-numeric or zero value', async () => {
      vi.resetModules();
      localStorage.setItem(KEYS.SIDEBAR_WIDTH, 'wide');
      let mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarWidth).toBe(mod.SIDEBAR_DEFAULT_WIDTH);

      vi.resetModules();
      localStorage.setItem(KEYS.SIDEBAR_WIDTH, '0');
      mod = await import('./uiStore');
      expect(mod.useUIStore.getState().sidebarWidth).toBe(mod.SIDEBAR_DEFAULT_WIDTH);
    });
  });
});
