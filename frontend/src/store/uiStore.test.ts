import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useUIStore } from './uiStore';

const KEYS = {
  TOOLTIPS: 'codefyui-tooltips',
  GRIDSNAP: 'codefyui-gridsnap',
  BEGINNER: 'codefyui-beginner-mode',
  LAYOUT_MODE: 'codefyui-last-layout-mode',
  FONT_SIZE: 'codefyui-font-size',
  GLOBAL_DEVICE: 'codefyui-global-device',
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
      draggingSourceType: null,
      beginnerMode: false,
      lastLayoutMode: 'experiments',
      fontSize: 'default',
      globalDevice: 'cpu',
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
  });
});
