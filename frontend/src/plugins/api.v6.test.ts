/**
 * The apiVersion 6 surface: the side dock (`dock: 'side'`) and
 * `ui.setPanelOpen` / `ui.isPanelOpen`, plus the state they drive.
 *
 * This file owns the exact version number (see `api.v3.test.ts:59-64` for
 * why only one file may pin it). The chrome is tested in
 * `PluginSideDock.test.tsx`; this is the contract and the persistence.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { buildPluginAPI } from './api';
import {
  _clearPluginPanels, getPluginPanel, resolveSideConfig, SIDE_PANEL_DEFAULTS,
} from './panels';
import {
  _resetSideDock, getSidePanels, isSidePanelOpen, resetSidePanelWidth,
  setSidePanelOpen, setSidePanelWidth, sidePanelWidth, toggleLastSidePanel,
  useSideDockStore,
} from './sideDock';

function freshApi(pluginId = 'agent') {
  return buildPluginAPI(pluginId, () => document.createElement('div'));
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
  useNodeDefStore.setState({ definitions: [] });
  window.localStorage.clear();
  _clearPluginPanels();
  _resetSideDock();
});

afterEach(() => {
  _clearPluginPanels();
  window.localStorage.clear();
  _resetSideDock();
});

describe('apiVersion 6', () => {
  it('reports 6 and offers setPanelOpen / isPanelOpen', () => {
    const api = freshApi();
    expect(api.apiVersion).toBe(6);
    expect(typeof api.ui.setPanelOpen).toBe('function');
    expect(typeof api.ui.isPanelOpen).toBe('function');
  });

  it('leaves every earlier ui member in place', () => {
    const api = freshApi();
    for (const member of ['addFloatingWidget', 'toast', 'addPanel', 'removePanel',
      'addToolbarButton', 'removeToolbarButton']) {
      expect(typeof (api.ui as unknown as Record<string, unknown>)[member], member).toBe('function');
    }
  });
});

describe('addPanel({ dock: "side" })', () => {
  it('registers a side panel with the default sizing, closed', () => {
    const api = freshApi();
    const el = api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    const panel = getPluginPanel('agent:chat')!;
    expect(panel.element).toBe(el);
    expect(panel.dock).toBe('side');
    expect(panel.side).toEqual(SIDE_PANEL_DEFAULTS);
    expect(api.ui.isPanelOpen('chat')).toBe(false);
  });

  it('keeps the widths consistent whatever the plugin passes', () => {
    expect(resolveSideConfig({ defaultWidth: 50, minWidth: 320, maxWidth: 200 }))
      .toEqual({ defaultWidth: 320, minWidth: 320, maxWidth: 320, defaultOpen: false });
    expect(resolveSideConfig({ defaultWidth: Number.NaN, minWidth: -5 }))
      .toEqual(SIDE_PANEL_DEFAULTS);
  });

  it('opens on first registration when defaultOpen is set', () => {
    const api = freshApi();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side', side: { defaultOpen: true } });
    expect(api.ui.isPanelOpen('chat')).toBe(true);
  });

  it('the user closing it beats defaultOpen on the next load', () => {
    const api = freshApi();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side', side: { defaultOpen: true } });
    api.ui.setPanelOpen('chat', false);
    // A reload: the registry and the dock state start over, storage stays.
    _clearPluginPanels();
    _resetSideDock();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side', side: { defaultOpen: true } });
    expect(api.ui.isPanelOpen('chat')).toBe(false);
  });

  it('reopens after a reload when it was open, and remembers its width', () => {
    const api = freshApi();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    expect(api.ui.setPanelOpen('chat', true)).toBe(true);
    setSidePanelWidth('agent:chat', 512);
    expect(window.localStorage.getItem('codefyui-side-panel-width:agent:chat')).toBe('512');
    expect(window.localStorage.getItem('codefyui-side-panel-open:agent:chat')).toBe('1');

    _clearPluginPanels();
    _resetSideDock();
    expect(useSideDockStore.getState().activeKey).toBeNull();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    expect(api.ui.isPanelOpen('chat')).toBe(true);
    expect(sidePanelWidth(getPluginPanel('agent:chat')!)).toBe(512);
  });

  it('a plugin unregistering does not forget that the user had it open', () => {
    const api = freshApi();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    api.ui.setPanelOpen('chat', true);
    api.ui.removePanel('chat');
    expect(useSideDockStore.getState().activeKey).toBeNull();
    expect(window.localStorage.getItem('codefyui-side-panel-open:agent:chat')).toBe('1');
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    expect(api.ui.isPanelOpen('chat')).toBe(true);
  });
});

describe('setPanelOpen', () => {
  it('shows one side panel at a time', () => {
    const a = freshApi('a');
    const b = freshApi('b');
    a.ui.addPanel({ id: 'p', title: 'A', dock: 'side' });
    b.ui.addPanel({ id: 'p', title: 'B', dock: 'side' });
    a.ui.setPanelOpen('p', true);
    b.ui.setPanelOpen('p', true);
    expect(a.ui.isPanelOpen('p')).toBe(false);
    expect(b.ui.isPanelOpen('p')).toBe(true);
    expect(window.localStorage.getItem('codefyui-side-panel-open:a:p')).toBe('0');
  });

  it('ignores ids that are not one of the plugin own side panels', () => {
    const api = freshApi();
    api.ui.addPanel({ id: 'bottom', title: 'Log' });
    expect(api.ui.setPanelOpen('bottom', true)).toBe(false);
    expect(api.ui.setPanelOpen('nope', true)).toBe(false);
    // Another plugin cannot open this plugin's panel by its bare id.
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    expect(freshApi('other').ui.setPanelOpen('chat', true)).toBe(false);
    expect(api.ui.isPanelOpen('chat')).toBe(false);
  });
});

describe('widths', () => {
  it('clamps to the panel range and resets to the default', () => {
    const api = freshApi();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side', side: { minWidth: 320, maxWidth: 600 } });
    const panel = getPluginPanel('agent:chat')!;
    setSidePanelWidth(panel.key, 9999);
    expect(sidePanelWidth(panel)).toBe(600);
    setSidePanelWidth(panel.key, 10);
    expect(sidePanelWidth(panel)).toBe(320);
    resetSidePanelWidth(panel.key);
    expect(sidePanelWidth(panel)).toBe(400);
    expect(window.localStorage.getItem('codefyui-side-panel-width:agent:chat')).toBeNull();
  });

  it('a stored width outside the range is clamped, not trusted', () => {
    window.localStorage.setItem('codefyui-side-panel-width:agent:chat', '5000');
    const api = freshApi();
    api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    expect(sidePanelWidth(getPluginPanel('agent:chat')!)).toBe(720);
  });
});

describe('Ctrl/Cmd+Shift+A target', () => {
  it('toggles the last used side panel, else the first one', () => {
    expect(toggleLastSidePanel()).toBe(false);
    const api = freshApi();
    api.ui.addPanel({ id: 'one', title: 'One', dock: 'side' });
    api.ui.addPanel({ id: 'two', title: 'Two', dock: 'side' });
    expect(getSidePanels().map((p) => p.key)).toEqual(['agent:one', 'agent:two']);
    expect(toggleLastSidePanel()).toBe(true);
    expect(isSidePanelOpen('agent:one')).toBe(true);
    setSidePanelOpen('agent:two', true);
    toggleLastSidePanel();
    expect(useSideDockStore.getState().activeKey).toBeNull();
    toggleLastSidePanel();
    expect(isSidePanelOpen('agent:two')).toBe(true);
  });
});

describe('storage that throws', () => {
  it('still opens and sizes panels for the page', () => {
    const original = window.localStorage;
    const broken = {
      getItem: () => { throw new Error('denied'); },
      setItem: () => { throw new Error('denied'); },
      removeItem: () => { throw new Error('denied'); },
    };
    Object.defineProperty(window, 'localStorage', { value: broken, configurable: true });
    try {
      const api = freshApi();
      api.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
      expect(api.ui.setPanelOpen('chat', true)).toBe(true);
      setSidePanelWidth('agent:chat', 450);
      expect(sidePanelWidth(getPluginPanel('agent:chat')!)).toBe(450);
    } finally {
      Object.defineProperty(window, 'localStorage', { value: original, configurable: true });
    }
  });
});
