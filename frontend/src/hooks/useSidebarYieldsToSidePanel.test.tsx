/**
 * Review round 1, F4: with a side panel open and a node selected on a window
 * narrower than 1440px, the left node panel collapses to its rail, and only a
 * collapse made that way is undone.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { act, cleanup, render } from '@testing-library/react';
import { useTabStore } from '../store/tabStore';
import { useUIStore } from '../store/uiStore';
import { _clearPluginPanels, registerPluginPanel } from '../plugins/panels';
import { _resetSideDock, setSidePanelOpen } from '../plugins/sideDock';
import { useSidebarYieldsToSidePanel } from './useSidebarYieldsToSidePanel';

function Probe() {
  useSidebarYieldsToSidePanel();
  return null;
}

function setWidth(width: number) {
  Object.defineProperty(window, 'innerWidth', { value: width, configurable: true, writable: true });
  window.dispatchEvent(new Event('resize'));
}

const select = (id: string | null) => {
  const s = useTabStore.getState();
  useTabStore.setState({
    tabs: s.tabs.map((t) => (t.id === s.activeTabId ? { ...t, selectedNodeId: id } : t)),
  });
};
const collapsed = () => useUIStore.getState().sidebarCollapsed;

beforeEach(() => {
  window.localStorage.clear();
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('t');
  useUIStore.setState({ sidebarCollapsed: false, sidebarYielded: false, sidebarYieldDeclined: false });
  _clearPluginPanels();
  _resetSideDock();
  registerPluginPanel('agent', { id: 'chat', title: 'Agent', dock: 'side' });
  setWidth(1280);
});

afterEach(() => {
  cleanup();
  _clearPluginPanels();
  _resetSideDock();
  window.localStorage.clear();
});

describe('useSidebarYieldsToSidePanel', () => {
  it('collapses only when a side panel is open AND a node is selected below 1440px', () => {
    render(<Probe />);
    act(() => { setSidePanelOpen('agent:chat', true); });
    expect(collapsed()).toBe(false);
    act(() => select('n1'));
    expect(collapsed()).toBe(true);
    // Not persisted: a reload restores the user's own choice.
    expect(window.localStorage.getItem('codefyui-sidebar-collapsed')).toBeNull();
    act(() => select(null));
    expect(collapsed()).toBe(false);
  });

  it('does nothing at 1440px and wider', () => {
    setWidth(1440);
    render(<Probe />);
    act(() => { setSidePanelOpen('agent:chat', true); select('n1'); });
    expect(collapsed()).toBe(false);
  });

  it('gives the panel back when the side panel closes or the window widens', () => {
    render(<Probe />);
    act(() => { setSidePanelOpen('agent:chat', true); select('n1'); });
    expect(collapsed()).toBe(true);
    act(() => { setSidePanelOpen('agent:chat', false); });
    expect(collapsed()).toBe(false);
    act(() => { setSidePanelOpen('agent:chat', true); });
    expect(collapsed()).toBe(true);
    act(() => setWidth(1600));
    expect(collapsed()).toBe(false);
  });

  it('leaves a panel the user collapsed collapsed afterwards', () => {
    useUIStore.getState().setSidebarCollapsed(true);
    render(<Probe />);
    act(() => { setSidePanelOpen('agent:chat', true); select('n1'); });
    act(() => select(null));
    expect(collapsed()).toBe(true);
    expect(window.localStorage.getItem('codefyui-sidebar-collapsed')).toBe('true');
  });

  it('a panel the user reopens during the crowding stays open until it ends', () => {
    render(<Probe />);
    act(() => { setSidePanelOpen('agent:chat', true); select('n1'); });
    act(() => useUIStore.getState().toggleSidebarCollapsed());
    expect(collapsed()).toBe(false);
    act(() => select('n2'));
    act(() => setWidth(1300));
    expect(collapsed()).toBe(false);
    // A new crowding after it ended yields again.
    act(() => select(null));
    act(() => select('n3'));
    expect(collapsed()).toBe(true);
  });
});
