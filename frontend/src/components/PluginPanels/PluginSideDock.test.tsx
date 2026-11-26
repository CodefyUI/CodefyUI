/**
 * The side dock's chrome (apiVersion 6): the column, its resize handle, the
 * tab strip, the overlay on a narrow window, the toolbar toggles and the
 * Ctrl/Cmd+Shift+X shortcut. State and persistence are in `api.v6.test.ts`.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useI18n } from '../../i18n';
import { buildPluginAPI } from '../../plugins/api';
import { _clearPluginPanels, getPluginPanel } from '../../plugins/panels';
import { _resetSideDock, sidePanelWidth, useSideDockStore } from '../../plugins/sideDock';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { SidePanelToggles } from '../Toolbar/SidePanelToggles';
import { PluginSideDock } from './PluginSideDock';

function api(pluginId = 'agent') {
  return buildPluginAPI(pluginId, () => document.createElement('div'));
}

function setWidth(width: number) {
  Object.defineProperty(window, 'innerWidth', { value: width, configurable: true, writable: true });
  window.dispatchEvent(new Event('resize'));
}

function Shortcuts() {
  useKeyboardShortcuts();
  return null;
}

function Harness() {
  return (
    <div>
      <SidePanelToggles />
      <Shortcuts />
      <input data-testid="field" />
      <PluginSideDock />
    </div>
  );
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  window.localStorage.clear();
  _clearPluginPanels();
  _resetSideDock();
  setWidth(1470);
});

afterEach(() => {
  cleanup();
  _clearPluginPanels();
  _resetSideDock();
  window.localStorage.clear();
  vi.restoreAllMocks();
});

describe('PluginSideDock', () => {
  it('renders nothing until a side panel is opened, then attaches its element', () => {
    const a = api();
    const el = a.ui.addPanel({ id: 'chat', title: 'Agent', icon: 'A', dock: 'side' });
    el.textContent = 'hello from the plugin';
    render(<Harness />);
    expect(screen.queryByTestId('plugin-side-dock')).toBeNull();
    act(() => { a.ui.setPanelOpen('chat', true); });
    const dock = screen.getByTestId('plugin-side-dock');
    expect(dock.style.width).toBe('400px');
    expect(screen.getByTestId('plugin-side-panel-agent:chat').contains(el)).toBe(true);
    // One panel: no tab strip, and the host draws no title bar of its own.
    expect(screen.queryByRole('tablist')).toBeNull();
    expect(dock.textContent).toBe('hello from the plugin');
  });

  it('fires onShow and onHide on open, close and tab switch, keeping the element', () => {
    const events: string[] = [];
    const a = api();
    const el = a.ui.addPanel({
      id: 'one', title: 'One', dock: 'side',
      onShow: () => events.push('one:show'), onHide: () => events.push('one:hide'),
    });
    a.ui.addPanel({
      id: 'two', title: 'Two', dock: 'side',
      onShow: () => events.push('two:show'), onHide: () => events.push('two:hide'),
    });
    const child = document.createElement('span');
    el.appendChild(child);
    render(<Harness />);
    act(() => { a.ui.setPanelOpen('one', true); });
    fireEvent.click(screen.getByTestId('plugin-side-tab-agent:two'));
    fireEvent.click(screen.getByTestId('plugin-side-tab-agent:one'));
    act(() => { a.ui.setPanelOpen('one', false); });
    expect(events).toEqual(['one:show', 'one:hide', 'two:show', 'two:hide', 'one:show', 'one:hide']);
    expect(el.isConnected).toBe(false);
    expect(el.firstChild).toBe(child);
  });

  it('shows a tab strip with the selected tab marked when several panels exist', () => {
    const a = api();
    a.ui.addPanel({ id: 'one', title: 'One', dock: 'side' });
    api('other').ui.addPanel({ id: 'two', title: 'Two', dock: 'side' });
    render(<Harness />);
    act(() => { a.ui.setPanelOpen('one', true); });
    const tabs = screen.getAllByRole('tab');
    expect(tabs.map((t) => t.textContent)).toEqual(['One', 'Two']);
    expect(tabs[0].getAttribute('aria-selected')).toBe('true');
    fireEvent.keyDown(tabs[0], { key: 'ArrowRight' });
    expect(useSideDockStore.getState().activeKey).toBe('other:two');
  });

  it('resizes from the keyboard, by dragging, and resets on double-click', () => {
    const a = api();
    a.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    render(<Harness />);
    act(() => { a.ui.setPanelOpen('chat', true); });
    const handle = screen.getByTestId('plugin-side-resize');
    expect(handle.getAttribute('role')).toBe('separator');
    expect(handle.getAttribute('aria-valuenow')).toBe('400');

    fireEvent.keyDown(handle, { key: 'ArrowLeft' });
    expect(screen.getByTestId('plugin-side-dock').style.width).toBe('416px');
    fireEvent.keyDown(handle, { key: 'ArrowRight' });
    fireEvent.keyDown(handle, { key: 'ArrowRight' });
    expect(screen.getByTestId('plugin-side-dock').style.width).toBe('384px');
    fireEvent.keyDown(handle, { key: 'End' });
    expect(screen.getByTestId('plugin-side-dock').style.width).toBe('720px');

    fireEvent.doubleClick(handle);
    expect(screen.getByTestId('plugin-side-dock').style.width).toBe('400px');

    // Dragging the left edge to the left widens the column.
    fireEvent.mouseDown(handle, { clientX: 1000 });
    fireEvent.mouseMove(document, { clientX: 900 });
    fireEvent.mouseUp(document);
    expect(screen.getByTestId('plugin-side-dock').style.width).toBe('500px');
    expect(window.localStorage.getItem('codefyui-side-panel-width:agent:chat')).toBe('500');
    expect(document.body.style.cursor).toBe('');
  });

  it('floats over the canvas with a close button below 1100px', () => {
    const a = api();
    a.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side' });
    render(<Harness />);
    act(() => { a.ui.setPanelOpen('chat', true); });
    expect(screen.getByTestId('plugin-side-dock').dataset.overlay).toBe('false');
    expect(screen.queryByTestId('plugin-side-close')).toBeNull();
    act(() => setWidth(1000));
    expect(screen.getByTestId('plugin-side-dock').dataset.overlay).toBe('true');
    fireEvent.click(screen.getByTestId('plugin-side-close'));
    expect(screen.queryByTestId('plugin-side-dock')).toBeNull();
    expect(window.localStorage.getItem('codefyui-side-panel-open:agent:chat')).toBe('0');
  });

  it('keeps a floating column narrower than the window', () => {
    const a = api();
    a.ui.addPanel({ id: 'chat', title: 'Agent', dock: 'side', side: { defaultWidth: 700 } });
    setWidth(600);
    render(<Harness />);
    act(() => { a.ui.setPanelOpen('chat', true); });
    expect(screen.getByTestId('plugin-side-dock').style.width).toBe('552px');
    expect(sidePanelWidth(getPluginPanel('agent:chat')!)).toBe(700);
  });
});

describe('toolbar toggles', () => {
  it('one per side panel, with aria-pressed following the open panel', () => {
    const a = api();
    a.ui.addPanel({ id: 'chat', title: 'Agent', icon: 'A', dock: 'side' });
    a.ui.addPanel({ id: 'log', title: 'Log' });
    render(<Harness />);
    const toggles = screen.getByTestId('side-panel-toggles').querySelectorAll('button');
    expect(toggles).toHaveLength(1);
    const toggle = screen.getByTestId('side-panel-toggle-agent:chat');
    expect(toggle.textContent).toBe('A');
    expect(toggle.getAttribute('aria-label')).toBe('Show or hide Agent');
    expect(toggle.getAttribute('aria-pressed')).toBe('false');
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByTestId('plugin-side-dock')).toBeTruthy();
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-pressed')).toBe('false');
    expect(screen.queryByTestId('plugin-side-dock')).toBeNull();
  });

  it('shows the panel badge as a dot and in the accessible name, updated by addPanel', () => {
    const a = api();
    a.ui.addPanel({ id: 'chat', title: 'Copilot', icon: 'C', dock: 'side' });
    render(<Harness />);
    const toggle = screen.getByTestId('side-panel-toggle-agent:chat');
    expect(screen.queryByTestId('side-panel-badge-agent:chat')).toBeNull();
    expect(toggle.getAttribute('aria-label')).toBe('Show or hide Copilot');

    let el: HTMLElement | null = null;
    act(() => { el = a.ui.addPanel({ id: 'chat', title: 'Copilot', icon: 'C', dock: 'side', badge: 'attention' }); });
    expect(el).toBe(getPluginPanel('agent:chat')!.element);
    expect(screen.getByTestId('side-panel-badge-agent:chat').dataset.badge).toBe('attention');
    expect(toggle.getAttribute('aria-label')).toBe('Show or hide Copilot: needs you');

    act(() => { a.ui.addPanel({ id: 'chat', title: 'Copilot', dock: 'side', badge: 'working' }); });
    expect(toggle.getAttribute('aria-label')).toBe('Show or hide Copilot: working');
    act(() => { a.ui.addPanel({ id: 'chat', title: 'Copilot', dock: 'side', badge: 'done' }); });
    expect(screen.getByTestId('side-panel-badge-agent:chat').dataset.badge).toBe('done');
    act(() => { a.ui.addPanel({ id: 'chat', title: 'Copilot', dock: 'side', badge: null }); });
    expect(screen.queryByTestId('side-panel-badge-agent:chat')).toBeNull();
  });

  it('reads the badge in Traditional Chinese, and ignores unknown values', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const a = api();
    a.ui.addPanel({ id: 'chat', title: 'Copilot', dock: 'side', badge: 'attention' });
    render(<Harness />);
    expect(screen.getByTestId('side-panel-toggle-agent:chat').getAttribute('aria-label'))
      .toBe('顯示或隱藏Copilot：需要你處理');
    act(() => {
      a.ui.addPanel({ id: 'chat', title: 'Copilot', dock: 'side', badge: 'blinking' as never });
    });
    expect(screen.queryByTestId('side-panel-badge-agent:chat')).toBeNull();
    // A badge on another dock means nothing and is dropped.
    act(() => { a.ui.addPanel({ id: 'log', title: 'Log', badge: 'attention' }); });
    expect(getPluginPanel('agent:log')!.badge).toBeNull();
  });

  it('renders nothing without side panels', () => {
    render(<SidePanelToggles />);
    expect(screen.queryByTestId('side-panel-toggles')).toBeNull();
  });
});

describe('Ctrl/Cmd+Shift+X', () => {
  it('toggles the last used side panel, but not from a text field', () => {
    const a = api();
    a.ui.addPanel({ id: 'one', title: 'One', dock: 'side' });
    a.ui.addPanel({ id: 'two', title: 'Two', dock: 'side' });
    render(<Harness />);
    act(() => { a.ui.setPanelOpen('two', true); });
    act(() => { a.ui.setPanelOpen('two', false); });

    fireEvent.keyDown(screen.getByTestId('field'), { key: 'X', ctrlKey: true, shiftKey: true });
    expect(screen.queryByTestId('plugin-side-dock')).toBeNull();

    fireEvent.keyDown(document.body, { key: 'X', metaKey: true, shiftKey: true });
    expect(useSideDockStore.getState().activeKey).toBe('agent:two');
    fireEvent.keyDown(document.body, { key: 'x', ctrlKey: true, shiftKey: true });
    expect(useSideDockStore.getState().activeKey).toBeNull();
  });

  it('does not run auto layout, and the old Shift+A chord does nothing', () => {
    const a = api();
    a.ui.addPanel({ id: 'one', title: 'One', dock: 'side' });
    render(<Harness />);
    fireEvent.keyDown(document.body, { key: 'A', ctrlKey: true, shiftKey: true });
    expect(useSideDockStore.getState().activeKey).toBeNull();
    fireEvent.keyDown(document.body, { key: 'L', ctrlKey: true, shiftKey: true });
    expect(useSideDockStore.getState().activeKey).toBeNull();
  });

  it('leaves the key alone when no side panel exists', () => {
    render(<Harness />);
    const event = new KeyboardEvent('keydown', { key: 'X', ctrlKey: true, shiftKey: true, bubbles: true, cancelable: true });
    document.body.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });
});
