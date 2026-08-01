import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { SidebarRail } from './SidebarRail';
import { useUIStore, SIDEBAR_DEFAULT_WIDTH } from '../../store/uiStore';
import { useI18n } from '../../i18n';

const TAB_LABELS = ['Nodes', 'Presets', 'Templates', 'Custom & Plugins'];

function tab(name: string) {
  return screen.getByRole('tab', { name });
}

beforeEach(() => {
  localStorage.clear();
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({
    sidebarTab: 'nodes',
    sidebarCollapsed: false,
    sidebarWidth: SIDEBAR_DEFAULT_WIDTH,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe('SidebarRail', () => {
  it('renders a vertical tablist with one labelled tab per section', () => {
    render(<SidebarRail />);
    const list = screen.getByRole('tablist', { name: 'Sidebar sections' });
    expect(list.getAttribute('aria-orientation')).toBe('vertical');
    for (const label of TAB_LABELS) expect(tab(label)).toBeTruthy();
  });

  // A tablist may only own `tab` children, and the collapse toggle is not one —
  // keeping it inside would both break that and add a second Tab stop the
  // roving tabindex is supposed to prevent.
  it('keeps the collapse toggle out of the tablist', () => {
    render(<SidebarRail />);
    const list = screen.getByRole('tablist', { name: 'Sidebar sections' });
    const toggle = screen.getByRole('button', { name: 'Collapse sidebar' });

    expect(within(list).getAllByRole('tab')).toHaveLength(TAB_LABELS.length);
    expect(list.contains(toggle)).toBe(false);
    // Every element the tablist owns is a tab.
    expect(
      Array.from(list.querySelectorAll('button')).every(
        (b) => b.getAttribute('role') === 'tab',
      ),
    ).toBe(true);
    // ...and the toggle is still in the rail beside it.
    expect(list.parentElement?.contains(toggle)).toBe(true);
  });

  it('marks only the active tab as selected', () => {
    render(<SidebarRail />);
    expect(tab('Nodes').getAttribute('aria-selected')).toBe('true');
    expect(tab('Presets').getAttribute('aria-selected')).toBe('false');
  });

  it('uses a roving tabindex so the rail is a single Tab stop', () => {
    render(<SidebarRail />);
    expect(tab('Nodes').getAttribute('tabindex')).toBe('0');
    expect(tab('Presets').getAttribute('tabindex')).toBe('-1');
    expect(tab('Templates').getAttribute('tabindex')).toBe('-1');
  });

  it('clicking another tab selects it and persists the choice', () => {
    render(<SidebarRail />);
    fireEvent.click(tab('Templates'));
    expect(useUIStore.getState().sidebarTab).toBe('templates');
    expect(localStorage.getItem('codefyui-sidebar-tab')).toBe('templates');
    expect(tab('Templates').getAttribute('aria-selected')).toBe('true');
  });

  it('clicking the already-open tab collapses the sidebar', () => {
    render(<SidebarRail />);
    fireEvent.click(tab('Nodes'));
    expect(useUIStore.getState().sidebarCollapsed).toBe(true);
    expect(useUIStore.getState().sidebarTab).toBe('nodes');
  });

  it('clicking the active tab again while collapsed re-opens it', () => {
    useUIStore.setState({ sidebarCollapsed: true });
    render(<SidebarRail />);
    fireEvent.click(tab('Nodes'));
    expect(useUIStore.getState().sidebarCollapsed).toBe(false);
  });

  it('clicking a different tab while collapsed opens the sidebar onto it', () => {
    useUIStore.setState({ sidebarCollapsed: true });
    render(<SidebarRail />);
    fireEvent.click(tab('Custom & Plugins'));
    expect(useUIStore.getState().sidebarTab).toBe('custom');
    expect(useUIStore.getState().sidebarCollapsed).toBe(false);
  });

  it('drops the active highlight (but not the selection) while collapsed', () => {
    const { rerender } = render(<SidebarRail />);
    const activeClass = tab('Nodes').className;
    useUIStore.setState({ sidebarCollapsed: true });
    rerender(<SidebarRail />);
    expect(tab('Nodes').className).not.toBe(activeClass);
    expect(tab('Nodes').getAttribute('aria-selected')).toBe('true');
  });

  // Exactly one panel exists at a time, and only while expanded, so only the
  // tab that owns it may name it — any other tab would be pointing aria-controls
  // at an id that is nowhere in the document.
  it('only points aria-controls at a panel that exists', () => {
    const { rerender } = render(<SidebarRail />);
    expect(tab('Nodes').getAttribute('aria-controls')).toBe('sidebar-panel-nodes');
    for (const label of ['Presets', 'Templates', 'Custom & Plugins']) {
      expect(tab(label).getAttribute('aria-controls')).toBeNull();
    }

    // Switching tabs moves the claim along with the panel.
    fireEvent.click(tab('Presets'));
    expect(tab('Presets').getAttribute('aria-controls')).toBe('sidebar-panel-presets');
    expect(tab('Nodes').getAttribute('aria-controls')).toBeNull();

    // Collapsed: no panel at all, so no tab claims one.
    useUIStore.setState({ sidebarCollapsed: true });
    rerender(<SidebarRail />);
    for (const label of TAB_LABELS) {
      expect(tab(label).getAttribute('aria-controls')).toBeNull();
    }
  });

  // ── Keyboard navigation ────────────────────────────────────────────────────

  it('ArrowDown moves to the next tab and focuses it', () => {
    render(<SidebarRail />);
    fireEvent.keyDown(tab('Nodes'), { key: 'ArrowDown' });
    expect(useUIStore.getState().sidebarTab).toBe('presets');
    expect(document.activeElement).toBe(tab('Presets'));
  });

  it('ArrowUp wraps from the first tab to the last', () => {
    render(<SidebarRail />);
    fireEvent.keyDown(tab('Nodes'), { key: 'ArrowUp' });
    expect(useUIStore.getState().sidebarTab).toBe('custom');
    expect(document.activeElement).toBe(tab('Custom & Plugins'));
  });

  it('ArrowDown wraps from the last tab back to the first', () => {
    useUIStore.setState({ sidebarTab: 'custom' });
    render(<SidebarRail />);
    fireEvent.keyDown(tab('Custom & Plugins'), { key: 'ArrowDown' });
    expect(useUIStore.getState().sidebarTab).toBe('nodes');
  });

  it('Home and End jump to the first and last tabs', () => {
    useUIStore.setState({ sidebarTab: 'presets' });
    render(<SidebarRail />);
    fireEvent.keyDown(tab('Presets'), { key: 'End' });
    expect(useUIStore.getState().sidebarTab).toBe('custom');

    fireEvent.keyDown(tab('Custom & Plugins'), { key: 'Home' });
    expect(useUIStore.getState().sidebarTab).toBe('nodes');
  });

  it('arrow navigation expands a collapsed sidebar onto the tab it lands on', () => {
    useUIStore.setState({ sidebarCollapsed: true });
    render(<SidebarRail />);
    fireEvent.keyDown(tab('Nodes'), { key: 'ArrowDown' });
    expect(useUIStore.getState().sidebarTab).toBe('presets');
    expect(useUIStore.getState().sidebarCollapsed).toBe(false);
  });

  it('ignores unrelated keys and leaves the default behaviour alone', () => {
    render(<SidebarRail />);
    const event = fireEvent.keyDown(tab('Nodes'), { key: 'a' });
    expect(useUIStore.getState().sidebarTab).toBe('nodes');
    expect(event).toBe(true); // not prevented
  });

  it('prevents the default for keys it handles (no page scroll)', () => {
    render(<SidebarRail />);
    const notPrevented = fireEvent.keyDown(tab('Nodes'), { key: 'ArrowDown' });
    expect(notPrevented).toBe(false);
  });

  // ── Collapse toggle ────────────────────────────────────────────────────────

  it('the collapse toggle collapses and restores the sidebar', () => {
    render(<SidebarRail />);
    const collapse = screen.getByRole('button', { name: 'Collapse sidebar' });
    expect(collapse.getAttribute('aria-expanded')).toBe('true');
    fireEvent.click(collapse);
    expect(useUIStore.getState().sidebarCollapsed).toBe(true);

    const expand = screen.getByRole('button', { name: 'Expand sidebar' });
    expect(expand.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(expand);
    expect(useUIStore.getState().sidebarCollapsed).toBe(false);
  });

  it('advertises the keyboard shortcut in the toggle tooltip', () => {
    render(<SidebarRail />);
    const collapse = screen.getByRole('button', { name: 'Collapse sidebar' });
    expect(collapse.getAttribute('title')).toMatch(/Collapse sidebar \((Ctrl|Cmd)\+B\)/);
  });

  it('translates the rail for a non-English locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<SidebarRail />);
    expect(screen.getByRole('tab', { name: '節點' })).toBeTruthy();
    expect(screen.getByRole('tab', { name: '自訂與外掛' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '收合側邊欄' })).toBeTruthy();
  });
});
