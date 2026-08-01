import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { NodePalette } from './NodePalette';
import { useNodeDefStore } from '../../store/nodeDefStore';
import {
  useUIStore,
  SIDEBAR_DEFAULT_WIDTH,
  SIDEBAR_MAX_WIDTH,
  SIDEBAR_MIN_WIDTH,
} from '../../store/uiStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import type { NodeDefinition, PresetDefinition } from '../../types';

/*
 * The sidebar SHELL (#126): icon rail plus the panel for the open tab. The
 * per-tab behaviours live in NodesTab/PresetsTab/TemplatesTab/CustomTab tests;
 * what is asserted here is the composition — which panel is mounted, that
 * collapsing removes it, that width survives a drag, and that the node
 * library still behaves exactly as it did before the split when it is the
 * open tab.
 */

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return {
    ...actual,
    listExamples: vi.fn(),
    listCustomNodes: vi.fn(),
    listPlugins: vi.fn(),
  };
});

const mockedRest = vi.mocked(rest);

function def(node_name: string, category: string): NodeDefinition {
  return {
    node_name,
    category,
    description: `${node_name} desc`,
    inputs: [],
    outputs: [],
    params: [],
  };
}

function preset(preset_name: string, category: string): PresetDefinition {
  return {
    preset_name,
    category,
    description: `${preset_name} desc`,
    tags: ['beginner'],
    nodes: [{ id: 'a', type: 'Linear', params: {} }],
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
  };
}

function panel() {
  return document.querySelector('[role="tabpanel"]') as HTMLElement | null;
}

beforeEach(() => {
  localStorage.clear();
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({
    tooltipsEnabled: true,
    beginnerMode: false,
    sidebarTab: 'nodes',
    sidebarCollapsed: false,
    sidebarWidth: SIDEBAR_DEFAULT_WIDTH,
  });
  // Installed through setState rather than vi.spyOn: zustand's set() clones the
  // state object, so a spy would survive into a NEW object that
  // restoreAllMocks() never reaches — and the next vi.spyOn would hand back the
  // same spy with the previous test's call history still on it.
  useNodeDefStore.setState({
    definitions: [def('Conv2d', 'CNN')],
    categorized: { CNN: [def('Conv2d', 'CNN')] },
    presets: [preset('CNNBlock', 'CNN')],
    presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] },
    loading: false,
    error: null,
    fetchDefinitions: vi.fn().mockResolvedValue(undefined),
  });
  // vi.fn()s from the module factory keep their call history across tests;
  // reset them so "was this tab fetched?" means "in THIS test".
  mockedRest.listExamples.mockReset().mockResolvedValue([]);
  mockedRest.listCustomNodes.mockReset().mockResolvedValue([]);
  mockedRest.listPlugins.mockReset().mockResolvedValue([]);
});

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
});

describe('NodePalette (sidebar shell)', () => {
  it('opens on the Nodes tab, with the node library behaving as before', () => {
    render(<NodePalette />);
    expect(screen.getByText('Nodes')).toBeTruthy();
    expect(screen.getByPlaceholderText('Search nodes...')).toBeTruthy();
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.getByText('Drag nodes onto the canvas')).toBeTruthy();

    // Drag-to-canvas payload is unchanged.
    const item = screen.getByText('Conv2d').closest('div')!.parentElement!;
    const setData = vi.fn();
    fireEvent.dragStart(item, { dataTransfer: { setData, effectAllowed: '' } });
    expect(setData).toHaveBeenCalledWith('application/codefyui-node', 'Conv2d');

    // Search still filters.
    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'zzz' },
    });
    expect(screen.getByText('No matching nodes')).toBeTruthy();
  });

  it('the rail always renders, collapsed or not', () => {
    const { rerender } = render(<NodePalette />);
    expect(screen.getByRole('tablist', { name: 'Sidebar sections' })).toBeTruthy();
    act(() => useUIStore.getState().setSidebarCollapsed(true));
    rerender(<NodePalette />);
    expect(screen.getByRole('tablist', { name: 'Sidebar sections' })).toBeTruthy();
  });

  it('mounts only the open tab, and swaps panels from the rail', async () => {
    render(<NodePalette />);
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.queryByText('CNNBlock')).toBeNull();

    fireEvent.click(screen.getByRole('tab', { name: 'Presets' }));
    expect(screen.getByText('CNNBlock')).toBeTruthy();
    expect(screen.queryByText('Conv2d')).toBeNull();
    expect(screen.getByPlaceholderText('Search presets...')).toBeTruthy();

    fireEvent.click(screen.getByRole('tab', { name: 'Templates' }));
    expect(await screen.findByText('No examples available')).toBeTruthy();

    fireEvent.click(screen.getByRole('tab', { name: 'Custom & Plugins' }));
    expect(await screen.findByText('No custom nodes yet')).toBeTruthy();

    fireEvent.click(screen.getByRole('tab', { name: 'Nodes' }));
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('a tab is only fetched when it is opened', () => {
    render(<NodePalette />);
    expect(mockedRest.listExamples).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('tab', { name: 'Templates' }));
    expect(mockedRest.listExamples).toHaveBeenCalledTimes(1);
  });

  // ── Catalog bootstrap ──────────────────────────────────────────────────────
  // The node/preset catalog is the whole app's, not the Nodes tab's: the canvas,
  // quick search, example loading and the plugin host all read it. The SHELL
  // must start the load, because the Nodes tab mounts only when it is the open
  // tab AND the sidebar is expanded — and both of those are persisted, so a
  // user who quit on another tab (or collapsed) would otherwise come back to an
  // app with an empty catalog for the whole session.

  it('starts the catalog load even when the sidebar is collapsed', () => {
    useNodeDefStore.setState({ definitions: [], categorized: {}, presets: [], presetCategorized: {} });
    const fetchDefinitions = useNodeDefStore.getState().fetchDefinitions as ReturnType<typeof vi.fn>;
    act(() => useUIStore.getState().setSidebarCollapsed(true));

    render(<NodePalette />);

    expect(panel()).toBeNull();
    expect(fetchDefinitions).toHaveBeenCalledTimes(1);
  });

  it('starts the catalog load when a tab other than Nodes is open', () => {
    useNodeDefStore.setState({ definitions: [], categorized: {}, presets: [], presetCategorized: {} });
    const fetchDefinitions = useNodeDefStore.getState().fetchDefinitions as ReturnType<typeof vi.fn>;
    act(() => useUIStore.getState().setSidebarTab('presets'));

    render(<NodePalette />);

    expect(screen.getByPlaceholderText('Search presets...')).toBeTruthy();
    expect(fetchDefinitions).toHaveBeenCalledTimes(1);
  });

  it('does not re-fetch the catalog when it is already loaded', () => {
    // The beforeEach seeds a non-empty catalog.
    const fetchDefinitions = useNodeDefStore.getState().fetchDefinitions as ReturnType<typeof vi.fn>;
    render(<NodePalette />);
    expect(fetchDefinitions).not.toHaveBeenCalled();
  });

  it('labels the panel with the tab that opened it', () => {
    render(<NodePalette />);
    expect(panel()?.id).toBe('sidebar-panel-nodes');
    expect(panel()?.getAttribute('aria-labelledby')).toBe('sidebar-tab-nodes');

    fireEvent.click(screen.getByRole('tab', { name: 'Presets' }));
    expect(panel()?.id).toBe('sidebar-panel-presets');
    expect(panel()?.getAttribute('aria-labelledby')).toBe('sidebar-tab-presets');
  });

  // ── Collapse ───────────────────────────────────────────────────────────────

  it('collapsing removes the panel from the DOM so the canvas gets the width', () => {
    render(<NodePalette />);
    expect(panel()).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Collapse sidebar' }));
    expect(panel()).toBeNull();
    expect(screen.queryByText('Conv2d')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Expand sidebar' }));
    expect(panel()).toBeTruthy();
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('exposes the collapsed state on the shell for styling and e2e', () => {
    const { container } = render(<NodePalette />);
    const shell = container.firstElementChild as HTMLElement;
    expect(shell.dataset.collapsed).toBe('false');
    fireEvent.click(screen.getByRole('button', { name: 'Collapse sidebar' }));
    expect(shell.dataset.collapsed).toBe('true');
  });

  it('restores the persisted tab, width and collapsed state', () => {
    // What a reload looks like: the store is rebuilt from localStorage before
    // the component mounts.
    act(() => {
      useUIStore.getState().setSidebarTab('presets');
      useUIStore.getState().setSidebarWidth(310);
      useUIStore.getState().setSidebarCollapsed(false);
    });
    expect(localStorage.getItem('codefyui-sidebar-tab')).toBe('presets');
    expect(localStorage.getItem('codefyui-sidebar-width')).toBe('310');
    expect(localStorage.getItem('codefyui-sidebar-collapsed')).toBe('false');

    render(<NodePalette />);
    expect(screen.getByText('CNNBlock')).toBeTruthy();
    expect(panel()?.style.width).toBe('310px');
  });

  // ── Resize ────────────────────────────────────────────────────────────────

  it('drags the resize handle to a new width and persists it', () => {
    const { container } = render(<NodePalette />);
    const handle = container.querySelector('[role="separator"]') as HTMLElement;
    expect(handle.getAttribute('aria-label')).toBe('Resize sidebar');

    fireEvent.mouseDown(handle, { clientX: 250 });
    expect(document.body.style.cursor).toBe('col-resize');

    fireEvent.mouseMove(document, { clientX: 310 });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_DEFAULT_WIDTH + 60);
    expect(panel()?.style.width).toBe(`${SIDEBAR_DEFAULT_WIDTH + 60}px`);

    fireEvent.mouseUp(document);
    expect(document.body.style.cursor).toBe('');
    expect(localStorage.getItem('codefyui-sidebar-width')).toBe(
      String(SIDEBAR_DEFAULT_WIDTH + 60),
    );

    // The listeners are gone: a stray move after the drag changes nothing.
    fireEvent.mouseMove(document, { clientX: 900 });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_DEFAULT_WIDTH + 60);
  });

  it('clamps a drag that would push the panel out of its usable range', () => {
    const { container } = render(<NodePalette />);
    const handle = container.querySelector('[role="separator"]') as HTMLElement;

    fireEvent.mouseDown(handle, { clientX: 250 });
    fireEvent.mouseMove(document, { clientX: -900 });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_MIN_WIDTH);

    fireEvent.mouseMove(document, { clientX: 4000 });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_MAX_WIDTH);
    fireEvent.mouseUp(document);
  });

  it('has no resize handle while collapsed', () => {
    act(() => useUIStore.getState().setSidebarCollapsed(true));
    const { container } = render(<NodePalette />);
    expect(container.querySelector('[role="separator"]')).toBeNull();
  });

  it('tears down a drag left in flight by an unmount', () => {
    const { container, unmount } = render(<NodePalette />);
    const handle = container.querySelector('[role="separator"]') as HTMLElement;
    fireEvent.mouseDown(handle, { clientX: 250 });
    expect(document.body.style.cursor).toBe('col-resize');

    // Unmounting mid-drag must not leave the page stuck under a col-resize
    // cursor with listeners still on `document`.
    unmount();
    expect(document.body.style.cursor).toBe('');
    expect(document.body.style.userSelect).toBe('');
    fireEvent.mouseMove(document, { clientX: 900 });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_DEFAULT_WIDTH);
  });

  // A focusable role="separator" has to be operable from the keyboard, per the
  // window-splitter pattern.
  it('is a keyboard-operable splitter', () => {
    const { container } = render(<NodePalette />);
    const handle = container.querySelector('[role="separator"]') as HTMLElement;
    expect(handle.getAttribute('tabindex')).toBe('0');
    expect(handle.getAttribute('aria-valuenow')).toBe(String(SIDEBAR_DEFAULT_WIDTH));
    expect(handle.getAttribute('aria-valuemin')).toBe(String(SIDEBAR_MIN_WIDTH));
    expect(handle.getAttribute('aria-valuemax')).toBe(String(SIDEBAR_MAX_WIDTH));

    fireEvent.keyDown(handle, { key: 'ArrowRight' });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_DEFAULT_WIDTH + 16);
    expect(handle.getAttribute('aria-valuenow')).toBe(String(SIDEBAR_DEFAULT_WIDTH + 16));

    fireEvent.keyDown(handle, { key: 'ArrowLeft' });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_DEFAULT_WIDTH);

    fireEvent.keyDown(handle, { key: 'Home' });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_MIN_WIDTH);

    fireEvent.keyDown(handle, { key: 'End' });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_MAX_WIDTH);

    // An unrelated key is left to the browser.
    const notPrevented = fireEvent.keyDown(handle, { key: 'a' });
    expect(useUIStore.getState().sidebarWidth).toBe(SIDEBAR_MAX_WIDTH);
    expect(notPrevented).toBe(true);
  });

  // ── Focus handoff on collapse ──────────────────────────────────────────────

  it('moves focus to the rail when collapsing away from a focused panel', () => {
    render(<NodePalette />);
    const categoryHeader = screen.getByText('CNN').closest('button')!;
    act(() => categoryHeader.focus());
    expect(document.activeElement).toBe(categoryHeader);

    act(() => useUIStore.getState().toggleSidebarCollapsed());

    // The panel that held focus is gone; focus landed on the open tab rather
    // than falling back to <body>.
    expect(panel()).toBeNull();
    expect(document.activeElement).toBe(screen.getByRole('tab', { name: 'Nodes' }));
  });

  it('leaves focus alone when collapsing from outside the panel', () => {
    render(<NodePalette />);
    const outside = document.createElement('button');
    document.body.appendChild(outside);
    act(() => outside.focus());

    act(() => useUIStore.getState().toggleSidebarCollapsed());

    // Ctrl+B while working on the canvas must not yank focus into the sidebar.
    expect(document.activeElement).toBe(outside);
    outside.remove();
  });

  it('leaves focus alone when it had moved out of the panel before collapsing', () => {
    render(<NodePalette />);
    const outside = document.createElement('button');
    document.body.appendChild(outside);

    // Focus goes into the panel and then back out — the handoff must follow
    // where focus IS, not where it once was, or Ctrl+B from the canvas would
    // yank focus to the rail and turn arrow keys into tab switches.
    act(() => screen.getByText('CNN').closest('button')!.focus());
    act(() => outside.focus());

    act(() => useUIStore.getState().toggleSidebarCollapsed());

    expect(document.activeElement).toBe(outside);
    outside.remove();
  });

  it('does not grab focus when it mounts already collapsed', () => {
    act(() => useUIStore.getState().setSidebarCollapsed(true));
    render(<NodePalette />);
    expect(document.activeElement).toBe(document.body);
  });

  // ── Keyboard shortcut integration ──────────────────────────────────────────

  it('reflects a Ctrl+B store toggle without a rail click', () => {
    render(<NodePalette />);
    expect(panel()).toBeTruthy();
    act(() => useUIStore.getState().toggleSidebarCollapsed());
    expect(panel()).toBeNull();
    act(() => useUIStore.getState().toggleSidebarCollapsed());
    expect(panel()).toBeTruthy();
  });
});
