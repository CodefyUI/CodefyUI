import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TabBar } from './TabBar';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { useDialogStore } from '../../store/dialogStore';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Reset the tab store to a single fresh tab named `Tab 1`. */
function resetToSingleTab() {
  useTabStore.setState({
    tabs: [],
    activeTabId: null as unknown as string,
    clipboard: null,
  });
  useTabStore.getState().addTab('Tab 1');
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useDialogStore.setState({ active: null, resolve: null });
  resetToSingleTab();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('TabBar', () => {
  it('renders the active tab name and the add button', () => {
    render(<TabBar />);
    expect(screen.getByText('Tab 1')).toBeTruthy();
    // Add button uses title from i18n.
    expect(screen.getByTitle('New tab')).toBeTruthy();
  });

  it('does not render a close button with only a single tab (close guard)', () => {
    render(<TabBar />);
    expect(screen.queryByText('×')).toBeNull();
  });

  it('adds a new tab when clicking the add button', () => {
    render(<TabBar />);
    fireEvent.click(screen.getByTitle('New tab'));
    expect(useTabStore.getState().tabs).toHaveLength(2);
    // Second tab is the new active tab.
    expect(useTabStore.getState().activeTabId).toBe(
      useTabStore.getState().tabs[1].id,
    );
  });

  it('selects a tab when clicking on it', () => {
    useTabStore.getState().addTab('Tab 2');
    const firstId = useTabStore.getState().tabs[0].id;
    render(<TabBar />);
    // Tab 2 is active after addTab; click Tab 1 to switch.
    fireEvent.click(screen.getByText('Tab 1'));
    expect(useTabStore.getState().activeTabId).toBe(firstId);
  });

  it('shows close buttons when there are 2+ tabs and closes a tab', () => {
    useTabStore.getState().addTab('Tab 2');
    render(<TabBar />);
    const closeButtons = screen.getAllByText('×');
    expect(closeButtons).toHaveLength(2);
    fireEvent.click(closeButtons[0]);
    expect(useTabStore.getState().tabs).toHaveLength(1);
  });

  it('applies active styling (font weight 600) to the active tab', () => {
    useTabStore.getState().addTab('Tab 2');
    render(<TabBar />);
    const activeTabEl = screen.getByText('Tab 2').closest('div')!;
    expect(activeTabEl.style.fontWeight).toBe('600');
    const inactiveTabEl = screen.getByText('Tab 1').closest('div')!;
    expect(inactiveTabEl.style.fontWeight).toBe('400');
  });

  it('shows a running indicator dot for tabs with running status', () => {
    useTabStore.getState().addTab('Tab 2');
    const tabs = useTabStore.getState().tabs;
    useTabStore.setState({
      tabs: tabs.map((t, i) => (i === 0 ? { ...t, status: 'running' } : t)),
    });
    const { container } = render(<TabBar />);
    // The running dot is a span sibling without text. jsdom keeps the hex in
    // the box-shadow shorthand (only `background`/`color` longhands normalize).
    const dots = Array.from(container.querySelectorAll('span')).filter((s) =>
      s.style.boxShadow.includes('#FFC107'),
    );
    expect(dots.length).toBe(1);
  });

  // ── Rename flow ────────────────────────────────────────────────────────────

  it('double-clicking a tab enters edit mode with the current name', () => {
    render(<TabBar />);
    fireEvent.doubleClick(screen.getByText('Tab 1'));
    const input = screen.getByDisplayValue('Tab 1') as HTMLInputElement;
    expect(input).toBeTruthy();
  });

  it('commits a rename on Enter', () => {
    render(<TabBar />);
    const id = useTabStore.getState().tabs[0].id;
    fireEvent.doubleClick(screen.getByText('Tab 1'));
    const input = screen.getByDisplayValue('Tab 1');
    fireEvent.change(input, { target: { value: 'Renamed' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(useTabStore.getState().tabs.find((t) => t.id === id)?.name).toBe(
      'Renamed',
    );
  });

  it('cancels a rename on Escape (keeps the original name)', () => {
    render(<TabBar />);
    const id = useTabStore.getState().tabs[0].id;
    fireEvent.doubleClick(screen.getByText('Tab 1'));
    const input = screen.getByDisplayValue('Tab 1');
    fireEvent.change(input, { target: { value: 'Discarded' } });
    fireEvent.keyDown(input, { key: 'Escape' });
    expect(useTabStore.getState().tabs.find((t) => t.id === id)?.name).toBe(
      'Tab 1',
    );
    // Back to display mode.
    expect(screen.getByText('Tab 1')).toBeTruthy();
  });

  it('commits a rename on blur', () => {
    render(<TabBar />);
    const id = useTabStore.getState().tabs[0].id;
    fireEvent.doubleClick(screen.getByText('Tab 1'));
    const input = screen.getByDisplayValue('Tab 1');
    fireEvent.change(input, { target: { value: 'BlurName' } });
    fireEvent.blur(input);
    expect(useTabStore.getState().tabs.find((t) => t.id === id)?.name).toBe(
      'BlurName',
    );
  });

  it('does not rename to a blank/whitespace name on commit', () => {
    render(<TabBar />);
    const id = useTabStore.getState().tabs[0].id;
    fireEvent.doubleClick(screen.getByText('Tab 1'));
    const input = screen.getByDisplayValue('Tab 1');
    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    // Name unchanged; editing closes.
    expect(useTabStore.getState().tabs.find((t) => t.id === id)?.name).toBe(
      'Tab 1',
    );
  });

  it('ignores unrelated keys while editing (no Enter/Escape)', () => {
    render(<TabBar />);
    fireEvent.doubleClick(screen.getByText('Tab 1'));
    const input = screen.getByDisplayValue('Tab 1');
    fireEvent.keyDown(input, { key: 'a' });
    // Still editing.
    expect(screen.getByDisplayValue('Tab 1')).toBeTruthy();
  });

  it('stops propagation when clicking inside the edit input (does not switch tab)', () => {
    useTabStore.getState().addTab('Tab 2');
    const firstId = useTabStore.getState().tabs[0].id;
    render(<TabBar />);
    // Edit the active (Tab 2) tab.
    fireEvent.doubleClick(screen.getByText('Tab 2'));
    const input = screen.getByDisplayValue('Tab 2');
    fireEvent.click(input);
    // activeTabId unchanged (still Tab 2, not the first one).
    expect(useTabStore.getState().activeTabId).not.toBe(firstId);
  });

  // ── Close guards & running confirm ──────────────────────────────────────────

  it('close guard: does nothing when only one tab remains even if click fires', () => {
    // With a single tab there is no close button, but exercise handleClose via
    // a second tab then closing down to one and asserting it stops.
    useTabStore.getState().addTab('Tab 2');
    render(<TabBar />);
    const closeButtons = screen.getAllByText('×');
    fireEvent.click(closeButtons[0]);
    expect(useTabStore.getState().tabs).toHaveLength(1);
    // No more close buttons.
    expect(screen.queryByText('×')).toBeNull();
  });

  it('closing a running tab asks for confirmation and removes it when confirmed', async () => {
    // Need the DialogContainer-like resolution: confirm() opens the dialog
    // store; drive it directly by resolving via close(true).
    useTabStore.getState().addTab('Tab 2');
    const tabs = useTabStore.getState().tabs;
    const runningId = tabs[0].id;
    useTabStore.setState({
      tabs: tabs.map((t) => (t.id === runningId ? { ...t, status: 'running' } : t)),
    });
    render(<TabBar />);
    const closeButtons = screen.getAllByText('×');
    fireEvent.click(closeButtons[0]);
    // A confirm dialog is now active.
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    expect(useDialogStore.getState().active?.title).toBe(
      'This tab is still running. Close it anyway?',
    );
    // Confirm → tab removed.
    useDialogStore.getState().close(true);
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(1);
    });
  });

  it('closing a running tab keeps it when the confirm is cancelled', async () => {
    useTabStore.getState().addTab('Tab 2');
    const tabs = useTabStore.getState().tabs;
    const runningId = tabs[0].id;
    useTabStore.setState({
      tabs: tabs.map((t) => (t.id === runningId ? { ...t, status: 'running' } : t)),
    });
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    useDialogStore.getState().close(false);
    // Tab still present after cancel.
    await waitFor(() => {
      expect(useDialogStore.getState().active).toBeNull();
    });
    expect(useTabStore.getState().tabs).toHaveLength(2);
  });

  // ── Close confirm for a tab with a graph in it (#331) ──────────────────────

  /**
   * Give the tab at `index` some content without going through the canvas:
   * `tabHasContent` only reads nodes/edges/subgraphs, so two nodes is enough
   * to make it the "would lose work" case.
   */
  function fillTab(index: number) {
    const tabs = useTabStore.getState().tabs;
    useTabStore.setState({
      tabs: tabs.map((t, i) =>
        i === index
          ? {
              ...t,
              nodes: [
                { id: 'n1', type: 'default', position: { x: 0, y: 0 }, data: { type: 'Dataset', params: {} } },
                { id: 'n2', type: 'default', position: { x: 90, y: 0 }, data: { type: 'Model', params: {} } },
              ] as never,
            }
          : t,
      ),
    });
  }

  it('closing an EMPTY tab does not ask (closes immediately)', () => {
    useTabStore.getState().addTab('Tab 2');
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    expect(useDialogStore.getState().active).toBeNull();
    expect(useTabStore.getState().tabs).toHaveLength(1);
  });

  it('closing a tab holding a graph asks first, with the tab name and node count', async () => {
    useTabStore.getState().addTab('Tab 2');
    fillTab(0);
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    const active = useDialogStore.getState().active!;
    expect(active.kind).toBe('confirm');
    expect(active.title).toBe('Close "Tab 1"?');
    expect(active.message).toContain('2 nodes');
    // Destructive action, so the danger styling the overwrite confirm uses.
    expect(active.kind === 'confirm' && active.variant).toBe('danger');
    expect(active.confirmText).toBe('Close tab');
    // Nothing removed while the dialog is up.
    expect(useTabStore.getState().tabs).toHaveLength(2);
  });

  it('cancelling the confirm keeps the tab and everything in it', async () => {
    useTabStore.getState().addTab('Tab 2');
    fillTab(0);
    const before = useTabStore.getState().tabs[0];
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    useDialogStore.getState().close(false);
    await waitFor(() => {
      expect(useDialogStore.getState().active).toBeNull();
    });
    const after = useTabStore.getState().tabs[0];
    expect(useTabStore.getState().tabs).toHaveLength(2);
    // Same tab object: nodes, file binding and undo stacks all untouched.
    expect(after).toBe(before);
    expect(after.nodes).toHaveLength(2);
    expect(screen.getByText('Tab 1')).toBeTruthy();
  });

  it('confirming the dialog removes the tab', async () => {
    useTabStore.getState().addTab('Tab 2');
    fillTab(0);
    const firstId = useTabStore.getState().tabs[0].id;
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    useDialogStore.getState().close(true);
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(1);
    });
    expect(useTabStore.getState().tabs.find((t) => t.id === firstId)).toBeUndefined();
  });

  it('a tab bound to a saved file still asks: dirty tracking cannot prove it is unchanged', async () => {
    // `dirtyNodeIds` is the partial-re-execution hint -- cleared at the start
    // of every run and never set by addNode -- so "clean and bound to a file"
    // does not mean "identical to what is on disk". Asking anyway is the only
    // answer that cannot silently discard work (#331).
    useTabStore.getState().addTab('Tab 2');
    fillTab(0);
    const tabs = useTabStore.getState().tabs;
    useTabStore.setState({
      tabs: tabs.map((t, i) =>
        i === 0 ? { ...t, currentGraphFile: 'saved-graph', dirtyNodeIds: new Set<string>() } : t,
      ),
    });
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    expect(useTabStore.getState().tabs).toHaveLength(2);
  });

  it('a running tab with a graph gets ONE dialog carrying both warnings', async () => {
    useTabStore.getState().addTab('Tab 2');
    fillTab(0);
    const tabs = useTabStore.getState().tabs;
    useTabStore.setState({
      tabs: tabs.map((t, i) => (i === 0 ? { ...t, status: 'running' as const } : t)),
    });
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    const active = useDialogStore.getState().active!;
    expect(active.title).toBe('This tab is still running. Close it anyway?');
    expect(active.message).toContain('2 nodes');
    useDialogStore.getState().close(true);
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(1);
    });
    // Only the one dialog: nothing re-opened behind it.
    expect(useDialogStore.getState().active).toBeNull();
  });

  it('uses the zh-TW copy when that locale is active', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    useTabStore.getState().addTab('Tab 2');
    fillTab(0);
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    const active = useDialogStore.getState().active!;
    expect(active.title).toBe('要關閉「Tab 1」嗎？');
    expect(active.confirmText).toBe('關閉分頁');
    expect(active.message).toContain('2 個節點');
  });

  it('clicking the close button stops propagation (does not activate the tab)', () => {
    useTabStore.getState().addTab('Tab 2');
    const firstId = useTabStore.getState().tabs[0].id;
    // Make sure active is the second tab.
    render(<TabBar />);
    const tab1El = screen.getByText('Tab 1').closest('div')!;
    const closeBtn = within(tab1El).getByText('×');
    fireEvent.click(closeBtn);
    // Tab 1 was removed; active should not have become Tab 1.
    expect(useTabStore.getState().tabs.find((t) => t.id === firstId)).toBeUndefined();
  });

  // ── Read-only badge and plugin provenance (#341) ───────────────────────────

  /** Mark the tab at `index` read-only and, optionally, plugin-opened. */
  function markTab(index: number, patch: Record<string, unknown>) {
    const tabs = useTabStore.getState().tabs;
    useTabStore.setState({
      tabs: tabs.map((t, i) => (i === index ? { ...t, ...patch } : t)),
    });
  }

  it('shows no badge on an ordinary tab', () => {
    render(<TabBar />);
    expect(screen.queryByText('Read-only')).toBeNull();
  });

  it('badges a read-only tab', () => {
    markTab(0, { readOnly: true });
    render(<TabBar />);
    expect(screen.getByText('Read-only')).toBeTruthy();
  });

  it('names the plugin that opened a tab in its tooltip', () => {
    markTab(0, { source: { kind: 'agent-variant', pluginId: 'graph-copilot' } });
    render(<TabBar />);
    expect(screen.getByTitle('Opened by graph-copilot')).toBeTruthy();
  });

  it('carries no tooltip on a tab the user opened', () => {
    render(<TabBar />);
    // The add button is the only titled control on a plain tab bar.
    expect(screen.queryByTitle(/Opened by/)).toBeNull();
  });

  it('shows both at once without hiding the tab name', () => {
    markTab(0, {
      readOnly: true,
      source: { kind: 'agent-variant', pluginId: 'graph-copilot' },
    });
    render(<TabBar />);
    const tab = screen.getByTitle('Opened by graph-copilot');
    expect(within(tab).getByText('Read-only')).toBeTruthy();
    expect(within(tab).getByText('Tab 1')).toBeTruthy();
  });

  it('uses the zh-TW copy for both', () => {
    useI18n.setState({ locale: 'zh-TW' });
    markTab(0, {
      readOnly: true,
      source: { kind: 'agent-variant', pluginId: 'graph-copilot' },
    });
    render(<TabBar />);
    expect(screen.getByText('唯讀')).toBeTruthy();
    expect(screen.getByTitle('由 graph-copilot 開啟')).toBeTruthy();
  });

  it('renaming a read-only tab still works -- the badge is chrome, not a lock', () => {
    // `readOnly` gates SAVING a newer-format file, and plugin writes. The tab
    // label is the user's own, and nothing here should take it away.
    markTab(0, { readOnly: true });
    render(<TabBar />);
    const id = useTabStore.getState().tabs[0].id;
    fireEvent.doubleClick(screen.getByText('Tab 1'));
    const input = screen.getByDisplayValue('Tab 1');
    fireEvent.change(input, { target: { value: 'Candidate A' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(useTabStore.getState().tabs.find((t) => t.id === id)?.name).toBe('Candidate A');
  });
});
