import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TabBar } from './TabBar';
import { DialogContainer } from '../shared/DialogContainer';
import { NO_ACTIVE_TAB, useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { useDialogStore } from '../../store/dialogStore';
import { useUIStore } from '../../store/uiStore';

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

  it('renders a close button on the only tab', () => {
    // Every tab is closable now, the last one included: closing it leaves the
    // workspace with no tab open, which is the welcome screen. Hiding the ×
    // here used to make a single leftover tab permanent.
    render(<TabBar />);
    expect(screen.getByText('×')).toBeTruthy();
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

  it('closes down to one tab, then closes that one too', () => {
    useTabStore.getState().addTab('Tab 2');
    render(<TabBar />);
    fireEvent.click(screen.getAllByText('×')[0]);
    expect(useTabStore.getState().tabs).toHaveLength(1);

    // The survivor keeps its own × -- that is the whole change. Both tabs are
    // empty here, so neither close asks for confirmation.
    fireEvent.click(screen.getByText('×'));
    expect(useTabStore.getState().tabs).toEqual([]);
    expect(useTabStore.getState().activeTabId).toBe('');
  });

  it('still asks before closing the last tab when it holds a graph', async () => {
    // The content confirm is what stops an accidental click from costing a
    // graph, and it is the ONLY thing standing between the × and an empty
    // workspace now -- so it has to fire on the last tab, not just on the
    // ones with a neighbour to fall back to.
    const only = useTabStore.getState().tabs[0];
    useTabStore.setState({
      tabs: [{ ...only, nodes: [{ id: 'n1' } as never] }],
    });
    render(<TabBar />);
    fireEvent.click(screen.getByText('×'));

    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    expect(useTabStore.getState().tabs).toHaveLength(1);
    useDialogStore.getState().resolve?.(true);
    await waitFor(() => expect(useTabStore.getState().tabs).toEqual([]));
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

  // ── Keyboard and screen readers (#402) ─────────────────────────────────────

  describe('keyboard and screen readers (#402)', () => {
    // Everything that reaches past the strip. React Flow deletes the canvas
    // selection on a Delete that gets to `document` and arms panning on a
    // Space that gets to `window`; the shortcut hook opens the selected
    // node's details on an Enter that gets to `document`.
    const onDocument = vi.fn();
    const onWindow = vi.fn();

    beforeEach(() => {
      onDocument.mockClear();
      onWindow.mockClear();
      document.addEventListener('keydown', onDocument);
      window.addEventListener('keydown', onWindow);
    });

    afterEach(() => {
      document.removeEventListener('keydown', onDocument);
      window.removeEventListener('keydown', onWindow);
    });

    /** A tab found the way a screen reader finds it: by role and name. */
    function tabNamed(name: string) {
      return screen.getByRole('tab', { name });
    }

    /** `Tab 1` to `Tab 3` with the FIRST one active; returns their ids. */
    function openThreeTabs() {
      useTabStore.getState().addTab('Tab 2');
      useTabStore.getState().addTab('Tab 3');
      const ids = useTabStore.getState().tabs.map((t) => t.id);
      useTabStore.getState().setActiveTab(ids[0]);
      return ids;
    }

    /**
     * A mouse press the way a browser runs it. jsdom dispatches the event but
     * skips its default action -- moving focus to the pressed element, or to
     * its nearest focusable ancestor -- so that step is done here, and only
     * when nothing prevented it. `focusTarget` is what the browser would
     * focus: the pressed element in Chromium; the tab around a pressed button
     * in Safari and macOS Firefox, which never focus a clicked button. A real
     * press reports its click count, `detail: 1`; one with `detail` 0 is what
     * a screen reader or a script sends. Returns whether the default was
     * allowed.
     */
    function press(el: HTMLElement, focusTarget: HTMLElement = el, init: MouseEventInit = {}) {
      const allowed = fireEvent.mouseDown(el, init);
      if (allowed) focusTarget.focus();
      return allowed;
    }

    /** A whole left click on one element: press, release, click. */
    function pointerClick(el: HTMLElement, count = 1, focusTarget: HTMLElement = el) {
      press(el, focusTarget, { detail: count });
      fireEvent.mouseUp(el, { detail: count });
      fireEvent.click(el, { detail: count });
    }

    it('is a named tablist whose tabs say which one is selected', () => {
      useTabStore.getState().addTab('Tab 2');
      const [firstId, secondId] = useTabStore.getState().tabs.map((t) => t.id);
      render(<TabBar />);

      const list = screen.getByRole('tablist', { name: 'Workspace tabs' });
      expect(within(list).getAllByRole('tab')).toHaveLength(2);
      expect(tabNamed('Tab 2').getAttribute('aria-selected')).toBe('true');
      expect(tabNamed('Tab 1').getAttribute('aria-selected')).toBe('false');
      // From the tab's own id, not its position, so it survives a close.
      expect(tabNamed('Tab 1').id).toBe(`workspace-tab-${firstId}`);
      expect(tabNamed('Tab 2').id).toBe(`workspace-tab-${secondId}`);
      // A tablist may only own tabs, so + sits beside it rather than in it.
      expect(list.contains(screen.getByRole('button', { name: 'New tab' }))).toBe(false);
    });

    it('names the + button, which is where focus goes when the last tab closes', () => {
      // Named by its content it would be "+"; the tooltip's words are its name.
      render(<TabBar />);
      expect(screen.getByRole('button', { name: 'New tab' })).toBe(screen.getByTitle('New tab'));
    });

    it('puts only the active tab in the Tab order', () => {
      openThreeTabs();
      render(<TabBar />);
      expect(tabNamed('Tab 1').tabIndex).toBe(0);
      expect(tabNamed('Tab 2').tabIndex).toBe(-1);
      expect(tabNamed('Tab 3').tabIndex).toBe(-1);
    });

    it('makes the first tab the Tab stop when no tab is active', () => {
      // A plugin can open tabs without activating any (`activate: 'none'`),
      // which leaves the welcome screen up with tabs in the strip. A strip
      // with no tab at 0 could not be reached at all, and a plugin's tab has
      // no other way in: the Graphs panel raises tabs by file.
      useTabStore.getState().addTab('Tab 2');
      useTabStore.setState({ activeTabId: NO_ACTIVE_TAB });
      render(<TabBar />);
      expect(tabNamed('Tab 1').tabIndex).toBe(0);
      expect(tabNamed('Tab 2').tabIndex).toBe(-1);
      expect(screen.queryByRole('tab', { selected: true })).toBeNull();
    });

    it('ArrowRight and ArrowLeft move focus to the neighbouring tab, wrapping at both ends', () => {
      const ids = openThreeTabs();
      render(<TabBar />);

      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'ArrowRight' });
      expect(document.activeElement).toBe(tabNamed('Tab 2'));
      // Focus only: switching a workspace re-renders the whole canvas, so the
      // arrows do not switch (manual activation).
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);
      // The Tab stop stays on the active tab, so leaving the strip and coming
      // back lands on the tab that is showing.
      expect(tabNamed('Tab 1').tabIndex).toBe(0);
      expect(tabNamed('Tab 2').tabIndex).toBe(-1);

      fireEvent.keyDown(tabNamed('Tab 2'), { key: 'ArrowLeft' });
      expect(document.activeElement).toBe(tabNamed('Tab 1'));

      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'ArrowLeft' });
      expect(document.activeElement).toBe(tabNamed('Tab 3'));

      fireEvent.keyDown(tabNamed('Tab 3'), { key: 'ArrowRight' });
      expect(document.activeElement).toBe(tabNamed('Tab 1'));
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);
    });

    it('Home and End move focus to the first and the last tab', () => {
      const ids = openThreeTabs();
      render(<TabBar />);

      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'End' });
      expect(document.activeElement).toBe(tabNamed('Tab 3'));
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);

      fireEvent.keyDown(tabNamed('Tab 3'), { key: 'Home' });
      expect(document.activeElement).toBe(tabNamed('Tab 1'));
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);
    });

    it('Enter and Space switch to the focused tab', () => {
      const ids = openThreeTabs();
      render(<TabBar />);

      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'ArrowRight' });
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);
      expect(fireEvent.keyDown(tabNamed('Tab 2'), { key: 'Enter' })).toBe(false);
      expect(useTabStore.getState().activeTabId).toBe(ids[1]);
      expect(document.activeElement).toBe(tabNamed('Tab 2'));

      fireEvent.keyDown(tabNamed('Tab 2'), { key: 'ArrowRight' });
      expect(useTabStore.getState().activeTabId).toBe(ids[1]);
      // `false` means the default was prevented: Space would scroll the page.
      expect(fireEvent.keyDown(tabNamed('Tab 3'), { key: ' ' })).toBe(false);
      expect(useTabStore.getState().activeTabId).toBe(ids[2]);
      expect(document.activeElement).toBe(tabNamed('Tab 3'));
    });

    it('keeps every key it answers away from document and window', () => {
      openThreeTabs();
      render(<TabBar />);
      for (const key of ['ArrowRight', 'ArrowLeft', 'Home', 'End', 'Enter', ' ']) {
        fireEvent.keyDown(screen.getByRole('tab', { selected: true }), { key });
      }
      expect(onDocument).not.toHaveBeenCalled();
      expect(onWindow).not.toHaveBeenCalled();
    });

    it('lets a key it does not answer through, so the global shortcuts still work', () => {
      // The control for the test above: the listeners do see what the strip
      // leaves alone, so "never called" there is not a listener that missed.
      render(<TabBar />);
      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'z', ctrlKey: true });
      expect(onDocument).toHaveBeenCalledTimes(1);
      expect(onWindow).toHaveBeenCalledTimes(1);
    });

    it('leaves modified keys to the browser and to the global shortcuts', () => {
      // Alt+Left and Alt+Right are Back and Forward, and Delete or Enter with
      // Ctrl, Shift or Meta is a different command from the plain key.
      const ids = openThreeTabs();
      render(<TabBar />);
      const tab = tabNamed('Tab 1');
      tab.focus();
      const chords = [
        { key: 'ArrowLeft', altKey: true },
        { key: 'Delete', ctrlKey: true },
        { key: 'Delete', shiftKey: true },
        { key: 'Enter', metaKey: true },
      ];
      for (const chord of chords) {
        expect(fireEvent.keyDown(tab, chord)).toBe(true);
      }
      expect(onDocument).toHaveBeenCalledTimes(chords.length);
      expect(useTabStore.getState().tabs).toHaveLength(3);
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);
      expect(document.activeElement).toBe(tab);
    });

    it('answers nothing while a modal is open', () => {
      // `?` opens the shortcuts sheet without taking focus, so a tab can still
      // hold it; its keys must not close or switch tabs behind the sheet.
      const ids = openThreeTabs();
      useUIStore.setState({ shortcutsModalOpen: true });
      try {
        render(<TabBar />);
        const tab = tabNamed('Tab 1');
        tab.focus();
        for (const key of ['ArrowRight', 'End', 'Enter', ' ', 'Delete']) {
          expect(fireEvent.keyDown(tab, { key })).toBe(true);
        }
        expect(useTabStore.getState().tabs).toHaveLength(3);
        expect(useTabStore.getState().activeTabId).toBe(ids[0]);
        expect(document.activeElement).toBe(tab);
      } finally {
        useUIStore.setState({ shortcutsModalOpen: false });
      }
    });

    it('after keyboard focus, Delete closes the tab and the canvas behind never sees the key', () => {
      // React Flow binds Delete on `document` and filters only inputs, so a
      // Delete that closed a tab would otherwise also delete the nodes
      // selected on the canvas.
      useTabStore.getState().addTab('Tab 2');
      render(<TabBar />);
      tabNamed('Tab 2').focus();
      fireEvent.keyDown(document.activeElement!, { key: 'Delete' });
      expect(useTabStore.getState().tabs.map((t) => t.name)).toEqual(['Tab 1']);
      expect(onDocument).not.toHaveBeenCalled();
      expect(onWindow).not.toHaveBeenCalled();
    });

    it('after a pointer click, the tab is switched to but Delete still goes to the canvas', () => {
      // Before the strip took the keyboard, a clicked tab could not hold
      // focus, so the next Delete deleted the canvas selection. A mouse user
      // keeps exactly that.
      const ids = openThreeTabs();
      render(<TabBar />);
      pointerClick(tabNamed('Tab 2'));
      expect(useTabStore.getState().activeTabId).toBe(ids[1]);
      expect(document.activeElement).toBe(document.body);

      fireEvent.keyDown(document.activeElement!, { key: 'Delete' });
      expect(onDocument).toHaveBeenCalledTimes(1);
      expect(useTabStore.getState().tabs).toHaveLength(3);
    });

    it('leaves focus alone on a click with no press before it', () => {
      // Focus is settled by the press. A click that comes without one -- a
      // script's `click()` -- has moved nothing, so it has nothing to undo.
      render(<TabBar />);
      tabNamed('Tab 1').focus();
      fireEvent.click(tabNamed('Tab 1'));
      expect(document.activeElement).toBe(tabNamed('Tab 1'));
    });

    it('a pointer double-click still opens the rename box with focus in it', () => {
      render(<TabBar />);
      pointerClick(tabNamed('Tab 1'), 1);
      pointerClick(tabNamed('Tab 1'), 2);
      fireEvent.doubleClick(tabNamed('Tab 1'));
      expect(document.activeElement).toBe(screen.getByDisplayValue('Tab 1'));
    });

    it('a pointer click on a close button leaves no focus behind, even when the confirm is cancelled', async () => {
      // The dialog hands focus back to what had it when the dialog opened.
      // Had the close button kept the click's focus, a later Enter or Space
      // would press it again instead of reaching the canvas.
      fillTab(0);
      render(
        <>
          <TabBar />
          <DialogContainer />
        </>,
      );
      pointerClick(screen.getByRole('button', { name: 'Close Tab 1' }));
      fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }));
      await waitFor(() => {
        expect(useDialogStore.getState().active).toBeNull();
      });
      expect(useTabStore.getState().tabs).toHaveLength(1);
      expect(document.activeElement).toBe(document.body);
    });

    it('in Safari and macOS Firefox, the press on a close button does not focus the tab either', async () => {
      // Those browsers never focus a clicked button: the press focuses its
      // nearest focusable ancestor, which is now the tab, and the close
      // button's click handler stops the click before the tab sees it. Only
      // the press itself can refuse that focus.
      fillTab(0);
      render(
        <>
          <TabBar />
          <DialogContainer />
        </>,
      );
      const tab = tabNamed('Tab 1');
      const close = screen.getByRole('button', { name: 'Close Tab 1' });
      expect(press(close, tab, { detail: 1 })).toBe(false);
      expect(document.activeElement).not.toBe(tab);

      fireEvent.mouseUp(close, { detail: 1 });
      fireEvent.click(close, { detail: 1 });
      fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }));
      await waitFor(() => {
        expect(useDialogStore.getState().active).toBeNull();
      });
      expect(document.activeElement).toBe(document.body);
      fireEvent.keyDown(document.activeElement!, { key: 'Delete' });
      expect(onDocument).toHaveBeenCalledTimes(1);
      expect(useTabStore.getState().tabs).toHaveLength(1);
    });

    it('a press released off the tab leaves no focus on it', () => {
      // No click lands on the tab, so nothing that waited for one would run.
      render(<TabBar />);
      const tab = tabNamed('Tab 1');
      expect(press(tab, tab, { detail: 1 })).toBe(false);
      fireEvent.mouseUp(document.body);
      expect(document.activeElement).not.toBe(tab);
    });

    it('a right-button press leaves no focus on the tab', () => {
      // The context menu follows and no click does, so a menu dismissed with
      // Esc must not leave the tab holding the next Delete.
      render(<TabBar />);
      const tab = tabNamed('Tab 1');
      expect(press(tab, tab, { button: 2, detail: 1 })).toBe(false);
      fireEvent.contextMenu(tab);
      expect(document.activeElement).not.toBe(tab);
    });

    it('keeps the focus a screen reader gives a tab when it activates it', () => {
      // The screen reader focuses the tab, then sends a simulated press and
      // click with no click count. Those are the users this strip was made
      // for: their place must hold, and the next Delete must reach the tab.
      useTabStore.getState().addTab('Tab 2');
      const firstId = useTabStore.getState().tabs[0].id;
      render(<TabBar />);
      const tab = tabNamed('Tab 1');
      tab.focus();
      expect(fireEvent.mouseDown(tab, { detail: 0 })).toBe(true);
      fireEvent.mouseUp(tab, { detail: 0 });
      fireEvent.click(tab, { detail: 0 });
      expect(useTabStore.getState().activeTabId).toBe(firstId);
      expect(document.activeElement).toBe(tab);

      fireEvent.keyDown(document.activeElement!, { key: 'Delete' });
      expect(onDocument).not.toHaveBeenCalled();
      expect(useTabStore.getState().tabs.map((t) => t.id)).not.toContain(firstId);
    });

    it('keeps the focus a screen reader gives a close button when it presses it', () => {
      // The same press as the tab's, on the button inside it: it reports no
      // click count, and the focus it lands with is the user's place.
      openThreeTabs();
      render(<TabBar />);
      const close = screen.getByRole('button', { name: 'Close Tab 2' });
      close.focus();
      expect(fireEvent.mouseDown(close, { detail: 0 })).toBe(true);
      expect(document.activeElement).toBe(close);
    });

    it('Delete on a focused close button closes its tab, and the canvas behind never sees the key', async () => {
      // Out of the Tab order, but a screen reader or a script can still put
      // focus on the close button. Its Delete used to go past the tab to React
      // Flow on `document`, which deleted the canvas selection instead.
      const ids = openThreeTabs();
      render(<TabBar />);
      const close = screen.getByRole('button', { name: 'Close Tab 2' });
      close.focus();
      expect(fireEvent.keyDown(close, { key: 'Delete' })).toBe(false);
      expect(useTabStore.getState().tabs.map((t) => t.id)).toEqual([ids[0], ids[2]]);
      expect(onDocument).not.toHaveBeenCalled();
      expect(onWindow).not.toHaveBeenCalled();
      // As a Delete on the tab would leave it: on the tab that took its place.
      await waitFor(() => {
        expect(document.activeElement).toBe(tabNamed('Tab 3'));
      });
    });

    it('after a screen reader closes a tab with its close button, focus moves to the neighbour', async () => {
      // A mouse close leaves focus nowhere, as it always did. A screen
      // reader's press -- and Enter or Space on the button -- reports no click
      // count, and the button it was on goes with the tab: left there, focus
      // would fall to the page body and the next Tab start from the top.
      const ids = openThreeTabs();
      render(<TabBar />);

      pointerClick(screen.getByRole('button', { name: 'Close Tab 1' }));
      // Past the close's own promise, where the move would happen.
      await act(async () => {});
      expect(useTabStore.getState().tabs.map((t) => t.id)).toEqual([ids[1], ids[2]]);
      expect(document.activeElement).toBe(document.body);

      const close = screen.getByRole('button', { name: 'Close Tab 2' });
      close.focus();
      fireEvent.mouseDown(close, { detail: 0 });
      fireEvent.mouseUp(close, { detail: 0 });
      fireEvent.click(close, { detail: 0 });
      expect(useTabStore.getState().tabs.map((t) => t.id)).toEqual([ids[2]]);
      await waitFor(() => {
        expect(document.activeElement).toBe(tabNamed('Tab 3'));
      });
    });

    it('pressing another tab still commits a rename in progress', () => {
      // The press refuses focus but still takes it from whatever had it, as a
      // press on the strip always did, and the rename box commits on blur.
      useTabStore.getState().addTab('Tab 2');
      const firstId = useTabStore.getState().tabs[0].id;
      render(<TabBar />);
      fireEvent.doubleClick(screen.getByText('Tab 1'));
      fireEvent.change(screen.getByDisplayValue('Tab 1'), { target: { value: 'Renamed' } });
      press(tabNamed('Tab 2'), tabNamed('Tab 2'), { detail: 1 });
      expect(useTabStore.getState().tabs.find((t) => t.id === firstId)?.name).toBe('Renamed');
    });

    it('a press inside the rename box keeps its default, so it can place the caret', () => {
      render(<TabBar />);
      fireEvent.doubleClick(screen.getByText('Tab 1'));
      const input = screen.getByDisplayValue('Tab 1');
      expect(press(input, input, { detail: 1 })).toBe(true);
      expect(document.activeElement).toBe(input);
    });

    it('a double-click inside the rename box keeps what was typed', () => {
      // Double-clicking a word to select it used to bubble up to the tab,
      // which started the rename over from the saved name.
      render(<TabBar />);
      fireEvent.doubleClick(screen.getByText('Tab 1'));
      const input = screen.getByDisplayValue('Tab 1');
      fireEvent.change(input, { target: { value: 'Half typed' } });
      fireEvent.doubleClick(input);
      expect(screen.getByDisplayValue('Half typed')).toBe(input);
    });

    it('F2 opens the rename box on the focused tab, with focus in it', () => {
      render(<TabBar />);
      const tab = tabNamed('Tab 1');
      tab.focus();
      expect(fireEvent.keyDown(tab, { key: 'F2' })).toBe(false);
      expect(document.activeElement).toBe(screen.getByDisplayValue('Tab 1'));
    });

    it('Enter commits an F2 rename and hands focus back to the tab', () => {
      const id = useTabStore.getState().tabs[0].id;
      render(<TabBar />);
      tabNamed('Tab 1').focus();
      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'F2' });
      const input = screen.getByDisplayValue('Tab 1');
      fireEvent.change(input, { target: { value: 'Renamed' } });
      fireEvent.keyDown(input, { key: 'Enter' });
      expect(useTabStore.getState().tabs.find((t) => t.id === id)?.name).toBe('Renamed');
      expect(document.activeElement).toBe(tabNamed('Renamed'));
    });

    it('Escape cancels an F2 rename and hands focus back to the tab', () => {
      // Focus moves only once the box is gone: the box commits on blur, so
      // taking focus from it first would save what Escape throws away.
      const id = useTabStore.getState().tabs[0].id;
      render(<TabBar />);
      tabNamed('Tab 1').focus();
      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'F2' });
      const input = screen.getByDisplayValue('Tab 1');
      fireEvent.change(input, { target: { value: 'Discarded' } });
      fireEvent.keyDown(input, { key: 'Escape' });
      expect(useTabStore.getState().tabs.find((t) => t.id === id)?.name).toBe('Tab 1');
      expect(document.activeElement).toBe(tabNamed('Tab 1'));
    });

    it('a rename a double-click started still ends with focus on the page body', () => {
      // Only a rename F2 started hands focus back; the mouse's is unchanged.
      render(<TabBar />);
      fireEvent.doubleClick(screen.getByText('Tab 1'));
      fireEvent.keyDown(screen.getByDisplayValue('Tab 1'), { key: 'Enter' });
      expect(document.activeElement).toBe(document.body);
    });

    it('an F2 rename left by a press elsewhere commits, and focus is not pulled back', () => {
      // Enter and Escape end a rename where it is; a press elsewhere is the
      // user taking focus away, and the tab has no claim on it then.
      useTabStore.getState().addTab('Tab 2');
      const firstId = useTabStore.getState().tabs[0].id;
      render(<TabBar />);
      tabNamed('Tab 1').focus();
      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'F2' });
      fireEvent.change(screen.getByDisplayValue('Tab 1'), { target: { value: 'Pressed away' } });
      press(tabNamed('Tab 2'), tabNamed('Tab 2'), { detail: 1 });
      expect(useTabStore.getState().tabs.find((t) => t.id === firstId)?.name).toBe('Pressed away');
      expect(document.activeElement).toBe(document.body);
    });

    it('F2 renames a read-only tab too, as a double-click does', () => {
      // `readOnly` refuses Save and plugin writes; the label stays the user's.
      markTab(0, { readOnly: true });
      render(<TabBar />);
      const tab = tabNamed('Tab 1, read-only');
      tab.focus();
      fireEvent.keyDown(tab, { key: 'F2' });
      expect(document.activeElement).toBe(screen.getByDisplayValue('Tab 1'));
    });

    it('F2 does nothing while a modal is open, or with a modifier held', () => {
      render(<TabBar />);
      const tab = tabNamed('Tab 1');
      tab.focus();
      expect(fireEvent.keyDown(tab, { key: 'F2', ctrlKey: true })).toBe(true);
      expect(screen.queryByDisplayValue('Tab 1')).toBeNull();

      useUIStore.setState({ shortcutsModalOpen: true });
      try {
        expect(fireEvent.keyDown(tab, { key: 'F2' })).toBe(true);
        expect(screen.queryByDisplayValue('Tab 1')).toBeNull();
        expect(document.activeElement).toBe(tab);
      } finally {
        useUIStore.setState({ shortcutsModalOpen: false });
      }
    });

    it('Delete goes through the same store action as the close button', () => {
      const ids = openThreeTabs();
      const realRemoveTab = useTabStore.getState().removeTab;
      // Installed through setState and passing through to the real action:
      // TabBar reads the action by selector, which a spy on getState() would
      // never reach.
      const removeTab = vi.fn(realRemoveTab);
      useTabStore.setState({ removeTab });
      try {
        render(<TabBar />);

        fireEvent.keyDown(tabNamed('Tab 1'), { key: 'Delete' });
        expect(removeTab).toHaveBeenLastCalledWith(ids[0]);
        fireEvent.click(screen.getByRole('button', { name: 'Close Tab 2' }));
        expect(removeTab).toHaveBeenLastCalledWith(ids[1]);
        expect(useTabStore.getState().tabs.map((t) => t.id)).toEqual([ids[2]]);
      } finally {
        useTabStore.setState({ removeTab: realRemoveTab });
      }
    });

    it('asks before a Delete closes a tab holding a graph, with the button\'s own question', async () => {
      useTabStore.getState().addTab('Tab 2');
      fillTab(0);
      render(<TabBar />);
      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'Delete' });

      await waitFor(() => {
        expect(useDialogStore.getState().active).not.toBeNull();
      });
      const active = useDialogStore.getState().active!;
      expect(active.title).toBe('Close "Tab 1"?');
      expect(active.message).toContain('2 nodes');
      expect(active.confirmText).toBe('Close tab');
      expect(useTabStore.getState().tabs).toHaveLength(2);

      useDialogStore.getState().close(false);
      await waitFor(() => {
        expect(useDialogStore.getState().active).toBeNull();
      });
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });

    it('after a keyboard close, focus moves to the tab that became active', async () => {
      const ids = openThreeTabs();
      useTabStore.getState().setActiveTab(ids[1]);
      render(<TabBar />);

      // The middle tab: its right-hand neighbour takes over.
      tabNamed('Tab 2').focus();
      fireEvent.keyDown(tabNamed('Tab 2'), { key: 'Delete' });
      expect(useTabStore.getState().activeTabId).toBe(ids[2]);
      await waitFor(() => {
        expect(document.activeElement).toBe(tabNamed('Tab 3'));
      });

      // The last tab: the one before it takes over.
      fireEvent.keyDown(tabNamed('Tab 3'), { key: 'Delete' });
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);
      await waitFor(() => {
        expect(document.activeElement).toBe(tabNamed('Tab 1'));
      });
    });

    it('after closing a focused tab that is not the active one, focus goes to its neighbour', async () => {
      // Not back to the active tab: a second Delete would then close the tab
      // the user is looking at instead of the next one along.
      const ids = openThreeTabs();
      render(<TabBar />);

      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'ArrowRight' });
      fireEvent.keyDown(tabNamed('Tab 2'), { key: 'Delete' });
      expect(useTabStore.getState().activeTabId).toBe(ids[0]);
      await waitFor(() => {
        expect(document.activeElement).toBe(tabNamed('Tab 3'));
      });

      // The last tab has no neighbour after it, so the one before it.
      fireEvent.keyDown(tabNamed('Tab 3'), { key: 'Delete' });
      await waitFor(() => {
        expect(document.activeElement).toBe(tabNamed('Tab 1'));
      });
      expect(useTabStore.getState().tabs.map((t) => t.id)).toEqual([ids[0]]);
    });

    it('after a confirmed keyboard close, focus does not follow the dialog back to the closed tab', async () => {
      // The dialog hands focus back to whatever opened it as it closes, which
      // here is the tab it is about to remove. Left there, focus would fall
      // to the page body with that tab, and the next Tab would start from the
      // top of the page.
      const ids = openThreeTabs();
      fillTab(0);
      render(
        <>
          <TabBar />
          <DialogContainer />
        </>,
      );
      tabNamed('Tab 1').focus();
      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'Delete' });
      fireEvent.click(await screen.findByRole('button', { name: 'Close tab' }));

      await waitFor(() => {
        expect(useTabStore.getState().tabs).toHaveLength(2);
      });
      expect(useTabStore.getState().activeTabId).toBe(ids[1]);
      await waitFor(() => {
        expect(document.activeElement).toBe(tabNamed('Tab 2'));
      });
    });

    it('puts focus on the New tab button once the last tab is closed', async () => {
      render(<TabBar />);
      tabNamed('Tab 1').focus();
      fireEvent.keyDown(tabNamed('Tab 1'), { key: 'Delete' });
      expect(useTabStore.getState().tabs).toEqual([]);
      await waitFor(() => {
        expect(document.activeElement).toBe(screen.getByRole('button', { name: 'New tab' }));
      });
    });

    it('leaves the keys typed into the rename box to the box', () => {
      // They bubble up through the tab, and they are the box's: Delete
      // deletes a character, Space types one, the arrows move the caret, and
      // F2 must not start the rename over.
      useTabStore.getState().addTab('Tab 2');
      render(<TabBar />);
      fireEvent.doubleClick(screen.getByText('Tab 2'));
      const input = screen.getByDisplayValue('Tab 2');
      for (const key of ['Delete', ' ', 'ArrowLeft', 'Home', 'F2']) {
        // `true` means nothing prevented the default.
        expect(fireEvent.keyDown(input, { key })).toBe(true);
      }
      expect(useTabStore.getState().tabs).toHaveLength(2);
      expect(document.activeElement).toBe(input);
    });

    it('names a tab with its title, the plugin that opened it and its read-only state', () => {
      useTabStore.getState().addTab('Tab 2');
      useTabStore.getState().addTab('Tab 3');
      const source = { kind: 'agent-variant', pluginId: 'graph-copilot' };
      markTab(0, { source });
      markTab(1, { readOnly: true });
      markTab(2, { readOnly: true, source });
      render(<TabBar />);

      expect(tabNamed('Tab 1, opened by graph-copilot')).toBeTruthy();
      expect(tabNamed('Tab 2, read-only')).toBeTruthy();
      const both = tabNamed('Tab 3, opened by graph-copilot, read-only');
      // The tooltip stays, for the mouse.
      expect(both.getAttribute('title')).toBe('Opened by graph-copilot');
    });

    it('says a tab is running in its name, which the dot alone never did (#504)', () => {
      // A background run is otherwise only a yellow dot, and the name set
      // above replaces the tab's content, so nothing else would say it.
      useTabStore.getState().addTab('Tab 2');
      useTabStore.getState().addTab('Tab 3');
      markTab(0, { status: 'running' });
      markTab(2, {
        status: 'running',
        readOnly: true,
        source: { kind: 'agent-variant', pluginId: 'graph-copilot' },
      });
      render(<TabBar />);

      expect(tabNamed('Tab 1, running')).toBeTruthy();
      // An idle tab says nothing about running.
      expect(tabNamed('Tab 2')).toBeTruthy();
      expect(tabNamed('Tab 3, opened by graph-copilot, read-only, running')).toBeTruthy();
    });

    it('drops "running" from the name once the run ends', () => {
      markTab(0, { status: 'running' });
      render(<TabBar />);
      expect(tabNamed('Tab 1, running')).toBeTruthy();

      act(() => markTab(0, { status: 'idle' }));
      expect(tabNamed('Tab 1')).toBeTruthy();
    });

    it('says a tab is running in zh-TW too', () => {
      useI18n.setState({ locale: 'zh-TW' });
      markTab(0, { status: 'running', readOnly: true });
      render(<TabBar />);
      expect(tabNamed('Tab 1，唯讀，執行中')).toBeTruthy();
    });

    it('keeps a tab name that happens to contain a placeholder as typed', () => {
      markTab(0, { name: '{plugin}', source: { kind: 'agent-variant', pluginId: 'graph-copilot' } });
      render(<TabBar />);
      expect(tabNamed('{plugin}, opened by graph-copilot')).toBeTruthy();
    });

    it('makes the close control a real button, named after its tab and out of the Tab order', () => {
      render(<TabBar />);
      const close = screen.getByRole('button', { name: 'Close Tab 1' });
      expect(close.tagName).toBe('BUTTON');
      expect(close.getAttribute('type')).toBe('button');
      // Delete closes the focused tab, so a second stop per tab would only
      // lengthen the way through the strip.
      expect(close.tabIndex).toBe(-1);
    });

    it('uses the zh-TW names when that locale is active', () => {
      useI18n.setState({ locale: 'zh-TW' });
      markTab(0, { readOnly: true, source: { kind: 'agent-variant', pluginId: 'graph-copilot' } });
      render(<TabBar />);
      expect(screen.getByRole('tablist', { name: '工作區分頁' })).toBeTruthy();
      expect(tabNamed('Tab 1，由 graph-copilot 開啟，唯讀')).toBeTruthy();
      expect(screen.getByRole('button', { name: '關閉「Tab 1」' })).toBeTruthy();
      expect(screen.getByRole('button', { name: '新增分頁' })).toBeTruthy();
    });
  });
});
