import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, renderHook, act } from '@testing-library/react';

// The name prompt (NIT 21) is an in-app modal driven by a promise; mocking
// the helper keeps these tests about the MENU's decisions -- refuse or ask,
// and in that order -- rather than about the dialog component.
vi.mock('../../utils/dialog', () => ({
  prompt: vi.fn(async () => 'Subgraph'),
  confirm: vi.fn(async () => true),
}));

import {
  NodeContextMenu,
  useNodeContextMenuItems,
  useNoteContextMenuItems,
  type ContextMenuPosition,
} from './NodeContextMenu';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import type { Node } from '@xyflow/react';
import type { NodeData } from '../../types';
import { prompt } from '../../utils/dialog';

function resetToSingleTab() {
  useTabStore.setState({
    tabs: [],
    activeTabId: null as unknown as string,
    clipboard: null,
  });
  useTabStore.getState().addTab('Tab 1');
}

function makeNoteNode(
  id: string,
  data: Partial<NodeData> = {},
): Node<NodeData> {
  return {
    id,
    type: 'noteNode',
    position: { x: 0, y: 0 },
    data: {
      label: 'Note',
      type: 'note',
      params: {},
      noteKind: 'text',
      noteContent: '',
      noteColor: '#3d3d1a',
      boundToNodeId: null,
      boundOffset: null,
      ...data,
    },
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  resetToSingleTab();
  useToastStore.setState({ toasts: [] });
  vi.mocked(prompt).mockReset();
  vi.mocked(prompt).mockResolvedValue('Subgraph');
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── NodeContextMenu (presentational) ─────────────────────────────────────────

describe('NodeContextMenu', () => {
  const position: ContextMenuPosition = { nodeId: 'n1', x: 100, y: 200 };

  it('renders all items at the given position and runs action + onClose on click', () => {
    const onClose = vi.fn();
    const action = vi.fn();
    render(
      <NodeContextMenu
        position={position}
        items={[{ label: 'Do Thing', action }]}
        onClose={onClose}
      />,
    );
    const btn = screen.getByText('Do Thing');
    fireEvent.click(btn);
    expect(action).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('positions the menu using the x/y from position', () => {
    const { container } = render(
      <NodeContextMenu
        position={position}
        items={[{ label: 'A', action: vi.fn() }]}
        onClose={vi.fn()}
      />,
    );
    // The menu element is the second child (after the backdrop) with left/top.
    const positioned = Array.from(container.querySelectorAll('div')).find(
      (d) => d.style.left === '100px' && d.style.top === '200px',
    );
    expect(positioned).toBeTruthy();
  });

  it('clicking the backdrop calls onClose', () => {
    const onClose = vi.fn();
    const { container } = render(
      <NodeContextMenu
        position={position}
        items={[{ label: 'A', action: vi.fn() }]}
        onClose={onClose}
      />,
    );
    // Backdrop is the first child div.
    const backdrop = container.querySelector('div')!;
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('right-clicking the backdrop prevents default and calls onClose', () => {
    const onClose = vi.fn();
    const { container } = render(
      <NodeContextMenu
        position={position}
        items={[{ label: 'A', action: vi.fn() }]}
        onClose={onClose}
      />,
    );
    const backdrop = container.querySelector('div')!;
    const evt = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    const prevented = !backdrop.dispatchEvent(evt);
    expect(prevented).toBe(true);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('renders a divider after items that set dividerAfter', () => {
    const { container } = render(
      <NodeContextMenu
        position={position}
        items={[
          { label: 'WithDivider', action: vi.fn(), dividerAfter: true },
          { label: 'NoDivider', action: vi.fn() },
        ]}
        onClose={vi.fn()}
      />,
    );
    // Exactly one divider div (class contains "divider").
    const dividers = Array.from(container.querySelectorAll('div')).filter((d) =>
      /divider/i.test(d.className),
    );
    expect(dividers).toHaveLength(1);
  });

  it('uses a custom color when provided, otherwise the default #ccc', () => {
    render(
      <NodeContextMenu
        position={position}
        items={[
          { label: 'Red', action: vi.fn(), color: '#F44336' },
          { label: 'Default', action: vi.fn() },
        ]}
        onClose={vi.fn()}
      />,
    );
    const red = screen.getByText('Red');
    const def = screen.getByText('Default');
    // jsdom normalizes color longhand to rgb().
    expect(red.style.color).toBe('rgb(244, 67, 54)');
    expect(def.style.color).toBe('rgb(204, 204, 204)');
  });
});

// ── useNodeContextMenuItems ──────────────────────────────────────────────────

describe('useNodeContextMenuItems', () => {
  it('returns open-details / rename / duplicate / delete with wired callbacks', () => {
    const onDelete = vi.fn();
    const onRename = vi.fn();
    const onDuplicate = vi.fn();
    const onOpenDetails = vi.fn();
    const { result } = renderHook(() =>
      useNodeContextMenuItems('node-1', { onDelete, onRename, onDuplicate, onOpenDetails }),
    );
    const items = result.current;
    expect(items.map((i) => i.label)).toEqual([
      'Open details',
      'Rename',
      'Duplicate',
      'Delete',
    ]);

    items[0].action();
    expect(onOpenDetails).toHaveBeenCalledWith('node-1');
    items[1].action();
    expect(onRename).toHaveBeenCalledWith('node-1');
    items[2].action();
    expect(onDuplicate).toHaveBeenCalledWith('node-1');
    items[3].action();
    expect(onDelete).toHaveBeenCalledWith('node-1');

    // Open details and Duplicate each have a divider after them; delete is red.
    expect(items[0].dividerAfter).toBe(true);
    expect(items[2].dividerAfter).toBe(true);
    expect(items[3].color).toBe('#F44336');
  });
});

// ── useNoteContextMenuItems ──────────────────────────────────────────────────

describe('useNoteContextMenuItems', () => {
  it('shows "Bind" when the note is unbound and calls bindNoteToNearest', () => {
    const noteId = 'note-1';
    // Put an unbound note and a target node on the active tab.
    const tabId = useTabStore.getState().activeTabId;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              nodes: [
                makeNoteNode(noteId, { boundToNodeId: null }),
                {
                  id: 'target',
                  type: 'baseNode',
                  position: { x: 100, y: 100 },
                  data: { label: 'T', type: 'X', params: {} },
                } as Node<NodeData>,
              ],
            }
          : t,
      ),
    });
    const bindSpy = vi.spyOn(useTabStore.getState(), 'bindNoteToNearest');

    const onDelete = vi.fn();
    const { result } = renderHook(() =>
      useNoteContextMenuItems(noteId, { onDelete }),
    );
    const items = result.current;
    expect(items[0].label).toBe('Bind to Nearest Node');
    items[0].action();
    expect(bindSpy).toHaveBeenCalledWith(noteId);
  });

  it('shows "Unbind" when the note is bound and calls unbindNote', () => {
    const noteId = 'note-2';
    const tabId = useTabStore.getState().activeTabId;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId
          ? { ...t, nodes: [makeNoteNode(noteId, { boundToNodeId: 'parent' })] }
          : t,
      ),
    });
    const unbindSpy = vi.spyOn(useTabStore.getState(), 'unbindNote');

    const { result } = renderHook(() =>
      useNoteContextMenuItems(noteId, { onDelete: vi.fn() }),
    );
    const items = result.current;
    expect(items[0].label).toBe('Unbind Note');
    items[0].action();
    expect(unbindSpy).toHaveBeenCalledWith(noteId);
  });

  it('color items invoke updateNoteData and highlight the active color', () => {
    const noteId = 'note-3';
    const tabId = useTabStore.getState().activeTabId;
    // Active color is Blue (#1a2d3d).
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId
          ? { ...t, nodes: [makeNoteNode(noteId, { noteColor: '#1a2d3d' })] }
          : t,
      ),
    });
    const updateSpy = vi.spyOn(useTabStore.getState(), 'updateNoteData');

    const { result } = renderHook(() =>
      useNoteContextMenuItems(noteId, { onDelete: vi.fn() }),
    );
    const items = result.current;
    // index 0 bind/unbind, 1 changeColor header, then 6 colors, then spacer, delete.
    const colorItems = items.filter((i) => /^ {2}/.test(i.label));
    expect(colorItems).toHaveLength(6);

    // The blue color item should be highlighted (#fff); the others muted (#999).
    const blue = colorItems.find((i) => i.label.trim() === 'Blue')!;
    const yellow = colorItems.find((i) => i.label.trim() === 'Yellow')!;
    expect(blue.color).toBe('#fff');
    expect(yellow.color).toBe('#999');

    // Selecting a color forwards to updateNoteData.
    yellow.action();
    expect(updateSpy).toHaveBeenCalledWith(noteId, { noteColor: '#3d3d1a' });

    // The "Change Color" header is a no-op action (exercise it for coverage).
    const header = items.find((i) => i.label === 'Change Color')!;
    expect(header.color).toBe('#888');
    header.action();
    // The spacer item with empty label is also a no-op.
    const spacer = items.find((i) => i.label === '' && i.dividerAfter)!;
    spacer.action();
  });

  it('delete item forwards to onDelete', () => {
    const noteId = 'note-4';
    const tabId = useTabStore.getState().activeTabId;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId ? { ...t, nodes: [makeNoteNode(noteId)] } : t,
      ),
    });
    const onDelete = vi.fn();
    const { result } = renderHook(() =>
      useNoteContextMenuItems(noteId, { onDelete }),
    );
    const del = result.current.find((i) => i.label === 'Delete')!;
    expect(del.color).toBe('#F44336');
    del.action();
    expect(onDelete).toHaveBeenCalledWith(noteId);
  });

  it('falls back gracefully when there is no active tab / note (isBound false, default color)', () => {
    // No matching note on the tab → note is undefined, isBound false.
    const { result } = renderHook(() =>
      useNoteContextMenuItems('missing', { onDelete: vi.fn() }),
    );
    const items = result.current;
    expect(items[0].label).toBe('Bind to Nearest Node');
    // All color items muted because note?.data.noteColor is undefined.
    const colorItems = items.filter((i) => /^ {2}/.test(i.label));
    expect(colorItems.every((i) => i.color === '#999')).toBe(true);
  });
});


// ── Bypass entry (core#128) ──────────────────────────────────────────────

/** Put a single node on the active tab and return its id. */
function seedNode(id: string, over: Partial<Node<NodeData>> = {}) {
  const tabId = useTabStore.getState().activeTabId;
  useTabStore.setState({
    tabs: useTabStore.getState().tabs.map((t) =>
      t.id === tabId
        ? {
            ...t,
            nodes: [
              {
                id,
                type: 'baseNode',
                position: { x: 0, y: 0 },
                data: { label: id, type: 'Dropout', params: {} },
                ...over,
              } as Node<NodeData>,
            ],
          }
        : t,
    ),
  });
  return id;
}

describe('useNodeContextMenuItems — bypass', () => {
  const callbacks = {
    onDelete: vi.fn(),
    onRename: vi.fn(),
    onDuplicate: vi.fn(),
    onOpenDetails: vi.fn(),
  };

  it('offers Bypass on an ordinary node and forwards to the store', () => {
    seedNode('n1');
    const spy = vi.spyOn(useTabStore.getState(), 'toggleNodeBypass');
    const { result } = renderHook(() => useNodeContextMenuItems('n1', callbacks));

    expect(result.current.map((i) => i.label)).toEqual([
      'Open details',
      'Rename',
      'Duplicate',
      'Bypass',
      'Delete',
    ]);
    result.current[3].action();
    expect(spy).toHaveBeenCalledWith('n1');
  });

  it('offers the inverse label once the node is muted', () => {
    seedNode('n1', { data: { label: 'n1', type: 'Dropout', params: {}, bypassed: true } });
    const { result } = renderHook(() => useNodeContextMenuItems('n1', callbacks));
    const item = result.current.find((i) => i.label === 'Remove Bypass');
    expect(item).toBeTruthy();
    expect(item!.color).toBe('#22d3ee');
  });

  it('omits the entry for node kinds bypass does not apply to', () => {
    for (const type of ['noteNode', 'start', 'presetNode']) {
      seedNode('n1', { type });
      const { result } = renderHook(() => useNodeContextMenuItems('n1', callbacks));
      expect(result.current.map((i) => i.label)).toEqual([
        'Open details',
        'Rename',
        'Duplicate',
        'Delete',
      ]);
      // Duplicate keeps the divider that would otherwise sit under Bypass, so
      // Delete stays visually separated either way.
      expect(result.current[2].dividerAfter).toBe(true);
    }
  });

  it('omits the entry for the graph I/O contract nodes (core#128 review)', () => {
    // GraphInput/GraphOutput render as ordinary baseNode cards, so the
    // component-type check cannot see them — the real node type can. Muting
    // one would leave a published contract advertising an input or output
    // the run cannot honour, which the backend refuses outright.
    for (const graphType of ['GraphInput', 'GraphOutput']) {
      seedNode('n1', { data: { label: 'n1', type: graphType, params: {} } });
      const { result } = renderHook(() => useNodeContextMenuItems('n1', callbacks));
      expect(result.current.map((i) => i.label)).toEqual([
        'Open details',
        'Rename',
        'Duplicate',
        'Delete',
      ]);
    }
  });
});

// ── Subgraph entries (core#137) ─────────────────────────────────────────

function makePlainNode(id: string, type = 'Dataset'): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: id,
      type,
      params: {},
      definition: {
        node_name: type, category: 'x', description: '',
        inputs: [], outputs: [], params: [],
      },
    },
  };
}

describe('subgraph context menu entries', () => {
  const callbacks = {
    onDelete: vi.fn(), onRename: vi.fn(),
    onDuplicate: vi.fn(), onOpenDetails: vi.fn(),
  };

  it('offers Collapse only once two or more nodes are selected', () => {
    const store = useTabStore.getState();
    store.setNodes([makePlainNode('a'), makePlainNode('b')]);

    const single = renderHook(() => useNodeContextMenuItems('a', callbacks));
    expect(single.result.current.map((i) => i.label)).not.toContain(
      'Collapse to subgraph',
    );

    store.setNodes(
      useTabStore.getState().getActiveTab().nodes.map((n) => ({
        ...n, selected: true,
      })),
    );
    const many = renderHook(() => useNodeContextMenuItems('a', callbacks));
    expect(many.result.current.map((i) => i.label)).toContain(
      'Collapse to subgraph',
    );
  });

  it('offers Enter and Expand only on an instance node', () => {
    const store = useTabStore.getState();
    store.setNodes([makePlainNode('a'), makePlainNode('blk', 'subgraph:sg')]);

    const plain = renderHook(() => useNodeContextMenuItems('a', callbacks));
    expect(plain.result.current.map((i) => i.label)).not.toContain(
      'Enter subgraph',
    );

    const instance = renderHook(() => useNodeContextMenuItems('blk', callbacks));
    const labels = instance.result.current.map((i) => i.label);
    expect(labels).toContain('Enter subgraph');
    expect(labels).toContain('Expand subgraph here');
    // Still a normal node otherwise.
    expect(labels).toContain('Delete');
  });

  it('Collapse reports a refusal as a toast instead of failing silently', () => {
    const store = useTabStore.getState();
    // a -> b -> c with only a and c selected: b is in the way.
    store.setNodes([
      { ...makePlainNode('a'), selected: true },
      makePlainNode('b'),
      { ...makePlainNode('c'), selected: true },
    ]);
    store.setEdges([
      { id: 'e1', source: 'a', target: 'b' },
      { id: 'e2', source: 'b', target: 'c' },
    ]);
    const { result } = renderHook(() => useNodeContextMenuItems('a', callbacks));
    const collapse = result.current.find(
      (i) => i.label === 'Collapse to subgraph',
    )!;
    collapse.action();

    const toasts = useToastStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].message).toContain('b');
    expect(useTabStore.getState().getActiveTab().subgraphs).toEqual([]);
  });
});

// ── Review MAJOR 8 / NIT 21: the collapse entry ──────────────────────────

/** A node whose id is a UUID and whose human name lives in `data.label`. */
function makeNamedNode(id: string, label: string): Node<NodeData> {
  const base = makePlainNode(id);
  return { ...base, data: { ...base.data, label } };
}

/** a -> b -> c with only a and c selected: b is the convexity blocker. */
function seedNonConvex() {
  const store = useTabStore.getState();
  store.setNodes([
    { ...makeNamedNode('3f2b1a44-9c1b-4a2f-9b6e-2d0f1a4c8e11', 'Encoder'), selected: true },
    makeNamedNode('7a1c0de2-55aa-4d7b-8f31-0c2b9e5d1a03', 'Bottleneck'),
    { ...makeNamedNode('c40b7f18-2e63-4a90-b1d5-9f7a6c3e2b44', 'Decoder'), selected: true },
  ]);
  store.setEdges([
    {
      id: 'e1',
      source: '3f2b1a44-9c1b-4a2f-9b6e-2d0f1a4c8e11',
      target: '7a1c0de2-55aa-4d7b-8f31-0c2b9e5d1a03',
    },
    {
      id: 'e2',
      source: '7a1c0de2-55aa-4d7b-8f31-0c2b9e5d1a03',
      target: 'c40b7f18-2e63-4a90-b1d5-9f7a6c3e2b44',
    },
  ]);
}

function collapseItem(nodeId: string) {
  const callbacks = {
    onDelete: vi.fn(), onRename: vi.fn(),
    onDuplicate: vi.fn(), onOpenDetails: vi.fn(),
  };
  const { result } = renderHook(() => useNodeContextMenuItems(nodeId, callbacks));
  return result.current.find((i) => i.label === 'Collapse to subgraph')!;
}

describe('collapse refusal names nodes the user can recognise (MAJOR 8)', () => {
  it('names the BLOCKER by its label, never by its raw id', async () => {
    seedNonConvex();
    const collapse = collapseItem('3f2b1a44-9c1b-4a2f-9b6e-2d0f1a4c8e11');
    await act(async () => { await collapse.action(); });

    const message = useToastStore.getState().toasts[0].message;
    // `buildFlowNode` ids are crypto.randomUUID()s; the human name is in
    // `data.label`. A UUID in a "add these nodes to the selection" message is
    // not something a user can act on.
    expect(message).toContain('Bottleneck');
    expect(message).not.toContain('7a1c0de2-55aa-4d7b-8f31-0c2b9e5d1a03');
  });

  it('adds the blockers to the selection so the retry is one click away', async () => {
    seedNonConvex();
    const collapse = collapseItem('3f2b1a44-9c1b-4a2f-9b6e-2d0f1a4c8e11');
    await act(async () => { await collapse.action(); });

    const selected = useTabStore.getState().getActiveTab()
      .nodes.filter((n) => n.selected).map((n) => n.data.label).sort();
    expect(selected).toEqual(['Bottleneck', 'Decoder', 'Encoder']);
  });

  it('falls back to the id when a node genuinely has no label', async () => {
    seedNonConvex();
    const store = useTabStore.getState();
    store.setNodes(
      store.getActiveTab().nodes.map((n) =>
        n.data.label === 'Bottleneck'
          ? { ...n, data: { ...n.data, label: '' } }
          : n,
      ),
    );
    const collapse = collapseItem('3f2b1a44-9c1b-4a2f-9b6e-2d0f1a4c8e11');
    await act(async () => { await collapse.action(); });
    expect(useToastStore.getState().toasts[0].message).toContain(
      '7a1c0de2-55aa-4d7b-8f31-0c2b9e5d1a03',
    );
  });

  it('never asks for a name for a selection it is about to refuse', async () => {
    seedNonConvex();
    const collapse = collapseItem('3f2b1a44-9c1b-4a2f-9b6e-2d0f1a4c8e11');
    await act(async () => { await collapse.action(); });
    expect(prompt).not.toHaveBeenCalled();
  });
});

describe('collapse asks for a name (NIT 21)', () => {
  /** Two connected nodes, both selected: a collapse that will be accepted. */
  function seedCollapsible() {
    const store = useTabStore.getState();
    store.setNodes([
      { ...makeNamedNode('n-1', 'Conv'), selected: true },
      { ...makeNamedNode('n-2', 'ReLU'), selected: true },
    ]);
    store.setEdges([{ id: 'e1', source: 'n-1', target: 'n-2' }]);
  }

  it('prompts, and the block carries the name the user entered', async () => {
    seedCollapsible();
    vi.mocked(prompt).mockResolvedValueOnce('Encoder block');
    const collapse = collapseItem('n-1');
    await act(async () => { await collapse.action(); });

    expect(prompt).toHaveBeenCalledTimes(1);
    // Prefilled with the name the block would otherwise silently get.
    expect(vi.mocked(prompt).mock.calls[0][0].defaultValue).toBe('Subgraph');
    const subgraphs = useTabStore.getState().getActiveTab().subgraphs;
    expect(subgraphs).toHaveLength(1);
    expect(subgraphs[0].name).toBe('Encoder block');
  });

  it('cancelling leaves the graph exactly as it was', async () => {
    seedCollapsible();
    vi.mocked(prompt).mockResolvedValueOnce(null);
    const collapse = collapseItem('n-1');
    await act(async () => { await collapse.action(); });

    const tab = useTabStore.getState().getActiveTab();
    expect(tab.subgraphs).toEqual([]);
    expect(tab.nodes.map((n) => n.id)).toEqual(['n-1', 'n-2']);
  });

  it('an empty name falls back to the default rather than an unnamed block', async () => {
    seedCollapsible();
    vi.mocked(prompt).mockResolvedValueOnce('   ');
    const collapse = collapseItem('n-1');
    await act(async () => { await collapse.action(); });
    expect(useTabStore.getState().getActiveTab().subgraphs[0].name).toBe('Subgraph');
  });

  it('refuses a read-only tab before prompting', async () => {
    seedCollapsible();
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) => ({ ...t, readOnly: true })),
    });
    const collapse = collapseItem('n-1');
    await act(async () => { await collapse.action(); });

    expect(prompt).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts[0].message).toBe(
      'This graph is open read-only',
    );
    expect(useTabStore.getState().getActiveTab().subgraphs).toEqual([]);
  });
});
