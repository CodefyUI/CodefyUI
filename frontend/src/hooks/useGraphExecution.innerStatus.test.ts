/**
 * #559: while a block is open, the nodes inside it show what the run is doing
 * to them. The server reports each one under the id the run gives it
 * (`<card>/<inner>`) on `inner_node_status`, beside the card's own unchanged
 * `node_status` frames, and names the inner node on a card's progress frame.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';

import { useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { discardTabNodeUpdates, flushTabNodeUpdates } from '../store/nodeUpdateQueue';
import { useI18n } from '../i18n';
import type { NodeData, NodeDefinition } from '../types';
import { subgraphIdOf } from '../utils/subgraph';
import { dismissValidationToasts } from '../utils/validationToasts';

vi.mock('../api/rest', () => ({
  validateGraph: vi.fn(),
  getRun: vi.fn(),
}));
vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));
import { getRun, validateGraph } from '../api/rest';

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

function def(name: string): NodeDefinition {
  return {
    node_name: name,
    category: 'x',
    description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function node(id: string, type: string, x: number): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type, params: {}, definition: def(type), executionStatus: 'idle' },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

const TRIGGER: Edge = {
  id: 't', source: 'start', target: 'a', sourceHandle: 'trigger',
  targetHandle: '__trigger', type: 'triggerEdge', data: { type: 'trigger' },
};

/**
 * start -> a -> [b -> c] -> sink with b and c collapsed into "Block", and a
 * second copy of the block beside it, on a tab whose socket is a stand-in.
 * Returns the two cards' ids.
 */
function graphWithTwoCopies(collapse = ['b', 'c']): { card: string; copy: string } {
  store().setNodes([
    { ...node('start', 'Start', 0), type: 'start' },
    node('a', 'A', 100), node('b', 'B', 200), node('c', 'C', 300), node('sink', 'S', 400),
  ]);
  store().setEdges([
    TRIGGER, dataEdge('e1', 'a', 'b'), dataEdge('e2', 'b', 'c'), dataEdge('e3', 'c', 'sink'),
  ]);
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: collapse.includes(n.id) })));
  expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
  const cardNode = tab().nodes.find((n) => subgraphIdOf(n.data.type))!;
  store().setNodes([
    ...tab().nodes.map((n) => ({ ...n, selected: false })),
    { ...cardNode, id: 'copy', selected: false, position: { x: 250, y: 200 } },
  ]);
  const ws = {
    connected: true,
    on: vi.fn(),
    off: vi.fn(),
    send: vi.fn(),
    connect: vi.fn(async () => {}),
  };
  useTabStore.setState({
    tabs: useTabStore.getState().tabs.map((t) => ({ ...t, ws: ws as never })),
  });
  return { card: cardNode.id, copy: 'copy' };
}

function on(type: string): (frame: unknown) => void {
  const calls = (tab().ws.on as ReturnType<typeof vi.fn>).mock.calls
    .filter(([registered]) => registered === type);
  return calls[calls.length - 1][1];
}

/** Send frames, then apply them as the next animation frame would. */
function frames(...sends: [string, unknown][]) {
  act(() => {
    for (const [type, frame] of sends) on(type)(frame);
    flushTabNodeUpdates();
  });
}

const data = (id: string) => tab().nodes.find((n) => n.id === id)?.data;
const status = (id: string) => data(id)?.executionStatus;

beforeEach(() => {
  vi.mocked(validateGraph).mockReset();
  vi.mocked(validateGraph).mockResolvedValue({ valid: true, errors: [] });
  vi.mocked(getRun).mockReset();
  vi.mocked(getRun).mockResolvedValue(null as never);
  useI18n.setState({ locale: 'en' });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C'), def('S')],
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
});

afterEach(() => {
  dismissValidationToasts();
  discardTabNodeUpdates();
});

describe('a run seen from inside a block', () => {
  it('paints each node inside the open block, and the card at the top level', () => {
    const { card } = graphWithTwoCopies();
    store().enterSubgraph(card);
    renderHook(() => useGraphExecution());

    frames(
      ['node_status', { node_id: card, status: 'running' }],
      ['inner_node_status', { node_id: `${card}/b`, container_id: card, status: 'completed' }],
      ['inner_node_status', { node_id: `${card}/c`, container_id: card, status: 'running' }],
    );

    expect(status('b')).toBe('completed');
    expect(status('c')).toBe('running');
    // Only the card's own frames reach the top level, and no log line comes
    // from an inner frame.
    expect(tab().logs).toEqual([]);

    frames(
      ['inner_node_status', {
        node_id: `${card}/c`, container_id: card, status: 'error', error: 'boom',
      }],
      ['node_status', { node_id: card, status: 'error', error: 'boom' }],
    );
    expect(status('c')).toBe('error');
    expect(data('c')?.error).toBe('boom');

    act(() => store().exitSubgraph());
    expect(status(card)).toBe('error');
  });

  it('never paints one copy of a block with what ran in the other', () => {
    const { card, copy } = graphWithTwoCopies();
    store().enterSubgraph(card);
    renderHook(() => useGraphExecution());

    frames(['inner_node_status', { node_id: `${copy}/b`, container_id: copy, status: 'error' }]);
    expect(status('b')).toBe('idle');

    frames(['inner_node_status', { node_id: `${card}/b`, container_id: card, status: 'completed' }]);
    expect(status('b')).toBe('completed');

    // The other copy, opened after: what ran in IT.
    act(() => store().exitSubgraph());
    act(() => { store().enterSubgraph(copy); });
    expect(status('b')).toBe('error');
    expect(status('c')).toBe('idle');
  });

  it('a block opened after its nodes reported shows what they reported', () => {
    const { card } = graphWithTwoCopies();
    renderHook(() => useGraphExecution());

    frames(
      ['inner_node_status', { node_id: `${card}/b`, container_id: card, status: 'cached' }],
      ['inner_node_status', { node_id: `${card}/c`, container_id: card, status: 'bypassed' }],
    );
    // Nothing at the top level is a node inside the block.
    expect(status('b')).toBeUndefined();

    act(() => { store().enterSubgraph(card); });
    expect(status('b')).toBe('cached');
    expect(status('c')).toBe('bypassed');
  });

  it("puts a card's progress on the node inside it that sent it", () => {
    const { card } = graphWithTwoCopies();
    store().enterSubgraph(card);
    renderHook(() => useGraphExecution());
    const progress = { event: 'batch', batch: 2, total_batches: 4 };

    frames(['node_status', {
      node_id: card, status: 'progress', inner_node_id: `${card}/c`,
      outputs: [{ output_kind: 'progress', progress }],
    }]);

    expect(data('c')?.progress).toEqual(progress);
    expect(data('b')?.progress).toBeUndefined();
    act(() => store().exitSubgraph());
    expect(data(card)?.progress).toEqual(progress);
  });

  it('paints a block nested inside the open one on the level it is drawn on', () => {
    const { card } = graphWithTwoCopies(['a', 'b', 'c']);
    store().enterSubgraph(card);
    // Inside Block: collapse b and c into a block of their own, and open it.
    store().setNodes(tab().nodes.map((n) => ({ ...n, selected: ['b', 'c'].includes(n.id) })));
    expect(store().collapseSelectionToSubgraph('Inner').ok).toBe(true);
    const nest = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    store().enterSubgraph(nest);
    renderHook(() => useGraphExecution());

    frames(
      ['inner_node_status', { node_id: `${card}/${nest}/c`, container_id: card, status: 'completed' }],
      ['inner_node_status', { node_id: `${card}/${nest}`, container_id: card, status: 'completed' }],
      ['inner_node_status', { node_id: `${card}/a`, container_id: card, status: 'running' }],
    );
    expect(status('c')).toBe('completed');

    // One level up, the nested card and its sibling took theirs while hidden.
    act(() => store().exitSubgraph());
    expect(status(nest)).toBe('completed');
    expect(status('a')).toBe('running');
  });

  it('a Stop settles a node inside a block that it left running', () => {
    const { card } = graphWithTwoCopies();
    store().enterSubgraph(card);
    renderHook(() => useGraphExecution());

    frames(
      ['inner_node_status', { node_id: `${card}/b`, container_id: card, status: 'completed' }],
      ['inner_node_status', { node_id: `${card}/c`, container_id: card, status: 'running' }],
    );
    frames(['execution_stopped', { run_id: 'run-1' }]);

    expect(status('c')).toBe('interrupted');
    expect(status('b')).toBe('completed');
    expect(tab().innerRunStates?.[`${card}/c`]?.executionStatus).toBe('interrupted');
  });

  it("a new run clears the last run's inner states, open or not", async () => {
    const { card, copy } = graphWithTwoCopies();
    store().enterSubgraph(card);
    const { result } = renderHook(() => useGraphExecution());
    frames(
      ['inner_node_status', { node_id: `${card}/b`, container_id: card, status: 'completed' }],
      ['inner_node_status', { node_id: `${copy}/b`, container_id: copy, status: 'error' }],
    );

    await act(async () => {
      await result.current.execute();
    });

    expect(status('b')).toBe('idle');
    expect(tab().innerRunStates).toEqual({});
    act(() => store().exitSubgraph());
    act(() => { store().enterSubgraph(copy); });
    expect(status('b')).toBe('idle');
  });

  it('ignores an inner frame that names no node or status', () => {
    const { card } = graphWithTwoCopies();
    store().enterSubgraph(card);
    renderHook(() => useGraphExecution());
    const before = tab();

    frames(
      ['inner_node_status', { status: 'completed' }],
      ['inner_node_status', { node_id: `${card}/b` }],
    );

    expect(tab()).toBe(before);
  });
});

describe('a bypassed node', () => {
  it('takes the status on its card without a log line', () => {
    graphWithTwoCopies();
    renderHook(() => useGraphExecution());

    frames(['node_status', { node_id: 'a', status: 'bypassed' }]);

    expect(status('a')).toBe('bypassed');
    expect(tab().logs).toEqual([]);
  });
});
