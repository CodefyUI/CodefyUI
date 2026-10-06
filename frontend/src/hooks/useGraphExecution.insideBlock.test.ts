/**
 * Run pressed while a block is open (found during the browser e2e of the
 * fixes for #618-#625). A Start node cannot go inside a block, and a Run from
 * inside one always refused with "no entry point". It runs the whole graph,
 * as a Run from the top level does.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';

import { useGraphExecution } from './useGraphExecution';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { discardTabNodeUpdates, flushTabNodeUpdates } from '../store/nodeUpdateQueue';
import { useToastStore } from '../store/toastStore';
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
const validateGraphMock = vi.mocked(validateGraph);

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();
const toasts = () => useToastStore.getState().toasts;

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
 * start -> a -> [b -> c] -> sink with b and c collapsed into "Block", on a tab
 * whose socket is a stand-in; returns the block's card id.
 */
function graphWithABlock({ withStart = true, collapse = ['b', 'c'] } = {}): string {
  store().setNodes([
    ...(withStart ? [{ ...node('start', 'Start', 0), type: 'start' }] : []),
    node('a', 'A', 100), node('b', 'B', 200), node('c', 'C', 300), node('sink', 'S', 400),
  ]);
  store().setEdges([
    ...(withStart ? [TRIGGER] : []),
    dataEdge('e1', 'a', 'b'), dataEdge('e2', 'b', 'c'), dataEdge('e3', 'c', 'sink'),
  ]);
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: collapse.includes(n.id) })));
  expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
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
  return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
}

const sent = () => (tab().ws.send as ReturnType<typeof vi.fn>).mock.calls.map((call) => call[0]);

async function run(result: { current: ReturnType<typeof useGraphExecution> }) {
  await act(async () => {
    await result.current.execute();
  });
}

beforeEach(() => {
  validateGraphMock.mockReset();
  validateGraphMock.mockResolvedValue({ valid: true, errors: [] });
  vi.mocked(getRun).mockReset();
  vi.mocked(getRun).mockResolvedValue(null as never);
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
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

describe('a run from inside a block', () => {
  it('runs the whole graph and sends the frame a run from the top level sends', async () => {
    const card = graphWithABlock();
    const blockId = tab().subgraphs[0].id;
    store().enterSubgraph(card);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(toasts()).toEqual([]);
    expect(sent()).toHaveLength(1);
    const inside = sent()[0];
    expect(inside.action).toBe('execute');
    expect(inside.nodes.map((n: { id: string }) => n.id).sort()).toEqual(
      ['a', card, 'sink', 'start'].sort(),
    );
    expect(inside.subgraphs.map((d: { id: string }) => d.id)).toEqual([blockId]);
    // What the server is asked to check is the same whole graph.
    expect(validateGraphMock.mock.calls[0][0].map((n: { id: string }) => n.id).sort()).toEqual(
      ['a', card, 'sink', 'start'].sort(),
    );

    // The same graph from the top level, with Main.
    act(() => store().exitAllSubgraphs());
    await run(result);
    const top = sent()[1];
    const graphOf = (frame: Record<string, unknown>) => ({
      nodes: frame.nodes, edges: frame.edges, presets: frame.presets,
      subgraphs: frame.subgraphs, settings: frame.settings, name: frame.name,
    });
    expect(graphOf(inside)).toEqual(graphOf(top));
  });

  it('runs when the Start is wired to the card of the open block', async () => {
    // The e2e graph: collapsing the node Start triggers moves the trigger
    // onto the card, and the block's inside then holds no trigger at all.
    const card = graphWithABlock({ collapse: ['a', 'b'] });
    expect(tab().edges.some((e) => e.source === 'start' && e.target === card)).toBe(true);
    store().enterSubgraph(card);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(toasts()).toEqual([]);
    expect(sent()).toHaveLength(1);
    expect(sent()[0].nodes.map((n: { id: string }) => n.id).sort()).toEqual(
      [card, 'c', 'sink', 'start'].sort(),
    );
  });

  it('names what changed as a run from the top level does', async () => {
    // Collapsing marks the new card changed; from the top level a run also
    // names what is downstream of it, along the whole graph's wires.
    const card = graphWithABlock();
    const fromTop = store().getDirtyWithDownstream();
    expect(fromTop).toEqual([card, 'sink']);
    store().enterSubgraph(card);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(sent()[0].changed_nodes).toEqual(fromTop);
  });

  it('names the open block by its current name when the server refuses the graph', async () => {
    const card = graphWithABlock();
    store().enterSubgraph(card);
    store().renameSubgraph(tab().subgraphStack[0].subgraphId, 'Renamed');
    validateGraphMock.mockResolvedValueOnce({
      valid: false,
      errors: [],
      issues: [{
        message: `Missing required input 'in' on node ${card}/b`,
        code: 'missing_input',
        node_id: `${card}/b`,
        params: { port: 'in', type: 'B' },
      }],
    } as never);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(toasts().map((toast) => toast.message)).toEqual(['Renamed: input "in" is not connected']);
    expect(sent()).toEqual([]);
  });

  describe('what a run reports while a block is open', () => {
    // The server reports a node inside a block under its outermost card's
    // id, so every id a run reports is the top level's.
    function on(type: string): (frame: unknown) => void {
      const calls = (tab().ws.on as ReturnType<typeof vi.fn>).mock.calls
        .filter(([registered]) => registered === type);
      return calls[calls.length - 1][1];
    }
    const status = (id: string) => tab().nodes.find((n) => n.id === id)?.data.executionStatus;

    it('names each node in the run log by its title', () => {
      const card = graphWithABlock();
      store().renameNode('sink', 'Output');
      store().enterSubgraph(card);
      store().renameSubgraph(tab().subgraphStack[0].subgraphId, 'Renamed');
      renderHook(() => useGraphExecution());

      act(() => {
        on('node_status')({ node_id: card, status: 'completed' });
        on('node_status')({ node_id: 'sink', status: 'completed' });
      });

      expect(tab().logs.map((entry) => entry.message)).toEqual([
        'Node Renamed completed',
        'Node Output completed',
      ]);
    });

    it('a run from inside a block clears the last run from the top-level cards', async () => {
      const card = graphWithABlock();
      act(() => store().setTabNodeExecutionStatus(tab().id, card, 'error', 'the last run'));
      store().enterSubgraph(card);
      const { result } = renderHook(() => useGraphExecution());

      await run(result);
      act(() => store().exitSubgraph());

      expect(status(card)).toBe('idle');
    });

    it('the top-level cards take the statuses it reports', () => {
      const card = graphWithABlock();
      store().enterSubgraph(card);
      renderHook(() => useGraphExecution());

      act(() => {
        on('node_status')({ node_id: card, status: 'completed' });
        flushTabNodeUpdates();
      });
      act(() => store().exitSubgraph());

      expect(status(card)).toBe('completed');
    });

    describe('a Stop', () => {
      // A block's card closes only when all its inner nodes have reported,
      // and a Stop leaves some of them never run: no frame ever ends the card.
      it('settles a card it left running, as a node that stopped early', () => {
        const card = graphWithABlock();
        renderHook(() => useGraphExecution());

        act(() => {
          on('node_status')({ node_id: 'a', status: 'completed' });
          on('node_status')({ node_id: card, status: 'running' });
          flushTabNodeUpdates();
          on('execution_stopped')({ run_id: 'run-1' });
          flushTabNodeUpdates();
        });

        expect(status(card)).toBe('interrupted');
        // A node that finished keeps what it reported.
        expect(status('a')).toBe('completed');
        expect(tab().status).toBe('idle');
      });

      it('settles it on a run from inside a block too', () => {
        const card = graphWithABlock();
        store().enterSubgraph(card);
        renderHook(() => useGraphExecution());

        act(() => {
          on('node_status')({ node_id: card, status: 'running' });
          flushTabNodeUpdates();
          on('execution_stopped')({ run_id: 'run-1' });
          flushTabNodeUpdates();
        });
        act(() => store().exitSubgraph());

        expect(status(card)).toBe('interrupted');
      });

      it('settles it when the server reports the run cancelled on attach', () => {
        const card = graphWithABlock();
        renderHook(() => useGraphExecution());

        act(() => {
          on('node_status')({ node_id: card, status: 'running' });
          flushTabNodeUpdates();
          on('attached')({ run_id: 'run-1', status: 'cancelled' });
          flushTabNodeUpdates();
        });

        expect(status(card)).toBe('interrupted');
      });
    });

    it('an interrupted run settles a top-level card it left running', () => {
      const card = graphWithABlock();
      store().enterSubgraph(card);
      renderHook(() => useGraphExecution());

      act(() => {
        on('node_status')({ node_id: card, status: 'running' });
        flushTabNodeUpdates();
        on('execution_stopped')({ run_id: 'run-1', reason: 'interrupted' });
        flushTabNodeUpdates();
      });
      act(() => store().exitSubgraph());

      expect(status(card)).toBe('interrupted');
    });
  });

  it('still refuses a graph with no Start node anywhere', async () => {
    const card = graphWithABlock({ withStart: false });
    store().enterSubgraph(card);
    const { result } = renderHook(() => useGraphExecution());

    await run(result);

    expect(toasts().map((toast) => toast.message)).toEqual([
      useI18n.getState().t('execution.error.noEntryPoints'),
    ]);
    expect(sent()).toEqual([]);
  });
});
