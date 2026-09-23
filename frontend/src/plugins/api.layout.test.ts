/**
 * The `auto_layout` op and the view (#401 item 3).
 *
 * The toolbar's Auto Layout lays the graph out and then fits the view to the
 * result. The op did only the first half, so a plugin that tidied a graph the
 * user had panned away from left them looking at empty canvas, and the plugin
 * API has no viewport call a plugin could use to correct it. Graph Copilot
 * ends every structural batch with `auto_layout`.
 *
 * A fit request names no tab, and the canvas on screen consumes it, so one is
 * only ever made for the tab on screen. A tab in the background forgets its
 * remembered pan and zoom instead, and the next switch to it fits its nodes.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useProjectStore } from '../store/projectStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import {
  _resetViewportMemory, recallViewport, rememberViewport,
} from '../utils/viewportMemory';
import { buildPluginAPI } from './api';
import type { GraphOp } from './ops';
import type { NodeDefinition } from '../types';

vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const fitRequest = () => useUIStore.getState().layoutFitRequest;

const DEFS: NodeDefinition[] = [
  {
    node_name: 'Source', category: 'Layer', description: '',
    inputs: [],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  },
  {
    node_name: 'Sink', category: 'Layer', description: '',
    inputs: [{ name: 'x', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [], params: [],
  },
];

/**
 * The size the layout and the fit give a node the canvas has not measured
 * (`NODE_W` x `NODE_H` in `autoLayout.ts`). Nothing in these tests is measured.
 */
const NODE_W = 200;
const NODE_H = 80;

function freshApi(pluginId = 'test-plugin') {
  return buildPluginAPI(pluginId, () => document.createElement('div'));
}

/** Two nodes far off the default view, wired, then laid out. */
const LAYOUT_BATCH: GraphOp[] = [
  { op: 'add_node', node_type: 'Source', ref: 'a', position: { x: 5000, y: 5000 } },
  { op: 'add_node', node_type: 'Sink', ref: 'b', position: { x: 9000, y: 9000 } },
  { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
  { op: 'auto_layout' },
];

/** The box the nodes fill, at the unmeasured size. */
function boxOf(nodes: Array<{ position: { x: number; y: number } }>) {
  const left = Math.min(...nodes.map((n) => n.position.x));
  const top = Math.min(...nodes.map((n) => n.position.y));
  const right = Math.max(...nodes.map((n) => n.position.x + NODE_W));
  const bottom = Math.max(...nodes.map((n) => n.position.y + NODE_H));
  return { x: left, y: top, width: right - left, height: bottom - top };
}

beforeEach(() => {
  // Reset FIRST: a case that opened a project would otherwise leave the
  // autosave scoped to it for every case after.
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('live');
  useNodeDefStore.setState({ definitions: DEFS, presets: [] } as never);
  useUIStore.setState({ layoutFitRequest: null });
  useToastStore.setState({ toasts: [] });
  _resetViewportMemory();
  window.localStorage.clear();
});

describe('auto_layout fits the view on the tab the user is looking at', () => {
  it('the legacy graph.applyOperations fits the view to the laid-out nodes', () => {
    const api = freshApi();

    const result = api.graph.applyOperations(LAYOUT_BATCH);

    expect(result.results.every((r) => r.ok)).toBe(true);
    const laid = store().getActiveTab().nodes;
    expect(laid).toHaveLength(2);
    // The layout really ran: neither node is where the batch put it.
    expect(laid.every((n) => n.position.x < 5000)).toBe(true);
    expect(fitRequest()).not.toBeNull();
    expect(fitRequest()!.bounds).toEqual(boxOf(laid));
  });

  it('workspace.applyOperations on the active tab fits the view as well', () => {
    const api = freshApi();

    const result = api.workspace.applyOperations({ operations: LAYOUT_BATCH });

    expect(result.committed).toBe(true);
    expect(fitRequest()).not.toBeNull();
    expect(fitRequest()!.bounds).toEqual(boxOf(store().getActiveTab().nodes));
  });

  it('the box takes in a note bound to a laid-out node and leaves an unbound note out', () => {
    // The rule of the toolbar's Layout All: a bound note moves with its node,
    // so it is part of what was laid out; an unbound note stays where it is,
    // and a box reaching for it could fit the graph down to a speck.
    const api = freshApi();

    const result = api.graph.applyOperations([
      { op: 'add_node', node_type: 'Source', ref: 'a', position: { x: 5000, y: 5000 } },
      { op: 'add_node', node_type: 'Sink', ref: 'b', position: { x: 9000, y: 9000 } },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
      { op: 'add_note', ref: 'explains', text: 'why a', bind_to: 'a' },
      { op: 'add_note', ref: 'aside', text: 'far away', position: { x: -20000, y: -20000 } },
      { op: 'auto_layout' },
    ]);

    expect(result.results.every((r) => r.ok)).toBe(true);
    const nodes = store().getActiveTab().nodes;
    const bound = nodes.find((n) => n.id === result.refs.explains)!;
    const loose = nodes.find((n) => n.id === result.refs.aside)!;
    const laidOut = nodes.filter((n) => n.type !== 'noteNode');
    expect(fitRequest()).not.toBeNull();
    const { bounds } = fitRequest()!;
    // The bound note rode along above its node, so it sets the top of the box.
    expect(bound.position.y).toBeLessThan(Math.min(...laidOut.map((n) => n.position.y)));
    expect(bounds).toEqual(boxOf([...laidOut, bound]));
    // The unbound note did not move, and the box does not reach for it.
    expect(loose.position).toEqual({ x: -20000, y: -20000 });
    expect(bounds.x).toBeGreaterThan(loose.position.x + NODE_W);
  });

  it('inside a block, the legacy path fits the block it laid out', () => {
    // The legacy path writes the canvas the user has open, and inside a block
    // that canvas is the block -- so its laid-out nodes are what to frame.
    store().setNodes([{
      id: 'inst', type: 'baseNode', position: { x: 0, y: 0 },
      data: { label: 'Encoder', type: 'subgraph:blk', params: {} },
    } as never]);
    store().setSubgraphs([{
      id: 'blk', name: 'Encoder', description: '',
      nodes: [
        { id: 'in1', type: 'Source', position: { x: 3000, y: 3000 }, data: { params: {} } },
        { id: 'in2', type: 'Sink', position: { x: 6000, y: 6000 }, data: { params: {} } },
      ],
      edges: [
        { id: 'ie1', source: 'in1', target: 'in2', sourceHandle: 'out', targetHandle: 'x' },
      ],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    } as never]);
    expect(store().enterSubgraph('inst')).toBe(true);
    const api = freshApi();

    const result = api.graph.applyOperations([{ op: 'auto_layout' }]);

    expect(result.results[0].ok).toBe(true);
    const inner = store().getActiveTab().nodes;
    expect(inner.map((n) => n.id).sort()).toEqual(['in1', 'in2']);
    expect(fitRequest()).not.toBeNull();
    expect(fitRequest()!.bounds).toEqual(boxOf(inner));
  });
});

describe('auto_layout on a tab in the background', () => {
  it('asks for no fit, and forgets that tab\'s viewport so the next visit fits it', () => {
    const api = freshApi();
    const live = store().activeTabId;
    const bg = store().createTab({ activate: false });
    rememberViewport(live, { x: 7, y: 8, zoom: 0.5 });
    rememberViewport(bg, { x: 1, y: 2, zoom: 3 });

    const result = api.workspace.applyOperations({ tabId: bg, operations: LAYOUT_BATCH });

    expect(result.committed).toBe(true);
    // A fit request would move the canvas on screen -- the ACTIVE tab's -- to
    // coordinates that belong to a graph the user is not looking at.
    expect(fitRequest()).toBeNull();
    // Nothing remembered, so switching to the tab fits its nodes rather than
    // restoring a pan and zoom aimed at where they used to be.
    expect(recallViewport(bg)).toBeUndefined();
    // The tab on screen keeps where the user left it.
    expect(recallViewport(live)).toEqual({ x: 7, y: 8, zoom: 0.5 });
  });
});

describe('what does not move the view', () => {
  it('a committed batch without auto_layout', () => {
    const api = freshApi();
    const result = api.workspace.applyOperations({
      operations: [{ op: 'add_node', node_type: 'Source' }],
    });
    expect(result.committed).toBe(true);
    expect(fitRequest()).toBeNull();
  });

  it('an auto_layout that moves nothing, which commits nothing either (#397)', () => {
    const api = freshApi();
    expect(api.workspace.applyOperations({ operations: LAYOUT_BATCH }).committed).toBe(true);
    useUIStore.setState({ layoutFitRequest: null });

    const again = api.workspace.applyOperations({ operations: [{ op: 'auto_layout' }] });

    expect(again.results).toEqual([{ index: 0, ok: true }]);
    expect(again.committed).toBe(false);
    expect(fitRequest()).toBeNull();
  });

  it('a batch a read-only tab refuses', () => {
    const api = freshApi();
    store().setTabReadOnly(true);

    const result = api.graph.applyOperations(LAYOUT_BATCH);

    expect(result.results.every((r) => !r.ok)).toBe(true);
    expect(fitRequest()).toBeNull();
  });

  it('an auto_layout with nothing but notes to lay out', () => {
    // The layout leaves notes alone, so a batch that only adds one has moved
    // nothing the view should follow.
    const api = freshApi();
    const result = api.workspace.applyOperations({
      operations: [{ op: 'add_note', text: 'only me' }, { op: 'auto_layout' }],
    });
    expect(result.committed).toBe(true);
    expect(fitRequest()).toBeNull();
  });

  it('a committed batch whose auto_layout moves nothing, on screen or behind', () => {
    // A label edit (or a `set_params`) that ends in `auto_layout`, on a graph
    // already laid out: the batch commits, no node moves, and a fit would pull
    // a user who had zoomed in back out for nothing.
    const api = freshApi();
    const bg = store().createTab({ activate: false });
    const onScreen = api.workspace.applyOperations({ operations: LAYOUT_BATCH });
    const behind = api.workspace.applyOperations({ tabId: bg, operations: LAYOUT_BATCH });
    useUIStore.setState({ layoutFitRequest: null });
    rememberViewport(bg, { x: 1, y: 2, zoom: 3 });

    const relabelled = api.workspace.applyOperations({
      operations: [
        { op: 'set_node_meta', node_id: onScreen.refs.a, label: 'Input' },
        { op: 'auto_layout' },
      ],
    });
    const relabelledBehind = api.workspace.applyOperations({
      tabId: bg,
      operations: [
        { op: 'set_node_meta', node_id: behind.refs.a, label: 'Input' },
        { op: 'auto_layout' },
      ],
    });

    expect([...relabelled.results, ...relabelledBehind.results].every((r) => r.ok)).toBe(true);
    expect(relabelled.committed).toBe(true);
    expect(relabelledBehind.committed).toBe(true);
    expect(fitRequest()).toBeNull();
    expect(recallViewport(bg)).toEqual({ x: 1, y: 2, zoom: 3 });
  });

  it('a note added or moved in a batch that ends in auto_layout, on screen or behind', () => {
    // The layout leaves notes where they are, so a note arriving or moving is
    // no reason to move the view: the laid-out graph is still where it was.
    // Counting notes as laid-out nodes would fit (or forget) on every note.
    const api = freshApi();
    const bg = store().createTab({ activate: false });
    api.workspace.applyOperations({ operations: LAYOUT_BATCH });
    api.workspace.applyOperations({ tabId: bg, operations: LAYOUT_BATCH });
    useUIStore.setState({ layoutFitRequest: null });
    rememberViewport(bg, { x: 1, y: 2, zoom: 3 });
    const noteThenLayout: GraphOp[] = [{ op: 'add_note', text: 'x' }, { op: 'auto_layout' }];

    const added = api.workspace.applyOperations({ operations: noteThenLayout });
    const addedBehind = api.workspace.applyOperations({ tabId: bg, operations: noteThenLayout });
    const moveThenLayout = (noteId: string): GraphOp[] => [
      { op: 'move_node', node_id: noteId, position: { x: -900, y: -900 } },
      { op: 'auto_layout' },
    ];
    const moved = api.workspace.applyOperations({
      operations: moveThenLayout(added.results[0].node_id!),
    });
    const movedBehind = api.workspace.applyOperations({
      tabId: bg, operations: moveThenLayout(addedBehind.results[0].node_id!),
    });

    for (const result of [added, addedBehind, moved, movedBehind]) {
      expect(result.committed).toBe(true);
      expect(result.results.every((r) => r.ok)).toBe(true);
    }
    expect(fitRequest()).toBeNull();
    expect(recallViewport(bg)).toEqual({ x: 1, y: 2, zoom: 3 });
  });
});

it('a batch that removes a node and lays out still fits the view to what is left', () => {
  // The node left standing does not move -- it is already where the layout
  // puts a graph of one -- but the graph shrank, and that is a change too.
  const api = freshApi();
  const laid = api.workspace.applyOperations({ operations: LAYOUT_BATCH });
  const before = store().getActiveTab().nodes.find((n) => n.id === laid.refs.a)!.position;
  useUIStore.setState({ layoutFitRequest: null });

  const result = api.workspace.applyOperations({
    operations: [{ op: 'remove_node', node_id: laid.refs.b }, { op: 'auto_layout' }],
  });

  expect(result.committed).toBe(true);
  const left = store().getActiveTab().nodes;
  expect(left.map((n) => n.id)).toEqual([laid.refs.a]);
  expect(left[0].position).toEqual(before);
  expect(fitRequest()).not.toBeNull();
  expect(fitRequest()!.bounds).toEqual(boxOf(left));
});

it('shows no toast, even with an unbound note on the canvas', () => {
  // The toolbar warns that an unbound note stays put. A plugin sends
  // auto_layout after every batch, so the same warning would answer each one:
  // a toast for a click the user never made. The reference states the rule
  // for plugin authors instead.
  const api = freshApi();
  api.graph.applyOperations([{ op: 'add_note', text: 'loose', position: { x: -500, y: -500 } }]);

  api.graph.applyOperations(LAYOUT_BATCH);

  expect(useToastStore.getState().toasts).toEqual([]);
});
