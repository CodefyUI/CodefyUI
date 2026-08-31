/**
 * `api.workspace` — the apiVersion 5 surface (#341).
 *
 * Four things a plugin could not do before: open a graph as a tab without
 * moving the user, read a tab that is not on screen, write to a named tab
 * only if it has not changed since it last looked, and hear about all of it.
 *
 * The read-only refusal on the LEGACY path is here too, because it is the one
 * behaviour an installed plugin can notice (#341 section 4.6).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useTabStore, _persistedTabsForTesting } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { buildPluginAPI } from './api';
import type { NodeDefinition } from '../types';

vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();

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

function freshApi(pluginId = 'test-plugin') {
  return buildPluginAPI(pluginId, () => document.createElement('div'));
}

/** A serialized two-node chain, in the shape `getGraph()` answers with. */
function candidateGraph(name = 'candidate') {
  return {
    name,
    description: '',
    nodes: [
      { id: 'a', type: 'Source', position: { x: 0, y: 0 }, data: { params: {} } },
      { id: 'b', type: 'Sink', position: { x: 200, y: 0 }, data: { params: {} } },
    ],
    edges: [
      { id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'x' },
    ],
    presets: [],
    segmentGroups: [],
    subgraphs: [],
  };
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('live');
  useNodeDefStore.setState({ definitions: DEFS, presets: [] } as never);
  window.localStorage.clear();
});

describe('meta', () => {
  it('reports at least apiVersion 5', () => {
    // The documented v5 feature check is `api.apiVersion >= 5`, so that is
    // what this asserts; the exact number lives in `api.view.test.ts`.
    expect(freshApi().apiVersion).toBeGreaterThanOrEqual(5);
  });

  it('offers the whole workspace surface as callable functions', () => {
    const api = freshApi();
    for (const member of ['openGraphs', 'tabs', 'snapshot', 'applyOperations', 'onChanged']) {
      expect(
        typeof (api.workspace as unknown as Record<string, unknown>)[member],
        `workspace.${member}`,
      ).toBe('function');
    }
  });

  it('every apiVersion 4 member is still there', () => {
    const V4_SHAPE: Record<string, string[]> = {
      ui: ['addFloatingWidget', 'toast', 'addPanel', 'removePanel', 'addToolbarButton', 'removeToolbarButton'],
      graph: ['getGraph', 'getNodeDefinitions', 'applyOperations', 'onGraphChanged', 'getView'],
      nodes: ['registerRenderer'],
      events: ['onExecution'],
      runs: ['list', 'get', 'metrics'],
      http: ['fetch'],
      storage: ['get', 'set', 'remove'],
    };
    const api = freshApi() as unknown as Record<string, Record<string, unknown>>;
    for (const [section, members] of Object.entries(V4_SHAPE)) {
      expect(api[section], `section ${section}`).toBeDefined();
      for (const member of members) {
        expect(typeof api[section][member], `${section}.${member}`).toBe('function');
      }
    }
  });
});

describe('workspace.openGraphs', () => {
  it('opens a labelled tab without disturbing the live graph', () => {
    const api = freshApi();
    const live = store().activeTabId;
    const [result] = api.workspace.openGraphs(
      [{ title: 'Variant A', graph: candidateGraph(), readOnly: true,
         source: { kind: 'agent-variant', pluginId: 'test-plugin', jobId: 'j1' } }],
      { activate: 'none' },
    );
    expect('tabId' in result).toBe(true);
    if (!('tabId' in result)) return;

    expect(store().activeTabId).toBe(live);
    const tab = store().getTab(result.tabId)!;
    expect(tab.name).toBe('Variant A');
    expect(tab.nodes.map((n) => n.id)).toEqual(['a', 'b']);
    expect(tab.edges).toHaveLength(1);
    expect(tab.readOnly).toBe(true);
    expect(tab.transient).toBe(true);
    expect(tab.source).toEqual({ kind: 'agent-variant', pluginId: 'test-plugin', jobId: 'j1' });
    expect(result.revision).toBe(tab.revision);
  });

  it('is positional: one bad entry does not sink its neighbours', () => {
    const api = freshApi();
    const results = api.workspace.openGraphs([
      { title: 'Good one', graph: candidateGraph('one') },
      { title: '   ', graph: candidateGraph('two') },
      { title: 'Good two', graph: candidateGraph('three') },
    ], { activate: 'none' });

    expect(results).toHaveLength(3);
    expect('tabId' in results[0]).toBe(true);
    expect(results[1]).toMatchObject({ code: 'invalid_graph' });
    expect('tabId' in results[2]).toBe(true);
  });

  it('refuses a graph over the 8 MiB limit with too_large', () => {
    const api = freshApi();
    const huge = candidateGraph();
    huge.description = 'x'.repeat(9 * 1024 * 1024);
    const [result] = api.workspace.openGraphs([{ title: 'Huge', graph: huge }], { activate: 'none' });
    expect(result).toMatchObject({ code: 'too_large' });
    expect(store().tabs).toHaveLength(1);
  });

  it('refuses past 32 tabs with too_many_tabs', () => {
    const api = freshApi();
    while (store().tabs.length < 32) store().createTab({ activate: false });
    const [result] = api.workspace.openGraphs(
      [{ title: 'One too many', graph: candidateGraph() }], { activate: 'none' },
    );
    expect(result).toMatchObject({ code: 'too_many_tabs' });
    expect(store().tabs).toHaveLength(32);
  });

  it('reports a reader failure as invalid_graph and opens no tab', () => {
    const api = freshApi();
    const [result] = api.workspace.openGraphs(
      // `nodes` is not a list, so the document reader cannot walk it.
      [{ title: 'Broken', graph: { nodes: 'not-a-list', edges: [] } as never }],
      { activate: 'none' },
    );
    expect(result).toMatchObject({ code: 'invalid_graph' });
    expect(store().tabs).toHaveLength(1);
  });

  it('honours activate: first, last and none', () => {
    const api = freshApi();
    const live = store().activeTabId;

    const none = api.workspace.openGraphs(
      [{ title: 'n1', graph: candidateGraph() }], { activate: 'none' });
    expect(store().activeTabId).toBe(live);

    const first = api.workspace.openGraphs([
      { title: 'f1', graph: candidateGraph() },
      { title: 'f2', graph: candidateGraph() },
    ]);
    expect(store().activeTabId).toBe((first[0] as { tabId: string }).tabId);

    const last = api.workspace.openGraphs([
      { title: 'l1', graph: candidateGraph() },
      { title: 'l2', graph: candidateGraph() },
    ], { activate: 'last' });
    expect(store().activeTabId).toBe((last[1] as { tabId: string }).tabId);
    void none;
  });

  it('persist: true opts the tab back into surviving a reload', () => {
    const api = freshApi();
    const [kept] = api.workspace.openGraphs(
      [{ title: 'Kept', graph: candidateGraph(), persist: true }], { activate: 'none' });
    expect(store().getTab((kept as { tabId: string }).tabId)!.transient).toBe(false);
  });

  it('an ACTIVATED transient tab leaves the next reload on a real tab', () => {
    // The one state where the persisted pointer names a tab that will not come
    // back: `saveTabs` writes the unfiltered `activeTabId` alongside records
    // `persistedTabsFor` has dropped the transient tab from. The reader's
    // fallback -- `tabs[0]` when no record carries the id (`loadTabs`) -- is
    // what keeps that from being a crash, so it is asserted here against the
    // records this state really produces.
    const api = freshApi();
    const live = store().activeTabId;
    const [opened] = api.workspace.openGraphs(
      [{ title: 'Candidate', graph: candidateGraph() }], { activate: 'first' });
    const { tabId } = opened as { tabId: string };
    expect(store().activeTabId).toBe(tabId);

    const records = _persistedTabsForTesting(store().tabs);
    expect(records.map((r) => r.id)).not.toContain(tabId);
    const restored = records.some((r) => r.id === store().activeTabId)
      ? store().activeTabId
      : records[0].id;
    expect(restored).toBe(live);
  });
});

describe('workspace.tabs and workspace.snapshot', () => {
  it('lists every tab in bar order, marking the active one', () => {
    const api = freshApi();
    const live = store().activeTabId;
    api.workspace.openGraphs([{ title: 'Second', graph: candidateGraph() }], { activate: 'none' });

    const tabs = api.workspace.tabs();
    expect(tabs).toHaveLength(2);
    expect(tabs.map((t) => t.tabId)).toEqual(store().tabs.map((t) => t.id));
    expect(tabs.filter((t) => t.active).map((t) => t.tabId)).toEqual([live]);
    expect(tabs[1].title).toBe('Second');
    expect(tabs[1].transient).toBe(true);
  });

  it('snapshots a background tab, whole graph included', () => {
    const api = freshApi();
    const [opened] = api.workspace.openGraphs(
      [{ title: 'Background', graph: candidateGraph(), source: { kind: 'k', pluginId: 'p' } }],
      { activate: 'none' },
    );
    const { tabId } = opened as { tabId: string };

    const snap = api.workspace.snapshot(tabId);
    expect('error' in snap).toBe(false);
    if ('error' in snap) return;
    expect(snap.tabId).toBe(tabId);
    expect(snap.active).toBe(false);
    expect(snap.source).toEqual({ kind: 'k', pluginId: 'p' });
    expect(snap.graph.nodes.map((n: { id: string }) => n.id)).toEqual(['a', 'b']);
    expect(snap.graph.edges).toHaveLength(1);
  });

  it('snapshot() with no id answers for the active tab', () => {
    const api = freshApi();
    const snap = api.workspace.snapshot();
    expect('error' in snap).toBe(false);
    if ('error' in snap) return;
    expect(snap.tabId).toBe(store().activeTabId);
    expect(snap.active).toBe(true);
  });

  it('an unknown id is an error result, never a throw', () => {
    const api = freshApi();
    expect(api.workspace.snapshot('no-such-tab')).toEqual({ error: 'unknown_tab' });
  });
});

describe('workspace.applyOperations', () => {
  function openEditable() {
    const api = freshApi();
    const [opened] = api.workspace.openGraphs(
      [{ title: 'Target', graph: candidateGraph() }], { activate: 'none' });
    return { api, tabId: (opened as { tabId: string }).tabId };
  }

  it('commits to a named tab and advances the revision by one', () => {
    const { api, tabId } = openEditable();
    const before = store().getTab(tabId)!.revision;

    const result = api.workspace.applyOperations({
      tabId,
      expectedRevision: before,
      operations: [{ op: 'add_node', node_type: 'Source' }],
    });

    expect(result.committed).toBe(true);
    expect(result.conflict).toBeUndefined();
    expect(result.tabId).toBe(tabId);
    expect(result.revision).toBe(before + 1);
    expect(result.node_count).toBe(3);
    expect(store().getTab(tabId)!.nodes).toHaveLength(3);
  });

  it('refuses on a stale revision, and hands back the current one', () => {
    const { api, tabId } = openEditable();
    const stale = store().getTab(tabId)!.revision;
    // Somebody else edits the tab in the meantime.
    api.workspace.applyOperations({ tabId, operations: [{ op: 'add_node', node_type: 'Sink' }] });
    const current = store().getTab(tabId)!.revision;

    const result = api.workspace.applyOperations({
      tabId,
      expectedRevision: stale,
      operations: [{ op: 'clear_graph' }],
    });

    expect(result.conflict).toBe('revision_mismatch');
    expect(result.committed).toBe(false);
    expect(result.results).toEqual([]);
    // The plugin can re-arm without a second read, and the tab is untouched.
    expect(result.revision).toBe(current);
    expect(store().getTab(tabId)!.nodes.length).toBeGreaterThan(0);
  });

  it('refuses a read-only tab before it reduces anything', () => {
    const api = freshApi();
    const [opened] = api.workspace.openGraphs(
      [{ title: 'Frozen', graph: candidateGraph(), readOnly: true }], { activate: 'none' });
    const { tabId } = opened as { tabId: string };
    const before = store().getTab(tabId)!.revision;

    const result = api.workspace.applyOperations({
      tabId, operations: [{ op: 'clear_graph' }],
    });

    expect(result.conflict).toBe('read_only');
    expect(result.committed).toBe(false);
    expect(result.results).toEqual([]);
    expect(result.revision).toBe(before);
    expect(store().getTab(tabId)!.nodes).toHaveLength(2);
  });

  it('an unknown tab is a conflict with revision 0', () => {
    const api = freshApi();
    const result = api.workspace.applyOperations({
      tabId: 'gone', operations: [{ op: 'clear_graph' }],
    });
    expect(result).toMatchObject({
      conflict: 'unknown_tab', committed: false, revision: 0, tabId: 'gone',
    });
    expect(result.results).toEqual([]);
  });

  it('atomic: one bad op in three commits nothing and still names the failure', () => {
    const { api, tabId } = openEditable();
    const before = store().getTab(tabId)!.revision;

    const result = api.workspace.applyOperations({
      tabId,
      atomic: true,
      operations: [
        { op: 'add_node', node_type: 'Source' },
        { op: 'add_node', node_type: 'Ghost' },
        { op: 'add_node', node_type: 'Sink' },
      ],
    });

    expect(result.committed).toBe(false);
    expect(result.conflict).toBeUndefined();
    // Full length, in input order: the model has to see WHICH op failed.
    expect(result.results).toHaveLength(3);
    expect(result.results.map((r) => r.ok)).toEqual([true, false, true]);
    expect(result.revision).toBe(before);
    expect(store().getTab(tabId)!.nodes).toHaveLength(2);
  });

  it('non-atomic still commits the ops that worked', () => {
    const { api, tabId } = openEditable();
    const result = api.workspace.applyOperations({
      tabId,
      operations: [
        { op: 'add_node', node_type: 'Ghost' },
        { op: 'add_node', node_type: 'Sink' },
      ],
    });
    expect(result.committed).toBe(true);
    expect(result.results.map((r) => r.ok)).toEqual([false, true]);
    expect(store().getTab(tabId)!.nodes).toHaveLength(3);
  });

  it('a batch that mutates nothing commits nothing and pushes no undo step', () => {
    const { api, tabId } = openEditable();
    const before = store().getTab(tabId)!;
    const result = api.workspace.applyOperations({
      tabId, operations: [{ op: 'add_node', node_type: 'Ghost' }],
    });
    expect(result.committed).toBe(false);
    expect(result.revision).toBe(before.revision);
    expect(store().getTab(tabId)!.undoStack).toHaveLength(before.undoStack.length);
  });

  it('move + segment + note + label is ONE undo step', () => {
    const { api, tabId } = openEditable();
    const result = api.workspace.applyOperations({
      tabId,
      atomic: true,
      operations: [
        { op: 'move_node', node_id: 'a', position: { x: 640, y: 320 } },
        { op: 'set_segment', segment_id: 's1', head_node_id: 'a', tail_node_id: 'b' },
        { op: 'add_note', text: 'ablation: no warmup', bind_to: 'a' },
        { op: 'set_node_meta', node_id: 'b', label: 'Readout' },
      ],
    });
    expect(result.results.every((r) => r.ok)).toBe(true);
    expect(result.committed).toBe(true);

    const after = store().getTab(tabId)!;
    expect(after.nodes.find((n) => n.id === 'a')!.position).toEqual({ x: 640, y: 320 });
    expect(after.segmentGroups).toHaveLength(1);
    expect(after.nodes.some((n) => n.type === 'noteNode')).toBe(true);
    expect(after.nodes.find((n) => n.id === 'b')!.data.label).toBe('Readout');

    // Undo is per tab, so activate it first -- exactly what the user does.
    store().setActiveTab(tabId);
    store().undo();
    const undone = store().getTab(tabId)!;
    expect(undone.nodes.find((n) => n.id === 'a')!.position).toEqual({ x: 0, y: 0 });
    expect(undone.segmentGroups).toHaveLength(0);
    expect(undone.nodes.some((n) => n.type === 'noteNode')).toBe(false);
    expect(undone.nodes.find((n) => n.id === 'b')!.data.label).not.toBe('Readout');
  });

  it('with no tabId, writes to the active tab', () => {
    const api = freshApi();
    const live = store().activeTabId;
    const result = api.workspace.applyOperations({
      operations: [{ op: 'add_node', node_type: 'Source' }],
    });
    expect(result.tabId).toBe(live);
    expect(store().getTab(live)!.nodes).toHaveLength(1);
  });
});

describe('workspace.applyOperations while the user is inside a block', () => {
  /**
   * Put the active tab inside an empty block, through the real action.
   *
   * The canvas arrays are the BLOCK's contents from here on, which is the
   * whole reason for the refusal below: `snapshot()` still describes the
   * flushed top level, so a write that landed on those arrays would be a
   * write to something the plugin never read (#341 section 4.4 step 3).
   */
  function enterEmptyBlock() {
    store().setNodes([{
      id: 'inst', type: 'baseNode', position: { x: 0, y: 0 },
      data: { label: 'Encoder', type: 'subgraph:blk', params: {} },
    } as never]);
    store().setSubgraphs([{
      id: 'blk', name: 'Encoder', description: '',
      nodes: [], edges: [], interface: { inputs: [], outputs: [], triggerTargets: [] },
    }]);
    expect(store().enterSubgraph('inst')).toBe(true);
  }

  /**
   * The same, but with two connected nodes inside the block -- enough for a
   * `set_segment` that WOULD be accepted on its merits, so what the test
   * catches is the guard and not the op's own validation.
   */
  function enterBlockWithChain() {
    store().setNodes([{
      id: 'inst', type: 'baseNode', position: { x: 0, y: 0 },
      data: { label: 'Encoder', type: 'subgraph:blk', params: {} },
    } as never]);
    store().setSubgraphs([{
      id: 'blk', name: 'Encoder', description: '',
      nodes: [
        { id: 'in1', type: 'Source', position: { x: 0, y: 0 }, data: { params: {} } },
        { id: 'in2', type: 'Sink', position: { x: 200, y: 0 }, data: { params: {} } },
      ],
      edges: [
        { id: 'ie1', source: 'in1', target: 'in2', sourceHandle: 'out', targetHandle: 'x' },
      ],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    } as never]);
    expect(store().enterSubgraph('inst')).toBe(true);
  }

  it('refuses with editing_subgraph instead of writing into the block', () => {
    const api = freshApi();
    enterEmptyBlock();
    const before = store().getActiveTab().revision;

    const result = api.workspace.applyOperations({
      operations: [{ op: 'add_node', node_type: 'Source' }],
    });

    expect(result.conflict).toBe('editing_subgraph');
    expect(result.committed).toBe(false);
    expect(result.results).toEqual([]);
    expect(result.revision).toBe(before);
    // Nothing was written anywhere: the block is still empty.
    expect(store().getActiveTab().nodes).toEqual([]);
  });

  it('read-only outranks it, and it outranks a stale revision', () => {
    const api = freshApi();
    enterEmptyBlock();
    expect(api.workspace.applyOperations({
      expectedRevision: 9999, operations: [{ op: 'clear_graph' }],
    }).conflict).toBe('editing_subgraph');

    store().setTabReadOnly(true);
    expect(api.workspace.applyOperations({
      operations: [{ op: 'clear_graph' }],
    }).conflict).toBe('read_only');
  });

  it('the LEGACY path still writes into the open block, exactly as before', () => {
    // The one place the two paths deliberately disagree. Moving an installed
    // plugin's writes out from under it is the change #341 refused to make
    // (core#200 item 7, maintainer decision); `workspace` is where a plugin
    // gets the stricter promise instead.
    const api = freshApi();
    enterEmptyBlock();
    const result = api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    expect(result.results[0].ok).toBe(true);
    expect(store().getActiveTab().nodes).toHaveLength(1);
    expect(store().getActiveTab().subgraphStack).toHaveLength(1);
  });

  it('...but refuses a legacy batch holding a segment op, whole batch', () => {
    // `enterSubgraph` captures `segmentGroups` rather than swapping them, so a
    // segment committed from in here is a TOP-LEVEL overlay naming inner node
    // ids: it survives the exit, is written to the saved file, and draws
    // nothing -- so the user cannot delete it either. The whole batch goes,
    // including the innocent op, because segments are committed wholesale and
    // there is no half of this batch that lands safely.
    const api = freshApi();
    enterBlockWithChain();
    const before = store().getActiveTab();
    const nodesBefore = before.nodes.length;

    const result = api.graph.applyOperations([
      { op: 'add_node', node_type: 'Source' },
      { op: 'set_segment', segment_id: 's1', head_node_id: 'in1', tail_node_id: 'in2' },
    ]);

    expect(result.results).toHaveLength(2);
    expect(result.results.every((r) => !r.ok)).toBe(true);
    expect(result.results.map((r) => r.index)).toEqual([0, 1]);
    expect(result.results[0].error).toContain('set_segment');

    const after = store().getActiveTab();
    expect(after.segmentGroups).toEqual([]);
    expect(after.nodes).toHaveLength(nodesBefore);
    expect(after.revision).toBe(before.revision);
    expect(after.undoStack).toHaveLength(before.undoStack.length);
  });

  it('the same batch at the TOP level commits, segment and all', () => {
    // The counterfactual: nothing is wrong with these two ops, and the legacy
    // path has no quarrel with segments as such. It is standing inside a block
    // -- where `segmentGroups` is still the top level's -- that makes the
    // write unsafe.
    const api = freshApi();
    const [opened] = api.workspace.openGraphs(
      [{ title: 'Top level', graph: candidateGraph() }], { activate: 'first' });
    const { tabId } = opened as { tabId: string };

    const result = api.graph.applyOperations([
      { op: 'add_node', node_type: 'Source' },
      { op: 'set_segment', segment_id: 's1', head_node_id: 'a', tail_node_id: 'b' },
    ]);

    expect(result.results.every((r) => r.ok)).toBe(true);
    expect(store().getTab(tabId)!.segmentGroups).toHaveLength(1);
    expect(store().getTab(tabId)!.nodes).toHaveLength(3);
  });

  /**
   * A graph with a real top-level overlay, standing inside a block.
   *
   * The overlay is installed through the real write path rather than pushed
   * into the store by hand, so what the tests below assert against is a
   * segment the editor itself produced.
   */
  function enterBlockOverSegmentedGraph() {
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 },
        data: { label: 'a', type: 'Source', params: {} } } as never,
      { id: 'b', type: 'baseNode', position: { x: 200, y: 0 },
        data: { label: 'b', type: 'Sink', params: {} } } as never,
      { id: 'inst', type: 'baseNode', position: { x: 400, y: 0 },
        data: { label: 'Encoder', type: 'subgraph:blk', params: {} } } as never,
    ]);
    store().setEdges([
      { id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'x' },
    ]);
    store().setSubgraphs([{
      id: 'blk', name: 'Encoder', description: '',
      nodes: [{ id: 'in1', type: 'Source', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    } as never]);

    const seeded = freshApi().workspace.applyOperations({
      operations: [
        { op: 'set_segment', segment_id: 's-top', head_node_id: 'a', tail_node_id: 'b' },
      ],
    });
    expect(seeded.committed).toBe(true);
    expect(store().getActiveTab().segmentGroups).toHaveLength(1);
    expect(store().enterSubgraph('inst')).toBe(true);
  }

  it('a legacy clear_graph in a block leaves the TOP LEVEL overlays alone', () => {
    // `clear_graph` empties the reducer's segment list, and while a block is
    // open that list is the GRAPH's, not the block's -- so committing the
    // outcome wholesale would delete every overlay on a canvas the user is not
    // even looking at, and the next save would write the loss to disk. Before
    // v5 the legacy commit never touched segments at all; from inside a block
    // it still does not.
    const api = freshApi();
    enterBlockOverSegmentedGraph();
    // The block's own canvas: one node, and no segments of its own.
    expect(store().getActiveTab().nodes).toHaveLength(1);
    expect(store().getActiveTab().segmentGroups).toHaveLength(1);

    const result = api.graph.applyOperations([{ op: 'clear_graph' }]);
    expect(result.results[0].ok).toBe(true);

    // Emptied the block, exactly as before (api.view.test.ts pins this too).
    expect(store().getActiveTab().nodes).toEqual([]);
    // ...and left the graph's overlay standing, through the exit and into
    // what the next save would write.
    expect(store().getActiveTab().segmentGroups).toHaveLength(1);
    store().exitSubgraph();
    expect(store().getActiveTab().segmentGroups).toHaveLength(1);
    expect(api.graph.getGraph().segmentGroups).toHaveLength(1);
  });

  it('the same clear_graph at the TOP level still empties the overlays', () => {
    // The companion: nothing about the legacy path spares segments as such.
    // Being inside a block is the whole of the difference.
    const api = freshApi();
    enterBlockOverSegmentedGraph();
    store().exitSubgraph();
    expect(store().getActiveTab().segmentGroups).toHaveLength(1);

    api.graph.applyOperations([{ op: 'clear_graph' }]);
    expect(store().getActiveTab().segmentGroups).toEqual([]);
    expect(store().getActiveTab().nodes).toEqual([]);
  });
});

describe('workspace.onChanged', () => {
  it('reports an added tab, a graph change, an activation and a close, in order', () => {
    const api = freshApi();
    const seen: string[] = [];
    const off = api.workspace.onChanged((e) => seen.push(`${e.type}${e.type === 'tabs' ? `:${e.removed}` : ''}`));

    const [opened] = api.workspace.openGraphs(
      [{ title: 'Watched', graph: candidateGraph() }], { activate: 'none' });
    const { tabId } = opened as { tabId: string };
    expect(seen[0]).toBe('tabs:false');
    expect(seen).toContain('graph');

    seen.length = 0;
    store().setActiveTab(tabId);
    expect(seen).toEqual(['active-tab']);

    seen.length = 0;
    store().removeTab(tabId);
    expect(seen).toContain('tabs:true');

    off();
  });

  it('carries the tabId and the new revision on a graph event', () => {
    const api = freshApi();
    const events: Array<{ type: string; tabId: string; revision: number }> = [];
    const off = api.workspace.onChanged((e) => events.push(e));
    store().addNote('text', { x: 0, y: 0 });
    const graphEvents = events.filter((e) => e.type === 'graph');
    // Exactly one: `addNote` snapshots and then writes, and only the write
    // touches a document field.
    expect(graphEvents).toHaveLength(1);
    expect(graphEvents[0].tabId).toBe(store().activeTabId);
    expect(graphEvents[0].revision).toBe(store().getActiveTab().revision);
    off();
  });

  it('attaches origin to a plugin write and to nothing else', () => {
    const api = freshApi('graph-copilot');
    const origins: Array<{ pluginId: string } | undefined> = [];
    const off = api.workspace.onChanged((e) => {
      if (e.type === 'graph') origins.push(e.origin);
    });

    api.workspace.applyOperations({ operations: [{ op: 'add_node', node_type: 'Source' }] });
    expect(origins).toEqual([{ pluginId: 'graph-copilot' }]);

    origins.length = 0;
    store().addNote('text', { x: 0, y: 0 });
    expect(origins).toEqual([undefined]);
    off();
  });

  it('unsubscribes cleanly, and drops a callback that throws', () => {
    const api = freshApi();
    let calls = 0;
    const off = api.workspace.onChanged(() => { calls += 1; });
    store().addNote('text', { x: 0, y: 0 });
    const seen = calls;
    off();
    store().addNote('text', { x: 0, y: 0 });
    expect(calls).toBe(seen);

    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    let thrown = 0;
    api.workspace.onChanged(() => { thrown += 1; throw new Error('plugin bug'); });
    store().addNote('text', { x: 0, y: 0 });
    store().addNote('text', { x: 0, y: 0 });
    // Called once, then unsubscribed -- a plugin cannot break the store.
    expect(thrown).toBe(1);
    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });
});

describe('the legacy graph.applyOperations on a read-only tab (#341 section 4.6)', () => {
  it('refuses every op instead of writing through', () => {
    const api = freshApi();
    store().setTabReadOnly(true);
    const before = store().getActiveTab().nodes.length;

    const result = api.graph.applyOperations([
      { op: 'add_node', node_type: 'Source' },
      { op: 'add_node', node_type: 'Sink' },
    ]);

    expect(result.results).toHaveLength(2);
    expect(result.results.every((r) => !r.ok)).toBe(true);
    expect(result.results[0].error).toContain('read-only');
    expect(result.results.map((r) => r.index)).toEqual([0, 1]);
    expect(store().getActiveTab().nodes).toHaveLength(before);
  });

  it('still writes through on an editable tab', () => {
    const api = freshApi();
    const result = api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    expect(result.results[0].ok).toBe(true);
    expect(store().getActiveTab().nodes).toHaveLength(1);
  });
});
