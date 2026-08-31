import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { buildPluginAPI } from './api';
import { getNodeRenderer, _clearNodeRenderers, type PluginNodeRenderer } from './nodeRenderers';
import type { NodeDefinition } from '../types';

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

function freshApi() {
  return buildPluginAPI('test-plugin', () => document.createElement('div'));
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
  useNodeDefStore.setState({ definitions: DEFS });
  window.localStorage.clear();
});

describe('graph surface', () => {
  it('applyOperations commits as a single undo step', () => {
    const api = freshApi();
    const result = api.graph.applyOperations([
      { op: 'add_node', node_type: 'Source', ref: 'a' },
      { op: 'add_node', node_type: 'Sink', ref: 'b' },
      { op: 'connect', source: 'a', source_handle: 'out', target: 'b', target_handle: 'x' },
    ]);
    expect(result.results.every((r) => r.ok)).toBe(true);
    expect(result.node_count).toBe(2);
    expect(result.edge_count).toBe(1);

    const tab = useTabStore.getState().getActiveTab();
    expect(tab.nodes).toHaveLength(2);
    expect(tab.edges).toHaveLength(1);

    useTabStore.getState().undo();
    const after = useTabStore.getState().getActiveTab();
    expect(after.nodes).toHaveLength(0);
    expect(after.edges).toHaveLength(0);
  });

  it('does not push an undo snapshot when nothing mutates', () => {
    const api = freshApi();
    const before = useTabStore.getState().getActiveTab().undoStack.length;
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Ghost' }]);
    expect(useTabStore.getState().getActiveTab().undoStack.length).toBe(before);
  });

  it('getGraph returns the serialized active tab', () => {
    const api = freshApi();
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    const g = api.graph.getGraph();
    expect(g.nodes).toHaveLength(1);
    expect(g.nodes[0].type).toBe('Source');
  });

  it('getNodeDefinitions returns the store definitions', () => {
    expect(freshApi().graph.getNodeDefinitions()).toEqual(DEFS);
  });

  it('onGraphChanged fires on graph mutations and unsubscribes cleanly', () => {
    const api = freshApi();
    let calls = 0;
    const off = api.graph.onGraphChanged(() => { calls += 1; });
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    expect(calls).toBeGreaterThan(0);
    const seen = calls;
    off();
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Sink' }]);
    expect(calls).toBe(seen);
  });

  it('onGraphChanged fires for a definition-only change (core#200 item 3)', () => {
    // Renaming a block, or editing its insides and stepping back out, changes
    // `subgraphs` and nothing else. The subscription compared only nodes and
    // edges, so a plugin watching the graph was never told -- even though
    // `graph.getGraph()` would have answered with different bytes.
    const api = freshApi();
    // Set the canvas up BEFORE subscribing, so the only thing that moves
    // afterwards is the definition list. An instance on the canvas is what
    // keeps the definition reachable -- `getSerializedGraph` drops orphans.
    const store = useTabStore.getState();
    store.setNodes([{
      id: 'inst', type: 'baseNode', position: { x: 0, y: 0 },
      data: { label: 'Encoder', type: 'subgraph:blk', params: {} },
    } as never]);
    store.setSubgraphs([{
      id: 'blk', name: 'Encoder', description: '',
      nodes: [], edges: [], interface: { inputs: [], outputs: [], triggerTargets: [] },
    }]);

    const before = JSON.stringify(api.graph.getGraph());
    let calls = 0;
    const off = api.graph.onGraphChanged(() => { calls += 1; });

    useTabStore.getState().renameSubgraph('blk', 'Decoder');
    expect(calls).toBe(1);
    // The premise: the graph a plugin can read really did change bytes.
    expect(JSON.stringify(api.graph.getGraph())).not.toBe(before);

    off();
    useTabStore.getState().renameSubgraph('blk', 'Encoder');
    expect(calls).toBe(1);
  });

  it('applyOperations still answers with an ApplyResult and nothing more', () => {
    // The legacy path now runs through the workspace commit, which knows
    // about tab ids, revisions and conflicts. None of that may leak out
    // here: an installed v1 plugin reads these four keys and no others.
    const api = freshApi();
    const result = api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    expect(Object.keys(result).sort()).toEqual(
      ['edge_count', 'node_count', 'refs', 'results'],
    );
  });

  it('onGraphChanged registers its unsubscribe with trackCleanup', () => {
    const tracked: Array<() => void> = [];
    const api = buildPluginAPI(
      'test-plugin',
      () => document.createElement('div'),
      (fn) => tracked.push(fn),
    );
    let calls = 0;
    api.graph.onGraphChanged(() => { calls += 1; });
    expect(tracked).toHaveLength(1);

    // Running the tracked cleanup unsubscribes — later mutations don't fire it.
    tracked[0]();
    api.graph.applyOperations([{ op: 'add_node', node_type: 'Source' }]);
    expect(calls).toBe(0);
  });
});

describe('storage surface', () => {
  it('namespaces keys per plugin', () => {
    const api = freshApi();
    api.storage.set('conversations', '[]');
    expect(window.localStorage.getItem('plugin:test-plugin:conversations')).toBe('[]');
    expect(api.storage.get('conversations')).toBe('[]');
    api.storage.remove('conversations');
    expect(api.storage.get('conversations')).toBeNull();
  });
});

describe('meta', () => {
  it('exposes apiVersion and pluginId', () => {
    const api = freshApi();
    // A floor, not an exact number: this file predates the versions that
    // followed, and what it is really asserting is that `apiVersion` is
    // present and has not gone BACKWARDS. The exact value is pinned once, in
    // the test file for the version that set it.
    expect(api.apiVersion).toBeGreaterThanOrEqual(3);
    expect(api.pluginId).toBe('test-plugin');
  });
});

describe('nodes surface', () => {
  afterEach(() => _clearNodeRenderers());

  it('registerRenderer registers, and the returned fn unregisters', () => {
    const api = freshApi();
    const renderer: PluginNodeRenderer = { mount: () => {} };
    const off = api.nodes.registerRenderer('test-plugin:Foo', renderer);
    expect(getNodeRenderer('test-plugin:Foo')).toBe(renderer);
    off();
    expect(getNodeRenderer('test-plugin:Foo')).toBeUndefined();
  });
});
