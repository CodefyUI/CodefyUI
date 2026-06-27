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
    expect(api.apiVersion).toBe(2);
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
