/**
 * `api.graph.getView()` — the read-only view context added at apiVersion 4
 * (core#200 item 7), and the write behaviour it exists to warn a plugin about.
 *
 * The item: the op surface reads `tab.nodes` / `tab.edges`, which while the
 * user is inside a block are the BLOCK's insides. A plugin could not tell, so
 * `clear_graph` issued at the wrong moment emptied a block instead of the
 * graph. The maintainer's decision (2026-08-12) was the gradual one: publish
 * where the user is, leave where writes land alone, and let a plugin choose.
 *
 * So the write tests below are not aspirational — they PIN today's behaviour,
 * which this change turns from an accident into documented API.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { buildPluginAPI } from './api';
import { subgraphIdOf, subgraphViewPath } from '../utils/subgraph';
import type { NodeData, NodeDefinition } from '../types';

vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

function def(name: string): NodeDefinition {
  return {
    node_name: name, category: 'x', description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function node(id: string, type: string, x = 0): Node<NodeData> {
  return {
    id, type: 'baseNode', position: { x, y: 0 },
    data: { label: id, type, params: {}, definition: def(type), executionStatus: 'idle' },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

function freshApi() {
  return buildPluginAPI('test-plugin', () => document.createElement('div'));
}

function select(...ids: string[]) {
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: ids.includes(n.id) })));
}

/** a -> b -> c -> d, with b/c/d collapsed into a block. Returns its instance id. */
function seedOneBlock(name = 'Block'): string {
  store().setNodes([node('a', 'A', 0), node('b', 'B', 100), node('c', 'C', 200), node('d', 'D', 300)]);
  store().setEdges([dataEdge('e1', 'a', 'b'), dataEdge('e2', 'b', 'c'), dataEdge('e3', 'c', 'd')]);
  select('b', 'c', 'd');
  expect(store().collapseSelectionToSubgraph(name).ok).toBe(true);
  return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
}

/** Enters `Block`, then collapses c/d inside it into `Inner` and enters that too. */
function seedTwoDeep(): void {
  store().enterSubgraph(seedOneBlock('Block'));
  select('c', 'd');
  expect(store().collapseSelectionToSubgraph('Inner').ok).toBe(true);
  const inner = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
  expect(store().enterSubgraph(inner)).toBe(true);
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C'), def('D'), def('S')],
    presets: [], categorized: {}, presetCategorized: {}, loading: false, error: null,
  } as never);
});

describe('graph.getView', () => {
  it('reports the top level when no block is open', () => {
    seedOneBlock();
    expect(freshApi().graph.getView()).toEqual({
      depth: 0, path: [], atTopLevel: true,
    });
  });

  it('reports one level deep, with the block name the breadcrumb shows', () => {
    const instanceId = seedOneBlock('Encoder');
    store().enterSubgraph(instanceId);

    const view = freshApi().graph.getView();
    expect(view.depth).toBe(1);
    expect(view.atTopLevel).toBe(false);
    expect(view.path).toHaveLength(1);
    expect(view.path[0].name).toBe('Encoder');
    expect(view.path[0].subgraphId).toBe(tab().subgraphStack[0].subgraphId);
  });

  it('reports two levels deep, outermost first', () => {
    seedTwoDeep();

    const view = freshApi().graph.getView();
    expect(view.depth).toBe(2);
    expect(view.atTopLevel).toBe(false);
    expect(view.path.map((level) => level.name)).toEqual(['Block', 'Inner']);
    expect(view.path.map((level) => level.subgraphId)).toEqual(
      tab().subgraphStack.map((frame) => frame.subgraphId),
    );
  });

  it('follows the user back out, one level at a time', () => {
    const api = freshApi();
    seedTwoDeep();
    expect(api.graph.getView().depth).toBe(2);

    store().exitSubgraph();
    expect(api.graph.getView().depth).toBe(1);
    expect(api.graph.getView().path.map((l) => l.name)).toEqual(['Block']);

    store().exitSubgraph();
    expect(api.graph.getView()).toEqual({ depth: 0, path: [], atTopLevel: true });
  });

  it('is back at the top level after exitAllSubgraphs', () => {
    seedTwoDeep();
    store().exitAllSubgraphs();
    expect(freshApi().graph.getView().atTopLevel).toBe(true);
  });

  it('answers live, so a rename while inside is reflected immediately', () => {
    const api = freshApi();
    const instanceId = seedOneBlock('Encoder');
    store().enterSubgraph(instanceId);
    // The name comes from the live definition list, not from the copy the
    // frame captured on entry (which exists for undo).
    store().renameSubgraph(tab().subgraphStack[0].subgraphId, 'Decoder');
    expect(api.graph.getView().path[0].name).toBe('Decoder');
  });

  it('falls back to the id when the definition has gone missing', () => {
    // Tested on the pure derivation rather than through the store, because no
    // store action can produce the state: `setSubgraphs` drops `subgraphStack`
    // as it installs a new list (the deliberate ordering contract of #200 item
    // 8), so "stack pointing at a definition that is gone" is unreachable from
    // outside. The fallback is still the right answer -- `exitSubgraph` guards
    // the same possibility, and an empty name would be worse than an ugly one:
    // a plugin's warning has to name something.
    expect(subgraphViewPath([{ subgraphId: 'gone' }], [])).toEqual([
      { subgraphId: 'gone', name: 'gone' },
    ]);
  });

  it('reports the top level again once a new document replaces the list', () => {
    const api = freshApi();
    store().enterSubgraph(seedOneBlock('Encoder'));
    // `setSubgraphs` is one of the two actions that drop the editing stack
    // (#200 item 8). Whatever else that does, the view must not go on claiming
    // the user is inside a block that no longer exists.
    store().setSubgraphs([]);
    expect(api.graph.getView()).toEqual({ depth: 0, path: [], atTopLevel: true });
  });

  it('answers the top level rather than throwing when there is no tab', () => {
    // A plugin can call this whenever it likes, including from an activation
    // that runs before the editor has restored its tabs. Throwing inside
    // third-party code over a state the host owns would be the host's bug.
    const api = freshApi();
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    expect(api.graph.getView()).toEqual({ depth: 0, path: [], atTopLevel: true });
  });

  it('is a fresh object each call, so nothing a plugin keeps goes stale silently', () => {
    const api = freshApi();
    const instanceId = seedOneBlock();
    const before = api.graph.getView();
    store().enterSubgraph(instanceId);
    expect(before.depth).toBe(0);
    expect(api.graph.getView().depth).toBe(1);
    expect(api.graph.getView()).not.toBe(before);
  });

  it('mutating the returned view cannot move the editor', () => {
    const api = freshApi();
    const instanceId = seedOneBlock();
    store().enterSubgraph(instanceId);
    const view = api.graph.getView();
    view.depth = 0;
    view.path.length = 0;
    view.atTopLevel = true;
    // Read-only means read-only: the editor is exactly where it was.
    expect(tab().subgraphStack).toHaveLength(1);
    expect(api.graph.getView().depth).toBe(1);
  });
});

describe('read/write asymmetry the view exists to expose', () => {
  it('getGraph answers with the whole graph while a block is open', () => {
    const instanceId = seedOneBlock();
    const api = freshApi();
    store().enterSubgraph(instanceId);

    const graph = api.graph.getGraph();
    expect(graph.nodes.map((n: { id: string }) => n.id).sort()).toEqual(
      ['a', instanceId].sort(),
    );
    // Reading must not close the editor behind the user.
    expect(api.graph.getView().depth).toBe(1);
  });

  it('applyOperations lands in the OPEN canvas, not at the top level', () => {
    const instanceId = seedOneBlock();
    const api = freshApi();
    store().enterSubgraph(instanceId);
    const topLevelBefore = api.graph.getGraph().nodes.map((n: { id: string }) => n.id).sort();

    const result = api.graph.applyOperations([{ op: 'add_node', node_type: 'S' }]);
    expect(result.results[0].ok).toBe(true);

    // The new node is on the canvas the user is looking at -- inside the block.
    const added = result.results[0].node_id!;
    expect(tab().nodes.map((n) => n.id)).toContain(added);
    expect(tab().subgraphStack).toHaveLength(1);
    // ...and the graph a plugin READS is unchanged, which is the whole trap:
    // read the graph, reason about it, write, and the write goes elsewhere.
    expect(api.graph.getGraph().nodes.map((n: { id: string }) => n.id).sort())
      .toEqual(topLevelBefore);
    // Exactly what `getView` is for: the plugin could have known.
    expect(api.graph.getView().atTopLevel).toBe(false);
  });

  it('clear_graph inside a block empties the block, not the graph', () => {
    const instanceId = seedOneBlock();
    const api = freshApi();
    store().enterSubgraph(instanceId);

    api.graph.applyOperations([{ op: 'clear_graph' }]);
    expect(tab().nodes).toEqual([]);
    // The graph itself still has its top-level nodes, block instance included.
    expect(api.graph.getGraph().nodes.map((n: { id: string }) => n.id).sort())
      .toEqual(['a', instanceId].sort());
  });

  it('an op naming a top-level node fails while a block is open', () => {
    const instanceId = seedOneBlock();
    const api = freshApi();
    store().enterSubgraph(instanceId);

    // 'a' exists at the top level and nowhere inside the block, so a plugin
    // that read `getGraph()` and wrote back blind gets a refusal per op.
    const result = api.graph.applyOperations([
      { op: 'set_params', node_id: 'a', params: { scale: 2 } },
    ]);
    expect(result.results[0].ok).toBe(false);
    expect(result.results[0].error).toBeTruthy();
  });

  it('onGraphChanged fires on the level change, so the view can be re-read', () => {
    const instanceId = seedOneBlock();
    const api = freshApi();
    let depthSeen = -1;
    const off = api.graph.onGraphChanged(() => {
      depthSeen = api.graph.getView().depth;
    });

    store().enterSubgraph(instanceId);
    expect(depthSeen).toBe(1);
    store().exitSubgraph();
    expect(depthSeen).toBe(0);
    off();
  });
});

describe('apiVersion 4', () => {
  it('reports 4 and offers getView', () => {
    const api = freshApi();
    expect(api.apiVersion).toBe(4);
    expect(typeof api.graph.getView).toBe('function');
  });

  it('leaves every earlier graph member in place', () => {
    // Additive: a v2/v3 plugin reaches for these four and nothing else.
    const api = freshApi();
    for (const member of ['getGraph', 'getNodeDefinitions', 'applyOperations', 'onGraphChanged']) {
      expect(
        typeof (api.graph as unknown as Record<string, unknown>)[member],
        `graph.${member}`,
      ).toBe('function');
    }
  });
});
