/**
 * The save path, end to end, over the WIRE — deliberately without mocking
 * `../api/rest`.
 *
 * Why this file exists at all (core#137 review, CRITICAL 1):
 * `saveActiveGraph` shipped destructuring only `{ nodes, edges, presets,
 * segmentGroups }` out of `getSerializedGraph()`, silently dropping
 * `subgraphs`. Every collapsed block's contents were therefore erased by the
 * one action a user takes to make their work durable, and the suite was
 * green throughout. It was green because the tests that "proved" definitions
 * propagate round-tripped through the serialization HELPER
 * (`getSerializedGraph` -> `setSubgraphs`), while the sibling
 * `saveActiveGraph.test.ts` mocked the whole `../api/rest` module — so no
 * test ever looked at what the application actually puts in the request body.
 * A helper that resembles the save path is not the save path.
 *
 * So this file stubs `globalThis.fetch` and nothing else, drives the store
 * with its OWN actions (add nodes -> select -> collapseSelectionToSubgraph),
 * and then asserts on the bytes of the POST that `rest.saveGraph` — the real
 * function — sends. The chain under test is "user collapsed a block" all the
 * way to "definition is in the request body", which is the only chain whose
 * breakage costs data.
 *
 * Keep future assertions here at the wire level. The moment this file mocks
 * a module between `saveActiveGraph` and `fetch`, it stops covering the
 * class of bug it was written for.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { saveActiveGraph } from './saveActiveGraph';
import { importGraphFile } from './importGraphFile';
import { useTabStore } from '../store/tabStore';
import { useProjectStore } from '../store/projectStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { _setSessionTokenForTesting } from '../api/_auth';
import { subgraphIdOf } from './subgraph';
import type { NodeData, NodeDefinition, PresetDefinition } from '../types';

// Persistence is IndexedDB-flavoured background noise for this test; the tab
// store's own suite covers it. Note this is BELOW the code under test, not
// between `saveActiveGraph` and `fetch`.
vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

interface CapturedRequest {
  url: string;
  method: string;
  body: Record<string, any>;
}

let requests: CapturedRequest[] = [];

/** Record every request and answer with a boring 200, like /graph/save does. */
function captureFetch() {
  g.fetch = vi.fn(async (input: any, init: any = {}) => {
    requests.push({
      url: String(input),
      method: String(init.method ?? 'GET').toUpperCase(),
      body: typeof init.body === 'string' ? JSON.parse(init.body) : {},
    });
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ status: 'ok' }),
      text: async () => '',
    } as unknown as Response;
  }) as unknown as typeof fetch;
}

/**
 * Answer the two list reads with these lists, as the server does when the
 * editor fetches them again after a change, and capture every other request
 * as before.
 */
function serveLists(definitions: NodeDefinition[], presets: PresetDefinition[] = []) {
  const capture = g.fetch;
  g.fetch = vi.fn(async (input: any, init?: any) => {
    const url = String(input);
    const list = url.endsWith('/api/nodes')
      ? definitions
      : url.endsWith('/api/presets')
        ? presets
        : null;
    if (list === null) return capture(input, init);
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => list,
    } as unknown as Response;
  }) as unknown as typeof fetch;
}

function savePost(): CapturedRequest {
  const post = requests.find(
    (r) => r.url.endsWith('/api/graph/save') && r.method === 'POST',
  );
  if (!post) {
    throw new Error(
      `no POST /api/graph/save was sent; captured: ${JSON.stringify(
        requests.map((r) => `${r.method} ${r.url}`),
      )}`,
    );
  }
  return post;
}

function def(name: string): NodeDefinition {
  return {
    node_name: name,
    category: 'x',
    description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      {
        name: 'scale', param_type: 'float', default: 1, description: '',
        options: [], min_value: null, max_value: null,
      },
    ],
  };
}

/** Like `def`, but the node declares a SECRET param (an LLM API key). */
function secretDef(name: string): NodeDefinition {
  const base = def(name);
  return {
    ...base,
    params: [
      ...base.params,
      {
        name: 'api_key', param_type: 'secret', default: '', description: '',
        options: [], min_value: null, max_value: null,
      },
    ],
  };
}

function node(
  id: string,
  type: string,
  x = 0,
  y = 0,
  definition: NodeDefinition = def(type),
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y },
    data: {
      label: id,
      type,
      params: { scale: 1 },
      definition,
      executionStatus: 'idle',
    },
  };
}

/** A canvas node whose attached definition declares a SECRET param. */
function keyedNode(id: string, type: string, x = 0, y = 0): Node<NodeData> {
  return node(id, type, x, y, secretDef(type));
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

/**
 * start -> a -> b -> c -> sink, then the user selects b + c and collapses
 * them. That leaves a real instance node with a real interface (one input
 * from a, one output to sink) and a real definition holding b, c and the
 * edge between them — the exact shape that had nowhere to go in the saved
 * file.
 */
function collapseABlock(): string {
  const store = useTabStore.getState();
  store.setNodes([
    { ...node('start', 'Start', 0, 0), type: 'start' },
    node('a', 'A', 100, 0),
    node('b', 'B', 200, 40),
    node('c', 'C', 300, 20),
    node('sink', 'S', 400, 0),
  ]);
  store.setEdges([
    {
      id: 't', source: 'start', target: 'a', sourceHandle: 'trigger',
      targetHandle: '__trigger', type: 'triggerEdge', data: { type: 'trigger' },
    },
    dataEdge('e1', 'a', 'b'),
    dataEdge('e2', 'b', 'c'),
    dataEdge('e3', 'c', 'sink'),
  ]);
  const selected = new Set(['b', 'c']);
  useTabStore.getState().setNodes(
    useTabStore.getState().getActiveTab().nodes.map((n) => ({
      ...n,
      selected: selected.has(n.id),
    })),
  );
  const result = useTabStore.getState().collapseSelectionToSubgraph('Block');
  // Guard the fixture itself: a collapse that quietly refused would leave a
  // plain graph behind and every assertion below would pass vacuously.
  if (!result.ok) {
    throw new Error(`fixture collapse failed: ${result.reason}`);
  }
  return result.definition.id;
}

beforeEach(() => {
  requests = [];
  originalFetch = g.fetch;
  captureFetch();
  // Pre-seed the session token so apiFetch does not spend a request on
  // /api/auth/bootstrap (which our stub would answer without a token).
  _setSessionTokenForTesting('test-token');
  localStorage.clear();
  useTabStore.setState({
    tabs: [], activeTabId: null as unknown as string, clipboard: null,
  });
  useTabStore.getState().addTab('wire');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C'), def('S')],
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
  // Project mode + a bound file is the no-prompt in-place save, so the test
  // needs no dialog stub — one less module standing between it and fetch.
  useProjectStore.setState({
    projectDir: '/proj', projectName: 'proj', loaded: true,
  });
  useTabStore.getState().setCurrentGraphFile('block-graph', 'block-graph');
});

afterEach(() => {
  g.fetch = originalFetch;
  _setSessionTokenForTesting(null);
  vi.restoreAllMocks();
});

describe('saveActiveGraph over the wire', () => {
  it('sends settings.device when the graph assigns one, and no settings key when it does not', async () => {
    useTabStore.getState().setNodes([node('a', 'A')]);
    await saveActiveGraph();
    expect('settings' in savePost().body).toBe(false);

    requests = [];
    useTabStore.getState().setGraphDevice('mps');
    await saveActiveGraph();
    expect(savePost().body.settings).toEqual({ device: 'mps' });
  });

  it('POSTs to /api/graph/save with the graph the canvas is showing', async () => {
    collapseABlock();
    await saveActiveGraph();

    const post = savePost();
    expect(post.body.name).toBe('block-graph');
    expect(post.body.nodes.map((n: any) => n.id).sort()).toEqual(
      ['a', 'sink', 'start'].concat(
        post.body.nodes
          .filter((n: any) => subgraphIdOf(n.type))
          .map((n: any) => n.id),
      ).sort(),
    );
  });

  it('carries every subgraph definition, with its inner nodes and interface', async () => {
    const definitionId = collapseABlock();
    await saveActiveGraph();

    const body = savePost().body;

    // The instance is on the canvas...
    const instance = body.nodes.find((n: any) => subgraphIdOf(n.type) === definitionId);
    expect(
      instance,
      'the collapsed block should be saved as a subgraph:<id> instance node',
    ).toBeDefined();

    // ...and the thing that gives it meaning must be in the same request.
    // Asserting the KEY exists is not enough: an empty array would satisfy
    // that while still losing the user's work.
    expect(
      Array.isArray(body.subgraphs) ? body.subgraphs.map((d: any) => d.id) : body.subgraphs,
      'the saved graph must carry the definition its instance node references, '
        + 'or reopening the file yields an instance with nothing inside it',
    ).toEqual([definitionId]);

    const definition = body.subgraphs[0];
    expect(definition.name).toBe('Block');
    expect(definition.nodes.map((n: any) => n.id).sort()).toEqual(['b', 'c']);
    expect(definition.edges.map((e: any) => e.id)).toEqual(['e2']);
    // The interface is what reconnects the block to the rest of the graph.
    // Without it a reopened file could not even draw the instance's handles.
    expect(definition.interface.inputs).toHaveLength(1);
    expect(definition.interface.inputs[0]).toMatchObject({ innerNode: 'b', innerPort: 'in' });
    expect(definition.interface.outputs).toHaveLength(1);
    expect(definition.interface.outputs[0]).toMatchObject({ innerNode: 'c', innerPort: 'out' });
    // Collapse recorded which inner node the incoming trigger fans out to;
    // losing it changes where the expanded graph starts executing.
    expect(definition.interface.triggerTargets).toEqual([]);
  });

  it('never puts a SECRET param value in the request body, in or out of a block', async () => {
    // The gap this file was written to close was "the body does not carry
    // `subgraphs`". Making it carry them opened the opposite one: the top
    // level is stripped and the definitions are not, so pressing Save writes
    // a live API key into `graphs/*.json` in cleartext. Pinning the BODY --
    // not a per-node slot -- is what makes this catch the next field too.
    // Both stores: the canvas node carries its own `definition` (what the
    // top-level strip reads), while a node serialized INTO a definition
    // keeps only its `type` string, so the in-block strip has to resolve the
    // type through the registry the way `resolveSerializedNodes` does.
    useNodeDefStore.setState({
      definitions: [secretDef('A'), secretDef('B'), secretDef('C'), secretDef('S')],
    } as never);
    const store = useTabStore.getState();
    store.setNodes([
      { ...node('start', 'Start', 0, 0), type: 'start' },
      keyedNode('a', 'A', 100, 0),
      keyedNode('b', 'B', 200, 40),
      keyedNode('c', 'C', 300, 20),
      keyedNode('sink', 'S', 400, 0),
    ]);
    store.setEdges([
      dataEdge('e1', 'a', 'b'),
      dataEdge('e2', 'b', 'c'),
      dataEdge('e3', 'c', 'sink'),
    ]);
    // One key outside the block, one inside it: the outside key is the
    // control that proves the assertion can see a leak at all.
    useTabStore.getState().updateNodeParams('a', { api_key: 'sk-OUTSIDE' });
    useTabStore.getState().updateNodeParams('b', { api_key: 'sk-INSIDE' });
    useTabStore.getState().setNodes(
      useTabStore.getState().getActiveTab().nodes.map((n) => ({
        ...n,
        selected: n.id === 'b' || n.id === 'c',
      })),
    );
    const collapsed = useTabStore.getState().collapseSelectionToSubgraph('Keyed');
    if (!collapsed.ok) throw new Error(`fixture collapse failed: ${collapsed.reason}`);
    // Fixture guard: the definition really did swallow the node holding the
    // key, so a pass below is about scrubbing and not about an empty block.
    expect(collapsed.definition.nodes.map((n: any) => n.id).sort()).toEqual(['b', 'c']);

    await saveActiveGraph();

    const body = savePost().body;
    expect(body.subgraphs.map((d: any) => d.id)).toEqual([collapsed.definition.id]);
    expect(JSON.stringify(body)).not.toContain('sk-INSIDE');
    expect(JSON.stringify(body)).not.toContain('sk-OUTSIDE');
    // Blanked, not deleted -- the slot survives so reopening the file shows
    // the field as empty rather than as absent.
    expect(body.subgraphs[0].nodes.find((n: any) => n.id === 'b').data.params)
      .toMatchObject({ api_key: '' });
  });

  // The in-block strip finds a node's SECRET params by its type in the node
  // list, and every fetch replaces that list. A custom node disabled in the
  // Custom Nodes manager drops out of it (#537 review).

  /**
   * a -> b -> c, b and c collapsed into a block, and a key typed into b from
   * inside it, where b shows the masked field. B is a custom node, listed
   * while the key is typed. Leaves the canvas inside the block.
   */
  function typeKeyInsideBlock(key: string) {
    useNodeDefStore.setState({
      definitions: [secretDef('A'), secretDef('B'), secretDef('C')],
    } as never);
    const store = useTabStore.getState();
    store.setNodes([
      keyedNode('a', 'A', 100, 0),
      keyedNode('b', 'B', 200, 40),
      keyedNode('c', 'C', 300, 20),
    ]);
    store.setEdges([dataEdge('e1', 'a', 'b'), dataEdge('e2', 'b', 'c')]);
    useTabStore.getState().setNodes(
      useTabStore.getState().getActiveTab().nodes.map((n) => ({
        ...n,
        selected: n.id === 'b' || n.id === 'c',
      })),
    );
    const collapsed = useTabStore.getState().collapseSelectionToSubgraph('Keyed');
    if (!collapsed.ok) throw new Error(`fixture collapse failed: ${collapsed.reason}`);
    expect(useTabStore.getState().enterSubgraph(collapsed.instanceId)).toBe(true);
    useTabStore.getState().updateNodeParams('b', { api_key: key });
  }

  /** The Custom Nodes manager disables B and fetches the lists again. */
  async function disableB() {
    serveLists([secretDef('A'), secretDef('C')]);
    await useNodeDefStore.getState().fetchDefinitions();
    // The premise: the fetch really left B out.
    expect(useNodeDefStore.getState().definitions.map((d) => d.node_name)).toEqual(['A', 'C']);
  }

  it('leaves a key out of a block after the node list stops listing its type', async () => {
    typeKeyInsideBlock('sk-DISABLED-TYPE');
    useTabStore.getState().exitSubgraph();
    await disableB();
    expect(JSON.stringify(useTabStore.getState().getActiveTab().subgraphs))
      .toContain('sk-DISABLED-TYPE');

    await saveActiveGraph();

    const body = savePost().body;
    expect(JSON.stringify(body)).not.toContain('sk-DISABLED-TYPE');
    expect(body.subgraphs[0].nodes.find((n: any) => n.id === 'b').data.params)
      .toEqual({ scale: 1, api_key: '' });
  });

  it('leaves that key out while the canvas is still inside the block', async () => {
    typeKeyInsideBlock('sk-SAVED-FROM-INSIDE');
    await disableB();
    // Still inside, where b still shows its masked field.
    expect(useTabStore.getState().getActiveTab().subgraphStack).toHaveLength(1);

    await saveActiveGraph();

    const body = savePost().body;
    expect(JSON.stringify(body)).not.toContain('sk-SAVED-FROM-INSIDE');
    expect(body.subgraphs[0].nodes.find((n: any) => n.id === 'b').data.params)
      .toEqual({ scale: 1, api_key: '' });
  });

  it('leaves a key out of a preset in a block after a fetch drops the preset a file brought', async () => {
    // A graph file that carries its own preset, one that still exposes its
    // inner key as a field. The server has no preset of that name.
    const filePreset: PresetDefinition = {
      preset_name: 'FileChat', category: 'LLM', description: '', tags: [],
      nodes: [{ id: 'chat', type: 'A', params: { scale: 1, api_key: '' } }],
      edges: [],
      exposed_inputs: [],
      exposed_outputs: [],
      exposed_params: [{
        internal_node: 'chat', param_name: 'api_key', display_name: 'A - api_key',
        group: 'A', param_def: secretDef('A').params[1],
      }],
    };
    const file = {
      nodes: [{ id: 'inst', type: 'subgraph:blk', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [],
      presets: [filePreset],
      subgraphs: [{
        id: 'blk', name: 'Block', description: '',
        nodes: [{
          id: 'p', type: 'preset:FileChat', position: { x: 0, y: 0 },
          data: { params: {}, internalParams: { chat: { scale: 1, api_key: '' } } },
        }],
        edges: [],
        interface: { inputs: [], outputs: [], triggerTargets: [] },
      }],
    };
    const picked = new File([JSON.stringify(file)], 'file-chat.json', { type: 'application/json' });
    expect(await importGraphFile(picked)).toBe(true);
    // The reader merged the file's preset into the list.
    expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual(['FileChat']);
    // Open the block and type the key into the preset's field.
    expect(useTabStore.getState().enterSubgraph('inst')).toBe(true);
    useTabStore.getState().updatePresetInternalParam('p', 'chat', 'api_key', 'sk-IN-FILE-PRESET');
    useTabStore.getState().exitSubgraph();
    // Export as preset, or any change on the server, fetches the lists again.
    serveLists([secretDef('A')], []);
    await useNodeDefStore.getState().fetchDefinitions();
    expect(JSON.stringify(useTabStore.getState().getActiveTab().subgraphs))
      .toContain('sk-IN-FILE-PRESET');
    // An imported graph is bound to no file; bind it so Save writes in place.
    useTabStore.getState().setCurrentGraphFile('block-graph', 'block-graph');

    await saveActiveGraph();

    expect(JSON.stringify(savePost().body)).not.toContain('sk-IN-FILE-PRESET');
  });

  it('sends an empty subgraphs list for a graph that has no blocks', async () => {
    useTabStore.getState().setNodes([node('a', 'A', 0, 0)]);
    await saveActiveGraph();
    expect(savePost().body.subgraphs).toEqual([]);
  });

  it('still carries definitions while the canvas is inside a block', async () => {
    // getSerializedGraph flushes the sub-canvas editing stack first, so a
    // save taken while the user is standing inside a block must produce the
    // same whole-graph payload as one taken from the top level. This is the
    // save most likely to happen right after an edit, and the one where
    // losing definitions would destroy work the user just did.
    const definitionId = collapseABlock();
    const instance = useTabStore
      .getState()
      .getActiveTab()
      .nodes.find((n) => subgraphIdOf(n.data.type) === definitionId)!;
    useTabStore.getState().enterSubgraph(instance.id);

    await saveActiveGraph();

    const body = savePost().body;
    expect(body.subgraphs.map((d: any) => d.id)).toEqual([definitionId]);
    expect(body.subgraphs[0].nodes.map((n: any) => n.id).sort()).toEqual(['b', 'c']);
    // The top-level graph, not the block's insides.
    expect(body.nodes.map((n: any) => n.id)).toContain('start');
  });
});
