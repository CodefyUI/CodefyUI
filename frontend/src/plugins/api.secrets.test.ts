/**
 * A plugin never reads an API key typed into the editor.
 *
 * `graph.getGraph()` and `workspace.snapshot().graph` answer `""` for every
 * SECRET param (plugin-frontend-extensions.md). The Run hook asks the
 * serializer to keep the keys for its own message; these pin that the plugin
 * surface still does not, in each of the three places a key can sit.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { buildPluginAPI, type SerializedGraph } from './api';
import type { NodeData, NodeDefinition, ParamDefinition, PresetDefinition } from '../types';

vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

const KEY_PARAM: ParamDefinition = {
  name: 'openai_api_key', param_type: 'secret', default: '', description: '',
  options: [], min_value: null, max_value: null,
};

const LLM_DEF: NodeDefinition = {
  node_name: 'LLMChat', category: 'LLM', description: '',
  inputs: [], outputs: [],
  params: [
    KEY_PARAM,
    {
      name: 'model', param_type: 'string', default: 'gpt-5.2', description: '',
      options: [], min_value: null, max_value: null,
    },
  ],
};

/** An old preset that still exposes its inner key as a field. */
const OLD_PRESET: PresetDefinition = {
  preset_name: 'KeyedChat', category: 'LLM', description: '', tags: [],
  nodes: [{ id: 'chat', type: 'LLMChat', params: { openai_api_key: '', model: 'gpt-5.2' } }],
  edges: [],
  exposed_inputs: [],
  exposed_outputs: [],
  exposed_params: [{
    internal_node: 'chat', param_name: 'openai_api_key',
    display_name: 'LLMChat - openai_api_key', group: 'LLMChat', param_def: KEY_PARAM,
  }],
};

const KEYS = ['sk-ON-CANVAS', 'sk-IN-PRESET', 'sk-IN-BLOCK'];

/** The active tab holds a typed key in each of the three places; returns the preset node's id. */
function typeAllThreeKeys(): string {
  const llm: Node<NodeData> = {
    id: 'llm', type: 'baseNode', position: { x: 0, y: 0 },
    data: {
      label: 'LLMChat', type: 'LLMChat',
      params: { openai_api_key: '', model: 'gpt-5.2' }, definition: LLM_DEF,
    },
  };
  const instance: Node<NodeData> = {
    id: 'inst', type: 'subgraphNode', position: { x: 400, y: 0 },
    data: { label: 'Block', type: 'subgraph:blk', params: {} },
  };
  store().setNodes([llm, instance]);
  store().setSubgraphs([{
    id: 'blk', name: 'Block', description: '',
    nodes: [{
      id: 'k', type: 'LLMChat', position: { x: 0, y: 0 },
      data: { params: { openai_api_key: 'sk-IN-BLOCK', model: 'gpt-5.2' } },
    }],
    edges: [],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
  }]);
  store().updateNodeParams('llm', { openai_api_key: 'sk-ON-CANVAS' });
  store().addPresetNode(OLD_PRESET, { x: 200, y: 0 });
  const presetId = tab().nodes.find((n) => n.data.isPreset)!.id;
  store().updatePresetInternalParam(presetId, 'chat', 'openai_api_key', 'sk-IN-PRESET');

  // The premise: the editor really holds all three.
  const held = JSON.stringify({ nodes: tab().nodes, subgraphs: tab().subgraphs });
  for (const key of KEYS) expect(held).toContain(key);
  return presetId;
}

function expectBlanked(graph: ReturnType<ReturnType<typeof useTabStore.getState>['getSerializedGraph']>, presetId: string) {
  const text = JSON.stringify(graph);
  for (const key of KEYS) expect(text).not.toContain(key);
  // Blanked in place, not dropped: the slots are there, empty.
  expect(graph.nodes.find((n) => n.id === 'llm')!.data.params.openai_api_key).toBe('');
  expect(graph.nodes.find((n) => n.id === presetId)!.data.internalParams.chat.openai_api_key).toBe('');
  expect(graph.subgraphs[0].nodes[0].data.params.openai_api_key).toBe('');
}

function freshApi() {
  return buildPluginAPI('test-plugin', () => document.createElement('div'));
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('keys');
  useNodeDefStore.setState({ definitions: [LLM_DEF], presets: [OLD_PRESET] } as never);
});

describe('a plugin never reads a typed key', () => {
  it('graph.getGraph() blanks it', () => {
    const presetId = typeAllThreeKeys();
    expectBlanked(freshApi().graph.getGraph(), presetId);
  });

  it('workspace.snapshot() blanks it for the active tab', () => {
    const presetId = typeAllThreeKeys();
    const snap = freshApi().workspace.snapshot();
    if ('error' in snap) throw new Error(snap.error);
    expectBlanked(snap.graph, presetId);
  });

  it('workspace.snapshot() blanks it for a background tab', () => {
    const presetId = typeAllThreeKeys();
    const keyed = store().activeTabId;
    store().addTab('other');
    expect(store().activeTabId).not.toBe(keyed);

    const snap = freshApi().workspace.snapshot(keyed);
    if ('error' in snap) throw new Error(snap.error);
    expect(snap.active).toBe(false);
    expectBlanked(snap.graph, presetId);
  });

  it('passing an options object gets a plugin no key either', () => {
    // The Run hook gets the keys by handing the serializer
    // `{ keepSecrets: true }`. A plugin surface that forwarded its arguments
    // would let a plugin hand it the same object, and the source guard in the
    // Run hook's tests only looks for the word `keepSecrets`.
    const presetId = typeAllThreeKeys();
    const api = freshApi();
    const getGraph = api.graph.getGraph as (options: { keepSecrets: boolean }) => SerializedGraph;
    expectBlanked(getGraph({ keepSecrets: true }), presetId);
    const snapshot = api.workspace.snapshot as (
      tabId: string | undefined,
      options: { keepSecrets: boolean },
    ) => ReturnType<typeof api.workspace.snapshot>;
    const snap = snapshot(undefined, { keepSecrets: true });
    if ('error' in snap) throw new Error(snap.error);
    expectBlanked(snap.graph, presetId);
  });

  it('blanks a key in a block whose node type has left the node list', () => {
    // A node inside a block has no definition of its own, so the strip finds
    // its SECRET params by its type in the node list. MyChat is a custom node,
    // listed while the key is typed and then disabled in the Custom Nodes
    // manager, which fetches the list again without it (#537 review).
    useNodeDefStore.setState({ definitions: [LLM_DEF, { ...LLM_DEF, node_name: 'MyChat' }] } as never);
    store().setNodes([{
      id: 'inst', type: 'subgraphNode', position: { x: 0, y: 0 },
      data: { label: 'Block', type: 'subgraph:blk', params: {} },
    }]);
    store().setSubgraphs([{
      id: 'blk', name: 'Block', description: '',
      nodes: [{
        id: 'k', type: 'MyChat', position: { x: 0, y: 0 },
        data: { params: { openai_api_key: 'sk-DROPPED-TYPE', model: 'gpt-5.2' } },
      }],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    }]);
    useNodeDefStore.setState({ definitions: [LLM_DEF] } as never);

    const api = freshApi();
    const snap = api.workspace.snapshot();
    if ('error' in snap) throw new Error(snap.error);
    for (const graph of [api.graph.getGraph(), snap.graph]) {
      expect(JSON.stringify(graph)).not.toContain('sk-DROPPED-TYPE');
      expect(graph.subgraphs[0].nodes[0].data.params)
        .toEqual({ openai_api_key: '', model: 'gpt-5.2' });
    }
  });
});
