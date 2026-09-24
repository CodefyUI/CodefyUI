/**
 * A key typed into a SECRET field reaches the run.
 *
 * The serializer blanks SECRET params (an LLM API key) for everything that
 * writes, exports or hands the graph out. The Run message is the one reader
 * that needs the key, and the server keeps its stored copy of the run
 * scrubbed (#251). Run used to get the blanked graph too, so a typed key never
 * reached the node: the run failed for a missing key, or quietly used (and
 * billed) the server's environment key instead.
 *
 * Each case types the key through the store action the field really calls,
 * into one of the three places a key can sit.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderHook, act } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import { useGraphExecution } from './useGraphExecution';
import {
  useTabStore,
  _buildPersistedTabForTesting,
  _tabFromPersistedForTesting,
} from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { discardTabNodeUpdates } from '../store/nodeUpdateQueue';
import { subgraphIdOf } from '../utils/subgraph';
import type { NodeData, NodeDefinition, ParamDefinition, PresetDefinition } from '../types';

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

const KEY_PARAM: ParamDefinition = {
  name: 'openai_api_key', param_type: 'secret', default: '', description: '',
  options: [], min_value: null, max_value: null,
};

/** LLMChat as the server describes it: a SECRET key beside an ordinary param. */
const LLM_DEF: NodeDefinition = {
  node_name: 'LLMChat', category: 'LLM', description: '',
  inputs: [{ name: 'in', data_type: 'ANY', description: '', optional: true }],
  outputs: [{ name: 'out', data_type: 'ANY', description: '', optional: false }],
  params: [
    KEY_PARAM,
    {
      name: 'model', param_type: 'string', default: 'gpt-5.2', description: '',
      options: [], min_value: null, max_value: null,
    },
  ],
};

const SOURCE_DEF: NodeDefinition = {
  node_name: 'TextInput', category: 'IO', description: '',
  inputs: [],
  outputs: [{ name: 'out', data_type: 'ANY', description: '', optional: false }],
  params: [],
};

/**
 * A preset saved before presets stopped exposing SECRET params: its exposed
 * `param_def` offers the inner key as a field, so a typed key lands in the
 * preset node's `internalParams`.
 */
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

/**
 * A type the server no longer has, such as a custom node disabled in the Custom
 * Nodes manager: its nodes keep this definition on the canvas, but the
 * registry `useNodeDefStore` last fetched does not list it.
 */
const DROPPED_DEF: NodeDefinition = { ...LLM_DEF, node_name: 'MyChat' };

/** An old preset whose inner node is of that dropped type. */
const DROPPED_PRESET: PresetDefinition = {
  ...OLD_PRESET,
  preset_name: 'DroppedChat',
  nodes: [{ id: 'chat', type: 'MyChat', params: { openai_api_key: '', model: 'gpt-5.2' } }],
};

/** LLMChat as plugin `name` registers it, with or without its SECRET key. */
function pluginChat(name: string, secret: boolean): NodeDefinition {
  return {
    ...LLM_DEF,
    node_name: name,
    params: LLM_DEF.params.filter((p) => secret || p.param_type !== 'secret'),
  };
}

/** An old preset whose inner node names a plugin type without its prefix. */
const BARE_TYPE_PRESET: PresetDefinition = {
  ...OLD_PRESET,
  preset_name: 'BareChat',
  nodes: [{ id: 'chat', type: 'SecretChat', params: { openai_api_key: '', model: 'gpt-5.2' } }],
};

const START: Node<NodeData> = {
  id: 'start', type: 'start', position: { x: -200, y: 0 },
  data: { label: 'Start', type: 'Start', params: {}, executionStatus: 'idle' },
};

/** A canvas node as `buildFlowNode` makes one: definition attached, params at their defaults. */
function canvasNode(id: string, definition: NodeDefinition, x = 0): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: {
      label: definition.node_name,
      type: definition.node_name,
      params: Object.fromEntries(definition.params.map((p) => [p.name, p.default])),
      definition,
      executionStatus: 'idle',
    },
  };
}

function trigger(target: string): Edge {
  return {
    id: `t-${target}`, source: 'start', target, sourceHandle: 'trigger',
    targetHandle: '__trigger', type: 'triggerEdge', data: { type: 'trigger' },
  };
}

/** A preset node as a block holds it, with `key` typed into its inner node. */
function presetEntry(id: string, preset: PresetDefinition, key: string) {
  return {
    id, type: `preset:${preset.preset_name}`, position: { x: 0, y: 0 },
    data: { params: {}, internalParams: { chat: { openai_api_key: key, model: 'gpt-5.2' } } },
  };
}

/** Start, triggering one block instance whose block holds `entries`. */
function seedBlock(entries: unknown[]) {
  store().setNodes([START, {
    id: 'inst', type: 'subgraphNode', position: { x: 200, y: 0 },
    data: { label: 'Block', type: 'subgraph:blk', params: {} },
  }]);
  store().setEdges([trigger('inst')]);
  store().setSubgraphs([{
    id: 'blk', name: 'Block', description: '',
    nodes: entries,
    edges: [],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
  }]);
}

/**
 * A page reload for the active tab: its nodes go through the autosave record
 * and come back with the definitions they were saved with. No list is
 * replayed, so no list this session names their types.
 */
function reload() {
  const record = _buildPersistedTabForTesting(tab());
  const restored = _tabFromPersistedForTesting(record, tab());
  useTabStore.setState({ tabs: [restored], activeTabId: restored.id });
}

/** Press Run on the active tab against a fake socket: what was sent, and what was validated. */
async function run() {
  const ws = {
    connected: true,
    on: vi.fn(),
    off: vi.fn(),
    send: vi.fn(),
    connect: vi.fn(async () => {}),
  };
  const { activeTabId } = useTabStore.getState();
  useTabStore.setState((s) => ({
    tabs: s.tabs.map((t) => (t.id === activeTabId ? { ...t, ws: ws as never } : t)),
  }));
  const { result } = renderHook(() => useGraphExecution());
  await act(async () => {
    await result.current.execute();
  });
  expect(ws.send).toHaveBeenCalledTimes(1);
  expect(validateGraphMock).toHaveBeenCalledTimes(1);
  return { message: ws.send.mock.calls[0][0], validated: validateGraphMock.mock.calls[0] };
}

beforeEach(() => {
  validateGraphMock.mockReset();
  validateGraphMock.mockResolvedValue({ valid: true, errors: [] });
  vi.mocked(getRun).mockReset();
  vi.mocked(getRun).mockResolvedValue(null);
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('keys');
  useNodeDefStore.setState({ definitions: [LLM_DEF, SOURCE_DEF], presets: [OLD_PRESET] } as never);
});

afterEach(() => {
  discardTabNodeUpdates();
});

describe('Run sends the key the user typed', () => {
  it('into a node on the canvas', async () => {
    store().setNodes([START, canvasNode('llm', LLM_DEF)]);
    store().setEdges([trigger('llm')]);
    store().updateNodeParams('llm', { openai_api_key: 'sk-TYPED-ON-CANVAS' });

    const { message, validated } = await run();

    const sent = message.nodes.find((n: any) => n.id === 'llm');
    expect(sent.data.params).toEqual({ openai_api_key: 'sk-TYPED-ON-CANVAS', model: 'gpt-5.2' });
    // Validation checks the same node with the key left out: it has no use
    // for the key, so it is never sent one.
    const checked = validated[0].find((n: any) => n.id === 'llm');
    expect(checked).toEqual({
      ...sent,
      data: { ...sent.data, params: { openai_api_key: '', model: 'gpt-5.2' } },
    });
    expect(JSON.stringify(validated)).not.toContain('sk-TYPED-ON-CANVAS');
  });

  it("into a preset node's internal params", async () => {
    store().setNodes([START]);
    store().addPresetNode(OLD_PRESET, { x: 0, y: 0 });
    const presetId = tab().nodes.find((n) => n.data.isPreset)!.id;
    store().setEdges([trigger(presetId)]);
    store().updatePresetInternalParam(presetId, 'chat', 'openai_api_key', 'sk-TYPED-IN-PRESET');

    const { message, validated } = await run();

    const sent = message.nodes.find((n: any) => n.id === presetId);
    expect(sent.data.internalParams.chat).toEqual({
      openai_api_key: 'sk-TYPED-IN-PRESET', model: 'gpt-5.2',
    });
    expect(JSON.stringify(validated)).not.toContain('sk-TYPED-IN-PRESET');
  });

  it('into a node inside a block', async () => {
    store().setNodes([START, canvasNode('a', SOURCE_DEF), canvasNode('k', LLM_DEF, 200)]);
    store().setEdges([
      trigger('a'),
      { id: 'e1', source: 'a', target: 'k', sourceHandle: 'out', targetHandle: 'in' },
    ]);
    // Start stays outside; its trigger moves onto the block instance.
    store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'start' })));
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);

    // Open the block, type the key into the node inside it, step back out.
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    expect(store().enterSubgraph(instanceId)).toBe(true);
    store().updateNodeParams('k', { openai_api_key: 'sk-TYPED-IN-BLOCK' });
    store().exitSubgraph();
    // The block definition holds the key now (the premise).
    expect(JSON.stringify(tab().subgraphs)).toContain('sk-TYPED-IN-BLOCK');

    const { message, validated } = await run();

    const inner = message.subgraphs[0].nodes.find((n: any) => n.id === 'k');
    expect(inner.data.params.openai_api_key).toBe('sk-TYPED-IN-BLOCK');
    expect(JSON.stringify(validated)).not.toContain('sk-TYPED-IN-BLOCK');
  });

  it('into a preset whose inner node names a plugin type without its prefix', async () => {
    // The server's registry finds a bare `SecretChat` as the alphabetically
    // first `<plugin>:SecretChat` (`NodeRegistry.get`), here `c9:SecretChat`,
    // and blanks its key in the stored run, so the run gets the key.
    useNodeDefStore.setState({
      definitions: [
        LLM_DEF, SOURCE_DEF, pluginChat('z9:SecretChat', false), pluginChat('c9:SecretChat', true),
      ],
    } as never);
    store().setNodes([START]);
    store().addPresetNode(BARE_TYPE_PRESET, { x: 0, y: 0 });
    const presetId = tab().nodes.find((n) => n.data.isPreset)!.id;
    store().setEdges([trigger(presetId)]);
    store().updatePresetInternalParam(presetId, 'chat', 'openai_api_key', 'sk-BARE-PLUGIN-TYPE');

    const { message } = await run();

    const sent = message.nodes.find((n: any) => n.id === presetId);
    expect(sent.data.internalParams.chat.openai_api_key).toBe('sk-BARE-PLUGIN-TYPE');
  });
});

// The server blanks a SECRET param in the run it stores only when its own
// registry calls it SECRET. For a type it has dropped, the key would be written
// to the run history as typed, and the run then fails as an unknown type
// anyway, so Run blanks it as Save does.
describe('Run still blanks a key the server could not keep out of its run history', () => {
  it('on a canvas node whose type the server no longer lists', async () => {
    store().setNodes([START, canvasNode('gone', DROPPED_DEF), canvasNode('llm', LLM_DEF, 200)]);
    store().setEdges([trigger('gone'), trigger('llm')]);
    store().updateNodeParams('gone', { openai_api_key: 'sk-DROPPED-TYPE' });
    store().updateNodeParams('llm', { openai_api_key: 'sk-KNOWN-TYPE' });

    const { message } = await run();

    const sent = (id: string) => message.nodes.find((n: any) => n.id === id);
    expect(sent('gone').data.params).toEqual({ openai_api_key: '', model: 'gpt-5.2' });
    expect(sent('llm').data.params.openai_api_key).toBe('sk-KNOWN-TYPE');
  });

  it("in a preset whose inner node's type the server no longer lists", async () => {
    store().setNodes([START]);
    store().addPresetNode(DROPPED_PRESET, { x: 0, y: 0 });
    store().addPresetNode(OLD_PRESET, { x: 0, y: 200 });
    const idOf = (name: string) =>
      tab().nodes.find((n) => n.data.presetDefinition?.preset_name === name)!.id;
    const dropped = idOf('DroppedChat');
    const known = idOf('KeyedChat');
    store().setEdges([trigger(dropped), trigger(known)]);
    store().updatePresetInternalParam(dropped, 'chat', 'openai_api_key', 'sk-DROPPED-INNER');
    store().updatePresetInternalParam(known, 'chat', 'openai_api_key', 'sk-KNOWN-INNER');

    const { message } = await run();

    const sent = (id: string) => message.nodes.find((n: any) => n.id === id);
    expect(sent(dropped).data.internalParams.chat.openai_api_key).toBe('');
    expect(sent(known).data.internalParams.chat.openai_api_key).toBe('sk-KNOWN-INNER');
  });

  it("in a preset inside a block whose inner node's type the server no longer lists", async () => {
    useNodeDefStore.setState({ presets: [OLD_PRESET, DROPPED_PRESET] } as never);
    seedBlock([
      presetEntry('dropped', DROPPED_PRESET, 'sk-DROPPED-IN-BLOCK'),
      presetEntry('known', OLD_PRESET, 'sk-KNOWN-IN-BLOCK'),
    ]);

    const { message } = await run();

    const inner = (id: string) => message.subgraphs[0].nodes.find((n: any) => n.id === id);
    expect(inner('dropped').data.internalParams.chat.openai_api_key).toBe('');
    expect(inner('known').data.internalParams.chat.openai_api_key).toBe('sk-KNOWN-IN-BLOCK');
  });

  it('in a preset whose bare inner type the server resolves to a plugin with no SECRET key', async () => {
    // The mirror of the bare-name case above: here `c9:SecretChat`, the
    // alphabetically first, has no SECRET param, so the server resolves the
    // bare `SecretChat` to it and would store the key as typed.
    useNodeDefStore.setState({
      definitions: [
        LLM_DEF, SOURCE_DEF, pluginChat('z9:SecretChat', true), pluginChat('c9:SecretChat', false),
      ],
    } as never);
    store().setNodes([START]);
    store().addPresetNode(BARE_TYPE_PRESET, { x: 0, y: 0 });
    const presetId = tab().nodes.find((n) => n.data.isPreset)!.id;
    store().setEdges([trigger(presetId)]);
    store().updatePresetInternalParam(presetId, 'chat', 'openai_api_key', 'sk-BARE-PLUGIN-TYPE');

    const { message } = await run();

    const sent = message.nodes.find((n: any) => n.id === presetId);
    expect(sent.data.internalParams.chat.openai_api_key).toBe('');
  });

  // A node inside a block keeps no definition of its own, so which of its
  // params are SECRET comes from the node list, and a preset's slots from the
  // preset list. Each fetch replaces both lists whole (#537 review).

  it('on a node inside a block whose type the node list no longer has', async () => {
    // MyChat is a custom node, listed while the keys are typed.
    useNodeDefStore.setState({ definitions: [LLM_DEF, SOURCE_DEF, DROPPED_DEF] } as never);
    store().setNodes([
      START, canvasNode('a', SOURCE_DEF), canvasNode('gone', DROPPED_DEF, 200),
      canvasNode('llm', LLM_DEF, 400),
    ]);
    store().setEdges([
      trigger('a'),
      { id: 'e1', source: 'a', target: 'gone', sourceHandle: 'out', targetHandle: 'in' },
      { id: 'e2', source: 'a', target: 'llm', sourceHandle: 'out', targetHandle: 'in' },
    ]);
    store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'start' })));
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
    const instanceId = tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
    expect(store().enterSubgraph(instanceId)).toBe(true);
    store().updateNodeParams('gone', { openai_api_key: 'sk-DROPPED-IN-BLOCK' });
    store().updateNodeParams('llm', { openai_api_key: 'sk-KNOWN-IN-BLOCK' });
    store().exitSubgraph();
    // The Custom Nodes manager disables MyChat and fetches the list again.
    useNodeDefStore.setState({ definitions: [LLM_DEF, SOURCE_DEF] } as never);

    const { message } = await run();

    const inner = (id: string) => message.subgraphs[0].nodes.find((n: any) => n.id === id);
    expect(inner('gone').data.params).toEqual({ openai_api_key: '', model: 'gpt-5.2' });
    expect(inner('llm').data.params.openai_api_key).toBe('sk-KNOWN-IN-BLOCK');
  });

  it('in a preset inside a block that the preset list no longer has', async () => {
    // FileChat came in with an opened file, whose reader merged it into the
    // list. Its inner node is an LLMChat, a type the server has.
    const filePreset: PresetDefinition = { ...OLD_PRESET, preset_name: 'FileChat' };
    useNodeDefStore.setState({ presets: [OLD_PRESET, filePreset] } as never);
    seedBlock([
      presetEntry('file', filePreset, 'sk-FILE-PRESET-IN-BLOCK'),
      presetEntry('known', OLD_PRESET, 'sk-KNOWN-IN-BLOCK'),
    ]);
    // A fetch replaces the list with the server's, which never had FileChat.
    useNodeDefStore.setState({ presets: [OLD_PRESET] } as never);

    const { message } = await run();

    // The server cannot place that slot: its preset registry has no FileChat,
    // and the message's `presets[]` no longer carries one.
    const inner = (id: string) => message.subgraphs[0].nodes.find((n: any) => n.id === id);
    expect(inner('file').data.internalParams.chat.openai_api_key).toBe('');
    expect(inner('known').data.internalParams.chat.openai_api_key).toBe('sk-KNOWN-IN-BLOCK');
  });

  // A node autosave brings back after a reload keeps the definition it was
  // saved with, whether or not a list this session names its type.

  it('on a node restored after a reload and collapsed into a block, of a type no list names', async () => {
    // RestoredChat is a custom node disabled before the reload.
    const restoredDef: NodeDefinition = { ...LLM_DEF, node_name: 'RestoredChat' };
    store().setNodes([
      START, canvasNode('a', SOURCE_DEF), canvasNode('gone', restoredDef, 200),
      canvasNode('llm', LLM_DEF, 400),
    ]);
    store().setEdges([
      trigger('a'),
      { id: 'e1', source: 'a', target: 'gone', sourceHandle: 'out', targetHandle: 'in' },
      { id: 'e2', source: 'a', target: 'llm', sourceHandle: 'out', targetHandle: 'in' },
    ]);
    reload();
    store().updateNodeParams('gone', { openai_api_key: 'sk-RESTORED-TYPE' });
    store().updateNodeParams('llm', { openai_api_key: 'sk-KNOWN-TYPE' });
    store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'start' })));
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);

    const { message } = await run();

    const inner = (id: string) => message.subgraphs[0].nodes.find((n: any) => n.id === id);
    expect(inner('gone').data.params).toEqual({ openai_api_key: '', model: 'gpt-5.2' });
    expect(inner('llm').data.params.openai_api_key).toBe('sk-KNOWN-TYPE');
  });

  it('in a preset node restored after a reload and collapsed into a block, of a preset no list names', async () => {
    // RestoredPreset came in with a file the previous session opened.
    const restoredPreset: PresetDefinition = { ...OLD_PRESET, preset_name: 'RestoredPreset' };
    store().setNodes([START]);
    store().addPresetNode(restoredPreset, { x: 0, y: 0 });
    store().addPresetNode(OLD_PRESET, { x: 0, y: 200 });
    const idOf = (name: string) =>
      tab().nodes.find((n) => n.data.presetDefinition?.preset_name === name)!.id;
    store().setEdges([trigger(idOf('RestoredPreset')), trigger(idOf('KeyedChat'))]);
    reload();
    store().updatePresetInternalParam(idOf('RestoredPreset'), 'chat', 'openai_api_key', 'sk-RESTORED-PRESET');
    store().updatePresetInternalParam(idOf('KeyedChat'), 'chat', 'openai_api_key', 'sk-KNOWN-PRESET');
    store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'start' })));
    expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);

    const { message } = await run();

    const inner = (name: string) =>
      message.subgraphs[0].nodes.find((n: any) => n.type === `preset:${name}`);
    expect(inner('RestoredPreset').data.internalParams.chat.openai_api_key).toBe('');
    expect(inner('KeyedChat').data.internalParams.chat.openai_api_key).toBe('sk-KNOWN-PRESET');
  });
});

// ── Nothing else asks for the keys ──────────────────────────────────────────
// `keepSecrets` is how the Run message gets them. A save, an export, autosave
// or a plugin surface that passed it too would put a typed key in a file or in
// third-party code, and each of those has its own test only for its own path.
// This reads the sources from disk, as lazyBoundaries.test.ts does.

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'test') sourceFiles(full, out);
    } else if (/\.tsx?$/.test(entry.name) && !/\.(test|d)\.tsx?$/.test(entry.name)) {
      out.push(full);
    }
  }
  return out;
}

describe('the keys go to Run and nowhere else', () => {
  it('only the store and the Run hook mention keepSecrets', () => {
    const users = sourceFiles(SRC)
      .filter((file) => readFileSync(file, 'utf8').includes('keepSecrets'))
      // Posix separators, so the list reads the same on Windows and Linux.
      .map((file) => relative(SRC, file).split(sep).join('/'))
      .sort();
    expect(users).toEqual(['hooks/useGraphExecution.ts', 'store/tabStore.ts']);
  });
});
