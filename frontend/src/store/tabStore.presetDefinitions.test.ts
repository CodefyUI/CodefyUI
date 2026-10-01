/**
 * #541: the preset definitions a document owns, as every serialized form
 * writes them.
 *
 *  - An inner node whose type the node list does not have at that moment --
 *    the list has not answered yet, its fetch failed, or the plugin that
 *    supplies the type is disabled -- keeps its settings. Only the values
 *    known to be keys are blanked, the rule a node inside a block gets. The
 *    stripped copy is what autosave stores and Save writes, and the document's
 *    copy is the one that runs, so a blanked setting could never come back.
 *  - A record built before the node list arrived is rebuilt once it does.
 *  - An owned definition the backend cannot read is kept as it came, so the
 *    server refuses it by name, and nothing on the way there throws.
 *  - Run keeps a preset key only where the definition it SENDS places a type
 *    that declares the param SECRET.
 *  - A key in an inner param the definition never exposed stays out of
 *    autosave and Save.
 *
 * Every value below is an obviously fake test string.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';

import {
  useTabStore,
  _buildPersistedTabForTesting,
  _persistedTabsForTesting,
  _tabFromPersistedForTesting,
} from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import { resolveExample } from '../utils/openExample';
import type {
  NodeData,
  NodeDefinition,
  ParamDefinition,
  PresetDefinition,
  SubgraphDefinition,
} from '../types';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

const LEGACY_KEY = 'sk-FAKE-541-LEGACY-DEFAULT';
const TYPED_KEY = 'sk-FAKE-541-TYPED-SLOT';
const STALE_KEY = 'sk-FAKE-541-STALE-PALETTE-SLOT';

function param(name: string, paramType: ParamDefinition['param_type']): ParamDefinition {
  return {
    name, param_type: paramType, default: '', description: '',
    options: [], min_value: null, max_value: null,
  };
}

/** A node type with two plain settings. */
function plainType(name: string): NodeDefinition {
  return {
    node_name: name, category: 'Layers', description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [param('in_features', 'int'), param('out_features', 'int')],
  };
}

/** A node type that declares `key` SECRET beside a plain `model`. */
function keyedType(name: string, key = 'api_key'): NodeDefinition {
  return {
    node_name: name, category: 'LLM', description: '', inputs: [],
    outputs: [{ name: 'value', data_type: 'ANY', description: '', optional: false }],
    params: [param('model', 'string'), param(key, 'secret')],
  };
}

function secretSlot(innerNode: string, name: string): PresetDefinition['exposed_params'][number] {
  return {
    internal_node: innerNode, param_name: name, display_name: name, group: 'Inner',
    param_def: param(name, 'secret'),
  };
}

/**
 * A definition whose inner nodes carry real settings, as an exported preset
 * does: a layer, and a chat node of an older preset that baked a key into its
 * default and exposes that slot as SECRET.
 */
function pipeline(name: string, chatType = 'PipelineChat'): PresetDefinition {
  return {
    preset_name: name, category: 'Owned', description: 'pipeline', tags: [],
    nodes: [
      { id: 'lin', type: 'PipelineLinear', params: { in_features: 784, out_features: 10 } },
      { id: 'chat', type: chatType, params: { model: 'small', api_key: LEGACY_KEY } },
    ],
    edges: [],
    exposed_inputs: [{
      name: 'x', internal_node: 'lin', internal_port: 'in', data_type: 'TENSOR', description: '',
    }],
    exposed_outputs: [{
      name: 'y', internal_node: 'lin', internal_port: 'out', data_type: 'TENSOR', description: '',
    }],
    exposed_params: [secretSlot('chat', 'api_key')],
  };
}

/** `pipeline`'s inner nodes with only the key blanked. */
const PIPELINE_KEPT = [
  { id: 'lin', type: 'PipelineLinear', params: { in_features: 784, out_features: 10 } },
  { id: 'chat', type: 'PipelineChat', params: { model: 'small', api_key: '' } },
];

/** A definition with one inner node of `type`, nothing exposed. */
function oneInner(name: string, type: string, params: Record<string, unknown>): PresetDefinition {
  return {
    preset_name: name, category: 'Owned', description: '', tags: [],
    nodes: [{ id: 'inner', type, params }],
    edges: [], exposed_inputs: [], exposed_outputs: [], exposed_params: [],
  };
}

/** A top-level card of `preset:<name>` attached to `attached`. */
function card(
  id: string,
  name: string,
  attached: PresetDefinition | undefined,
  internalParams: Record<string, Record<string, any>> = {},
): Node<NodeData> {
  return {
    id, type: 'presetNode', position: { x: 0, y: 0 },
    data: {
      label: name, type: `preset:${name}`, params: {}, isPreset: true,
      presetDefinition: attached, internalParams, executionStatus: 'idle',
    },
  };
}

/** A block instance and its definition, holding `entries` as a block stores them. */
function block(entries: unknown[]): { instance: Node<NodeData>; definition: SubgraphDefinition } {
  return {
    instance: {
      id: 'block', type: 'subgraphNode', position: { x: 300, y: 0 },
      data: { label: 'Block', type: 'subgraph:blk', params: {} },
    },
    definition: {
      id: 'blk', name: 'Block', description: '', edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
      nodes: entries,
    } as unknown as SubgraphDefinition,
  };
}

function select(...ids: string[]) {
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: ids.includes(n.id) })));
}

/** The serialized node `id`, at the top level or inside the one block. */
function serializedNode(graph: { nodes: any[]; subgraphs?: { nodes: any[] }[] }, id: string): any {
  return graph.nodes.find((n) => n.id === id)
    ?? graph.subgraphs?.[0]?.nodes.find((n: any) => n.id === id);
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  useNodeDefStore.setState({ definitions: [], presets: [], loading: false, error: null } as never);
});

// -- Inner types the node list does not have ----------------------------------

describe('an owned definition serialized while the node list does not have its inner types', () => {
  it('keeps every setting and blanks only the exposed SECRET slot, in every form', () => {
    // The node list is EMPTY: /api/nodes has not answered, or its fetch failed.
    const owned = pipeline('Pipeline');
    store().loadGraphDocument({
      nodes: [card('card', 'Pipeline', owned, { lin: { in_features: 784, out_features: 10 } })],
      edges: [], boundFile: null, presets: [owned],
    });
    select('card');
    store().copySelectedNodes();

    const forms: Array<[string, PresetDefinition[]]> = [
      ['autosave', _buildPersistedTabForTesting(tab()).presets ?? []],
      ['Save', store().getSerializedGraph().presets],
      ['Run', store().getSerializedGraph({ keepSecrets: true }).presets],
      ['clipboard', store().clipboard!.presets ?? []],
    ];
    for (const [form, presets] of forms) {
      expect(presets.map((p) => p.preset_name), form).toEqual(['Pipeline']);
      expect(presets[0].nodes, form).toEqual(PIPELINE_KEPT);
      expect(JSON.stringify(presets), form).not.toContain(LEGACY_KEY);
    }
    // The tab's own copy is untouched: only what is written is stripped.
    expect(tab().presets[0]).toBe(owned);
  });

  it('a record built before the node list answered restores the definition whole', () => {
    // A reload whose first autosave runs before /api/nodes answers: the record
    // it writes is what the next session restores.
    const owned = pipeline('Pipeline');
    store().loadGraphDocument({
      nodes: [card('card', 'Pipeline', owned)], edges: [], boundFile: null, presets: [owned],
    });
    const record = _buildPersistedTabForTesting(tab());
    const restored = _tabFromPersistedForTesting(record, tab());
    useTabStore.setState({ tabs: [restored], activeTabId: restored.id });
    // The node list arrives, with the installed preset of the same name.
    useNodeDefStore.setState({
      definitions: [plainType('PipelineLinear'), keyedType('PipelineChat')],
      presets: [pipeline('Pipeline')],
    } as never);

    // The tab's copy wins over the installed one, so it is the one a new
    // card and the next Save use: a blanked setting here could never return.
    expect(tab().presets[0].nodes).toEqual(PIPELINE_KEPT);
    store().addPresetNode(useNodeDefStore.getState().presets[0], { x: 0, y: 300 });
    const fresh = tab().nodes[tab().nodes.length - 1];
    expect(fresh.data.internalParams).toEqual({
      lin: { in_features: 784, out_features: 10 },
      chat: { model: 'small', api_key: '' },
    });
    expect(store().getSerializedGraph().presets[0].nodes).toEqual(PIPELINE_KEPT);
  });

  it('blanks what an earlier list declared SECRET on a type the list has since dropped, as a block does', () => {
    // A plugin's chat node, listed when the session started, then disabled.
    useNodeDefStore.setState({ definitions: [keyedType('DroppedChat')] } as never);
    useNodeDefStore.setState({ definitions: [plainType('Other')] } as never);
    const owned = oneInner('Dropped', 'DroppedChat', { model: 'small', api_key: LEGACY_KEY });
    const inBlock = block([{
      id: 'plain', type: 'DroppedChat', position: { x: 0, y: 0 },
      data: { params: { model: 'small', api_key: LEGACY_KEY } },
    }]);
    store().loadGraphDocument({
      nodes: [card('card', 'Dropped', owned), inBlock.instance],
      edges: [], boundFile: null, presets: [owned], subgraphs: [inBlock.definition],
    });

    const saved = store().getSerializedGraph();
    expect(saved.presets[0].nodes[0].params).toEqual({ model: 'small', api_key: '' });
    // The same values on a node of the same type inside a block come out the same.
    expect(serializedNode(saved, 'plain').data.params).toEqual(saved.presets[0].nodes[0].params);
    const record = _buildPersistedTabForTesting(tab());
    expect(record.presets![0].nodes[0].params).toEqual({ model: 'small', api_key: '' });
    expect(JSON.stringify(record)).not.toContain(LEGACY_KEY);
  });
});

// -- A slot the definition itself exposes as SECRET ---------------------------

describe('a slot the definition exposes as SECRET, on an inner type the node list has', () => {
  const BAKED_KEY = 'sk-FAKE-541-BAKED-DEFAULT';

  it('has its inner default blanked in every form, like the exposed default and the card value', () => {
    // The list has the type and does not call `in_features` SECRET; the
    // definition does, and baked a key into the default.
    useNodeDefStore.setState({ definitions: [plainType('PipelineLinear')] } as never);
    const baked: PresetDefinition = {
      ...oneInner('Baked', 'PipelineLinear', { in_features: BAKED_KEY, out_features: 10 }),
      exposed_params: [{
        ...secretSlot('inner', 'in_features'),
        param_def: { ...param('in_features', 'secret'), default: BAKED_KEY },
      }],
    };
    store().loadGraphDocument({
      nodes: [card('card', 'Baked', baked, { inner: { in_features: BAKED_KEY, out_features: 10 } })],
      edges: [], boundFile: null, presets: [baked],
    });
    select('card');
    store().copySelectedNodes();

    const forms: Array<[string, unknown]> = [
      ['autosave', _buildPersistedTabForTesting(tab())],
      ['Save', store().getSerializedGraph()],
      ['Run', store().getSerializedGraph({ keepSecrets: true })],
      ['clipboard', store().clipboard!.presets],
    ];
    for (const [form, written] of forms) {
      expect(JSON.stringify(written), form).not.toContain(BAKED_KEY);
    }
    const saved = store().getSerializedGraph().presets[0];
    expect(saved.nodes[0].params).toEqual({ in_features: '', out_features: 10 });
    expect(saved.exposed_params[0].param_def!.default).toBe('');
  });
});

// -- The node list arriving after the first autosave --------------------------

describe('the node list arriving after a record was built', () => {
  /** A document owning a definition whose inner type no list has named yet. */
  function loadLateDocument(name: string, type: string) {
    const owned = oneInner(name, type, { model: 'small', api_key: LEGACY_KEY });
    // A legacy block whose preset the tab does not own: only the installed
    // list can supply its definition.
    const legacy = block([{
      id: 'inside', type: 'preset:InstalledOnly', position: { x: 0, y: 0 },
      data: { params: {}, internalParams: {} },
    }]);
    store().loadGraphDocument({
      nodes: [card('card', name, owned), legacy.instance],
      edges: [], boundFile: null, presets: [owned], subgraphs: [legacy.definition],
    });
  }

  it('is rebuilt from the list rather than handed back from the cache', () => {
    loadLateDocument('Late', 'LateChat');
    const [first] = _persistedTabsForTesting(useTabStore.getState().tabs);
    // The premise: before the list answers, nothing says `api_key` is a key,
    // and the installed preset the block uses is not known either.
    expect(first.presets!.map((p) => p.preset_name)).toEqual(['Late']);
    expect(first.presets![0].nodes[0].params).toEqual({ model: 'small', api_key: LEGACY_KEY });

    useNodeDefStore.setState({
      definitions: [keyedType('LateChat'), plainType('PipelineLinear')],
      presets: [oneInner('InstalledOnly', 'PipelineLinear', { in_features: 3 })],
    } as never);
    const [second] = _persistedTabsForTesting(useTabStore.getState().tabs);

    expect(second).not.toBe(first);
    expect(second.presets!.map((p) => p.preset_name)).toEqual(['Late', 'InstalledOnly']);
    expect(second.presets![0].nodes[0].params).toEqual({ model: 'small', api_key: '' });
    expect(JSON.stringify(second)).not.toContain(LEGACY_KEY);
  });

  describe('through the autosave timer (module reload)', () => {
    const STORAGE_KEY = 'codefyui-tabs';

    beforeEach(() => {
      vi.resetModules();
      localStorage.clear();
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
      localStorage.clear();
    });

    it('rewrites a record the first autosave wrote, with nothing on the tab changed', async () => {
      // A fresh module: no list this session has named `TimerChat` yet.
      const tabs = await import('./tabStore');
      const defs = await import('./nodeDefStore');
      defs.useNodeDefStore.setState({ definitions: [], presets: [] } as never);
      const owned = oneInner('Late', 'TimerChat', { model: 'small', api_key: LEGACY_KEY });
      tabs.useTabStore.getState().loadGraphDocument({
        nodes: [card('card', 'Late', owned)], edges: [], boundFile: null, presets: [owned],
      });
      const writtenPresets = () =>
        JSON.parse(localStorage.getItem(STORAGE_KEY)!).tabs[0].presets as PresetDefinition[];

      // The first autosave runs before /api/nodes answers.
      vi.advanceTimersByTime(300);
      expect(writtenPresets()[0].nodes[0].params).toEqual({ model: 'small', api_key: LEGACY_KEY });

      // The list arrives. Nothing on the tab changes.
      defs.useNodeDefStore.setState({ definitions: [keyedType('TimerChat')] } as never);
      vi.advanceTimersByTime(300);

      expect(writtenPresets()[0].nodes[0].params).toEqual({ model: 'small', api_key: '' });
      expect(localStorage.getItem(STORAGE_KEY)).not.toContain(LEGACY_KEY);
    });
  });
});

// -- An entry that leaves out only what the server fills in -------------------

describe('an entry that leaves out only the fields the server fills in', () => {
  beforeEach(() => {
    useNodeDefStore.setState({ definitions: [plainType('PipelineLinear')] } as never);
  });

  /** Only what the server requires: a name, `nodes` and `edges`. */
  function bare(name: string): PresetDefinition {
    return {
      preset_name: name,
      nodes: [{ id: 'lin', type: 'PipelineLinear', params: { in_features: 1 } }],
      edges: [],
    } as unknown as PresetDefinition;
  }

  /** `bare` as the server reads it: the defaults of its `PresetDefinition`. */
  function withServerDefaults(name: string): PresetDefinition {
    return {
      ...bare(name),
      category: 'Preset', description: '', tags: [],
      exposed_inputs: [], exposed_outputs: [], exposed_params: [],
    };
  }

  it('opens with those defaults, and the card, the palette and every form read them', () => {
    const read = resolveExample({
      nodes: [{
        id: 'card', type: 'preset:Bare', position: { x: 0, y: 0 },
        data: { params: {}, internalParams: { lin: { in_features: 1 } } },
      }],
      edges: [],
      presets: [bare('Bare')],
    });
    store().loadGraphDocument({
      nodes: read.nodes, edges: read.edges, boundFile: null, presets: read.presets,
    });

    expect(tab().presets).toStrictEqual([withServerDefaults('Bare')]);
    // The card runs, draws and configures the filled entry.
    expect(tab().nodes[0].data.presetDefinition).toBe(tab().presets[0]);
    expect(tab().nodes[0].data.definition).toMatchObject({
      category: 'Preset', inputs: [], outputs: [],
    });
    expect(useNodeDefStore.getState().presets).toStrictEqual([withServerDefaults('Bare')]);
    expect(_buildPersistedTabForTesting(tab()).presets).toStrictEqual([withServerDefaults('Bare')]);
    expect(store().getSerializedGraph().presets).toStrictEqual([withServerDefaults('Bare')]);
    expect(store().getSerializedGraph({ keepSecrets: true }).presets)
      .toStrictEqual([withServerDefaults('Bare')]);
  });

  it('is filled in at every other door a tab adopts it through', () => {
    // A document installed directly, with a card attached to the bare entry.
    store().loadGraphDocument({
      nodes: [card('direct', 'Direct', bare('Direct'))], edges: [], boundFile: null,
      presets: [bare('Direct')],
    });
    expect(tab().presets).toStrictEqual([withServerDefaults('Direct')]);
    expect(tab().nodes[0].data.presetDefinition).toBe(tab().presets[0]);

    // A record that holds the bare entry, restored.
    const record = _buildPersistedTabForTesting(tab());
    record.presets = [bare('Direct')];
    const restored = _tabFromPersistedForTesting(record, tab());
    expect(restored.presets).toStrictEqual([withServerDefaults('Direct')]);
    expect(restored.nodes[0].data.presetDefinition).toBe(restored.presets[0]);

    // A clipboard that carries it, pasted.
    useTabStore.setState({
      clipboard: {
        nodes: [card('pasted', 'Pasted', bare('Pasted'))], edges: [], subgraphs: [],
        presets: [bare('Pasted')],
      },
    });
    store().addTab('paste');
    store().pasteNodes();
    expect(tab().presets).toStrictEqual([withServerDefaults('Pasted')]);
    expect(tab().nodes[0].data.presetDefinition).toBe(tab().presets[0]);

    // A template that ships it, inserted.
    store().addTab('insert');
    store().insertGraph(
      [card('inserted', 'Inserted', bare('Inserted'))], [], [], undefined, [bare('Inserted')],
    );
    expect(tab().presets).toStrictEqual([withServerDefaults('Inserted')]);
    expect(tab().nodes[0].data.presetDefinition).toBe(tab().presets[0]);

    // A palette entry, dropped onto the canvas.
    store().addTab('palette');
    store().addPresetNode(bare('Palette'), { x: 0, y: 0 });
    expect(tab().presets).toStrictEqual([withServerDefaults('Palette')]);
    expect(tab().nodes[0].data.presetDefinition).toBe(tab().presets[0]);
    expect(tab().nodes[0].data.internalParams).toEqual({ lin: { in_features: 1 } });
  });
});

// -- An owned definition the backend cannot read ------------------------------

describe('an owned definition the backend cannot read is kept as it came', () => {
  beforeEach(() => {
    useNodeDefStore.setState({ definitions: [plainType('PipelineLinear')] } as never);
  });

  /** `pipeline` with no `nodes`, as an unvalidated file can carry it. */
  function withoutNodes(name: string): PresetDefinition {
    const broken: Partial<PresetDefinition> = { ...pipeline(name) };
    delete broken.nodes;
    return broken as PresetDefinition;
  }

  it('a card of a definition with no nodes: every tab autosaves, and Save, Run and paste keep the entry', () => {
    store().loadGraphDocument({
      nodes: [card('other', 'Pipeline', pipeline('Pipeline'))],
      edges: [], boundFile: null, presets: [pipeline('Pipeline')],
    });
    store().addTab('broken');
    const broken = withoutNodes('Broken');
    store().loadGraphDocument({
      nodes: [card('card', 'Broken', broken, { lin: { in_features: 1 } })],
      edges: [], boundFile: null, presets: [broken],
    });

    const records = _persistedTabsForTesting(useTabStore.getState().tabs);
    expect(records.map((r) => r.name)).toEqual(['test', 'broken']);
    expect(records[1].presets).toStrictEqual([broken]);
    expect(store().getSerializedGraph().presets).toStrictEqual([broken]);
    expect(store().getSerializedGraph({ keepSecrets: true }).presets).toStrictEqual([broken]);
    expect(_tabFromPersistedForTesting(records[1], tab()).presets).toStrictEqual([broken]);

    select('card');
    store().copySelectedNodes();
    store().addTab('pasted');
    store().pasteNodes();
    expect(tab().presets).toStrictEqual([broken]);
    expect(store().getSerializedGraph().presets).toStrictEqual([broken]);
  });

  it('a definition with no nodes referenced only inside a block', () => {
    const broken = withoutNodes('Broken');
    const inBlock = block([{
      id: 'inside', type: 'preset:Broken', position: { x: 0, y: 0 },
      data: { params: {}, internalParams: { chat: { api_key: TYPED_KEY } } },
    }]);
    store().loadGraphDocument({
      nodes: [inBlock.instance], edges: [], boundFile: null,
      presets: [broken], subgraphs: [inBlock.definition],
    });

    expect(_buildPersistedTabForTesting(tab()).presets).toStrictEqual([broken]);
    const saved = store().getSerializedGraph();
    expect(saved.presets).toStrictEqual([broken]);
    // The slot the entry still exposes as SECRET is blanked.
    expect(serializedNode(saved, 'inside').data.internalParams).toEqual({ chat: { api_key: '' } });
    expect(store().getSerializedGraph({ keepSecrets: true }).presets).toStrictEqual([broken]);
    expect(store().enterSubgraph('block')).toBe(true);
  });

  it('is not filled in, nor is a field of the wrong type replaced', () => {
    // The server refuses both: one has no `nodes`, the other an
    // `exposed_params` that is not a list.
    const noNodes = { preset_name: 'NoNodes', edges: [] } as unknown as PresetDefinition;
    const wrongType = {
      ...oneInner('WrongType', 'PipelineLinear', { in_features: 3 }), exposed_params: null,
    } as unknown as PresetDefinition;
    store().loadGraphDocument({
      nodes: [card('a', 'NoNodes', noNodes), card('b', 'WrongType', wrongType, { inner: { in_features: 3 } })],
      edges: [], boundFile: null, presets: [noNodes, wrongType],
    });

    expect(tab().presets[0]).toBe(noNodes);
    expect(tab().presets[1]).toBe(wrongType);
    expect(store().getSerializedGraph().presets).toStrictEqual([noNodes, wrongType]);
    expect(store().getSerializedGraph({ keepSecrets: true }).presets).toStrictEqual([noNodes, wrongType]);
    expect(_buildPersistedTabForTesting(tab()).presets).toStrictEqual([noNodes, wrongType]);
  });

  it('stays out of the palette, which offers presets to drop anywhere, and the document keeps it', () => {
    const wrongTags = { ...pipeline('WrongTags'), tags: 'vision' } as unknown as PresetDefinition;
    const read = resolveExample({
      nodes: [{
        id: 'card', type: 'preset:Broken', position: { x: 0, y: 0 },
        data: { params: {}, internalParams: {} },
      }],
      edges: [],
      presets: [{ preset_name: 'Broken' }, wrongTags, pipeline('Fine')],
    });

    expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual(['Fine']);
    store().loadGraphDocument({
      nodes: read.nodes, edges: read.edges, boundFile: null, presets: read.presets,
    });
    expect(tab().presets.map((p) => p.preset_name)).toEqual(['Broken', 'WrongTags', 'Fine']);
    expect(store().getSerializedGraph().presets).toStrictEqual([{ preset_name: 'Broken' }]);
  });

  it('a definition with no nodes added from the palette', () => {
    const broken = withoutNodes('Broken');
    useNodeDefStore.setState({ presets: [broken] } as never);
    store().addPresetNode(broken, { x: 0, y: 0 });
    expect(tab().nodes[0].data.presetDefinition).toBe(broken);
    expect(tab().nodes[0].data.internalParams).toEqual({});
    expect(store().getSerializedGraph().presets).toStrictEqual([broken]);
  });

  it('an entry with no name is ignored, as the server ignores it', () => {
    const fine = pipeline('Fine');
    const raw = [null, 42, { category: 'nameless' }, fine] as unknown as PresetDefinition[];
    const read = resolveExample({
      nodes: [{
        id: 'card', type: 'preset:Fine', position: { x: 0, y: 0 },
        data: { params: {}, internalParams: {} },
      }],
      edges: [],
      presets: raw,
    });
    expect(read.nodes[0].data.presetDefinition).toBe(fine);
    expect(useNodeDefStore.getState().presets).toEqual([fine]);
    store().loadGraphDocument({
      nodes: read.nodes, edges: read.edges, boundFile: null, presets: read.presets,
    });

    expect(_buildPersistedTabForTesting(tab()).presets!.map((p) => p.preset_name)).toEqual(['Fine']);
    expect(store().getSerializedGraph().presets.map((p) => p.preset_name)).toEqual(['Fine']);
    select('card');
    store().copySelectedNodes();
    store().addTab('pasted');
    store().pasteNodes();
    expect(tab().nodes[0].data.presetDefinition!.preset_name).toBe('Fine');
  });
});

// -- Run trusts only the definition it sends ----------------------------------

describe('Run keeps a preset key only where the definition it sends places a SECRET type', () => {
  /** A same-name definition whose node at `inner` is `type`, exposing `key`. */
  function variant(type: string, key: string, category: string): PresetDefinition {
    return {
      preset_name: 'Shared', category, description: category, tags: [],
      nodes: [{ id: 'inner', type, params: { api_key: '', installed_key: '' } }],
      edges: [], exposed_inputs: [], exposed_outputs: [],
      exposed_params: [secretSlot('inner', key)],
    };
  }

  it('blanks a slot only the palette copy of the installed preset places', () => {
    // The palette still lists the installed preset, whose type at `inner`
    // declares `installed_key` SECRET. The server may no longer have it -- an
    // uninstall in another window -- and then it consults only the definition
    // the run sends, whose type at `inner` has no such key.
    useNodeDefStore.setState({
      definitions: [keyedType('InstalledKeyed', 'installed_key'), keyedType('DocumentKeyed', 'api_key')],
      presets: [variant('InstalledKeyed', 'installed_key', 'installed')],
    } as never);
    const owned = variant('DocumentKeyed', 'api_key', 'document');
    const values = () => ({ inner: { api_key: TYPED_KEY, installed_key: STALE_KEY } });
    const inBlock = block([{
      id: 'inside', type: 'preset:Shared', position: { x: 0, y: 0 },
      data: { params: {}, internalParams: values() },
    }]);
    store().loadGraphDocument({
      nodes: [card('card', 'Shared', owned, values()), inBlock.instance],
      edges: [], boundFile: null, presets: [owned], subgraphs: [inBlock.definition],
    });

    const run = store().getSerializedGraph({ keepSecrets: true });

    // The premise: the run sends the document's definition.
    expect(run.presets.map((p) => p.category)).toEqual(['document']);
    for (const id of ['card', 'inside']) {
      expect(serializedNode(run, id).data.internalParams, id).toEqual({
        inner: { api_key: TYPED_KEY, installed_key: '' },
      });
    }
    expect(JSON.stringify(run)).not.toContain(STALE_KEY);
  });

  it('blanks an unexposed key in a slot only the installed inner type calls SECRET', () => {
    // The same, with neither definition exposing anything: the keys came in
    // as inner defaults, which a card copies when it is dropped. The type the
    // run's own definition puts at `inner` declares only `api_key` SECRET, so
    // the server blanks that one in the run it stores, and only that one. A
    // name of its own: the session remembers every slot a list exposed.
    const unexposed = (definition: PresetDefinition) => ({
      ...definition, preset_name: 'Unexposed', exposed_params: [],
    });
    useNodeDefStore.setState({
      definitions: [keyedType('InstalledKeyed', 'installed_key'), keyedType('DocumentKeyed', 'api_key')],
      presets: [unexposed(variant('InstalledKeyed', 'installed_key', 'installed'))],
    } as never);
    const owned = unexposed(variant('DocumentKeyed', 'api_key', 'document'));
    const values = () => ({ inner: { api_key: TYPED_KEY, installed_key: STALE_KEY } });
    const inBlock = block([{
      id: 'inside', type: 'preset:Unexposed', position: { x: 0, y: 0 },
      data: { params: {}, internalParams: values() },
    }]);
    store().loadGraphDocument({
      nodes: [card('card', 'Unexposed', owned, values()), inBlock.instance],
      edges: [], boundFile: null, presets: [owned], subgraphs: [inBlock.definition],
    });

    const run = store().getSerializedGraph({ keepSecrets: true });
    for (const id of ['card', 'inside']) {
      expect(serializedNode(run, id).data.internalParams, id).toEqual({
        inner: { api_key: TYPED_KEY, installed_key: '' },
      });
    }
    // Save keeps neither.
    const saved = store().getSerializedGraph();
    for (const id of ['card', 'inside']) {
      expect(serializedNode(saved, id).data.internalParams, id).toEqual({
        inner: { api_key: '', installed_key: '' },
      });
    }
  });
});

// -- A key in an inner param the definition never exposed ---------------------

describe('a key in an inner param the definition does not expose', () => {
  it('stays out of autosave and Save, and Run keeps it where the server blanks it', () => {
    // An older, hand-made definition: the key is a default of a param it never
    // exposed, and dropping it from the palette copies every inner param into
    // the card.
    const legacy = oneInner('Legacy', 'LegacyChat', { model: 'small', api_key: LEGACY_KEY });
    useNodeDefStore.setState({ definitions: [keyedType('LegacyChat')], presets: [legacy] } as never);
    store().addPresetNode(legacy, { x: 0, y: 0 });
    const cardId = tab().nodes[0].id;
    // The premise: the card holds the key.
    expect(tab().nodes[0].data.internalParams).toEqual({ inner: { model: 'small', api_key: LEGACY_KEY } });
    const inBlock = block([{
      id: 'inside', type: 'preset:Legacy', position: { x: 0, y: 0 },
      data: { params: {}, internalParams: { inner: { model: 'small', api_key: LEGACY_KEY } } },
    }]);
    store().setNodes([...tab().nodes, inBlock.instance]);
    store().setSubgraphs([inBlock.definition]);

    const record = _buildPersistedTabForTesting(tab());
    expect(record.nodes[0].data.internalParams).toEqual({ inner: { model: 'small', api_key: '' } });
    expect(JSON.stringify(record)).not.toContain(LEGACY_KEY);
    const saved = store().getSerializedGraph();
    for (const id of [cardId, 'inside']) {
      expect(serializedNode(saved, id).data.internalParams, id).toEqual({
        inner: { model: 'small', api_key: '' },
      });
    }
    expect(JSON.stringify(saved)).not.toContain(LEGACY_KEY);

    // The definition the run sends places `LegacyChat` at `inner`, whose key
    // the server blanks in the run it stores, so the run gets it.
    const run = store().getSerializedGraph({ keepSecrets: true });
    expect(serializedNode(run, cardId).data.internalParams.inner.api_key).toBe(LEGACY_KEY);
  });
});
