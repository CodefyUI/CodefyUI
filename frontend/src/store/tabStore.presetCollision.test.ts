/**
 * #541 fix round: a preset card pasted or inserted into a tab whose document
 * owns a same-name definition.
 *
 * #541 attaches such a card to the document's definition. These tests pin
 * what has to follow the attachment:
 *
 *  - D1: the SECRET scrub of a TOP-LEVEL preset node takes the conservative
 *    union a node inside a block gets -- the attached definition, the
 *    document's own, the installed one and the slots remembered this session
 *    (Run uses the remembered slots only when no definition exists). A value
 *    typed against the installed definition must not become durable once the
 *    document's definition, which does not declare that slot, is attached
 *    instead -- nor later, once the installed preset leaves the list.
 *  - D2: the card draws the definition it is attached to -- description,
 *    category and ports -- exactly as the palette draws that definition.
 *
 * Every value below is an obviously fake test string.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import {
  useTabStore,
  _buildPersistedTabForTesting,
  _persistedTabsForTesting,
  _tabFromPersistedForTesting,
} from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import { resolveSerializedNodes } from '../utils';
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

const DOCUMENT_VALUE = 'sk-FAKE-541-DOCUMENT-SLOT';
const INSTALLED_VALUE = 'sk-FAKE-541-INSTALLED-ONLY-SLOT';

function secretParam(name: string): ParamDefinition {
  return {
    name, param_type: 'secret', default: '', description: '',
    options: [], min_value: null, max_value: null,
  };
}

/**
 * A registry node type that declares both keys SECRET, as the inner types of
 * the acceptance fixture do.
 */
function keyedType(name: string): NodeDefinition {
  return {
    node_name: name, category: 'E2E', description: '',
    inputs: [{ name: 'in', data_type: 'ANY', description: '', optional: true }],
    outputs: [{ name: 'value', data_type: 'ANY', description: '', optional: false }],
    params: [
      {
        name: 'marker', param_type: 'string', default: '', description: '',
        options: [], min_value: null, max_value: null,
      },
      secretParam('api_key'),
      secretParam('installed_only_secret'),
    ],
  };
}

/** A plain node that feeds an IMAGE into whatever it is wired to. */
const FEEDER: NodeDefinition = {
  node_name: 'Feeder', category: 'Data', description: '', inputs: [],
  outputs: [{ name: 'out', data_type: 'IMAGE', description: '', optional: false }],
  params: [],
};

function slot(param: string): PresetDefinition['exposed_params'][number] {
  return {
    internal_node: 'inner', param_name: param, display_name: param, group: 'Inner',
    param_def: secretParam(param),
  };
}

/** The document's own definition: it exposes `api_key` and nothing else. */
function ownedDefinition(name = 'Collision'): PresetDefinition {
  return {
    preset_name: name,
    category: 'Owned',
    description: 'owned-description',
    tags: [],
    nodes: [{
      id: 'inner', type: 'PortableValue',
      params: { marker: 'owned', api_key: '', installed_only_secret: '' },
    }],
    edges: [],
    exposed_inputs: [{
      name: 'owned_in', internal_node: 'inner', internal_port: 'in',
      data_type: 'TENSOR', description: 'owned input',
    }],
    exposed_outputs: [{
      name: 'owned_out', internal_node: 'inner', internal_port: 'value',
      data_type: 'ANY', description: 'owned output',
    }],
    exposed_params: [slot('api_key')],
  };
}

/** The installed same-name definition: a disjoint SECRET slot and other ports. */
function installedDefinition(name = 'Collision'): PresetDefinition {
  return {
    preset_name: name,
    category: 'Installed',
    description: 'installed-description',
    tags: [],
    nodes: [{
      id: 'inner', type: 'InstalledValue',
      params: { api_key: '', installed_only_secret: '' },
    }],
    edges: [],
    exposed_inputs: [{
      name: 'installed_in', internal_node: 'inner', internal_port: 'in',
      data_type: 'IMAGE', description: 'installed input',
    }],
    exposed_outputs: [{
      name: 'installed_out', internal_node: 'inner', internal_port: 'value',
      data_type: 'ANY', description: 'installed output',
    }],
    exposed_params: [slot('installed_only_secret')],
  };
}

/**
 * The card `addPresetNode` draws for `preset`, spelled out here rather than
 * borrowed from the code under test.
 */
function cardOf(preset: PresetDefinition): NodeDefinition {
  return {
    node_name: preset.preset_name,
    category: preset.category,
    description: preset.description,
    inputs: preset.exposed_inputs.map((p) => ({
      name: p.name, data_type: p.data_type, description: p.description, optional: false,
    })),
    outputs: preset.exposed_outputs.map((p) => ({
      name: p.name, data_type: p.data_type, description: p.description, optional: false,
    })),
    params: [],
  };
}

/** Both slots typed, as a user would type them. */
function typedValues(): Record<string, Record<string, any>> {
  return {
    inner: { marker: 'owned', api_key: DOCUMENT_VALUE, installed_only_secret: INSTALLED_VALUE },
  };
}

/** A top-level preset card of `preset:<name>`, attached to `attached`. */
function presetCard(
  id: string,
  name: string,
  attached: PresetDefinition | undefined,
  internalParams: Record<string, Record<string, any>>,
): Node<NodeData> {
  return {
    id,
    type: 'presetNode',
    position: { x: 0, y: 0 },
    data: {
      label: name,
      type: `preset:${name}`,
      params: {},
      ...(attached ? { definition: cardOf(attached) } : {}),
      isPreset: true,
      presetDefinition: attached,
      internalParams,
      executionStatus: 'idle',
    },
  };
}

/** A block whose one inner node is a `preset:<name>` holding `internalParams`. */
function blockHolding(
  name: string,
  internalParams: Record<string, Record<string, any>>,
): { instance: Node<NodeData>; definition: SubgraphDefinition } {
  return {
    instance: {
      id: 'block', type: 'subgraphNode', position: { x: 300, y: 0 },
      data: { label: 'Block', type: 'subgraph:blk', params: {} },
    },
    definition: {
      id: 'blk', name: 'Block', description: '', edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
      nodes: [{
        id: 'inside', type: `preset:${name}`, position: { x: 0, y: 0 },
        data: { params: {}, internalParams },
      }],
    },
  };
}

function select(...ids: string[]) {
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: ids.includes(n.id) })));
}

/** The serialized node `id`, at the top level or inside the one block. */
function serializedInternalParams(
  graph: { nodes: any[]; subgraphs: { nodes: any[] }[] },
  id: string,
): Record<string, Record<string, any>> {
  const top = graph.nodes.find((n) => n.id === id);
  if (top) return top.data.internalParams;
  return graph.subgraphs[0].nodes.find((n) => n.id === id).data.internalParams;
}

const BOTH_BLANK = { inner: { marker: 'owned', api_key: '', installed_only_secret: '' } };

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  // Installed through the store's state, the way a /api/presets fetch lands.
  useNodeDefStore.setState({
    definitions: [keyedType('PortableValue'), keyedType('InstalledValue'), FEEDER],
    presets: [installedDefinition()],
    categorized: {},
    loading: false,
    error: null,
  } as never);
});

// -- D1 ---------------------------------------------------------------------

describe('D1: a top-level preset node is scrubbed by the union a block node gets', () => {
  /** The tab of the reproduction: the document owns `owned`; one card, both slots typed. */
  function loadOwningDocumentWithTypedCard(): PresetDefinition {
    const owned = ownedDefinition();
    store().loadGraphDocument({
      nodes: [presetCard('top', 'Collision', owned, typedValues())],
      edges: [], boundFile: null, presets: [owned],
    });
    // The premise: the card runs the document's definition, which does not
    // declare the installed slot, and both values are really there.
    expect(tab().nodes[0].data.presetDefinition).toBe(owned);
    expect(tab().nodes[0].data.internalParams).toEqual(typedValues());
    return owned;
  }

  it('getSerializedGraphOf (Save, JSON and workspace export, plugin snapshots) blanks both slots', () => {
    loadOwningDocumentWithTypedCard();

    const saved = store().getSerializedGraphOf(tab());

    expect(serializedInternalParams(saved, 'top')).toEqual(BOTH_BLANK);
    expect(JSON.stringify(saved)).not.toContain(INSTALLED_VALUE);
    expect(JSON.stringify(saved)).not.toContain(DOCUMENT_VALUE);
  });

  it('the record autosave writes (IndexedDB and localStorage) blanks both slots', () => {
    loadOwningDocumentWithTypedCard();

    const record = _buildPersistedTabForTesting(tab());
    expect(record.nodes[0].data.internalParams).toEqual(BOTH_BLANK);
    // What `saveTabs` hands to IndexedDB, or to localStorage without it.
    const written = JSON.stringify(_persistedTabsForTesting(useTabStore.getState().tabs));
    expect(written).not.toContain(INSTALLED_VALUE);
    expect(written).not.toContain(DOCUMENT_VALUE);
  });

  // Run (`keepSecrets`) keeps a value only where the server blanks it in the
  // run it stores. The rule is the block path's, so the same values typed
  // into the same preset at the top level and inside a block come out of
  // both serializers identically -- whichever definitions are known.
  describe('keeps and blanks exactly what the block path does', () => {
    function serializeBothPlaces(name: string, attached: PresetDefinition | undefined, owned: PresetDefinition[]) {
      const values = typedValues();
      const block = blockHolding(name, typedValues());
      store().loadGraphDocument({
        nodes: [presetCard('top', name, attached, values), block.instance],
        edges: [], boundFile: null, presets: owned, subgraphs: [block.definition],
      });
      const run = store().getSerializedGraphOf(tab(), { keepSecrets: true });
      const saved = store().getSerializedGraphOf(tab());
      return {
        run: { top: serializedInternalParams(run, 'top'), inside: serializedInternalParams(run, 'inside') },
        saved: { top: serializedInternalParams(saved, 'top'), inside: serializedInternalParams(saved, 'inside') },
      };
    }

    it('when the server knows both inner types, Run keeps both values and Save blanks both', () => {
      const owned = ownedDefinition();
      const out = serializeBothPlaces('Collision', owned, [owned]);

      expect(out.run.top).toEqual(out.run.inside);
      expect(out.saved.top).toEqual(out.saved.inside);
      expect(out.run.top).toEqual(typedValues());
      expect(out.saved.top).toEqual(BOTH_BLANK);
    });

    it('when the server knows neither inner type, Run blanks the slot of the definition it runs', () => {
      useNodeDefStore.setState({ definitions: [FEEDER] } as never);
      const owned = ownedDefinition();
      const out = serializeBothPlaces('Collision', owned, [owned]);

      expect(out.run.top).toEqual(out.run.inside);
      expect(out.saved.top).toEqual(out.saved.inside);
      expect(out.run.top).toEqual({
        inner: { marker: 'owned', api_key: '', installed_only_secret: INSTALLED_VALUE },
      });
      expect(out.saved.top).toEqual(BOTH_BLANK);
    });

    it('when no list has the preset any more, both fall back to the slots remembered this session', () => {
      // Named only in this case: the record of remembered slots only grows.
      useNodeDefStore.setState({
        presets: [ownedDefinition('Forgotten'), installedDefinition('Forgotten')],
      } as never);
      useNodeDefStore.setState({ presets: [] } as never);
      const out = serializeBothPlaces('Forgotten', undefined, []);

      expect(out.run.top).toEqual(out.run.inside);
      expect(out.saved.top).toEqual(out.saved.inside);
      expect(out.run.top).toEqual(BOTH_BLANK);
      expect(out.saved.top).toEqual(BOTH_BLANK);
    });
  });

  it('a palette-built installed preset copied from another tab and pasted at the top level of an owning tab', () => {
    // Source tab: dragged from the palette (the palette hands over the
    // installed definition), configured, a value typed into its slot.
    store().addPresetNode(installedDefinition(), { x: 0, y: 0 });
    const source = tab().nodes[0];
    store().updatePresetInternalParam(source.id, 'inner', 'installed_only_secret', INSTALLED_VALUE);
    select(source.id);
    store().copySelectedNodes();

    // Owning tab: a document that owns the api_key-only definition.
    store().addTab('owner');
    store().loadGraphDocument({
      nodes: [], edges: [], boundFile: null, presets: [ownedDefinition()],
    });
    store().pasteNodes();
    const pasted = tab().nodes[0];
    // The premise: the paste attached the document's definition, and the
    // value typed against the installed one travelled with the card.
    expect(pasted.data.presetDefinition).toBe(tab().presets[0]);
    expect(pasted.data.internalParams!.inner.installed_only_secret).toBe(INSTALLED_VALUE);
    // The owned definition's Configure field, typed after the paste.
    store().updatePresetInternalParam(pasted.id, 'inner', 'api_key', DOCUMENT_VALUE);

    const saved = store().getSerializedGraph();
    expect(serializedInternalParams(saved, pasted.id).inner).toMatchObject({
      api_key: '', installed_only_secret: '',
    });
    expect(JSON.stringify(saved)).not.toContain(INSTALLED_VALUE);
    expect(JSON.stringify(saved)).not.toContain(DOCUMENT_VALUE);

    const record = _buildPersistedTabForTesting(tab());
    expect(record.nodes[0].data.internalParams!.inner).toMatchObject({
      api_key: '', installed_only_secret: '',
    });
    const written = JSON.stringify(_persistedTabsForTesting(useTabStore.getState().tabs));
    expect(written).not.toContain(INSTALLED_VALUE);
    expect(written).not.toContain(DOCUMENT_VALUE);
  });

  // A plugin disabled or a preset deleted: the next list no longer names the
  // installed definition the pasted value was typed against, and the document's
  // own definition never declared that slot. The slots this session remembers
  // for the name still blank it.
  describe('after a collision paste, the installed preset leaves the list', () => {
    /**
     * A palette-built card in a source tab -- the list `beforeEach` installed
     * with `setState` hands it the installed definition -- with a value typed
     * into the slot only that definition declares, copied.
     */
    function copyTypedInstalledCard() {
      store().addPresetNode(installedDefinition(), { x: 0, y: 0 });
      const source = tab().nodes[0];
      store().updatePresetInternalParam(source.id, 'inner', 'installed_only_secret', INSTALLED_VALUE);
      select(source.id);
      store().copySelectedNodes();
    }

    /** The list set again, without that preset. */
    const dropInstalledPreset = () => useNodeDefStore.setState({ presets: [] } as never);

    it('a top-level card: Save and the autosave record still blank the installed-only slot', () => {
      copyTypedInstalledCard();
      store().addTab('owner');
      store().loadGraphDocument({
        nodes: [], edges: [], boundFile: null, presets: [ownedDefinition()],
      });
      store().pasteNodes();
      const pasted = tab().nodes[0];
      dropInstalledPreset();
      // The premise: the card still holds the value, and neither the tab nor
      // the list has a definition declaring its slot.
      expect(pasted.data.internalParams!.inner.installed_only_secret).toBe(INSTALLED_VALUE);
      expect(tab().presets.map((p) => p.exposed_params.map((ep) => ep.param_name)))
        .toEqual([['api_key']]);

      const saved = store().getSerializedGraph();
      expect(serializedInternalParams(saved, pasted.id).inner.installed_only_secret).toBe('');
      expect(JSON.stringify(saved)).not.toContain(INSTALLED_VALUE);
      // What runs is still the document's definition: the remembered slots
      // only blank.
      expect(saved.presets.map((p) => p.description)).toEqual(['owned-description']);
      expect(saved.presets[0].exposed_params.map((ep) => ep.param_name)).toEqual(['api_key']);

      const record = _buildPersistedTabForTesting(tab());
      expect(record.nodes[0].data.internalParams!.inner.installed_only_secret).toBe('');
      const written = JSON.stringify(_persistedTabsForTesting(useTabStore.getState().tabs));
      expect(written).not.toContain(INSTALLED_VALUE);
      // Nor does the card draw from them after a reload.
      expect(_tabFromPersistedForTesting(record, tab()).nodes[0].data.definition)
        .toEqual(cardOf(ownedDefinition()));
    });

    it('a card pasted into a block: Save and the autosave record still blank the installed-only slot', () => {
      copyTypedInstalledCard();
      store().addTab('owner');
      const block = blockHolding('Collision', { inner: { marker: 'owned', api_key: '' } });
      store().loadGraphDocument({
        nodes: [block.instance], edges: [], boundFile: null,
        presets: [ownedDefinition()], subgraphs: [block.definition],
      });
      expect(store().enterSubgraph('block')).toBe(true);
      store().pasteNodes();
      const pastedId = tab().nodes.find((n) => n.data.isPreset && n.selected)!.id;
      store().exitSubgraph();
      dropInstalledPreset();
      // The premise: the block definition now holds the value.
      expect(JSON.stringify(tab().subgraphs)).toContain(INSTALLED_VALUE);

      const saved = store().getSerializedGraph();
      expect(serializedInternalParams(saved, pastedId).inner.installed_only_secret).toBe('');
      expect(JSON.stringify(saved)).not.toContain(INSTALLED_VALUE);

      const record = _buildPersistedTabForTesting(tab());
      expect(serializedInternalParams(
        { nodes: record.nodes, subgraphs: record.subgraphs ?? [] },
        pastedId,
      ).inner.installed_only_secret).toBe('');
      const written = JSON.stringify(_persistedTabsForTesting(useTabStore.getState().tabs));
      expect(written).not.toContain(INSTALLED_VALUE);
    });
  });
});

// -- D2 ---------------------------------------------------------------------

describe('D2: a preset card draws the definition it is attached to', () => {
  /**
   * A source tab holding a palette-built installed card fed by a Feeder wired
   * into `installed_in`, a port only the installed definition has. Both are
   * copied. Returns the internalParams the source card holds.
   */
  function copyInstalledCardWithWire(): Record<string, Record<string, any>> {
    store().addNode(FEEDER, { x: -200, y: 0 });
    store().addPresetNode(installedDefinition(), { x: 0, y: 0 });
    const [feeder, card] = tab().nodes;
    store().setEdges([{
      id: 'wire', source: feeder.id, target: card.id,
      sourceHandle: 'out', targetHandle: 'installed_in',
    }]);
    // The premise: the source card draws the installed definition.
    expect(card.data.definition).toEqual(cardOf(installedDefinition()));
    select(feeder.id, card.id);
    store().copySelectedNodes();
    return card.data.internalParams!;
  }

  const pastedCard = () => tab().nodes.find((n) => n.data.isPreset && n.selected)!;

  it('paste at the top level: the card, its autosave record and a reload draw the owned definition', () => {
    const copiedInternalParams = copyInstalledCardWithWire();
    store().addTab('owner');
    const owned = ownedDefinition();
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, presets: [owned] });
    store().pasteNodes();

    const card = pastedCard();
    expect(card.data.presetDefinition).toBe(owned);
    expect(card.data.definition).toEqual(cardOf(owned));
    expect(card.data.label).toBe(owned.preset_name);
    // Node-level overrides are kept as they are.
    expect(card.data.internalParams).toEqual(copiedInternalParams);
    // The card the palette draws in this tab is the same one.
    store().addPresetNode(installedDefinition(), { x: 0, y: 300 });
    const fromPalette = tab().nodes[tab().nodes.length - 1];
    expect(card.data.definition).toEqual(fromPalette.data.definition);
    expect(card.data.label).toBe(fromPalette.data.label);

    // A wire on a handle the owned definition lacks is kept exactly as it was
    // pasted, which is what the subgraph re-render does with a pasted
    // instance's wires: nothing prunes it on the way in.
    const wire = tab().edges.find((e: Edge) => e.target === card.id)!;
    expect(wire.targetHandle).toBe('installed_in');

    const record = _buildPersistedTabForTesting(tab());
    const persisted = record.nodes.find((n) => n.id === card.id)!;
    expect(persisted.data.definition).toEqual(cardOf(owned));
    expect(JSON.stringify(record)).not.toContain('installed-description');

    const restored = _tabFromPersistedForTesting(record, tab());
    expect(restored.nodes.find((n) => n.id === card.id)!.data.definition)
      .toEqual(cardOf(owned));
  });

  it('paste into an open block of the owning document draws the owned definition', () => {
    copyInstalledCardWithWire();
    store().addTab('owner');
    const owned = ownedDefinition();
    const block = blockHolding('Collision', { inner: { marker: 'owned', api_key: '' } });
    store().loadGraphDocument({
      nodes: [block.instance], edges: [], boundFile: null,
      presets: [owned], subgraphs: [block.definition],
    });
    expect(store().enterSubgraph('block')).toBe(true);
    store().pasteNodes();

    const card = pastedCard();
    expect(card.data.presetDefinition).toBe(owned);
    expect(card.data.definition).toEqual(cardOf(owned));
    expect(card.data.label).toBe(owned.preset_name);
  });

  it('insert draws the owned definition', () => {
    const owned = ownedDefinition();
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, presets: [owned] });
    const installed = installedDefinition();
    const incoming = presetCard('incoming', 'Collision', installed, { inner: { api_key: '' } });

    store().insertGraph([incoming], [], [], undefined, [installed]);

    const card = tab().nodes[0];
    expect(card.data.presetDefinition).toBe(owned);
    expect(card.data.definition).toEqual(cardOf(owned));
    expect(card.data.label).toBe(owned.preset_name);
    expect(card.data.internalParams).toEqual({ inner: { api_key: '' } });
    expect(_buildPersistedTabForTesting(tab()).nodes[0].data.definition).toEqual(cardOf(owned));
  });

  it('restore redraws a record written with the installed rendering from the definition it attaches', () => {
    const owned = ownedDefinition();
    store().loadGraphDocument({
      nodes: [presetCard('p', 'Collision', owned, { inner: { api_key: '' } })],
      edges: [], boundFile: null, presets: [owned],
    });
    const record = _buildPersistedTabForTesting(tab());
    // What a top-level pasted card's record held before this fix: the
    // document's definition in `presets`, the installed rendering on the node.
    record.nodes[0] = {
      ...record.nodes[0],
      data: { ...record.nodes[0].data, definition: cardOf(installedDefinition()) },
    };

    const restored = _tabFromPersistedForTesting(record, tab());

    expect(restored.nodes[0].data.presetDefinition).toBe(record.presets![0]);
    expect(restored.nodes[0].data.definition).toEqual(cardOf(owned));
    expect(restored.nodes[0].data.label).toBe(owned.preset_name);
  });

  it('a saved graph read back draws the same card as the palette', () => {
    const owned = ownedDefinition();
    const [read] = resolveSerializedNodes(
      [{
        id: 'p', type: 'preset:Collision', position: { x: 0, y: 0 },
        data: { params: {}, internalParams: {} },
      }],
      useNodeDefStore.getState().definitions,
      [owned],
    );
    expect(read.data.definition).toEqual(cardOf(owned));

    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, presets: [owned] });
    store().addPresetNode(installedDefinition(), { x: 0, y: 0 });
    expect(read.data.definition).toEqual(tab().nodes[0].data.definition);
  });

  it('a label the user gave the card stays its own', () => {
    // Every same-name definition renders the same label -- the preset's name
    // -- so the only label the re-render could change is a rename.
    store().addPresetNode(installedDefinition(), { x: 0, y: 0 });
    const source = tab().nodes[0];
    store().renameNode(source.id, 'My key holder');
    select(source.id);
    store().copySelectedNodes();
    store().addTab('owner');
    const owned = ownedDefinition();
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, presets: [owned] });
    store().pasteNodes();

    const card = pastedCard();
    expect(card.data.label).toBe('My key holder');
    expect(card.data.definition).toEqual(cardOf(owned));
  });
});
