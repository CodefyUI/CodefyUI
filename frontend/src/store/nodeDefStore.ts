import { create } from 'zustand';
import type { NodeDefinition, PresetDefinition } from '../types';
import { fetchNodeDefinitions, fetchPresetDefinitions, reloadNodes } from '../api/rest';

interface NodeDefState {
  definitions: NodeDefinition[];
  loading: boolean;
  error: string | null;
  categorized: Record<string, NodeDefinition[]>;
  presets: PresetDefinition[];
  fetchDefinitions: () => Promise<void>;
  reload: () => Promise<void>;
}

export const useNodeDefStore = create<NodeDefState>((set, get) => ({
  definitions: [],
  loading: false,
  error: null,
  categorized: {},
  presets: [],

  fetchDefinitions: async () => {
    set({ loading: true, error: null });
    try {
      const [defs, presets] = await Promise.all([
        fetchNodeDefinitions(),
        fetchPresetDefinitions(),
      ]);
      const categorized: Record<string, NodeDefinition[]> = {};
      for (const def of defs) {
        if (!categorized[def.category]) categorized[def.category] = [];
        categorized[def.category].push(def);
      }
      // Presets ship flat: the Nodes tab pins them into one group of their own
      // rather than spreading them across the node categories, so there is no
      // second `categorized` map to build.
      set({ definitions: defs, categorized, presets, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  reload: async () => {
    await reloadNodes();
    await get().fetchDefinitions();
  },
}));

// ── Every SECRET this session has seen ──
//
// The serializer blanks a SECRET param (an API key) by what the node's
// definition says. A node inside a block has no definition of its own -- a
// block holds its nodes in their saved shape -- so the strip looks its type up
// in `definitions`, and a preset's SECRET slots in `presets`. Every fetch
// replaces both lists whole, though: a custom node disabled in the Custom
// Nodes manager, or a preset that came in with an opened file, is missing from
// the next list, and the lookup then finds nothing to blank.
//
// So every list the store receives is remembered here: per node type, each
// param a list declared SECRET on it, and per preset name, each exposed slot a
// list declared SECRET. So is the definition a node carries at the moment it
// is folded into a block (`rememberNodeSecrets`): a node autosave brought back
// after a reload keeps the definition it was saved with, and no list this
// session may name its type. The record only grows. A later list without the
// type does not make its key stop being a key. It lives in memory for as long
// as the page does, and nothing writes it anywhere.

/** Per node type: a definition holding every param declared SECRET on it. */
const secretsOfType = new Map<string, NodeDefinition>();
/** Per preset name: a preset holding every exposed slot declared SECRET. */
const secretSlotsOfPreset = new Map<string, PresetDefinition>();

/** Add the SECRET ones among `params` to what `nodeType` is remembered to have. */
function addSecretParams(nodeType: string, params: unknown): void {
  if (!Array.isArray(params)) return;
  const known = secretsOfType.get(nodeType)?.params ?? [];
  const added = (params as NodeDefinition['params']).filter(
    (p) => p?.param_type === 'secret' && !known.some((k) => k.name === p.name),
  );
  if (!added.length) return;
  secretsOfType.set(nodeType, {
    node_name: nodeType, category: '', description: '', inputs: [], outputs: [],
    params: [...known, ...added],
  });
}

/** Add the SECRET ones among `slots` to what the preset `presetName` is remembered to have. */
function addSecretSlots(presetName: string, slots: unknown): void {
  if (!Array.isArray(slots)) return;
  const known = secretSlotsOfPreset.get(presetName)?.exposed_params ?? [];
  const added = (slots as PresetDefinition['exposed_params']).filter(
    (ep) =>
      ep?.param_def?.param_type === 'secret'
      && !known.some((k) => k.internal_node === ep.internal_node && k.param_name === ep.param_name),
  );
  if (!added.length) return;
  secretSlotsOfPreset.set(presetName, {
    preset_name: presetName, category: '', description: '', tags: [],
    nodes: [], edges: [], exposed_inputs: [], exposed_outputs: [],
    exposed_params: [...known, ...added],
  });
}

function rememberDefinitions(definitions: NodeDefinition[]): void {
  for (const definition of Array.isArray(definitions) ? definitions : []) {
    if (typeof definition?.node_name === 'string') {
      addSecretParams(definition.node_name, definition.params);
    }
  }
}

function rememberPresets(presets: PresetDefinition[]): void {
  for (const preset of Array.isArray(presets) ? presets : []) {
    if (typeof preset?.preset_name === 'string') {
      addSecretSlots(preset.preset_name, preset.exposed_params);
    }
  }
}

// Every list, whoever sets it: `fetchDefinitions` above, and the readers of an
// opened file or example, which merge its presets in with `setState`.
useNodeDefStore.subscribe((state, previous) => {
  if (state.definitions !== previous.definitions) rememberDefinitions(state.definitions);
  if (state.presets !== previous.presets) rememberPresets(state.presets);
});

/**
 * Remember what a node's own definition declares SECRET, under the type the
 * node is saved as. The folds that put a node into a block call this before
 * they drop the definition (`rememberFoldedSecrets` in the tab store).
 */
export function rememberNodeSecrets(nodeType: string, definition: NodeDefinition | undefined): void {
  addSecretParams(nodeType, definition?.params);
}

/**
 * Remember the SECRET slots a preset node's own preset definition exposes,
 * under the preset name its type names. Called at the same folds.
 */
export function rememberPresetNodeSecrets(
  presetName: string,
  preset: PresetDefinition | undefined,
): void {
  addSecretSlots(presetName, preset?.exposed_params);
}

/**
 * The params declared SECRET on `nodeType` this session, by a node list or by
 * the definition a node of that type carried into a block, as a definition
 * that holds only them. Undefined when neither has.
 */
export function rememberedSecrets(nodeType: string): NodeDefinition | undefined {
  return secretsOfType.get(nodeType);
}

/**
 * The exposed slots declared SECRET on the preset `presetName` this session,
 * by a preset list or by the preset definition a preset node carried into a
 * block, as a preset that holds only them. Undefined when neither has.
 */
export function rememberedPresetSecrets(presetName: string): PresetDefinition | undefined {
  return secretSlotsOfPreset.get(presetName);
}
