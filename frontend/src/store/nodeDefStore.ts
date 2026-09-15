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
