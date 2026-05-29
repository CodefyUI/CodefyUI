import { create } from 'zustand';
import type { NodeDefinition, PresetDefinition, ChapterPack } from '../types';
import {
  fetchNodeDefinitions,
  fetchPresetDefinitions,
  fetchChapterPacks,
  reloadNodes,
} from '../api/rest';

interface NodeDefState {
  definitions: NodeDefinition[];
  loading: boolean;
  error: string | null;
  categorized: Record<string, NodeDefinition[]>;
  presets: PresetDefinition[];
  presetCategorized: Record<string, PresetDefinition[]>;
  chapterPacks: ChapterPack[];
  fetchDefinitions: () => Promise<void>;
  reload: () => Promise<void>;
}

export const useNodeDefStore = create<NodeDefState>((set, get) => ({
  definitions: [],
  loading: false,
  error: null,
  categorized: {},
  presets: [],
  presetCategorized: {},
  chapterPacks: [],

  fetchDefinitions: async () => {
    set({ loading: true, error: null });
    try {
      const [defs, presets, chapterPacks] = await Promise.all([
        fetchNodeDefinitions(),
        fetchPresetDefinitions(),
        // Chapter packs are an optional virtual grouping — never block the
        // palette on them. If the endpoint 404s on an older backend, fall
        // back to empty so the rest of the UI keeps working.
        fetchChapterPacks().catch(() => [] as ChapterPack[]),
      ]);
      const categorized: Record<string, NodeDefinition[]> = {};
      for (const def of defs) {
        if (!categorized[def.category]) categorized[def.category] = [];
        categorized[def.category].push(def);
      }
      const presetCategorized: Record<string, PresetDefinition[]> = {};
      for (const p of presets) {
        if (!presetCategorized[p.category]) presetCategorized[p.category] = [];
        presetCategorized[p.category].push(p);
      }
      set({
        definitions: defs,
        categorized,
        presets,
        presetCategorized,
        chapterPacks,
        loading: false,
      });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  reload: async () => {
    await reloadNodes();
    await get().fetchDefinitions();
  },
}));
