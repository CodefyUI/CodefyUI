import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock the REST seam the store depends on. Each test configures the mock
// implementations, so we can exercise both the success and catch branches.
vi.mock('../api/rest', () => ({
  fetchNodeDefinitions: vi.fn(),
  fetchPresetDefinitions: vi.fn(),
  reloadNodes: vi.fn(),
}));

import { useNodeDefStore } from './nodeDefStore';
import {
  fetchNodeDefinitions,
  fetchPresetDefinitions,
  reloadNodes,
} from '../api/rest';

const mockFetchDefs = vi.mocked(fetchNodeDefinitions);
const mockFetchPresets = vi.mocked(fetchPresetDefinitions);
const mockReload = vi.mocked(reloadNodes);

// Minimal shapes — only the fields the store reads (category) matter, plus the
// identifying name field so assertions can verify ordering within a category.
const def = (name: string, category: string) =>
  ({ node_name: name, category }) as unknown as Awaited<ReturnType<typeof fetchNodeDefinitions>>[number];
const preset = (name: string, category: string) =>
  ({ preset_name: name, category }) as unknown as Awaited<ReturnType<typeof fetchPresetDefinitions>>[number];

describe('useNodeDefStore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useNodeDefStore.setState({
      definitions: [],
      loading: false,
      error: null,
      categorized: {},
      presets: [],
      presetCategorized: {},
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('fetchDefinitions — success', () => {
    it('categorizes definitions and presets, grouping shared categories', async () => {
      // Two defs in "io" exercise both the `!categorized[cat]` true and false
      // branches; one in "train". Same idea for presets.
      mockFetchDefs.mockResolvedValue([
        def('Dataset', 'io'),
        def('DataLoader', 'io'),
        def('Trainer', 'train'),
      ]);
      mockFetchPresets.mockResolvedValue([
        preset('p1', 'starters'),
        preset('p2', 'starters'),
        preset('p3', 'advanced'),
      ]);

      await useNodeDefStore.getState().fetchDefinitions();

      const state = useNodeDefStore.getState();
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
      expect(state.definitions).toHaveLength(3);
      expect(state.categorized.io.map((d) => d.node_name)).toEqual(['Dataset', 'DataLoader']);
      expect(state.categorized.train.map((d) => d.node_name)).toEqual(['Trainer']);
      expect(state.presets).toHaveLength(3);
      expect(state.presetCategorized.starters.map((p) => p.preset_name)).toEqual(['p1', 'p2']);
      expect(state.presetCategorized.advanced.map((p) => p.preset_name)).toEqual(['p3']);
    });

    it('sets loading=true while the requests are in flight', async () => {
      let resolveDefs!: (v: never[]) => void;
      mockFetchDefs.mockReturnValue(
        new Promise((res) => {
          resolveDefs = res as (v: never[]) => void;
        }),
      );
      mockFetchPresets.mockResolvedValue([]);

      const p = useNodeDefStore.getState().fetchDefinitions();
      expect(useNodeDefStore.getState().loading).toBe(true);
      expect(useNodeDefStore.getState().error).toBeNull();

      resolveDefs([]);
      await p;
      expect(useNodeDefStore.getState().loading).toBe(false);
    });

    it('produces empty maps when both lists are empty', async () => {
      mockFetchDefs.mockResolvedValue([]);
      mockFetchPresets.mockResolvedValue([]);
      await useNodeDefStore.getState().fetchDefinitions();
      const state = useNodeDefStore.getState();
      expect(state.categorized).toEqual({});
      expect(state.presetCategorized).toEqual({});
      expect(state.loading).toBe(false);
    });
  });

  describe('fetchDefinitions — error', () => {
    it('captures the error message and clears loading on rejection', async () => {
      mockFetchDefs.mockRejectedValue(new Error('network down'));
      mockFetchPresets.mockResolvedValue([]);

      await useNodeDefStore.getState().fetchDefinitions();

      const state = useNodeDefStore.getState();
      expect(state.error).toBe('network down');
      expect(state.loading).toBe(false);
      // Prior data is left untouched (defaults here).
      expect(state.definitions).toEqual([]);
    });
  });

  describe('reload', () => {
    it('calls reloadNodes then re-fetches definitions', async () => {
      mockReload.mockResolvedValue(undefined as never);
      mockFetchDefs.mockResolvedValue([def('Dataset', 'io')]);
      mockFetchPresets.mockResolvedValue([]);

      await useNodeDefStore.getState().reload();

      expect(mockReload).toHaveBeenCalledTimes(1);
      expect(mockFetchDefs).toHaveBeenCalledTimes(1);
      expect(mockFetchPresets).toHaveBeenCalledTimes(1);
      const state = useNodeDefStore.getState();
      expect(state.definitions).toHaveLength(1);
      expect(state.error).toBeNull();
    });
  });
});
