import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useNodeDefinitions } from './useNodeDefinitions';
import { useNodeDefStore } from '../store/nodeDefStore';

// Mock the REST layer the store calls so the effect never hits the network.
vi.mock('../api/rest', () => ({
  fetchNodeDefinitions: vi.fn(),
  fetchPresetDefinitions: vi.fn(),
  reloadNodes: vi.fn(),
}));

import {
  fetchNodeDefinitions,
  fetchPresetDefinitions,
} from '../api/rest';

const fetchDefsMock = vi.mocked(fetchNodeDefinitions);
const fetchPresetsMock = vi.mocked(fetchPresetDefinitions);

const sampleDef = {
  node_name: 'Dataset',
  category: 'Data',
  description: 'd',
  inputs: [],
  outputs: [],
  params: [],
} as any;

beforeEach(() => {
  // Reset store to a clean, empty state for each test.
  useNodeDefStore.setState({
    definitions: [],
    loading: false,
    error: null,
    categorized: {},
    presets: [],
    presetCategorized: {},
  });
  fetchDefsMock.mockReset();
  fetchPresetsMock.mockReset();
  fetchDefsMock.mockResolvedValue([sampleDef]);
  fetchPresetsMock.mockResolvedValue([]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useNodeDefinitions', () => {
  it('fetches definitions on mount when store is empty and not loading', async () => {
    const { result } = renderHook(() => useNodeDefinitions());

    // Triggers fetchDefinitions which resolves with our mocked data.
    await waitFor(() => {
      expect(useNodeDefStore.getState().definitions.length).toBe(1);
    });

    expect(fetchDefsMock).toHaveBeenCalledTimes(1);
    expect(fetchPresetsMock).toHaveBeenCalledTimes(1);
    expect(result.current.definitions).toEqual([sampleDef]);
    expect(result.current.categorized).toEqual({ Data: [sampleDef] });
    expect(typeof result.current.refetch).toBe('function');
  });

  it('does NOT fetch when definitions are already loaded', () => {
    useNodeDefStore.setState({ definitions: [sampleDef] });

    renderHook(() => useNodeDefinitions());

    expect(fetchDefsMock).not.toHaveBeenCalled();
  });

  it('does NOT fetch when a load is already in progress', () => {
    useNodeDefStore.setState({ loading: true });

    const { result } = renderHook(() => useNodeDefinitions());

    expect(fetchDefsMock).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(true);
  });

  it('exposes refetch bound to the store fetchDefinitions action', async () => {
    const { result } = renderHook(() => useNodeDefinitions());
    await waitFor(() => expect(fetchDefsMock).toHaveBeenCalledTimes(1));

    // Calling refetch invokes the store action again.
    await result.current.refetch();
    expect(fetchDefsMock).toHaveBeenCalledTimes(2);
  });

  it('surfaces error state from the store', async () => {
    fetchDefsMock.mockRejectedValueOnce(new Error('boom'));

    const { result } = renderHook(() => useNodeDefinitions());

    await waitFor(() => {
      expect(result.current.error).toBe('boom');
    });
    expect(result.current.loading).toBe(false);
  });
});
