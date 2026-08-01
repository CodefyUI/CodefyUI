import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useNodeDefinitionsBootstrap } from './useNodeDefinitionsBootstrap';
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

describe('useNodeDefinitionsBootstrap', () => {
  it('fetches nodes AND presets on mount when the store is empty and idle', async () => {
    renderHook(() => useNodeDefinitionsBootstrap());

    await waitFor(() => {
      expect(useNodeDefStore.getState().definitions.length).toBe(1);
    });

    expect(fetchDefsMock).toHaveBeenCalledTimes(1);
    expect(fetchPresetsMock).toHaveBeenCalledTimes(1);
    expect(useNodeDefStore.getState().categorized).toEqual({ Data: [sampleDef] });
  });

  it('does NOT fetch when definitions are already loaded', () => {
    useNodeDefStore.setState({ definitions: [sampleDef] });

    renderHook(() => useNodeDefinitionsBootstrap());

    expect(fetchDefsMock).not.toHaveBeenCalled();
  });

  it('does NOT fetch when a load is already in progress', () => {
    useNodeDefStore.setState({ loading: true });

    renderHook(() => useNodeDefinitionsBootstrap());

    expect(fetchDefsMock).not.toHaveBeenCalled();
  });

  // The guard reads getState() rather than closure-captured flags, so a second
  // caller overlapping the first in-flight load does not fire a second one.
  it('is idempotent across two overlapping callers', async () => {
    renderHook(() => useNodeDefinitionsBootstrap());
    renderHook(() => useNodeDefinitionsBootstrap());

    await waitFor(() => expect(useNodeDefStore.getState().definitions.length).toBe(1));
    expect(fetchDefsMock).toHaveBeenCalledTimes(1);
  });

  it('leaves the store error set when the load fails', async () => {
    fetchDefsMock.mockRejectedValueOnce(new Error('boom'));

    renderHook(() => useNodeDefinitionsBootstrap());

    await waitFor(() => {
      expect(useNodeDefStore.getState().error).toBe('boom');
    });
    expect(useNodeDefStore.getState().loading).toBe(false);
  });
});
