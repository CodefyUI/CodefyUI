import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import * as rest from '../api/rest';
import type { PackCatalog } from '../api/rest';

vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listPacks: vi.fn(),
    getPackJobEvents: vi.fn(),
    fetchHealth: vi.fn(),
  };
});

import { usePackCatalogBootstrap } from './usePackCatalogBootstrap';
import { _resetPackStoreForTesting, usePackStore } from '../store/packStore';

const api = vi.mocked(rest);

const catalog: PackCatalog = {
  packs: [],
  active_job: null,
  last_restart_job: null,
  remote_install_allowed: true,
  launch_mode: 'start',
  restart_available: false,
  gpu: null,
};

beforeEach(() => {
  _resetPackStoreForTesting();
  sessionStorage.clear();
  api.listPacks.mockResolvedValue(catalog);
});

afterEach(() => {
  _resetPackStoreForTesting();
  vi.clearAllMocks();
});

describe('usePackCatalogBootstrap', () => {
  it('reads the catalog once on mount', async () => {
    renderHook(() => usePackCatalogBootstrap());

    await waitFor(() => expect(usePackStore.getState().loaded).toBe(true));
    expect(api.listPacks).toHaveBeenCalledTimes(1);
  });

  it('reads it once even when two callers mount', async () => {
    // StrictMode mounts every effect twice, and the sidebar shell can be
    // remounted by a layout change; neither may cost a second catalog read.
    renderHook(() => usePackCatalogBootstrap());
    renderHook(() => usePackCatalogBootstrap());

    await waitFor(() => expect(usePackStore.getState().loaded).toBe(true));
    expect(api.listPacks).toHaveBeenCalledTimes(1);
  });

  it('does not re-read a catalog that is already loaded', () => {
    usePackStore.setState({ loaded: true });

    renderHook(() => usePackCatalogBootstrap());

    expect(api.listPacks).not.toHaveBeenCalled();
  });

  it('does not start a second read while one is in flight', () => {
    usePackStore.setState({ loading: true });

    renderHook(() => usePackCatalogBootstrap());

    expect(api.listPacks).not.toHaveBeenCalled();
  });
});
