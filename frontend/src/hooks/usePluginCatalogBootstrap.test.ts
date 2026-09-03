import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import * as rest from '../api/rest';
import type { PluginCatalog } from '../api/rest';
import { ApiError } from '../api/rest';

vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listPluginCatalog: vi.fn(),
    getPluginJobEvents: vi.fn(),
  };
});

// The store calls it after a settled change; nothing here settles one, but a
// real import of the host would pull the whole widget stack into this file.
vi.mock('../plugins/PluginHost', () => ({
  reloadPluginFrontends: vi.fn(),
}));

import { usePluginCatalogBootstrap } from './usePluginCatalogBootstrap';
import { _resetPluginStoreForTesting, usePluginStore } from '../store/pluginStore';
import { useToastStore } from '../store/toastStore';

const api = vi.mocked(rest);

const catalog: PluginCatalog = {
  entries: [],
  active_job: null,
  remote_install_allowed: true,
  generation: 1,
};

beforeEach(() => {
  _resetPluginStoreForTesting();
  useToastStore.setState({ toasts: [] });
  api.listPluginCatalog.mockResolvedValue(catalog);
});

afterEach(() => {
  _resetPluginStoreForTesting();
  vi.clearAllMocks();
});

describe('usePluginCatalogBootstrap', () => {
  it('reads the catalog once on mount', async () => {
    renderHook(() => usePluginCatalogBootstrap());

    await waitFor(() => expect(usePluginStore.getState().loaded).toBe(true));
    expect(api.listPluginCatalog).toHaveBeenCalledTimes(1);
  });

  it('reads it once even when two callers mount', async () => {
    // StrictMode mounts every effect twice, and the sidebar shell can be
    // remounted by a layout change; neither may cost a second catalog read.
    renderHook(() => usePluginCatalogBootstrap());
    renderHook(() => usePluginCatalogBootstrap());

    await waitFor(() => expect(usePluginStore.getState().loaded).toBe(true));
    expect(api.listPluginCatalog).toHaveBeenCalledTimes(1);
  });

  it('does not re-read a catalog that is already loaded', () => {
    usePluginStore.setState({ loaded: true });

    renderHook(() => usePluginCatalogBootstrap());

    expect(api.listPluginCatalog).not.toHaveBeenCalled();
  });

  it('does not start a second read while one is in flight', () => {
    usePluginStore.setState({ loading: true });

    renderHook(() => usePluginCatalogBootstrap());

    expect(api.listPluginCatalog).not.toHaveBeenCalled();
  });

  it('says nothing at all on a server without the route', async () => {
    // The whole point of booting this at all costs on an old server: a 404 is
    // an answer, not a failure, and the user never hears about it.
    api.listPluginCatalog.mockRejectedValue(new ApiError(404, 'Not Found'));

    renderHook(() => usePluginCatalogBootstrap());

    await waitFor(() => expect(usePluginStore.getState().unsupported).toBe(true));
    expect(useToastStore.getState().toasts).toEqual([]);
    expect(usePluginStore.getState().error).toBeNull();
  });
});
