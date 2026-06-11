import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { loadPluginFrontends } from './PluginHost';
import { useNodeDefStore } from '../store/nodeDefStore';

beforeEach(() => {
  useNodeDefStore.setState({
    definitions: [{
      node_name: 'X', category: 'c', description: '',
      inputs: [], outputs: [], params: [],
    }],
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockPluginsResponse(plugins: unknown[]) {
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: true,
    json: async () => plugins,
  })) as unknown as typeof fetch);
}

describe('loadPluginFrontends', () => {
  it('activates enabled plugins with a frontend entry', async () => {
    mockPluginsResponse([
      { id: 'a', enabled: true, frontend_entry: '/plugins/a/frontend/index.js' },
      { id: 'b', enabled: true, frontend_entry: null },
      { id: 'c', enabled: false, frontend_entry: '/plugins/c/frontend/index.js' },
    ]);
    const activate = vi.fn();
    const importer = vi.fn(async () => ({ default: activate }));
    const loaded = await loadPluginFrontends(
      () => document.createElement('div'), importer,
    );
    expect(importer).toHaveBeenCalledTimes(1);
    expect(importer).toHaveBeenCalledWith('/plugins/a/frontend/index.js');
    expect(activate).toHaveBeenCalledTimes(1);
    expect(activate.mock.calls[0][0].pluginId).toBe('a');
    expect(loaded).toEqual(['a']);
  });

  it('isolates a failing plugin without breaking the rest', async () => {
    mockPluginsResponse([
      { id: 'bad', enabled: true, frontend_entry: '/plugins/bad/frontend/index.js' },
      { id: 'good', enabled: true, frontend_entry: '/plugins/good/frontend/index.js' },
    ]);
    const activate = vi.fn();
    const importer = vi.fn(async (url: string) => {
      if (url.includes('bad')) throw new Error('boom');
      return { default: activate };
    });
    const loaded = await loadPluginFrontends(
      () => document.createElement('div'), importer,
    );
    expect(loaded).toEqual(['good']);
    expect(activate).toHaveBeenCalledTimes(1);
  });

  it('rejects entries whose default export is not a function', async () => {
    mockPluginsResponse([
      { id: 'a', enabled: true, frontend_entry: '/plugins/a/frontend/index.js' },
    ]);
    const importer = vi.fn(async () => ({ default: 42 }));
    const loaded = await loadPluginFrontends(
      () => document.createElement('div'), importer,
    );
    expect(loaded).toEqual([]);
  });
});
