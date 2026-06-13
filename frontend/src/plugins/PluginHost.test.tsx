import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { loadPluginFrontends } from './PluginHost';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';

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
  useToastStore.setState({ toasts: [] });
});

function mockPluginsResponse(plugins: unknown) {
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

  it('returns [] for non-array payloads', async () => {
    mockPluginsResponse({ not: 'an array' } as unknown as unknown[]);
    const importer = vi.fn();
    expect(await loadPluginFrontends(() => document.createElement('div'), importer)).toEqual([]);
    expect(importer).not.toHaveBeenCalled();
  });

  it('skips malformed array elements without aborting the rest', async () => {
    mockPluginsResponse([
      null,
      'garbage',
      { id: 42, enabled: true, frontend_entry: '/x.js' },
      { id: 'ok', enabled: true, frontend_entry: '/plugins/ok/frontend/index.js' },
    ]);
    const activate = vi.fn();
    const importer = vi.fn(async () => ({ default: activate }));
    expect(await loadPluginFrontends(() => document.createElement('div'), importer)).toEqual(['ok']);
  });

  it('returns [] when fetch fails or response not ok', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false })) as unknown as typeof fetch);
    expect(await loadPluginFrontends(() => document.createElement('div'), vi.fn())).toEqual([]);
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('net'); }) as unknown as typeof fetch);
    expect(await loadPluginFrontends(() => document.createElement('div'), vi.fn())).toEqual([]);
  });

  it('a hanging import times out and later plugins still activate', async () => {
    vi.useFakeTimers();
    try {
      mockPluginsResponse([
        { id: 'hang', enabled: true, frontend_entry: '/plugins/hang/frontend/index.js' },
        { id: 'ok', enabled: true, frontend_entry: '/plugins/ok/frontend/index.js' },
      ]);
      const activate = vi.fn();
      const importer = vi.fn((url: string) =>
        url.includes('hang')
          ? new Promise<never>(() => {})
          : Promise.resolve({ default: activate }),
      );
      const resultP = loadPluginFrontends(() => document.createElement('div'), importer);
      await vi.advanceTimersByTimeAsync(11000);
      expect(await resultP).toEqual(['ok']);
      expect(activate).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
