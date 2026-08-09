import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { loadPluginFrontends, unloadPluginFrontends } from './PluginHost';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import { useTabStore } from '../store/tabStore';
import type { CodefyUIPluginAPI } from './api';
import {
  _clearPluginPanels, getPluginPanels, registerPluginPanel,
  subscribePluginPanels,
} from './panels';
import {
  _clearPluginToolbarButtons, getPluginToolbarButtons,
  registerPluginToolbarButton,
} from './toolbarButtons';
import {
  _resetExecutionEvents, executionEventSubscriberCount, executionEventTapCount,
} from './executionEvents';
import { _clearNodeRenderers, getNodeRenderer } from './nodeRenderers';

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

/**
 * The unload contract (#132): activating a plugin and tearing it down again
 * must leave the editor exactly as it found it.
 *
 * This is the mount/unmount cycle the acceptance criterion asks for, run
 * repeatedly — a leak that adds one subscriber per activation is invisible in
 * a single round trip and obvious over five.
 */
describe('plugin unload leaves nothing behind', () => {
  /** Everything a plugin can leave registered, counted in one place. */
  function residue() {
    return {
      panels: getPluginPanels().length,
      buttons: getPluginToolbarButtons().length,
      executionSubscribers: executionEventSubscriberCount(),
      socketTaps: executionEventTapCount(),
      renderer: getNodeRenderer('demo:Node') !== undefined,
    };
  }

  const EMPTY = {
    panels: 0, buttons: 0, executionSubscribers: 0, socketTaps: 0, renderer: false,
  };

  /** A plugin that uses every part of the API that outlives activate(). */
  function greedyPlugin(panels: HTMLElement[] = []) {
    return (api: CodefyUIPluginAPI) => {
      panels.push(api.ui.addPanel({ id: 'dock', title: 'Dock' }));
      panels.push(api.ui.addPanel({ id: 'side', title: 'Side', dock: 'right' }));
      api.ui.addToolbarButton({ id: 'go', icon: '*', tooltip: 'Go', onClick: () => {} });
      api.ui.addFloatingWidget({ id: 'fab' });
      api.events.onExecution(() => {});
      api.graph.onGraphChanged(() => {});
      api.nodes.registerRenderer('demo:Node', { mount: () => {} });
    };
  }

  beforeEach(() => {
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('test');
    _clearPluginPanels();
    _clearPluginToolbarButtons();
    _resetExecutionEvents();
    _clearNodeRenderers();
  });

  afterEach(() => {
    _clearPluginPanels();
    _clearPluginToolbarButtons();
    _resetExecutionEvents();
    _clearNodeRenderers();
  });

  it('registers everything on activate and unregisters everything on unload', async () => {
    mockPluginsResponse([
      { id: 'demo', enabled: true, frontend_entry: '/plugins/demo/frontend/index.js' },
    ]);
    const created: HTMLElement[] = [];
    const importer = vi.fn(async () => ({ default: greedyPlugin(created) }));
    await loadPluginFrontends(() => document.createElement('div'), importer);

    expect(residue()).toEqual({
      panels: 2, buttons: 1, executionSubscribers: 1, socketTaps: 1, renderer: true,
    });
    // Attach a panel the way the dock would, to prove unload detaches it.
    document.body.appendChild(created[0]);

    unloadPluginFrontends();
    expect(residue()).toEqual(EMPTY);
    expect(created[0].isConnected).toBe(false);
  });

  it('does not accumulate across repeated activate/unload cycles', async () => {
    mockPluginsResponse([
      { id: 'demo', enabled: true, frontend_entry: '/plugins/demo/frontend/index.js' },
    ]);
    const importer = vi.fn(async () => ({ default: greedyPlugin() }));

    for (let i = 0; i < 5; i += 1) {
      await loadPluginFrontends(() => document.createElement('div'), importer);
      expect(residue(), `after activation ${i + 1}`).toEqual({
        panels: 2, buttons: 1, executionSubscribers: 1, socketTaps: 1, renderer: true,
      });
      unloadPluginFrontends();
      expect(residue(), `after unload ${i + 1}`).toEqual(EMPTY);
    }
  });

  it('unregisters a plugin whose own cleanup throws', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    mockPluginsResponse([
      { id: 'demo', enabled: true, frontend_entry: '/plugins/demo/frontend/index.js' },
    ]);
    // Registering the panel LAST means its tracked cleanup runs last, after
    // the throwing one has already aborted... nothing: each cleanup is
    // isolated, and the id sweep is the second line of defence.
    const importer = vi.fn(async () => ({
      default: (api: CodefyUIPluginAPI) => {
        api.graph.onGraphChanged(() => {});
        api.ui.addPanel({ id: 'dock', title: 'Dock' });
        api.ui.addToolbarButton({ id: 'go', icon: '*', tooltip: 'Go', onClick: () => {} });
        // A plugin that breaks the host's teardown contract.
        api.events.onExecution(() => {});
        throw new Error('activate exploded after registering');
      },
    }));
    await loadPluginFrontends(() => document.createElement('div'), importer);
    expect(getPluginPanels()).toHaveLength(1);

    unloadPluginFrontends();
    expect(getPluginPanels()).toHaveLength(0);
    expect(getPluginToolbarButtons()).toHaveLength(0);
    expect(warn).toHaveBeenCalled();
  });

  /**
   * The id sweep is the second line of defence behind the tracked cleanups,
   * and until now it only survived because the tracked path had already
   * emptied the registries before it ran — so it was never the thing doing the
   * work, and a throw inside it would have skipped every LATER plugin's sweep.
   *
   * This drives it directly: registrations the host never tracked (the case
   * the sweep exists for) plus a host subscriber that throws out of the
   * registry's own `notify()`.
   */
  it('a sweep that throws does not cancel the rest of the teardown', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    mockPluginsResponse([
      { id: 'noisy', enabled: true, frontend_entry: '/plugins/noisy/frontend/index.js' },
      { id: 'quiet', enabled: true, frontend_entry: '/plugins/quiet/frontend/index.js' },
    ]);
    await loadPluginFrontends(
      () => document.createElement('div'),
      vi.fn(async () => ({ default: () => {} })),
    );

    registerPluginPanel('noisy', { id: 'p', title: 'P' });
    registerPluginToolbarButton('noisy', {
      id: 'b', icon: '*', tooltip: 'B', onClick: () => {},
    });
    registerPluginPanel('quiet', { id: 'p', title: 'P' });

    const off = subscribePluginPanels(() => {
      throw new Error('host subscriber exploded');
    });
    try {
      expect(() => unloadPluginFrontends()).not.toThrow();
    } finally {
      off();
    }

    // 'noisy' is swept first and throws. Its toolbar button must still go
    // (the step after the throw), and so must 'quiet' (the iteration after).
    expect(getPluginToolbarButtons()).toHaveLength(0);
    expect(getPluginPanels()).toHaveLength(0);
    expect(warn).toHaveBeenCalled();
  });

  it('unload is idempotent', async () => {
    mockPluginsResponse([
      { id: 'demo', enabled: true, frontend_entry: '/plugins/demo/frontend/index.js' },
    ]);
    await loadPluginFrontends(
      () => document.createElement('div'),
      vi.fn(async () => ({ default: greedyPlugin() })),
    );
    unloadPluginFrontends();
    expect(() => unloadPluginFrontends()).not.toThrow();
    expect(residue()).toEqual(EMPTY);
  });
});
