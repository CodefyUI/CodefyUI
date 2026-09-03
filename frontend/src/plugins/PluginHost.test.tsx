import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import {
  PluginHost, _resetPluginHostForTesting, loadPluginFrontends,
  reloadPluginFrontends, startPluginFrontends, unloadPluginFrontends,
} from './PluginHost';
import { useI18n } from '../i18n';
import en from '../i18n/locales/en';
import zhTW from '../i18n/locales/zh-TW';
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

/** One node definition, in the shape `waitForNodeDefinitions` counts. */
const ONE_DEFINITION = [{
  node_name: 'X', category: 'c', description: '',
  inputs: [], outputs: [], params: [],
}];

/**
 * A fetch that answers the two URLs the host asks for.
 *
 * `/api/plugins` walks the payload list, one per call, and repeats the last
 * one -- that is how "the second answer no longer lists this plugin" is set
 * up. `/api/plugins/generation` answers `generation`, or 404s when it is null
 * (a server too old to have the route).
 */
function mockPluginHostFetch(options: {
  pluginPayloads: unknown[];
  generation: number | null;
}) {
  const calls: string[] = [];
  let index = 0;
  const fetchMock = vi.fn(async (url: string) => {
    calls.push(url);
    if (url === '/api/plugins/generation') {
      return options.generation === null
        ? { ok: false, json: async () => ({}) }
        : { ok: true, json: async () => ({ generation: options.generation }) };
    }
    const payload = options.pluginPayloads[
      Math.min(index, options.pluginPayloads.length - 1)
    ];
    index += 1;
    return { ok: true, json: async () => payload };
  });
  vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);
  return { calls };
}

/**
 * Re-activation without a page reload: the Plugin Center installs, enables or
 * removes a plugin and the editor has to show the result at once, without
 * ever running two activations over the same module-level registries.
 */
describe('reloadPluginFrontends', () => {
  const A = { id: 'a', enabled: true, frontend_entry: '/plugins/a/frontend/index.js' };
  const B = { id: 'b', enabled: true, frontend_entry: '/plugins/b/frontend/index.js' };

  const container = () => document.createElement('div');

  /**
   * A plugin that registers one of everything a reload has to clear first.
   * The execution subscriber is the one that cannot hide a duplicate: two
   * registrations of the same panel id replace each other, two subscriptions
   * stack.
   */
  function smallPlugin(api: CodefyUIPluginAPI) {
    api.ui.addPanel({ id: 'dock', title: 'Dock' });
    api.ui.addToolbarButton({ id: 'go', icon: '*', tooltip: 'Go', onClick: () => {} });
    api.events.onExecution(() => {});
  }

  function reset() {
    _resetPluginHostForTesting();
    _clearPluginPanels();
    _clearPluginToolbarButtons();
    _resetExecutionEvents();
  }

  beforeEach(reset);
  afterEach(reset);

  it('cache-busts the import with the reload generation', async () => {
    mockPluginHostFetch({ pluginPayloads: [[A]], generation: 7 });
    const activate = vi.fn();
    const importer = vi.fn(async () => ({ default: activate }));

    expect(await reloadPluginFrontends(container, importer)).toEqual(['a']);
    expect(importer).toHaveBeenCalledTimes(1);
    expect(importer).toHaveBeenCalledWith('/plugins/a/frontend/index.js?v=7');
  });

  it('busts with a timestamp when the server has no generation route', async () => {
    mockPluginHostFetch({ pluginPayloads: [[A]], generation: null });
    const urls: string[] = [];
    const importer = vi.fn(async (url: string) => {
      urls.push(url);
      return { default: vi.fn() };
    });

    const before = Date.now();
    await reloadPluginFrontends(container, importer);
    const url = urls[0] ?? '';

    expect(url).toMatch(/^\/plugins\/a\/frontend\/index\.js\?v=\d+$/);
    const stamp = Number(url.split('?v=')[1]);
    expect(stamp).toBeGreaterThanOrEqual(before);
    expect(stamp).toBeLessThanOrEqual(Date.now());
  });

  it('tears the old frontends down before activating the new ones', async () => {
    mockPluginHostFetch({ pluginPayloads: [[A]], generation: 3 });
    await loadPluginFrontends(container, vi.fn(async () => ({ default: smallPlugin })));
    expect(getPluginPanels()).toHaveLength(1);
    expect(getPluginToolbarButtons()).toHaveLength(1);

    // What the registries hold at the moment the new bundle arrives: if the
    // unload ran, nothing.
    const whenImported: number[] = [];
    const importer = vi.fn(async () => {
      whenImported.push(getPluginPanels().length, getPluginToolbarButtons().length);
      return { default: smallPlugin };
    });

    await reloadPluginFrontends(container, importer);

    expect(whenImported).toEqual([0, 0]);
    // One copy of each, not two.
    expect(getPluginPanels()).toHaveLength(1);
    expect(getPluginToolbarButtons()).toHaveLength(1);
    expect(executionEventSubscriberCount()).toBe(1);
  });

  it('does not re-activate a plugin the server no longer lists', async () => {
    mockPluginHostFetch({ pluginPayloads: [[A, B], [A]], generation: 4 });
    const importer = vi.fn(async () => ({ default: smallPlugin }));

    await loadPluginFrontends(container, importer);
    expect(getPluginPanels().map((p) => p.pluginId)).toEqual(['a', 'b']);

    // 'b' was uninstalled (or disabled) between the two answers.
    expect(await reloadPluginFrontends(container, importer)).toEqual(['a']);
    expect(getPluginPanels().map((p) => p.pluginId)).toEqual(['a']);
    expect(getPluginToolbarButtons().map((b) => b.pluginId)).toEqual(['a']);
  });

  it('hosts widgets in the document body when the stack is not mounted', async () => {
    mockPluginHostFetch({ pluginPayloads: [[A]], generation: 5 });
    const importer = vi.fn(async () => ({
      default: (api: CodefyUIPluginAPI) => { api.ui.addFloatingWidget({ id: 'fab' }); },
    }));

    // No <PluginHost /> has mounted: a reload fired from the Plugin Center
    // before the stack exists must still activate.
    await reloadPluginFrontends(undefined, importer);

    const widget = document.getElementById('plugin-widget-a-fab');
    expect(widget?.parentElement).toBe(document.body);
    widget?.remove();
  });

  /**
   * The race the promise chain exists for: an install settles while the boot
   * load is still parked in `waitForNodeDefinitions()`. Unserialised, the
   * reload would tear down nothing (the boot has registered nothing yet) and
   * both loads would then activate every plugin.
   */
  it('queues a reload issued during the boot load behind it', async () => {
    vi.useFakeTimers();
    try {
      useNodeDefStore.setState({ definitions: [] });
      mockPluginHostFetch({ pluginPayloads: [[A]], generation: 9 });

      const order: string[] = [];
      const activate = vi.fn((api: CodefyUIPluginAPI) => {
        order.push('activate');
        smallPlugin(api);
      });
      const importer = vi.fn(async (url: string) => {
        order.push(url);
        return { default: activate };
      });

      const boot = startPluginFrontends(container, importer);
      const reload = reloadPluginFrontends(container, importer);

      await vi.advanceTimersByTimeAsync(300);
      expect(importer, 'the boot is still waiting for node definitions')
        .not.toHaveBeenCalled();

      useNodeDefStore.setState({ definitions: ONE_DEFINITION });
      await vi.advanceTimersByTimeAsync(300);

      expect(await boot).toEqual(['a']);
      expect(await reload).toEqual(['a']);
      expect(activate).toHaveBeenCalledTimes(2);
      expect(order).toEqual([
        '/plugins/a/frontend/index.js', 'activate',
        '/plugins/a/frontend/index.js?v=9', 'activate',
      ]);
      // Twice activated, once registered: the reload's teardown ran between
      // the two passes. Unserialised, the reload would have torn down an
      // empty registry before the boot filled it, and both passes would have
      // stacked -- two of every subscription, panel and button.
      expect(executionEventSubscriberCount()).toBe(1);
      expect(getPluginPanels()).toHaveLength(1);
      expect(getPluginToolbarButtons()).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps serving later callers after an activation rejects', async () => {
    // The queue is shared with the boot and the dev poller, so a task that
    // throws must not be the end of it. Driven through the one call the host
    // does not guard: the toast it fires for a plugin that failed to load.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const realAddToast = useToastStore.getState().addToast;
    mockPluginHostFetch({ pluginPayloads: [[A]], generation: 6 });
    try {
      useToastStore.setState({
        addToast: () => { throw new Error('toast exploded'); },
      });
      await expect(
        reloadPluginFrontends(container, vi.fn(async () => { throw new Error('boom'); })),
      ).rejects.toThrow('toast exploded');
    } finally {
      useToastStore.setState({ addToast: realAddToast });
      warn.mockRestore();
    }

    expect(
      await reloadPluginFrontends(container, vi.fn(async () => ({ default: vi.fn() }))),
    ).toEqual(['a']);
  });
});

describe('plugin host toasts', () => {
  beforeEach(_resetPluginHostForTesting);
  afterEach(_resetPluginHostForTesting);

  it('names a plugin whose UI failed in the reader language', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const locale = useI18n.getState().locale;
    try {
      mockPluginsResponse([
        { id: 'a', enabled: true, frontend_entry: '/plugins/a/frontend/index.js' },
      ]);
      useI18n.setState({ locale: 'zh-TW' });
      await loadPluginFrontends(
        () => document.createElement('div'),
        vi.fn(async () => { throw new Error('boom'); }),
      );

      const [toast] = useToastStore.getState().toasts;
      expect(toast.type).toBe('error');
      expect(toast.message).toBe(
        zhTW['pluginCenter.toast.frontendFailed'].replace('{plugin}', 'a'),
      );
      expect(toast.message).not.toContain('{plugin}');
    } finally {
      useI18n.setState({ locale });
      warn.mockRestore();
    }
  });
});

/**
 * The dev-only hot reload: still armed by a linked plugin at boot, still
 * triggered by a generation bump, now going through the same reload the
 * Plugin Center uses.
 */
describe('dev hot-reload poller', () => {
  beforeEach(_resetPluginHostForTesting);
  afterEach(_resetPluginHostForTesting);

  it('reloads the frontends when the generation bumps', async () => {
    vi.useFakeTimers();
    try {
      let generation = 1;
      const calls: string[] = [];
      // A linked plugin with no frontend bundle: enough to arm the poller,
      // and nothing for the reload to import.
      const linked = [{
        id: 'linked', enabled: true, source_kind: 'local', frontend_entry: null,
      }];
      vi.stubGlobal('fetch', vi.fn(async (url: string) => {
        calls.push(url);
        return url === '/api/plugins/generation'
          ? { ok: true, json: async () => ({ generation }) }
          : { ok: true, json: async () => linked };
      }) as unknown as typeof fetch);

      render(<PluginHost />);
      await vi.advanceTimersByTimeAsync(10);
      expect(calls, 'the poller read the generation at boot')
        .toContain('/api/plugins/generation');

      await vi.advanceTimersByTimeAsync(1600);
      expect(useToastStore.getState().toasts, 'no bump, no reload').toHaveLength(0);

      generation = 2;
      const before = calls.length;
      await vi.advanceTimersByTimeAsync(1600);

      expect(calls.slice(before), 'the reload re-read the plugin list')
        .toContain('/api/plugins');
      expect(useToastStore.getState().toasts.map((t) => t.message))
        .toEqual([en['pluginCenter.toast.frontendsReloaded']]);
    } finally {
      vi.useRealTimers();
    }
  });
});
