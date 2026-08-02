/**
 * The apiVersion 3 surface (#132), and the proof that adding it left
 * apiVersion 2 alone.
 *
 * Kept beside `api.test.ts` rather than merged into it so the v2 file stays
 * exactly what it was: if a future change to the API breaks a v2 expectation,
 * it fails in the file that predates this work.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import { _setSessionTokenForTesting } from '../api/_auth';
import { buildPluginAPI } from './api';
import { _clearPluginPanels, getPluginPanel } from './panels';
import { _clearPluginToolbarButtons, getPluginToolbarButtons } from './toolbarButtons';
import { _resetExecutionEvents, executionEventSubscriberCount } from './executionEvents';
import {
  _clearNodeRenderers, getNodeRenderer, type PluginNodeRenderer,
} from './nodeRenderers';
import type { GraphOp } from './ops';
import type { NodeDefinition } from '../types';

const DEFS: NodeDefinition[] = [
  {
    node_name: 'Source', category: 'Layer', description: '',
    inputs: [],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  },
];

function freshApi() {
  return buildPluginAPI('test-plugin', () => document.createElement('div'));
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
  useNodeDefStore.setState({ definitions: DEFS });
  useToastStore.setState({ toasts: [] });
  window.localStorage.clear();
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
  vi.unstubAllGlobals();
  _setSessionTokenForTesting(null);
});

describe('meta', () => {
  it('reports apiVersion 3', () => {
    expect(freshApi().apiVersion).toBe(3);
  });
});

describe('ui.addPanel', () => {
  it('registers a panel and hands back its element', () => {
    const api = freshApi();
    const el = api.ui.addPanel({ id: 'metrics', title: 'Metrics', icon: '#' });
    expect(el).toBeInstanceOf(HTMLElement);
    const panel = getPluginPanel('test-plugin:metrics');
    expect(panel?.title).toBe('Metrics');
    expect(panel?.icon).toBe('#');
    expect(panel?.dock).toBe('bottom');
    expect(panel?.element).toBe(el);
  });

  it('keeps the element stable across re-registration', () => {
    const api = freshApi();
    const first = api.ui.addPanel({ id: 'metrics', title: 'Metrics' });
    first.appendChild(document.createElement('span'));
    const second = api.ui.addPanel({ id: 'metrics', title: 'Renamed' });
    expect(second).toBe(first);
    expect(second.childNodes).toHaveLength(1);
    expect(getPluginPanel('test-plugin:metrics')?.title).toBe('Renamed');
  });

  it('accepts the right dock', () => {
    freshApi().ui.addPanel({ id: 'side', title: 'Side', dock: 'right' });
    expect(getPluginPanel('test-plugin:side')?.dock).toBe('right');
  });

  it('removePanel unregisters it', () => {
    const api = freshApi();
    api.ui.addPanel({ id: 'metrics', title: 'Metrics' });
    api.ui.removePanel('metrics');
    expect(getPluginPanel('test-plugin:metrics')).toBeUndefined();
  });

  it('cannot reach another plugin panel of the same name', () => {
    buildPluginAPI('other', () => document.createElement('div'))
      .ui.addPanel({ id: 'metrics', title: 'Theirs' });
    freshApi().ui.removePanel('metrics');
    expect(getPluginPanel('other:metrics')).toBeDefined();
  });

  it('registers its removal with trackCleanup', () => {
    const tracked: Array<() => void> = [];
    const api = buildPluginAPI(
      'test-plugin', () => document.createElement('div'), (fn) => tracked.push(fn),
    );
    api.ui.addPanel({ id: 'metrics', title: 'Metrics' });
    expect(tracked).toHaveLength(1);
    tracked[0]();
    expect(getPluginPanel('test-plugin:metrics')).toBeUndefined();
  });
});

describe('ui.addToolbarButton', () => {
  it('registers a button and returns a remove fn', () => {
    const onClick = vi.fn();
    const remove = freshApi().ui.addToolbarButton({
      id: 'go', icon: '*', tooltip: 'Go', onClick,
    });
    const [button] = getPluginToolbarButtons();
    expect(button.key).toBe('test-plugin:go');
    expect(button.tooltip).toBe('Go');
    button.onClick();
    expect(onClick).toHaveBeenCalledTimes(1);
    remove();
    expect(getPluginToolbarButtons()).toHaveLength(0);
  });

  it('removeToolbarButton unregisters by id', () => {
    const api = freshApi();
    api.ui.addToolbarButton({ id: 'go', icon: '*', tooltip: 'Go', onClick: () => {} });
    api.ui.removeToolbarButton('go');
    expect(getPluginToolbarButtons()).toHaveLength(0);
  });

  it('registers its removal with trackCleanup', () => {
    const tracked: Array<() => void> = [];
    const api = buildPluginAPI(
      'test-plugin', () => document.createElement('div'), (fn) => tracked.push(fn),
    );
    api.ui.addToolbarButton({ id: 'go', icon: '*', tooltip: 'Go', onClick: () => {} });
    expect(tracked).toHaveLength(1);
    tracked[0]();
    expect(getPluginToolbarButtons()).toHaveLength(0);
  });
});

describe('events surface', () => {
  it('onExecution subscribes and the returned fn unsubscribes', () => {
    const off = freshApi().events.onExecution(() => {});
    expect(executionEventSubscriberCount()).toBe(1);
    off();
    expect(executionEventSubscriberCount()).toBe(0);
  });

  it('registers its unsubscribe with trackCleanup', () => {
    const tracked: Array<() => void> = [];
    const api = buildPluginAPI(
      'test-plugin', () => document.createElement('div'), (fn) => tracked.push(fn),
    );
    api.events.onExecution(() => {});
    expect(tracked).toHaveLength(1);
    tracked[0]();
    expect(executionEventSubscriberCount()).toBe(0);
  });
});

describe('runs facade', () => {
  const PAGE = { runs: [{ id: 'r1' }], total: 1, limit: 50, offset: 0 };

  function mockFetch(body: unknown, status = 200) {
    // Parameters are declared so `mock.calls[0]` is a two-element tuple: the
    // assertions below are about the URL the facade built and the headers the
    // HOST attached, and both live in that tuple.
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ({
      ok: status >= 200 && status < 300,
      status,
      statusText: 'x',
      json: async () => body,
    }));
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);
    return fetchMock;
  }

  it('list() proxies the run list endpoint', async () => {
    const fetchMock = mockFetch(PAGE);
    expect(await freshApi().runs.list()).toEqual(PAGE);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/runs');
  });

  it('list() forwards status, limit and offset', async () => {
    const fetchMock = mockFetch(PAGE);
    await freshApi().runs.list({ status: ['running'], limit: 5, offset: 10 });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/runs?status=running&limit=5&offset=10');
  });

  it('get() proxies one run, and answers null for an unknown id', async () => {
    const fetchMock = mockFetch({ id: 'r1' });
    expect(await freshApi().runs.get('r1')).toEqual({ id: 'r1' });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/runs/r1');

    mockFetch({}, 404);
    expect(await freshApi().runs.get('gone')).toBeNull();
  });

  it('metrics() proxies the metrics endpoint, optionally for one series', async () => {
    const body = { run_id: 'r1', names: ['loss'], metrics: [] };
    const fetchMock = mockFetch(body);
    expect(await freshApi().runs.metrics('r1')).toEqual(body);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/runs/r1/metrics');

    const withName = mockFetch(body);
    await freshApi().runs.metrics('r1', 'train/loss');
    expect(withName.mock.calls[0][0]).toBe('/api/runs/r1/metrics?name=train%2Floss');
  });

  it('escapes a hostile run id rather than letting it shape the URL', async () => {
    const fetchMock = mockFetch({ id: 'x' });
    await freshApi().runs.get('../../secret?x=1');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/runs/..%2F..%2Fsecret%3Fx%3D1');
  });

  it('surfaces a server failure as a rejection', async () => {
    mockFetch({}, 503);
    await expect(freshApi().runs.list()).rejects.toThrow(/Failed to list runs/);
  });

  it('is read-only: nothing on it submits, cancels or deletes', () => {
    expect(Object.keys(freshApi().runs).sort()).toEqual(['get', 'list', 'metrics']);
  });

  it('never exposes the session token to plugin code', async () => {
    _setSessionTokenForTesting('super-secret-token');
    mockFetch(PAGE);
    const api = freshApi();
    await api.runs.list();
    const reachable = JSON.stringify(
      api, (_k, v) => (typeof v === 'function' ? '[fn]' : v),
    );
    expect(reachable).not.toContain('super-secret-token');
    expect((api.runs as unknown as Record<string, unknown>).token).toBeUndefined();
  });

  it('the HOST attaches the token: a plugin request carries it unasked', async () => {
    _setSessionTokenForTesting('super-secret-token');
    const fetchMock = mockFetch({});
    await freshApi().http.fetch('/api/anything', { method: 'POST' });
    const init = fetchMock.mock.calls[0][1] as unknown as RequestInit;
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('super-secret-token');
  });
});

/**
 * apiVersion 3 was specified as purely additive, and Graph Copilot — which
 * declares `requires_codefyui = ">=1.3.0"` and was written against v2 — has to
 * keep working with no changes at all. `contract.v2.assert.ts` proves that at
 * the type level; this proves it at runtime by exercising every member the v2
 * contract published, from a plugin that knows nothing about v3.
 */
describe('apiVersion 2 compatibility', () => {
  const V2_SHAPE: Record<string, string[]> = {
    ui: ['addFloatingWidget', 'toast'],
    graph: ['getGraph', 'getNodeDefinitions', 'applyOperations', 'onGraphChanged'],
    nodes: ['registerRenderer'],
    http: ['fetch'],
    storage: ['get', 'set', 'remove'],
  };

  it('still offers every v2 member as a callable function', () => {
    const api = freshApi() as unknown as Record<string, Record<string, unknown>>;
    for (const [section, members] of Object.entries(V2_SHAPE)) {
      expect(api[section], `section ${section}`).toBeDefined();
      for (const member of members) {
        expect(typeof api[section][member], `${section}.${member}`).toBe('function');
      }
    }
  });

  it('is not a version a v2 feature check would refuse', () => {
    // The documented v2-era check was `api.apiVersion >= 2`.
    expect(freshApi().apiVersion).toBeGreaterThanOrEqual(2);
  });

  it('a v2-only plugin activates and drives the editor unchanged', () => {
    const widget = document.createElement('div');

    // Typed against the v2 contract alone — no events, runs or panels in
    // sight, exactly like a bundle built before this release.
    const activate = (api: {
      apiVersion: number;
      ui: { addFloatingWidget(o: { id: string }): HTMLElement; toast(m: string): void };
      graph: {
        getGraph(): { nodes: unknown[] };
        getNodeDefinitions(): unknown[];
        applyOperations(ops: GraphOp[]): { node_count: number };
        onGraphChanged(cb: () => void): () => void;
      };
      nodes: { registerRenderer(t: string, r: PluginNodeRenderer): () => void };
      storage: { get(k: string): string | null; set(k: string, v: string): void };
    }) => {
      if (api.apiVersion < 2) throw new Error('needs apiVersion 2');
      const el = api.ui.addFloatingWidget({ id: 'fab' });
      el.textContent = 'copilot';
      api.ui.toast('hello');
      let changes = 0;
      const off = api.graph.onGraphChanged(() => { changes += 1; });
      const applied = api.graph.applyOperations([
        { op: 'add_node', node_type: 'Source' },
      ]);
      api.nodes.registerRenderer('Source', { mount: () => {} });
      api.storage.set('convo', '[]');
      return {
        widgetText: el.textContent,
        defs: api.graph.getNodeDefinitions().length,
        nodes: api.graph.getGraph().nodes.length,
        applied: applied.node_count,
        changes: () => changes,
        stored: api.storage.get('convo'),
        off,
      };
    };

    const result = activate(buildPluginAPI('graph-copilot', () => widget));
    expect(result.widgetText).toBe('copilot');
    expect(result.defs).toBe(DEFS.length);
    expect(result.nodes).toBe(1);
    expect(result.applied).toBe(1);
    expect(result.changes()).toBeGreaterThan(0);
    expect(result.stored).toBe('[]');
    expect(getNodeRenderer('Source')).toBeDefined();
    expect(useToastStore.getState().toasts.some((t) => t.message === 'hello')).toBe(true);
    result.off();
  });
});
