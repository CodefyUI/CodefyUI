/**
 * The plugin API and the graph's seed (2.8.9). `getGraph()` serializes the
 * seed as `settings.seed`, so a seed change is a change a plugin can read and
 * must be told about, and `workspace.openGraphs` honours a seed in the graph
 * it is handed the way it honours `settings.device`.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { buildPluginAPI } from './api';

vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();

function freshApi() {
  return buildPluginAPI('test-plugin', () => document.createElement('div'));
}

function graph(settings?: { device?: string; seed?: number }) {
  return {
    nodes: [{ id: 'a', type: 'Source', position: { x: 0, y: 0 }, data: { params: {} } }],
    edges: [],
    ...(settings ? { settings } : {}),
  };
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('live');
  useNodeDefStore.setState({ definitions: [], presets: [] } as never);
  window.localStorage.clear();
});

describe('graph.onGraphChanged', () => {
  it('fires once for a seed change, which getGraph serializes', () => {
    const api = freshApi();
    let calls = 0;
    const off = api.graph.onGraphChanged(() => { calls += 1; });

    store().setSeed(7);
    expect(calls).toBe(1);
    expect((api.graph.getGraph() as { settings?: unknown }).settings).toEqual({ seed: 7 });

    // Same value again changes nothing a plugin can read.
    store().setSeed(7);
    expect(calls).toBe(1);
    off();
  });
});

describe('workspace.openGraphs', () => {
  it('installs settings.seed from the graph', () => {
    const api = freshApi();
    const [result] = api.workspace.openGraphs(
      [{ title: 'Seeded', graph: graph({ seed: 0 }) }],
      { activate: 'none' },
    );
    if (!('tabId' in result)) throw new Error('open failed');
    expect(store().getTab(result.tabId)!.seed).toBe(0);
  });

  it('gives a seedless graph the seed of the tab that was active', () => {
    store().setSeed(4);
    const api = freshApi();
    const [result] = api.workspace.openGraphs(
      [{ title: 'Seedless', graph: graph() }],
      { activate: 'none' },
    );
    if (!('tabId' in result)) throw new Error('open failed');
    expect(store().getTab(result.tabId)!.seed).toBe(4);
  });
});
