import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  GraphMissingError,
  readSavedGraphDocument,
  reloadTabFromDisk,
  resolveSavedGraph,
} from './openSavedGraph';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import type { PresetDefinition } from '../types';

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

function mockFetch(status: number, body: unknown) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'mock',
    json: async () => body,
  } as unknown as Response;
  g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
  return g.fetch as unknown as ReturnType<typeof vi.fn>;
}

/** A serialized node, in the shape `/api/graph/load/<name>` returns. */
function raw(id: string, over: Record<string, unknown> = {}) {
  return { id, type: 'Add', position: { x: 0, y: 0 }, data: { params: {} }, ...over };
}

function preset(name: string): PresetDefinition {
  return {
    preset_name: name,
    category: 'c',
    description: '',
    tags: [],
    nodes: [],
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
  } as unknown as PresetDefinition;
}

const tabs = () => useTabStore.getState().tabs;

beforeEach(() => {
  originalFetch = g.fetch;
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
});

afterEach(() => {
  g.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe('resolveSavedGraph', () => {
  it('builds the document the tab store installs, bound to the file', () => {
    const doc = resolveSavedGraph(
      {
        nodes: [raw('a'), raw('b')],
        edges: [
          { id: 'e1', source: 'a', target: 'b', sourceHandle: 'tensor', targetHandle: 'tensor' },
        ],
        description: 'a saved graph',
        segmentGroups: [{ id: 's1' }] as never,
        format_version: 1,
      },
      'alpha',
    );
    expect(doc.nodes.map((n) => n.id)).toEqual(['a', 'b']);
    expect(doc.edges).toHaveLength(1);
    expect(doc.boundFile).toBe('alpha');
    expect(doc.description).toBe('a saved graph');
    expect(doc.segmentGroups).toHaveLength(1);
    expect(doc.formatVersion).toBe(1);
    // No `name`: a saved graph is bound to its file by `currentGraphFile`,
    // and adopting the file's name here would rename a tab the user named.
    expect('name' in doc).toBe(false);
  });

  it('takes the binding decision from the caller, not from the file', () => {
    expect(resolveSavedGraph({ nodes: [], edges: [] }, null).boundFile).toBeNull();
  });

  it('falls back for every field a file can be missing', () => {
    const doc = resolveSavedGraph({}, 'alpha');
    expect(doc.nodes).toEqual([]);
    expect(doc.edges).toEqual([]);
    expect(doc.subgraphs).toEqual([]);
    expect(doc.segmentGroups).toEqual([]);
    expect(doc.description).toBe('');
    expect(doc.formatVersion).toBeUndefined();
  });

  it('merges presets the running server has never seen, keeping the ones it has', () => {
    useNodeDefStore.setState({ presets: [preset('Existing')] });
    resolveSavedGraph({ nodes: [], edges: [], presets: [preset('Existing'), preset('Fresh')] }, 'a');
    expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual([
      'Existing',
      'Fresh',
    ]);
  });

  it('writes no presets at all when the file carries none', () => {
    const before = useNodeDefStore.getState().presets;
    resolveSavedGraph({ nodes: [raw('a')], edges: [] }, 'a');
    expect(useNodeDefStore.getState().presets).toBe(before);
  });

  it('lays the graph out itself when the layout file was missing', () => {
    // 9999,9999 is not a placement dagre's ranked layout would ever produce
    // for a two-node graph, so any change proves a real layout ran.
    const doc = resolveSavedGraph(
      {
        nodes: [
          raw('n1', { position: { x: 9999, y: 9999 } }),
          raw('n2', { position: { x: 9999, y: 9999 } }),
          { id: 'note1', type: 'note', position: { x: 999, y: 999 }, data: {} },
        ],
        edges: [{ id: 'e1', source: 'n1', target: 'n2', sourceHandle: '', targetHandle: '' }],
        layout_missing: true,
      },
      'alpha',
    );
    const at = (id: string) => doc.nodes.find((n) => n.id === id)!.position;
    expect(at('n1')).not.toEqual({ x: 9999, y: 9999 });
    expect(at('n1').x).not.toBe(at('n2').x);
    // The lone unbound note is placed deterministically beside the graph.
    expect(at('note1')).toEqual({ x: -320, y: 0 });
  });

  it('keeps the file positions when the layout file was there', () => {
    const doc = resolveSavedGraph(
      { nodes: [raw('n1', { position: { x: 123, y: 456 } })], edges: [] },
      'alpha',
    );
    expect(doc.nodes[0].position).toEqual({ x: 123, y: 456 });
  });
});

describe('readSavedGraphDocument', () => {
  it('url-encodes the file name and resolves the body', async () => {
    const fetchMock = mockFetch(200, { nodes: [raw('a')], edges: [] });
    const doc = await readSavedGraphDocument('My Graph/v2', 'My Graph/v2');
    expect(fetchMock).toHaveBeenCalledWith('/api/graph/load/My%20Graph%2Fv2');
    expect(doc.nodes.map((n) => n.id)).toEqual(['a']);
    expect(doc.boundFile).toBe('My Graph/v2');
  });

  it('names the file in a GraphMissingError when the server has none', async () => {
    mockFetch(404, { detail: "Graph 'alpha' not found" });
    const err = await readSavedGraphDocument('alpha', 'alpha').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(GraphMissingError);
    expect((err as GraphMissingError).file).toBe('alpha');
    expect((err as Error).name).toBe('GraphMissingError');
  });

  it('reports any other refusal the way every load failure reads', async () => {
    mockFetch(500, {});
    const err = await readSavedGraphDocument('alpha', 'alpha').catch((e: unknown) => e);
    expect(err).not.toBeInstanceOf(GraphMissingError);
    expect((err as Error).message).toMatch(/Load failed/);
  });
});

describe('reloadTabFromDisk', () => {
  it('installs into the named tab, not the active one, and keeps it bound', async () => {
    const stale = tabs()[0].id;
    // Stamped while it is still the active tab: `projectOrigin` is what the
    // git store's affected-tab filter reads, so a reload that dropped it would
    // quietly take the tab out of every future offer.
    useTabStore.getState().stampActiveTabProject('D:/work/demo');
    useTabStore.getState().addTab('Tab 2');
    const active = useTabStore.getState().activeTabId;
    expect(active).not.toBe(stale);
    mockFetch(200, { nodes: [raw('fromDisk')], edges: [] });

    await reloadTabFromDisk(stale, 'alpha');

    const reloaded = tabs().find((t) => t.id === stale)!;
    expect(reloaded.nodes.map((n) => n.id)).toEqual(['fromDisk']);
    expect(reloaded.currentGraphFile).toBe('alpha');
    // The tab keeps the label the user is looking at, and the project it
    // belongs to.
    expect(reloaded.name).toBe('Tab 1');
    expect(reloaded.projectOrigin).toBe('D:/work/demo');
    // And the tab in front of the user is untouched.
    expect(useTabStore.getState().activeTabId).toBe(active);
    expect(tabs().find((t) => t.id === active)!.nodes).toEqual([]);
  });

  it('leaves the tab exactly as it is when the file is gone', async () => {
    const tabId = tabs()[0].id;
    useTabStore.getState().setNodes([raw('onScreen')] as never);
    mockFetch(404, { detail: 'not found' });

    await expect(reloadTabFromDisk(tabId, 'alpha')).rejects.toBeInstanceOf(GraphMissingError);

    // Whatever the graph on screen is, it is the only copy left.
    expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['onScreen']);
  });

  it('reports a file written by a newer build, which opens read-only', async () => {
    const tabId = tabs()[0].id;
    mockFetch(200, { nodes: [], edges: [], format_version: 99 });
    expect(await reloadTabFromDisk(tabId, 'alpha')).toBe(true);
    expect(tabs()[0].readOnly).toBe(true);
  });

  it('is a no-op for a tab that was closed before the reload was accepted', async () => {
    mockFetch(200, { nodes: [raw('fromDisk')], edges: [] });
    await expect(reloadTabFromDisk('gone', 'alpha')).resolves.toBe(false);
    expect(tabs()).toHaveLength(1);
    expect(tabs()[0].nodes).toEqual([]);
  });
});
