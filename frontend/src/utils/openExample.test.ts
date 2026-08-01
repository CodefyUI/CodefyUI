import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { insertExample, openExample, openExampleInNewTab } from './openExample';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import * as rest from '../api/rest';

// Only the network call is stubbed: resolution, preset merging and the store
// writes all run for real, which is the point — these three entry points are
// supposed to differ ONLY in where the resolved graph lands.
vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/rest')>();
  return { ...actual, loadExample: vi.fn() };
});

const mockedRest = vi.mocked(rest);
const store = () => useTabStore.getState();
const activeTab = () => useTabStore.getState().getActiveTab();

/** A serialized example node, in the shape `/api/examples/load` returns. */
function raw(id: string, over: Record<string, unknown> = {}) {
  return { id, type: 'Dropout', position: { x: 0, y: 0 }, data: { params: {} }, ...over };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
  mockedRest.loadExample.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('openExample', () => {
  it('replaces the active tab graph and mirrors the example name onto the tab', async () => {
    mockedRest.loadExample.mockResolvedValue({
      name: '  My Model  ',
      nodes: [raw('a'), raw('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'tensor', targetHandle: 'tensor' }],
    });

    await expect(openExample('Usage_Example/Foo')).resolves.toBe(true);
    expect(mockedRest.loadExample).toHaveBeenCalledWith('Usage_Example/Foo');
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['a', 'b']);
    expect(activeTab().edges).toHaveLength(1);
    expect(activeTab().name).toBe('My Model');
  });

  it('leaves the tab name alone when the example ships without one', async () => {
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });
    await openExample('x');
    expect(activeTab().name).toBe('Tab 1');
  });

  it('merges presets the running server has never seen', async () => {
    const preset = { preset_name: 'Fresh', category: 'c', description: '', tags: [], nodes: [], edges: [], exposed_inputs: [], exposed_outputs: [], exposed_params: [] };
    mockedRest.loadExample.mockResolvedValue({ nodes: [], edges: [], presets: [preset] });
    await openExample('x');
    expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual(['Fresh']);
  });

  it('toasts and leaves the graph alone when the load fails', async () => {
    store().setNodes([{ id: 'keep', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'k', type: 'K', params: {} } }] as never);
    mockedRest.loadExample.mockRejectedValue(new Error('nope'));

    await expect(openExample('x')).resolves.toBe(false);
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['keep']);
    expect(useToastStore.getState().toasts[0].message).toBe('Failed to load example');
  });
});

describe('openExampleInNewTab', () => {
  it('opens the example in a fresh tab, leaving the original untouched', async () => {
    store().setNodes([{ id: 'keep', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'k', type: 'K', params: {} } }] as never);
    const firstTabId = store().activeTabId;
    mockedRest.loadExample.mockResolvedValue({ name: 'Template', nodes: [raw('a')], edges: [] });

    await expect(openExampleInNewTab('x')).resolves.toBe(true);

    expect(store().tabs).toHaveLength(2);
    expect(store().activeTabId).not.toBe(firstTabId);
    expect(activeTab().name).toBe('Template');
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['a']);
    // The graph that was already open is exactly as it was.
    expect(store().getTab(firstTabId)!.nodes.map((n) => n.id)).toEqual(['keep']);
  });

  it('does not leave an empty tab behind when the load fails', async () => {
    mockedRest.loadExample.mockRejectedValue(new Error('nope'));
    await expect(openExampleInNewTab('x')).resolves.toBe(false);
    expect(store().tabs).toHaveLength(1);
    expect(useToastStore.getState().toasts[0].message).toBe('Failed to load example');
  });
});

describe('insertExample', () => {
  it('merges into the current canvas without clobbering colliding ids', async () => {
    // The canvas and the template both use "a" — the collision the id remap
    // exists for.
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'MINE', type: 'K', params: {} } },
    ] as never);
    mockedRest.loadExample.mockResolvedValue({
      name: 'Template',
      nodes: [raw('a'), raw('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'tensor', targetHandle: 'tensor' }],
    });

    await expect(insertExample('x')).resolves.toBe(true);

    const tab = activeTab();
    expect(tab.nodes).toHaveLength(3);
    expect(tab.nodes.find((n) => n.id === 'a')!.data.label).toBe('MINE');
    expect(new Set(tab.nodes.map((n) => n.id)).size).toBe(3);
    // The inserted edge joins the two INSERTED nodes, not the user's own "a".
    const insertedIds = new Set(tab.nodes.filter((n) => n.selected).map((n) => n.id));
    expect(insertedIds.size).toBe(2);
    expect(insertedIds.has(tab.edges[0].source)).toBe(true);
    expect(insertedIds.has(tab.edges[0].target)).toBe(true);
    // Inserting joins a graph rather than replacing it, so the tab keeps its
    // own name.
    expect(tab.name).toBe('Tab 1');
  });

  it('reports false for a template with no nodes', async () => {
    mockedRest.loadExample.mockResolvedValue({ nodes: [], edges: [] });
    await expect(insertExample('x')).resolves.toBe(false);
    expect(activeTab().nodes).toHaveLength(0);
  });

  it('toasts and leaves the graph alone when the load fails', async () => {
    store().setNodes([{ id: 'keep', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'k', type: 'K', params: {} } }] as never);
    mockedRest.loadExample.mockRejectedValue(new Error('nope'));

    await expect(insertExample('x')).resolves.toBe(false);
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['keep']);
    expect(useToastStore.getState().toasts[0].message).toBe('Failed to load example');
  });

  it('round-trips a bypassed node from the example file onto the canvas', async () => {
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('muted', { data: { params: {}, bypassed: true } }), raw('plain')],
      edges: [],
    });
    await insertExample('x');
    const [muted, plain] = activeTab().nodes;
    expect(muted.data.bypassed).toBe(true);
    expect(plain.data.bypassed).toBeUndefined();
  });
});
