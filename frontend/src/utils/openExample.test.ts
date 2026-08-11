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

  // -- #200 item 4: the two fields this reader used to drop --

  it('carries the example description onto the tab', async () => {
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('a')], edges: [], description: 'What this template teaches',
    });

    await openExample('x');

    expect(activeTab().description).toBe('What this template teaches');
  });

  it('clears the previous graph description for an example that ships none', async () => {
    // `description` is persisted through save, so the leftover was not just
    // cosmetic: it got written to disk as the new graph's description.
    store().setDescription('the graph that was here before');
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });

    await openExample('x');

    expect(activeTab().description).toBe('');
  });

  it('opens an example written by a newer CodefyUI read-only, and says so', async () => {
    // The gate must fail CLOSED. This path used to skip it entirely, so a
    // plugin-shipped template from a newer build opened fully editable and
    // the next save silently down-converted it.
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('a')], edges: [], format_version: 999,
    });

    await expect(openExample('x')).resolves.toBe(true);

    expect(activeTab().readOnly).toBe(true);
    expect(
      useToastStore.getState().toasts.some(
        (t) => t.type === 'warning' && t.message.includes('v999'),
      ),
    ).toBe(true);
  });

  it('opening a current-format example into a read-only tab makes it editable again', async () => {
    store().setTabReadOnly(true);
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [], format_version: 1 });

    await openExample('x');

    expect(activeTab().readOnly).toBe(false);
    expect(useToastStore.getState().toasts.some((t) => t.type === 'warning')).toBe(false);
  });

  it('an example with no format_version opens editable', async () => {
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });
    await openExample('x');
    expect(activeTab().readOnly).toBe(false);
  });

  // -- #200 item 9: an example is a template, bound to no file --

  it('unbinds the tab so the example cannot be saved over the file that was open', async () => {
    // The bug: the tab still claimed to be bound to my_graph, and a bound tab
    // is exactly the case Save skips the overwrite prompt for -- so the next
    // Save wrote the EXAMPLE into my_graph without asking.
    store().setCurrentGraphFile('my_graph');
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });

    await openExample('x');

    expect(activeTab().currentGraphFile).toBeNull();
  });

  it('leaves an already-unbound tab unbound', async () => {
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });
    await openExample('x');
    expect(activeTab().currentGraphFile).toBeNull();
  });

  it('installs the whole example in ONE store update', async () => {
    // #200 item 8: the ordering contract between setNodes and setSubgraphs
    // is enforced by construction now -- there is no intermediate state in
    // which the tab holds the new nodes and the old definitions.
    mockedRest.loadExample.mockResolvedValue(exampleWithBlock());
    let emissions = 0;
    const unsubscribe = useTabStore.subscribe(() => {
      emissions += 1;
    });

    await openExample('x');
    unsubscribe();

    expect(emissions).toBe(1);
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['plain', 'inst']);
    expect(activeTab().subgraphs.map((d) => d.id)).toEqual(['blk']);
    expect(activeTab().name).toBe('Blocky');
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

  // -- #200 item 10: a merge cannot be answered with read-only, so it refuses --

  it('refuses a template written by a newer CodefyUI, leaving the canvas untouched', async () => {
    store().setNodes([
      { id: 'mine', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'k', type: 'K', params: {} } },
    ] as never);
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('a'), raw('b')],
      edges: [],
      format_version: 999,
    });

    await expect(insertExample('x')).resolves.toBe(false);

    // Nothing merged: the node count is the user's own graph, unchanged.
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['mine']);
    expect(activeTab().edges).toEqual([]);
    // ...and the tab is NOT dragged into read-only for what the template is.
    expect(activeTab().readOnly).toBe(false);
    const toasts = useToastStore.getState().toasts;
    const toast = toasts[toasts.length - 1];
    expect(toast.type).toBe('error');
    expect(toast.message).toContain('v999');
  });

  it("does not install a refused template's presets on the way to refusing it", async () => {
    // Resolution merges unknown presets into the palette, so the gate has to
    // run on the RAW payload -- before anything is resolved.
    const preset = { preset_name: 'FromTheFuture', category: 'c', description: '', tags: [], nodes: [], edges: [], exposed_inputs: [], exposed_outputs: [], exposed_params: [] };
    mockedRest.loadExample.mockResolvedValue({
      nodes: [raw('a')], edges: [], presets: [preset], format_version: 999,
    });

    await expect(insertExample('x')).resolves.toBe(false);

    expect(useNodeDefStore.getState().presets).toEqual([]);
  });

  it('inserts a current-format template, and one with no format_version', async () => {
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [], format_version: 1 });
    await expect(insertExample('x')).resolves.toBe(true);
    expect(activeTab().nodes).toHaveLength(1);

    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('b')], edges: [] });
    await expect(insertExample('x')).resolves.toBe(true);
    expect(activeTab().nodes).toHaveLength(2);
    expect(useToastStore.getState().toasts).toEqual([]);
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

// ── Subgraph definitions (core#137) ─────────────────────────────────────

/** A one-node definition an example can ship, plus the instance for it. */
function exampleWithBlock() {
  return {
    name: 'Blocky',
    nodes: [
      raw('plain'),
      { id: 'inst', type: 'subgraph:blk', position: { x: 10, y: 10 }, data: { params: {} } },
    ],
    edges: [],
    subgraphs: [
      {
        id: 'blk',
        name: 'Block',
        description: '',
        nodes: [raw('inner')],
        edges: [],
        interface: { inputs: [], outputs: [], triggerTargets: [] },
      },
    ],
  };
}

describe('examples that ship subgraph definitions', () => {
  it('openExample lands the definitions, not just the instance node', async () => {
    mockedRest.loadExample.mockResolvedValue(exampleWithBlock());

    await expect(openExample('x')).resolves.toBe(true);

    expect(activeTab().subgraphs.map((d) => d.id)).toEqual(['blk']);
    // Serialization is what a save/run sees; an instance with no definition
    // renders "definition missing" and fails `Unknown subgraph` on the server.
    expect(store().getSerializedGraph().subgraphs.map((d) => d.id)).toEqual(['blk']);
    // The instance renders from the definition it names, not from a stub.
    const instance = activeTab().nodes.find((n) => n.id === 'inst')!;
    expect(instance.data.label).toBe('Block');
  });

  it('openExample replaces segment overlays instead of leaving stale ones', async () => {
    // Opening an example REPLACES the graph, so a segment left over from the
    // graph that was there names ids nothing wears any more -- and
    // `segmentGroups` is persisted through save, so it would be written to
    // disk. Both Toolbar readers already replace it; this is the third door.
    store().setSegmentGroups([
      { id: 'old', headNodeId: 'gone', tailNodeId: 'also-gone' } as never,
    ]);
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });

    await openExample('x');

    expect(activeTab().segmentGroups).toEqual([]);
  });

  it('openExampleInNewTab lands them in the new tab', async () => {
    mockedRest.loadExample.mockResolvedValue(exampleWithBlock());
    const firstTabId = store().activeTabId;

    await expect(openExampleInNewTab('x')).resolves.toBe(true);

    expect(activeTab().subgraphs.map((d) => d.id)).toEqual(['blk']);
    // ...and only in the new tab.
    expect(store().getTab(firstTabId)!.subgraphs).toEqual([]);
  });

  it('insertExample merges them into the canvas it joined', async () => {
    mockedRest.loadExample.mockResolvedValue(exampleWithBlock());

    await expect(insertExample('x')).resolves.toBe(true);

    expect(activeTab().subgraphs.map((d) => d.id)).toEqual(['blk']);
    expect(store().getSerializedGraph().subgraphs.map((d) => d.id)).toEqual(['blk']);
  });

  it('insertExample keeps the LOCAL definition on an id collision', async () => {
    // Same rule paste follows: the definition already here is the one this
    // tab's other instances share, so a template must not edit them.
    store().setSubgraphs([
      {
        id: 'blk', name: 'Mine', description: '',
        nodes: [], edges: [],
        interface: { inputs: [], outputs: [], triggerTargets: [] },
      },
    ] as never);
    mockedRest.loadExample.mockResolvedValue(exampleWithBlock());

    await insertExample('x');

    expect(activeTab().subgraphs).toHaveLength(1);
    expect(activeTab().subgraphs[0].name).toBe('Mine');
    // The inserted instance is drawn from the definition that WON, so the
    // canvas never paints ports the block does not have.
    const inserted = activeTab().nodes.find((n) => n.selected && n.data.subgraphId)!;
    expect(inserted.data.label).toBe('Mine');
  });
});
