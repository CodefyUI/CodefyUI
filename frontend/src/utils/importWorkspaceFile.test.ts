import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';
import { importWorkspaceFile, type WorkspaceImportResult } from './importWorkspaceFile';
import type { ParsedWorkspace, ParsedWorkspaceTab, WorkspaceRunSettings } from './workspaceFile';
import { useI18n } from '../i18n';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useProjectStore } from '../store/projectStore';
import { useTabStore, type TabState } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import type { NodeData, NodeDefinition } from '../types';

const store = () => useTabStore.getState();
const tabs = () => useTabStore.getState().tabs;
const names = () => tabs().map((t) => t.name);
const messages = (type: string) =>
  useToastStore.getState().toasts.filter((t) => t.type === type).map((t) => t.message);

const ADD: NodeDefinition = {
  node_name: 'Add', category: 'Math', description: '', inputs: [], outputs: [], params: [],
};

/** A serialized node, in the shape `Export -> JSON` writes. */
function rawNode(id: string, type = 'Add') {
  return { id, type, position: { x: 0, y: 0 }, data: { params: {} } };
}

/** A serialized graph, in the shape a workspace file's `tabs[i].graph` has. */
function graphOf(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    name: 'g',
    description: '',
    nodes: [rawNode('n1')],
    edges: [],
    presets: [],
    segmentGroups: [],
    subgraphs: [],
    format_version: 1,
    ...over,
  };
}

function entry(
  title: unknown,
  graph: unknown = graphOf(),
  run: Partial<WorkspaceRunSettings> = {},
): ParsedWorkspaceTab {
  return { title, graph, run };
}

function workspace(
  entries: ParsedWorkspaceTab[],
  over: Partial<ParsedWorkspace> = {},
): ParsedWorkspace {
  return {
    version: 1, appVersion: null, exportedAt: null, active: null, preferences: {},
    tabs: entries,
    ...over,
  };
}

function flowNode(id: string): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: 'Add', params: {} } };
}

/** Put a graph in the tab that is already open, so it is not the "lone empty tab". */
function holdAGraph(): TabState {
  store().setNodes([flowNode('mine')]);
  return tabs()[0];
}

function patchFirstTab(patch: Partial<TabState>): void {
  useTabStore.setState({ tabs: tabs().map((t, i) => (i === 0 ? { ...t, ...patch } : t)) });
}

function tabIdAt(result: WorkspaceImportResult, index: number): string {
  const entryResult = result.results[index];
  if (!('tabId' in entryResult)) {
    throw new Error(`entry ${index} was skipped: ${entryResult.skipped}`);
  }
  return entryResult.tabId;
}

beforeEach(() => {
  localStorage.clear();
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
  useUIStore.setState({
    fontSize: 'default',
    edgeStyle: 'circuit',
    gridSnapEnabled: false,
    tooltipsEnabled: true,
    beginnerMode: false,
  });
  // One EMPTY tab, which is what a fresh browser holds.
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('importWorkspaceFile: where the tabs land', () => {
  it('adds the tabs beside a tab that holds a graph, which keeps its content and its identity', async () => {
    const mine = holdAGraph();
    const result = await importWorkspaceFile(workspace([entry('A'), entry('B')]));

    expect(result.imported).toBe(2);
    expect(names()).toEqual(['Tab 1', 'A', 'B']);
    // The same OBJECT: nothing about the import rewrote the user's own tab.
    expect(tabs()[0]).toBe(mine);
    expect(tabs()[1].nodes.map((n) => n.id)).toEqual(['n1']);
    expect(messages('success')).toEqual(['Imported 2 tab(s).']);
  });

  it('replaces a lone empty tab, so a fresh browser ends up with exactly the exported set', async () => {
    await importWorkspaceFile(workspace([entry('A'), entry('B')]));
    expect(names()).toEqual(['A', 'B']);
  });

  it.each<[string, Partial<TabState>]>([
    ['running', { status: 'running' }],
    ['transient', { transient: true }],
  ])('keeps a lone empty tab that is %s', async (_label, patch) => {
    patchFirstTab(patch);
    await importWorkspaceFile(workspace([entry('A')]));
    expect(names()).toEqual(['Tab 1', 'A']);
  });

  it('activates the tab that was active in the file', async () => {
    holdAGraph();
    const result = await importWorkspaceFile(
      workspace([entry('A'), entry('B'), entry('C')], { active: 1 }),
    );
    expect(store().activeTabId).toBe(tabIdAt(result, 1));
  });

  it('counts the refused entries when it reads `active`, because the file does', async () => {
    holdAGraph();
    // `active` indexes the FILE's tabs, so an entry refused BEFORE it shifts
    // nothing: the tab that was active is still the one that comes up.
    const result = await importWorkspaceFile(
      workspace([entry('Broken', { nodes: 'oops', edges: [] }), entry('A'), entry('B')], {
        active: 2,
      }),
    );
    expect(store().activeTabId).toBe(tabIdAt(result, 2));
    expect(store().getTab(store().activeTabId)!.name).toBe('B');
  });

  it('activates the first imported tab when the active one was skipped, or none was named', async () => {
    holdAGraph();
    const skippedActive = await importWorkspaceFile(
      workspace([entry('A'), entry('Broken', { nodes: 'oops', edges: [] })], { active: 1 }),
    );
    expect(store().activeTabId).toBe(tabIdAt(skippedActive, 0));

    const noneNamed = await importWorkspaceFile(workspace([entry('C'), entry('D')]));
    expect(store().activeTabId).toBe(tabIdAt(noneNamed, 0));
  });

  it('names a tab with a blank or non-string title the way a new tab is named', async () => {
    holdAGraph();
    await importWorkspaceFile(workspace([entry('   '), entry(42)]));
    expect(names()).toEqual(['Tab 1', 'Tab 2', 'Tab 3']);
  });
});

describe('importWorkspaceFile: what an imported tab carries', () => {
  it('restores the run settings it was given, and keeps the defaults for the rest', async () => {
    const run: WorkspaceRunSettings = {
      seed: 7,
      deterministic: true,
      recordOutputs: false,
      verboseMode: true,
      weightsPersistent: false,
      backwardMode: true,
      autoBackward: true,
    };
    const result = await importWorkspaceFile(
      workspace([entry('Tuned', graphOf(), run), entry('Plain')]),
    );
    expect(store().getTab(tabIdAt(result, 0))).toMatchObject(run);
    expect(store().getTab(tabIdAt(result, 1))).toMatchObject({
      seed: null,
      deterministic: false,
      recordOutputs: true,
      verboseMode: false,
      weightsPersistent: true,
      backwardMode: false,
      autoBackward: false,
    });
  });

  it('is bound to no file, and gets a graph id of its own every time', async () => {
    // A hand-edited file cannot smuggle a binding or an id in through the graph.
    const copied = workspace([
      entry('Copy', graphOf({ graphId: 'copied-id', currentGraphFile: 'other', currentGraphName: 'Other' })),
    ]);
    const first = await importWorkspaceFile(copied);
    const second = await importWorkspaceFile(copied);
    const a = store().getTab(tabIdAt(first, 0))!;
    const b = store().getTab(tabIdAt(second, 0))!;

    expect(a.currentGraphFile).toBeNull();
    expect(a.currentGraphName).toBeNull();
    expect(a.graphId).not.toBe('copied-id');
    // Two tabs sharing an id would share trained weights on the server.
    expect(a.graphId).not.toBe(b.graphId);
  });

  it('opens read-only, with the usual notice, when its graph format is newer', async () => {
    const result = await importWorkspaceFile(
      workspace([entry('Newer', graphOf({ format_version: 99 })), entry('Also newer', graphOf({ format_version: 99 }))]),
    );
    expect(store().getTab(tabIdAt(result, 0))!.readOnly).toBe(true);
    expect(store().getTab(tabIdAt(result, 1))!.readOnly).toBe(true);
    // One notice per format version, not one per tab.
    expect(messages('warning')).toEqual([
      'Opened read-only: this graph uses a newer format (v99) than this CodefyUI build.',
    ]);
  });

  it('is stamped with no project, exactly like a freshly opened example', async () => {
    useProjectStore.setState({ projectDir: 'D:/proj', projectName: 'proj', loaded: true });
    const result = await importWorkspaceFile(workspace([entry('In a project')]));
    expect(store().getTab(tabIdAt(result, 0))!.projectOrigin).toBeNull();
  });

  it('belongs to no plugin and is not scratch, so the next export carries it', async () => {
    const result = await importWorkspaceFile(workspace([entry('Mine')]));
    expect(store().getTab(tabIdAt(result, 0))).toMatchObject({ source: null, transient: false });
  });
});

describe('importWorkspaceFile: entries that cannot open', () => {
  it('skips one bad entry, imports the rest, says so, and keeps its presets out of the palette', async () => {
    holdAGraph();
    const result = await importWorkspaceFile(
      workspace([
        entry('Good', graphOf({ presets: [{ preset_name: 'Kept' }] })),
        entry('Bad', { nodes: 'oops', edges: [], presets: [{ preset_name: 'Smuggled' }] }),
        entry('Also good'),
      ]),
    );

    expect(result.results).toEqual([
      { tabId: expect.any(String) },
      { skipped: 'invalid_graph' },
      { tabId: expect.any(String) },
    ]);
    expect(result.imported).toBe(2);
    expect(names()).toEqual(['Tab 1', 'Good', 'Also good']);
    expect(messages('warning')).toEqual(['Imported 2 of 3 tabs. Skipped: not a graph.']);
    expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual(['Kept']);
  });

  it.each([
    ['not an object', 'nope'],
    ['missing', undefined],
    ['an array', []],
    ['without edges', { nodes: [] }],
    ['holding a node that is not an object', { nodes: [null], edges: [] }],
  ])('refuses a graph that is %s', async (_label, graph) => {
    holdAGraph();
    // A literal, not `entry('Bad', graph)`: an explicit `undefined` would
    // trigger that helper's default parameter and hand over a VALID graph.
    const bad: ParsedWorkspaceTab = { title: 'Bad', graph, run: {} };
    const result = await importWorkspaceFile(workspace([bad]));
    expect(result.results).toEqual([{ skipped: 'invalid_graph' }]);
    expect(names()).toEqual(['Tab 1']);
  });

  it('imports a graph however large it is, because the export set no cap', async () => {
    // 9 MiB of description: over the plugin API's per-graph limit, which is
    // deliberately not this importer's. The 64 MiB FILE cap bounds the input.
    const result = await importWorkspaceFile(
      workspace([entry('Small'), entry('Huge', graphOf({ description: 'x'.repeat(9 * 1024 * 1024) }))]),
    );
    expect(result.imported).toBe(2);
    expect(names()).toEqual(['Small', 'Huge']);
    expect(messages('success')).toEqual(['Imported 2 tab(s).']);
  });

  it('refuses past 32 tabs BEFORE reading the entry, so its presets stay out', async () => {
    while (tabs().length < 32) store().createTab({ activate: false });
    const result = await importWorkspaceFile(
      workspace([entry('One too many', graphOf({ presets: [{ preset_name: 'Smuggled' }] }))]),
    );
    expect(result.results).toEqual([{ skipped: 'too_many_tabs' }]);
    expect(tabs()).toHaveLength(32);
    expect(useNodeDefStore.getState().presets).toEqual([]);
    expect(messages('error')).toEqual(['No tabs were imported: 32-tab limit reached.']);
  });

  it('round-trips a full 32-tab export into a fresh browser', async () => {
    // The lone empty tab is about to be closed, so it does not count.
    const full = Array.from({ length: 32 }, (_unused, i) => entry(`T${i + 1}`));
    const result = await importWorkspaceFile(workspace(full));
    expect(result.imported).toBe(32);
    expect(tabs()).toHaveLength(32);
    expect(messages('success')).toEqual(['Imported 32 tab(s).']);
  });

  it('reports one error when nothing opened, and applies no preferences', async () => {
    const lone = tabs()[0];
    const result = await importWorkspaceFile(
      workspace([entry('Bad', { nodes: 'oops', edges: [] })], {
        preferences: { fontSize: 'large', locale: 'zh-TW' },
      }),
    );
    expect(result.imported).toBe(0);
    // The lone empty tab is only closed when something took its place.
    expect(tabs()).toHaveLength(1);
    expect(tabs()[0]).toBe(lone);
    expect(messages('error')).toEqual(['No tabs were imported: not a graph.']);
    expect(useUIStore.getState().fontSize).toBe('default');
    expect(useI18n.getState().locale).toBe('en');
  });
});

describe('importWorkspaceFile: preferences', () => {
  it('applies and persists them once a tab has opened', async () => {
    await importWorkspaceFile(
      workspace([entry('A')], {
        preferences: {
          locale: 'zh-TW',
          fontSize: 'large',
          edgeStyle: 'curve',
          gridSnap: true,
          tooltips: false,
          beginnerMode: true,
        },
      }),
    );
    expect(useI18n.getState().locale).toBe('zh-TW');
    expect(localStorage.getItem('codefyui-locale')).toBe('zh-TW');
    expect(useUIStore.getState()).toMatchObject({
      fontSize: 'large',
      edgeStyle: 'curve',
      gridSnapEnabled: true,
      tooltipsEnabled: false,
      beginnerMode: true,
    });
    expect(localStorage.getItem('codefyui-font-size')).toBe('large');
    expect(localStorage.getItem('codefyui-edge-style')).toBe('curve');
    expect(localStorage.getItem('codefyui-gridsnap')).toBe('true');
    expect(localStorage.getItem('codefyui-tooltips')).toBe('false');
    expect(localStorage.getItem('codefyui-beginner-mode')).toBe('true');
    // The result toast comes after the language is applied, so it reads in
    // the language the file just switched to.
    expect(messages('success')).toEqual(['已匯入 1 個分頁。']);
  });

  it('leaves alone what the file does not mention', async () => {
    await importWorkspaceFile(workspace([entry('A')], { preferences: { fontSize: 'small' } }));
    expect(useUIStore.getState().fontSize).toBe('small');
    expect(useI18n.getState().locale).toBe('en');
    expect(localStorage.getItem('codefyui-locale')).toBeNull();
    expect(localStorage.getItem('codefyui-edge-style')).toBeNull();
  });

  it('still reports the import as a success when the browser refuses to store them', async () => {
    // Both preference writes go through `localStorage.setItem`, which throws
    // where storage is blocked or full. The tabs are open by the time that
    // happens, so reporting the import as a failure would send the user back
    // to import the same file again -- and end up with every tab twice.
    const PREFERENCE_KEYS = [
      'codefyui-locale',
      'codefyui-font-size',
      'codefyui-edge-style',
      'codefyui-gridsnap',
      'codefyui-tooltips',
      'codefyui-beginner-mode',
    ];
    const setItem = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (
      this: Storage,
      key: string,
      value: string,
    ) {
      if (PREFERENCE_KEYS.includes(key)) {
        throw new DOMException('storage is full', 'QuotaExceededError');
      }
      setItem.call(this, key, value);
    });

    const pending = importWorkspaceFile(
      workspace([entry('A')], { preferences: { fontSize: 'large', locale: 'zh-TW' } }),
    );

    await expect(pending).resolves.toMatchObject({ imported: 1 });
    expect(names()).toEqual(['A']);
    expect(messages('success')).toEqual(['Imported 1 tab(s).']);
    // A preference that could not be stored is simply not applied, and is
    // worth no toast of its own: the tabs are what the import was for.
    expect(useUIStore.getState().fontSize).toBe('default');
    expect(useI18n.getState().locale).toBe('en');
    expect(messages('warning')).toEqual([]);
    expect(messages('error')).toEqual([]);
  });
});

describe('importWorkspaceFile: node types this install lacks', () => {
  const graphWithStrangers = () =>
    graphOf({
      nodes: [
        rawNode('a', 'Add'),
        rawNode('b', 'deep:Foo'),
        rawNode('c', 'deep:Bar'),
        rawNode('d', 'Baz'),
        rawNode('e', 'deep:Foo'),
        rawNode('s', 'Start'),
        rawNode('p', 'preset:Thing'),
        rawNode('i', 'subgraph:blk'),
        { id: 'note1', type: 'note', position: { x: 0, y: 0 }, data: { noteKind: 'text', noteContent: 'hi' } },
      ],
      subgraphs: [
        {
          id: 'blk', name: 'Block', description: '',
          nodes: [rawNode('in1', 'Qux'), rawNode('in2', 'Quux'), rawNode('in3', 'Add')],
          edges: [],
          interface: { inputs: [], outputs: [], triggerTargets: [] },
        },
      ],
    });

  it('names up to three, counts the rest, and still opens the tab with placeholders', async () => {
    useNodeDefStore.setState({ definitions: [ADD], presets: [] });
    const result = await importWorkspaceFile(workspace([entry('Strangers', graphWithStrangers())]));

    expect(result.imported).toBe(1);
    expect(store().getTab(tabIdAt(result, 0))!.nodes).toHaveLength(9);
    // Distinct, in first-seen order, canvas first and block insides after.
    // Notes, Start, presets and block instances are never in the catalog.
    expect(messages('warning')).toEqual([
      'Node types not installed here: deep:Foo, deep:Bar, Baz, +2 more. They open as placeholders.',
    ]);
  });

  it('says nothing when every type is installed', async () => {
    useNodeDefStore.setState({ definitions: [ADD], presets: [] });
    await importWorkspaceFile(workspace([entry('Known')]));
    expect(messages('warning')).toEqual([]);
  });

  it('says nothing before the node catalog has loaded', async () => {
    // An empty catalog means "not loaded", not "everything is missing".
    await importWorkspaceFile(workspace([entry('Strangers', graphWithStrangers())]));
    expect(messages('warning')).toEqual([]);
  });
});
