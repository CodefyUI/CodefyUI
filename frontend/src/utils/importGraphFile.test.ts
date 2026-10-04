import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';
import { nodesBoundingBox } from './autoLayout';
import { importFile, importGraphFile } from './importGraphFile';
import { importWorkspaceFile } from './importWorkspaceFile';
import { buildWorkspaceFile } from './workspaceFile';
import { MAX_WORKSPACE_FILE_BYTES, MAX_WORKSPACE_TABS } from './workspaceLimits';
import { useNodeDefStore } from '../store/nodeDefStore';
import { NO_ACTIVE_TAB, useTabStore, whenTabsHydrated, type TabState } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import { useI18n } from '../i18n';
import type { NodeData } from '../types';

// The workspace importer runs for real everywhere below -- what the router
// hands it and what the tabs become is the point of these tests. The spy is
// here for the one test that needs it to REJECT, which no input of its own
// can make it do.
vi.mock('./importWorkspaceFile', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./importWorkspaceFile')>();
  return { ...actual, importWorkspaceFile: vi.fn(actual.importWorkspaceFile) };
});

// The same arrangement for hydration: real everywhere, where jsdom has no
// IndexedDB and it has long settled, and held open by the one test about the
// window before it does.
vi.mock('../store/tabStore', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../store/tabStore')>();
  return { ...actual, whenTabsHydrated: vi.fn(actual.whenTabsHydrated) };
});

const store = () => useTabStore.getState();
const tabs = () => useTabStore.getState().tabs;
const activeId = () => useTabStore.getState().activeTabId;
const toasts = () => useToastStore.getState().toasts;
const errorToasts = () => toasts().filter((t) => t.type === 'error');

/** A picked file holding exactly `body`. */
function jsonFile(body: string) {
  return new File([body], 'g.json', { type: 'application/json' });
}

/** A serialized node, in the shape `Export -> JSON` writes. */
function raw(id: string) {
  return { id, type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } };
}

/** A graph file the way starters are handed out: a name of its own, one node. */
function starter(over: Record<string, unknown> = {}) {
  return JSON.stringify({ name: 'Lab 3 starter', nodes: [raw('n1')], edges: [], ...over });
}

/**
 * Put a finished task in the active tab -- saved, so bound to its file -- and
 * hand back the tab as it then stands.
 */
function holdWork(): TabState {
  store().setNodes([raw('mine')] as never);
  store().setCurrentGraphFile('task-1', 'Task 1');
  return store().getActiveTab();
}

/**
 * A reader that fails the way an unreadable file makes the real one fail:
 * `error` fires and `load` never does. jsdom will happily read anything
 * handed to `new File()`, so the only way to reach that branch is to stand
 * in for the reader itself.
 */
function installFailingReader(readerError: { message: string } | null) {
  class FailingFileReader {
    error = readerError;
    onerror: (() => void) | null = null;
    onload: (() => void) | null = null;
    readAsText() {
      this.onerror?.();
    }
  }
  vi.stubGlobal('FileReader', FailingFileReader as unknown as typeof FileReader);
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({ layoutFitRequest: null });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
});

afterEach(() => {
  vi.unstubAllGlobals();
  // Back to the real hydration, and any one-shot a failed test left queued is
  // dropped rather than handed to whichever import runs next.
  vi.mocked(whenTabsHydrated).mockReset();
});

describe('importGraphFile', () => {
  it('installs the file onto the canvas', async () => {
    const body = JSON.stringify({
      nodes: [raw('n1')],
      edges: [],
      description: 'an imported graph',
      segmentGroups: [{ id: 's1' }],
    });
    expect(await importGraphFile(jsonFile(body))).toBe(true);
    expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
    expect(tabs()[0].description).toBe('an imported graph');
    expect(tabs()[0].segmentGroups).toHaveLength(1);
  });

  it('unbinds the tab, so the import cannot be saved over the file that was open', async () => {
    useTabStore.getState().setCurrentGraphFile('bound-graph', 'bound-graph');
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [] })));
    expect(tabs()[0].currentGraphFile).toBeNull();
  });

  it('merges presets the running server has never seen, keeping the ones it has', async () => {
    useNodeDefStore.setState({ presets: [{ preset_name: 'Existing' } as never] });
    const body = JSON.stringify({
      nodes: [],
      edges: [],
      presets: [{ preset_name: 'Existing' }, { preset_name: 'Fresh', nodes: [], edges: [] }],
    });
    await importGraphFile(jsonFile(body));
    expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual([
      'Existing',
      'Fresh',
    ]);
  });

  it('writes no presets at all when the file carries none', async () => {
    const before = useNodeDefStore.getState().presets;
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [raw('n1')], edges: [] })));
    expect(useNodeDefStore.getState().presets).toBe(before);
  });

  it.each([
    ['settings.device', { settings: { device: 'cuda:1' } }, 'cuda:1'],
    ['no settings block', {}, null],
    ['a device outside the pattern', { settings: { device: 'gpu' } }, null],
  ])('installs the graph device from %s', async (_label, extra, expected) => {
    useTabStore.getState().setGraphDevice('mps');
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [], ...extra })));
    expect(tabs()[0].graphDevice).toBe(expected);
  });

  it('opens a file written by a newer build read-only, and says so', async () => {
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [], format_version: 99 })));
    expect(tabs()[0].readOnly).toBe(true);
    expect(toasts().some((t) => t.type === 'warning' && t.message.includes('v99'))).toBe(true);
  });

  it('clears a stale read-only flag when an ordinary file is imported', async () => {
    useTabStore.getState().setTabReadOnly(true);
    await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [], format_version: 1 })));
    expect(tabs()[0].readOnly).toBe(false);
  });

  it('reports a file whose nodes are not a list', async () => {
    await importGraphFile(jsonFile(JSON.stringify({ nodes: 'oops', edges: [] })));
    expect(errorToasts()[0].message).toContain('Invalid graph format');
  });

  it('reports a file that is not JSON at all', async () => {
    expect(await importGraphFile(jsonFile('{not valid json'))).toBe(false);
    expect(errorToasts()).toHaveLength(1);
    expect(errorToasts()[0].message).toContain('Import failed');
  });

  // -- The two bugs the move fixed --

  describe('refusing JSON that is not a graph', () => {
    // `{}` passed every check the toolbar's import made: `data.nodes ?? []`
    // is an empty array. Picking a package.json therefore REPLACED the
    // canvas with nothing, through an install that pushes no undo frame --
    // so there was no way back to the graph that had been there.
    it.each([
      ['an empty object', '{}'],
      ['some other JSON file', JSON.stringify({ name: 'pkg', version: '1.0.0' })],
      ['a bare array', '[]'],
      ['a bare string', '"hello"'],
      ['literal null', 'null'],
      ['a number', '42'],
    ])('leaves the canvas alone for %s', async (_label, body) => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      expect(await importGraphFile(jsonFile(body))).toBe(false);
      expect(tabs()).toHaveLength(1);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['onScreen']);
      expect(errorToasts()[0].message).toContain('Not a graph file');
    });

    // The complement, and the reason the check is the KEY and not the
    // length: a graph the user really did empty is still a graph, and
    // importing it has to open it -- beside the graph on screen, which it
    // must not replace any more than a full one may.
    it('still imports a graph that genuinely holds no nodes', async () => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      expect(await importGraphFile(jsonFile(JSON.stringify({ nodes: [], edges: [] })))).toBe(true);
      expect(tabs()).toHaveLength(2);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['onScreen']);
      expect(tabs()[1].nodes).toEqual([]);
      expect(errorToasts()).toHaveLength(0);
    });
  });

  describe('an unreadable file', () => {
    // No `onerror` handler meant `readAsText` failing did nothing at all:
    // no canvas change, no message, nothing to tell the user their click
    // had been received.
    it('reports what the reader said', async () => {
      installFailingReader({ message: 'device is not readable' });
      expect(await importGraphFile(jsonFile('{}'))).toBe(false);
      expect(errorToasts()[0].message).toBe('Import failed: device is not readable');
    });

    it('still says something when the reader said nothing', async () => {
      installFailingReader(null);
      await importGraphFile(jsonFile('{}'));
      expect(errorToasts()[0].message).toBe('Import failed: Could not read the file');
    });
  });
});

describe('importFile', () => {
  /** A picked file with a name of the caller's choosing. */
  function namedFile(body: string, name: string) {
    return new File([body], name, { type: 'application/octet-stream' });
  }

  /** A valid one-tab workspace file, as parsed JSON the caller may then break. */
  function workspaceJson(): Record<string, unknown> {
    const file = buildWorkspaceFile({
      appVersion: '2.8.2',
      exportedAt: '2026-09-18T07:30:00.000Z',
      active: 0,
      tabs: [
        {
          title: 'Moved',
          graph: {
            name: 'Moved', description: '', nodes: [raw('n1')], edges: [],
            presets: [], segmentGroups: [], subgraphs: [], format_version: 1,
          },
          run: {
            seed: 7, deterministic: false, recordOutputs: true, verboseMode: false,
            weightsPersistent: true, backwardMode: false, autoBackward: false,
          },
        },
      ],
    });
    return JSON.parse(JSON.stringify(file)) as Record<string, unknown>;
  }

  /** Report a size over the cap without allocating 64 MiB. */
  function oversized(file: File): File {
    Object.defineProperty(file, 'size', { value: MAX_WORKSPACE_FILE_BYTES + 1 });
    return file;
  }

  /** A reader that counts what it is asked to read, and answers with `body`. */
  function installCountingReader(body: string): File[] {
    const reads: File[] = [];
    class CountingFileReader {
      error = null;
      onerror: (() => void) | null = null;
      onload: ((event: { target: { result: string } }) => void) | null = null;
      readAsText(file: File) {
        reads.push(file);
        this.onload?.({ target: { result: body } });
      }
    }
    vi.stubGlobal('FileReader', CountingFileReader as unknown as typeof FileReader);
    return reads;
  }

  it.each(['g.json', 'renamed.cduiworkspace'])(
    'sends a plain graph in %s down the graph path: a tab of its own beside the graph on screen',
    async (name) => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      const body = JSON.stringify({ nodes: [raw('n1')], edges: [] });
      expect(await importFile(namedFile(body, name))).toBe(true);
      expect(tabs()).toHaveLength(2);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['onScreen']);
      expect(tabs()[1].nodes.map((n) => n.id)).toEqual(['n1']);
      // The workspace importer reports every import it makes; the graph path
      // raises nothing on success.
      expect(toasts()).toEqual([]);
    },
  );

  it.each(['moved.cduiworkspace', 'renamed.json'])(
    'opens a workspace called %s as tabs: the content decides, not the name',
    async (name) => {
      useTabStore.getState().setNodes([raw('onScreen')] as never);
      expect(await importFile(namedFile(JSON.stringify(workspaceJson()), name))).toBe(true);
      expect(tabs().map((t) => t.name)).toEqual(['Tab 1', 'Moved']);
      // The graph that was on screen is still there.
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['onScreen']);
      expect(tabs()[1].nodes.map((n) => n.id)).toEqual(['n1']);
      expect(tabs()[1].seed).toBe(7);
    },
  );

  it('keeps the tab default for a run value of the wrong type', async () => {
    const json = workspaceJson();
    const [first] = json.tabs as Record<string, unknown>[];
    json.tabs = [{ ...first, run: { seed: 'seven', verboseMode: true } }];
    await importFile(namedFile(JSON.stringify(json), 'w.cduiworkspace'));
    const moved = tabs().find((t) => t.name === 'Moved')!;
    expect(moved.seed).toBeNull();
    expect(moved.verboseMode).toBe(true);
  });

  it.each(['g.json', 'w.cduiworkspace'])('reports invalid JSON in %s', async (name) => {
    expect(await importFile(namedFile('{not valid json', name))).toBe(false);
    expect(errorToasts()).toHaveLength(1);
    expect(errorToasts()[0].message).toContain('Import failed');
  });

  it('reports a file that cannot be read', async () => {
    installFailingReader({ message: 'device is not readable' });
    expect(await importFile(namedFile('{}', 'w.cduiworkspace'))).toBe(false);
    expect(errorToasts()[0].message).toBe('Import failed: device is not readable');
  });

  it('refuses a workspace written by a newer CodefyUI, and creates nothing', async () => {
    const before = tabs();
    const body = JSON.stringify({ ...workspaceJson(), version: 2 });
    expect(await importFile(namedFile(body, 'w.cduiworkspace'))).toBe(false);
    expect(tabs()).toBe(before);
    expect(errorToasts().map((t) => t.message)).toEqual([
      'This workspace was written by a newer CodefyUI. Nothing was imported.',
    ]);
  });

  it('refuses a workspace with no tabs in it', async () => {
    const body = JSON.stringify({ ...workspaceJson(), tabs: [] });
    expect(await importFile(namedFile(body, 'w.cduiworkspace'))).toBe(false);
    expect(errorToasts().map((t) => t.message)).toEqual(['Not a valid workspace file.']);
  });

  // The importer raises its own toasts for everything it can see coming, but
  // its store calls sit outside its try: a rejection would otherwise escape
  // the `void importFile(file)` the panel makes, and the user would be told
  // nothing at all.
  it('reports an importer that rejects, rather than letting it escape', async () => {
    vi.mocked(importWorkspaceFile).mockClear();
    vi.mocked(importWorkspaceFile).mockRejectedValueOnce(new Error('the tab store is gone'));
    const body = JSON.stringify(workspaceJson());
    expect(await importFile(namedFile(body, 'w.cduiworkspace'))).toBe(false);
    expect(errorToasts().map((t) => t.message)).toEqual([
      'Import failed: the tab store is gone',
    ]);
    // Called once, so the queued one-shot rejection is spent here and cannot
    // leak into whichever test runs next.
    expect(vi.mocked(importWorkspaceFile)).toHaveBeenCalledTimes(1);
  });

  describe('the 64 MiB cap', () => {
    // The second row is the extension check's case: a name off a Windows
    // disk can arrive in any case, and the check is what decides whether the
    // file is read into memory at all.
    it.each(['big.cduiworkspace', 'BIG.CDUIWORKSPACE'])(
      'refuses an oversize %s without reading it',
      async (name) => {
        const reads = installCountingReader(JSON.stringify(workspaceJson()));
        const file = oversized(namedFile('ignored', name));
        expect(await importFile(file)).toBe(false);
        expect(reads).toHaveLength(0);
        expect(errorToasts().map((t) => t.message)).toEqual([
          'The workspace file is over 64 MiB.',
        ]);
      },
    );

    it('refuses an oversize workspace under another name as soon as it sees what it is', async () => {
      const before = tabs();
      const file = oversized(namedFile(JSON.stringify(workspaceJson()), 'renamed.json'));
      expect(await importFile(file)).toBe(false);
      expect(tabs()).toBe(before);
      expect(errorToasts().map((t) => t.message)).toEqual([
        'The workspace file is over 64 MiB.',
      ]);
    });

    it('does not apply to a plain graph, whose import is unchanged', async () => {
      const file = oversized(jsonFile(JSON.stringify({ nodes: [raw('n1')], edges: [] })));
      expect(await importFile(file)).toBe(true);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
    });
  });
});

/**
 * #550. A graph file used to be installed over the active tab: no question, no
 * undo step, and autosave wrote the result within 250 ms. A student who
 * imported the next task's starter into the tab holding the last task lost it.
 */
describe('where an imported graph goes', () => {
  // Both doors, because they are one placement: neither may be the one that
  // still replaces work.
  describe.each([
    ['importFile', importFile],
    ['importGraphFile', importGraphFile],
  ])('%s', (_door, importer) => {
    it('opens beside a tab with work in it, in a new tab it makes active', async () => {
      const mine = holdWork();
      expect(await importer(jsonFile(starter()))).toBe(true);
      expect(tabs()).toHaveLength(2);
      // The very same object: no write so much as touched the tab with the
      // work in it, its binding included.
      expect(tabs()[0]).toBe(mine);
      const opened = tabs()[1];
      expect(activeId()).toBe(opened.id);
      expect(opened.nodes.map((n) => n.id)).toEqual(['n1']);
      expect(opened.name).toBe('Lab 3 starter');
      expect(opened.currentGraphFile).toBeNull();
    });

    it('fills the active tab when it is empty, and opens no other', async () => {
      const [empty] = tabs();
      expect(await importer(jsonFile(starter()))).toBe(true);
      expect(tabs().map((t) => t.id)).toEqual([empty.id]);
      expect(activeId()).toBe(empty.id);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
      // Not renamed: the import into an existing tab keeps the label on it.
      expect(tabs()[0].name).toBe('Tab 1');
    });
  });

  it('counts a tab holding nothing but a note as work', async () => {
    const note = {
      id: 'memo',
      type: 'noteNode',
      position: { x: 0, y: 0 },
      data: { label: 'Note', type: 'note', params: {}, noteContent: 'what the task asked' },
    };
    store().setNodes([note] as never);
    const mine = store().getActiveTab();
    await importFile(jsonFile(starter()));
    expect(tabs()).toHaveLength(2);
    expect(tabs()[0]).toBe(mine);
  });

  it('counts the graph waiting outside an open block, though the canvas in front is empty', async () => {
    // Standing inside a block whose insides were deleted: `nodes` is empty --
    // all the empty-canvas overlay looks at -- while the whole graph waits one
    // level up.
    const outside = [raw('outer')] as unknown as Node<NodeData>[];
    useTabStore.setState({
      tabs: tabs().map((t) => ({
        ...t,
        nodes: [],
        subgraphStack: [
          {
            subgraphId: 'blk',
            nodes: outside,
            edges: [],
            presets: [],
            undoStack: [],
            redoStack: [],
            selectedNodeId: null,
            subgraphs: [],
            segmentGroups: [],
            activeSegment: null,
          },
        ],
      })),
    });
    const mine = store().getActiveTab();
    await importFile(jsonFile(starter()));
    expect(tabs()).toHaveLength(2);
    expect(tabs()[0]).toBe(mine);
  });

  it.each<[string, Partial<TabState>]>([
    // A reload would take the import away with it: the tab is never saved.
    ['a tab a plugin opened for this session only', { transient: true }],
    // Kept across reloads, but its strip entry and accessible name say
    // "opened by" the plugin, and would go on saying it over the user's graph.
    ['a tab a plugin opened and kept', { source: { kind: 'agent-variant', pluginId: 'copilot' } }],
    ['a tab that is running', { status: 'running' }],
  ])('does not fill %s, though it is empty', async (_label, patch) => {
    useTabStore.setState({ tabs: tabs().map((t) => ({ ...t, ...patch })) });
    const theirs = store().getActiveTab();
    expect(await importFile(jsonFile(starter()))).toBe(true);
    expect(tabs()).toHaveLength(2);
    expect(tabs()[0]).toBe(theirs);
    expect(activeId()).toBe(tabs()[1].id);
    expect(tabs()[1].source).toBeNull();
  });

  describe('the view', () => {
    /** Resolves after the frames an import that opened a tab waits out. */
    const twoFrames = () =>
      new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));

    // A new tab's first-visit fit uses the canvas size React Flow last
    // measured. Opening the tab can resize the canvas -- the config panel of
    // a node selected in the tab before closes -- and a fit asked for at once
    // is consumed against that stale size too: the graph lands small, at the
    // left edge. So the request waits until the canvas has been measured.
    it('is fitted to the graph in a new tab once the canvas has its new size', async () => {
      holdWork();
      await importFile(jsonFile(starter()));
      expect(useUIStore.getState().layoutFitRequest).toBeNull();
      await twoFrames();
      expect(useUIStore.getState().layoutFitRequest).toEqual({
        bounds: nodesBoundingBox(tabs()[1].nodes),
      });
    });

    it('is not moved for a new tab the user has already left', async () => {
      const mine = holdWork();
      await importFile(jsonFile(starter()));
      store().setActiveTab(mine.id);
      await twoFrames();
      expect(useUIStore.getState().layoutFitRequest).toBeNull();
    });

    // A filled tab is already on screen, at the size it had, and keeps the
    // view it had, which after a pan can show nothing of the graph that just
    // arrived.
    it('is fitted to the graph a filled tab now holds', async () => {
      const far = { ...raw('far'), position: { x: 2400, y: 1800 } };
      await importFile(jsonFile(JSON.stringify({ nodes: [far], edges: [] })));
      expect(useUIStore.getState().layoutFitRequest).toEqual({
        bounds: nodesBoundingBox(tabs()[0].nodes),
      });
      expect(useUIStore.getState().layoutFitRequest?.bounds).toMatchObject({ x: 2400, y: 1800 });
    });

    it('is left alone when the graph that filled the tab has no nodes', async () => {
      await importFile(jsonFile(JSON.stringify({ nodes: [], edges: [] })));
      expect(useUIStore.getState().layoutFitRequest).toBeNull();
    });
  });

  it('opens a tab when none is open', async () => {
    useTabStore.setState({ tabs: [], activeTabId: NO_ACTIVE_TAB });
    expect(await importFile(jsonFile(starter()))).toBe(true);
    expect(tabs()).toHaveLength(1);
    expect(activeId()).toBe(tabs()[0].id);
    expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
    expect(tabs()[0].name).toBe('Lab 3 starter');
  });

  it.each([
    ['no name', {}],
    ['a blank name', { name: '   ' }],
    ['a name that is not a string', { name: 42 }],
  ])('names a new tab after the file when the graph has %s', async (_label, over) => {
    holdWork();
    const body = JSON.stringify({ nodes: [raw('n1')], edges: [], ...over });
    await importFile(new File([body], 'unet_no_skip.json', { type: 'application/json' }));
    expect(tabs()[1].name).toBe('unet_no_skip');
  });

  it('opens a file written by a newer build read-only in the new tab, not in the tab with work', async () => {
    const mine = holdWork();
    await importFile(jsonFile(starter({ format_version: 99 })));
    expect(tabs()[0]).toBe(mine);
    expect(tabs()[0].readOnly).toBe(false);
    expect(tabs()[1].readOnly).toBe(true);
    expect(toasts().some((t) => t.type === 'warning' && t.message.includes('v99'))).toBe(true);
  });

  describe('the tab limit', () => {
    /** Open tabs in the background until the editor holds the most it will. */
    function fillToTheLimit() {
      while (tabs().length < MAX_WORKSPACE_TABS) store().createTab({ activate: false });
    }

    it('refuses a graph that needs a new tab, and changes nothing', async () => {
      holdWork();
      fillToTheLimit();
      const before = useTabStore.getState();
      const palette = useNodeDefStore.getState().presets;
      // Reading the graph would merge this preset into the palette.
      const body = starter({ presets: [{ preset_name: 'Smuggled', nodes: [], edges: [] }] });
      expect(await importFile(jsonFile(body))).toBe(false);
      expect(tabs()).toBe(before.tabs);
      expect(activeId()).toBe(before.activeTabId);
      expect(useNodeDefStore.getState().presets).toBe(palette);
      expect(errorToasts().map((t) => t.message)).toEqual([
        '32-tab limit reached. Close a tab, then import again.',
      ]);
    });

    it('still fills an empty active tab, which needs no new one', async () => {
      fillToTheLimit();
      expect(await importFile(jsonFile(starter()))).toBe(true);
      expect(tabs()).toHaveLength(MAX_WORKSPACE_TABS);
      expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['n1']);
      expect(errorToasts()).toHaveLength(0);
    });
  });

  describe('a file it refuses leaves no tab behind', () => {
    it.each([
      ['not a graph', '{}'],
      ['nodes that are not a list', JSON.stringify({ nodes: 'oops', edges: [] })],
      ['a node the reader cannot resolve', JSON.stringify({ nodes: [null], edges: [] })],
      ['not JSON at all', '{not valid json'],
    ])('beside a tab with work: %s', async (_label, body) => {
      const mine = holdWork();
      expect(await importFile(jsonFile(body))).toBe(false);
      expect(tabs()).toHaveLength(1);
      expect(tabs()[0]).toBe(mine);
      expect(errorToasts()).toHaveLength(1);
    });

    it('with no tab open', async () => {
      useTabStore.setState({ tabs: [], activeTabId: NO_ACTIVE_TAB });
      expect(await importFile(jsonFile('{}'))).toBe(false);
      expect(tabs()).toEqual([]);
    });
  });

  it('waits for the saved tabs to load before deciding where the graph goes', async () => {
    // What hydration is about to install: last session's tab, holding work.
    const placeholder = { tabs: tabs(), activeTabId: activeId() };
    const savedId = store().createTab({ title: 'Saved' });
    store().setNodes([raw('saved')] as never);
    const saved = { tabs: tabs().filter((t) => t.id === savedId), activeTabId: savedId };
    // Until then the screen holds only the empty tab the editor boots with.
    useTabStore.setState(placeholder);

    let settle!: () => void;
    vi.mocked(whenTabsHydrated).mockReturnValueOnce(
      new Promise((resolve) => {
        settle = () => resolve('loaded');
      }),
    );
    const importing = importFile(jsonFile(starter()));
    await vi.waitFor(() => expect(whenTabsHydrated).toHaveBeenCalled());
    // Hydration lands, writing the tabs wholesale as it does.
    useTabStore.setState(saved);
    settle();

    expect(await importing).toBe(true);
    expect(tabs().map((t) => t.name)).toEqual(['Saved', 'Lab 3 starter']);
    expect(tabs()[0].nodes.map((n) => n.id)).toEqual(['saved']);
    expect(tabs()[1].nodes.map((n) => n.id)).toEqual(['n1']);
  });
});
