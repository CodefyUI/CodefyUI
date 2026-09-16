import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Node } from '@xyflow/react';
import type { NodeData } from '../types';

vi.mock('../api/rest', () => ({
  saveGraph: vi.fn().mockResolvedValue({}),
  listGraphs: vi.fn().mockResolvedValue([]),
}));
vi.mock('./dialog', () => ({
  prompt: vi.fn(),
  confirm: vi.fn().mockResolvedValue(true),
}));

import { saveActiveGraph } from './saveActiveGraph';
import { saveGraph, listGraphs } from '../api/rest';
import { confirm, prompt } from './dialog';
import { useTabStore } from '../store/tabStore';
import { useProjectStore } from '../store/projectStore';
import { useToastStore } from '../store/toastStore';
import { setGraphsWriteListener } from './graphsWrite';
import { sanitizeGraphName } from './index';
import { buildInstanceNode } from './subgraph';

function freshTab() {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
}

/**
 * The body of the last `saveGraph` call -- the request as it was sent.
 *
 * Read as a bag of keys rather than through `objectContaining`, because half
 * of what these tests pin is a key that must NOT be there: an unbound save
 * sends no `file` at all, and `objectContaining` cannot say so.
 */
function lastSaveBody(): Record<string, unknown> {
  const calls = vi.mocked(saveGraph).mock.calls;
  expect(calls.length, 'saveGraph was never called').toBeGreaterThan(0);
  return calls[calls.length - 1][0] as unknown as Record<string, unknown>;
}

/** One node on a canvas, so a save has something identifiable to write. */
function node(id: string): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Add', params: {} },
  } as Node<NodeData>;
}

beforeEach(() => {
  vi.clearAllMocks();
  // Per-project storage keys (e.g. last-saved-graph memory) leak across the
  // '/proj'/'/a'/'/b' describe blocks below without this (issue #88).
  localStorage.clear();
  freshTab();
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
});

describe('saveActiveGraph', () => {
  // This used to assert the opposite -- "non-project mode ALWAYS prompts
  // (legacy behavior)" -- and that is the behaviour the toolbar's save icon
  // could not live with: a graph opened from the Graphs panel with no
  // project directory open asked for its name on every single save, then
  // asked again whether it should overwrite the file it had been read from.
  // The binding, not the project directory, decides now.
  it('a bound tab outside project mode overwrites its own file with no prompt', async () => {
    useTabStore.getState().setCurrentGraphFile('bound', 'bound');
    await saveActiveGraph();
    expect(prompt).not.toHaveBeenCalled();
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'bound' }));
  });

  it('Save As on a bound tab outside project mode still prompts', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('bound-copy');
    useTabStore.getState().setCurrentGraphFile('bound', 'bound');
    await saveActiveGraph({ saveAs: true });
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'bound-copy' }));
  });

  // The regression guard on the other side of the new rule: dropping the
  // project-mode half of the condition must not turn "save" into "silently
  // write somewhere" for a canvas that has never had a file of its own.
  it('an unbound tab outside project mode still prompts for a name', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('brand-new');
    await saveActiveGraph();
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'brand-new' }));
  });

  it('project mode + bound overwrites IN PLACE with no prompt', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier', 'classifier');
    await saveActiveGraph();
    expect(prompt).not.toHaveBeenCalled();
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'classifier' }));
  });

  it('project mode Save As prompts even when bound', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier', 'classifier');
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('copy');
    await saveActiveGraph({ saveAs: true });
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'copy' }));
  });

  it('project mode + unbound (gallery/import) prompts', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('fresh');
    await saveActiveGraph();
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'fresh' }));
  });

  // #200 item 9. The binding is the whole of what decides "overwrite in
  // place, no prompt", so a document installed with no binding has to take
  // the prompt path even when the tab was bound a moment earlier -- which is
  // what opening an example into a tab bound to a file now does.
  it('project mode: a document opened with no file binding prompts instead of overwriting the file that was open', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier', 'classifier');
    // Exactly what `openExample` hands the store now.
    useTabStore.getState().loadGraphDocument({ nodes: [], edges: [], boundFile: null });
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('from-template');

    await saveActiveGraph();

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'from-template' }));
    expect(saveGraph).not.toHaveBeenCalledWith(expect.objectContaining({ name: 'classifier' }));
  });

  it('empty prompt aborts the save', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('');
    await saveActiveGraph();
    expect(saveGraph).not.toHaveBeenCalled();
  });

  // core#137 review, CRITICAL 1. The payload used to omit `subgraphs`
  // entirely, so saving a graph with a collapsed block wrote out the
  // instance node and none of its contents. This is the cheap half of the
  // guard; `saveActiveGraph.wire.test.ts` is the half that matters, because
  // it checks the real request body rather than a mocked call argument.
  //
  // The instance node is NOT decoration in this fixture: serialization prunes
  // definitions nothing on the canvas can reach (review finding 9), so a
  // definition with no instance would be dropped on purpose and this test
  // would be asserting against a state the app cannot produce.
  it('forwards subgraph definitions to saveGraph', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('with-blocks');
    const definition = {
      id: 'blk',
      name: 'Block',
      description: '',
      nodes: [],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    useTabStore.getState().setSubgraphs([definition]);
    useTabStore.getState().setNodes([
      buildInstanceNode(definition, { x: 0, y: 0 }, 'blk-1'),
    ]);
    await saveActiveGraph();
    expect(saveGraph).toHaveBeenCalledWith(
      expect.objectContaining({ subgraphs: [definition] }),
    );
  });
});

describe('saveActiveGraph cross-project guard (ID10)', () => {
  it('refuses to save a tab whose origin differs from the open project', async () => {
    useProjectStore.setState({ projectDir: '/b', projectName: 'b', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier', 'classifier');
    useTabStore.getState().stampActiveTabProject('/a'); // belongs to project A
    await saveActiveGraph();
    expect(saveGraph).not.toHaveBeenCalled();
  });

  it('stamps the origin after a successful project save', async () => {
    useProjectStore.setState({ projectDir: '/b', projectName: 'b', loaded: true });
    useTabStore.getState().setCurrentGraphFile('classifier', 'classifier');
    await saveActiveGraph();
    const st = useTabStore.getState();
    const tab = st.tabs.find((t) => t.id === st.activeTabId)!;
    expect(tab.projectOrigin).toBe('/b');
  });
});

describe('saveActiveGraph read-only guard (ID8)', () => {
  it('refuses to save a read-only (newer-format) graph', async () => {
    useTabStore.getState().setTabReadOnly(true);
    await saveActiveGraph();
    expect(saveGraph).not.toHaveBeenCalled();
  });

  // ID8 fast-follow (task 16 review Adjudication B / Important finding 1):
  // clear() must reset readOnly, otherwise a cleared (fresh, empty) graph is
  // stuck refusing Save forever even though it is trivially current-format.
  it('clear() resets readOnly so a subsequent save is no longer refused', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('fresh-after-clear');
    useTabStore.getState().setTabReadOnly(true);

    useTabStore.getState().clear();

    await saveActiveGraph();
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'fresh-after-clear' }));
  });
});

/**
 * The sidebar's Graphs panel reads the saved-graph list once and then shows
 * it; nothing else tells it that a save happened. Without this signal a
 * graph saved from the toolbar was simply missing from the sidebar -- and an
 * in-place save of a graph already in the list is worse, because the list
 * looks right while the row's timestamp is stale.
 *
 * The negative cases are the half worth pinning: announcing a write that did
 * not happen makes the panel re-read the server on every refused, cancelled
 * or failed save, which is a flicker the user has no way to explain.
 */
describe('saveActiveGraph announces a saved-graph write', () => {
  // Typed by its initialiser rather than annotated: `ReturnType<typeof
  // vi.fn>` widens to a mock that is also constructable, which the
  // listener slot rightly refuses.
  let heard = vi.fn(() => undefined);

  beforeEach(() => {
    heard = vi.fn(() => undefined);
    setGraphsWriteListener(heard);
  });

  afterEach(() => {
    setGraphsWriteListener(null);
  });

  it('says so after a save the server accepted', async () => {
    useTabStore.getState().setCurrentGraphFile('bound', 'bound');
    await saveActiveGraph();

    expect(saveGraph).toHaveBeenCalledTimes(1);
    expect(heard).toHaveBeenCalledTimes(1);
  });

  // Outside a project too: the panel lists saved graphs wherever they live,
  // so gating this on project mode -- the way the worktree signal beside it
  // is gated -- would leave the common case unannounced.
  it('says so for a prompted save with no project directory open', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('named-by-hand');
    await saveActiveGraph();

    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'named-by-hand' }));
    expect(heard).toHaveBeenCalledTimes(1);
  });

  it('says nothing when the save failed', async () => {
    vi.mocked(saveGraph).mockRejectedValueOnce(new Error('disk full'));
    useTabStore.getState().setCurrentGraphFile('bound', 'bound');
    await saveActiveGraph();

    expect(saveGraph).toHaveBeenCalledTimes(1);
    expect(heard).not.toHaveBeenCalled();
  });

  it('says nothing when the user cancelled the name prompt', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(null);
    await saveActiveGraph();

    expect(saveGraph).not.toHaveBeenCalled();
    expect(heard).not.toHaveBeenCalled();
  });

  it('says nothing when a read-only tab refused the save outright', async () => {
    useTabStore.getState().setTabReadOnly(true);
    useTabStore.getState().setCurrentGraphFile('bound', 'bound');
    await saveActiveGraph();

    expect(saveGraph).not.toHaveBeenCalled();
    expect(heard).not.toHaveBeenCalled();
  });
});

/**
 * An in-place save used to write the tab's BINDING back to the server as the
 * graph's name, and the binding is the sanitized file stem. `POST
 * /api/graph/save` stores the whole payload, name included, and `GET
 * /api/graph/list` reports that stored name -- so sending the stem renamed the
 * graph: "My Graph" became "My_Graph" in the Graphs panel and in the toast the
 * first time the toolbar's save icon was pressed. Every title with a space in
 * it was renamed by being saved.
 *
 * The name now travels on the tab, beside the stem, because the stem cannot
 * be turned back into it and the server cannot reliably be asked: a
 * case-insensitive filesystem leaves the listed file and the tab's stem
 * differing by case, and `GET /api/graph/list` answers 409 in a project
 * holding a base in both its legacy and its canonical form.
 */
describe('saveActiveGraph in-place saves under the name the TAB carries', () => {
  it.each([
    ['a space', 'My Graph', 'My_Graph'],
    ['a slash', 'My Graph/v2', 'My_Graph_v2'],
    ['a dot', 'model.v2', 'model_v2'],
    ['CJK and a space', '我的圖表 v2', '我的圖表_v2'],
  ])('sends the display name, not the stem, for a name holding %s', async (_what, name, stem) => {
    useTabStore.getState().setCurrentGraphFile(stem, name);

    await saveActiveGraph();

    expect(prompt).not.toHaveBeenCalled();
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name }));
    expect(saveGraph).not.toHaveBeenCalledWith(expect.objectContaining({ name: stem }));
    // And without asking the server anything. The list read this path used to
    // make was three bugs at once -- a case-only miss, a 409 in project mode,
    // and an await that let the user switch tabs before the canvas was read.
    expect(listGraphs).not.toHaveBeenCalled();
  });

  it('writes the NEW title after the Graphs panel renamed the graph', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    // What `handleRename` does once the server answers with the stem it wrote.
    useTabStore.getState().rebindGraphFile('alpha', {
      file: 'Renamed_Graph', name: 'Renamed Graph',
    });

    await saveActiveGraph();

    expect(prompt).not.toHaveBeenCalled();
    expect(saveGraph).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'Renamed Graph' }),
    );
    // Not the old title, which would rename the graph back by saving it.
    expect(saveGraph).not.toHaveBeenCalledWith(expect.objectContaining({ name: 'alpha' }));
  });

  it('asks for a name again after the Graphs panel deleted the graph', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'Alpha Graph');
    useTabStore.getState().rebindGraphFile('alpha', null);
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('alpha-again');

    await saveActiveGraph();

    // The file is gone, so an in-place save would silently recreate what the
    // user just deleted -- under a name the tab is the last holder of.
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'alpha-again' }));
  });

  it('keeps the binding on the tab, so the next save is in place too', async () => {
    useTabStore.getState().setCurrentGraphFile('My_Graph', 'My Graph');

    await saveActiveGraph();
    await saveActiveGraph();

    const tab = useTabStore.getState().getActiveTab();
    expect(tab.currentGraphFile).toBe('My_Graph');
    expect(tab.currentGraphName).toBe('My Graph');
    expect(vi.mocked(saveGraph).mock.calls.map(([body]) => (body as { name: string }).name))
      .toEqual(['My Graph', 'My Graph']);
  });
});

/**
 * The shape a tab restored from a record 2.8.0 wrote comes back in: a file,
 * and no name, because that build persisted the stem alone.
 *
 * Falling back to the stem for these would rename exactly the graphs the fix
 * is for -- they are the tabs holding the titles that needed sanitizing. They
 * are asked once instead, and never again.
 *
 * What the answer MEANS is the part that changed with `file`. While the typed
 * name chose the address too, this prompt forked the graph: type anything but
 * the file's exact original title and a new file was written, the bound one
 * left stale, the tab silently moved onto the copy. The address is pinned to
 * the binding now, so the answer is only a title.
 */
describe('saveActiveGraph on a tab bound to a file whose name it does not know', () => {
  it('asks for a name once, then saves in place under it without asking again', async () => {
    // The file is in the list, so a collision guard on this path would fire
    // on the typed name -- which is one reason this path runs none: the
    // question is "what is this file called?", not "where should this go?".
    vi.mocked(listGraphs).mockResolvedValue([{ name: 'My Graph', file: 'My_Graph' }]);
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('My Graph');
    useTabStore.getState().setCurrentGraphFile('My_Graph', null);

    await saveActiveGraph();

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(listGraphs).not.toHaveBeenCalled();
    expect(confirm).not.toHaveBeenCalled();
    expect(lastSaveBody()).toMatchObject({ file: 'My_Graph', name: 'My Graph' });
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.currentGraphFile).toBe('My_Graph');
    expect(tab.currentGraphName).toBe('My Graph');

    // Second save: the tab now carries both halves, so it is silent.
    await saveActiveGraph();

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(lastSaveBody()).toMatchObject({ file: 'My_Graph', name: 'My Graph' });
  });

  // The fork, pinned from the other side: the user types something that is
  // NOT the file's old title -- a typo, a better name, or simply what they
  // think the graph is called. That used to write `Something_Else.json` and
  // leave `My_Graph.json` holding the older copy this tab was showing.
  it('saves to the BOUND file even when the typed title sanitizes elsewhere', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('Something Else');
    useTabStore.getState().setCurrentGraphFile('My_Graph', null);

    await saveActiveGraph();

    expect(lastSaveBody()).toMatchObject({ file: 'My_Graph', name: 'Something Else' });
    const tab = useTabStore.getState().getActiveTab();
    // Still on the same file, now with a title to save it under.
    expect(tab.currentGraphFile).toBe('My_Graph');
    expect(tab.currentGraphName).toBe('Something Else');
  });

  it('writes nothing when the one-time question is cancelled', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(null);
    useTabStore.getState().setCurrentGraphFile('My_Graph', null);

    await saveActiveGraph();

    expect(saveGraph).not.toHaveBeenCalled();
    // And the tab is left exactly as it was, so the next Save asks again
    // rather than writing under a title nobody agreed to.
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.currentGraphFile).toBe('My_Graph');
    expect(tab.currentGraphName).toBeNull();
  });

  it('never writes the stem as the name on the way there', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('My Graph');
    useTabStore.getState().setCurrentGraphFile('My_Graph', null);

    await saveActiveGraph();

    expect(saveGraph).not.toHaveBeenCalledWith(expect.objectContaining({ name: 'My_Graph' }));
  });
});

/**
 * WHERE a save lands.
 *
 * `POST /api/graph/save` took one string for both jobs: `name` was sanitized
 * into the file stem AND stored inside the file as the graph's title. Nothing
 * can be both, and the two rounds of fixing this from the frontend alone each
 * traded one bug for the other -- sending the stem renamed every graph whose
 * title needed sanitizing, sending the title wrote to `sanitize(title)`,
 * which for a file whose stem is not `sanitize(its own stored title)` is A
 * DIFFERENT FILE. The address travels as `file` now, and these are the
 * assertions that say so.
 */
describe('saveActiveGraph targets the file the tab is bound to', () => {
  it('sends the bound stem as `file` and the title as `name`, with no dialog', async () => {
    useTabStore.getState().setCurrentGraphFile('My_Graph', 'My Graph');

    await saveActiveGraph();

    expect(prompt).not.toHaveBeenCalled();
    expect(lastSaveBody()).toMatchObject({ file: 'My_Graph', name: 'My Graph' });
  });

  // The failure this change exists for, reproduced end to end against the
  // real backend: the tab is bound to `Beta`, the graph inside it is called
  // "Alpha", and a save whose address came from the title overwrote
  // `Alpha.json` -- a file this tab had never been near -- while `Beta.json`
  // kept the older copy. In project mode `write_graph_pair`'s legacy path
  // deleted `Alpha.json` outright. The assertion is on `file` because `name`
  // was never what broke here.
  it('writes the bound file when the title sanitizes to a different one', async () => {
    useTabStore.getState().setCurrentGraphFile('Beta', 'Alpha');

    await saveActiveGraph();

    expect(lastSaveBody().file).toBe('Beta');
    expect(lastSaveBody().name).toBe('Alpha');
    // And the tab did not drift onto the file its title names, either.
    expect(useTabStore.getState().getActiveTab().currentGraphFile).toBe('Beta');
  });

  it.each([
    ['a space', 'My_Graph', 'My Graph'],
    ['CJK and a space', '我的圖表_v2', '我的圖表 v2'],
    ['nothing in common with the stem', 'graph-7', 'Quarterly Report'],
  ])('keeps targeting the bound file for a title with %s', async (_what, file, name) => {
    useTabStore.getState().setCurrentGraphFile(file, name);

    await saveActiveGraph();

    expect(prompt).not.toHaveBeenCalled();
    expect(lastSaveBody()).toMatchObject({ file, name });
  });

  // The other half of the contract: an omitted `file` is the server's
  // original behaviour, byte for byte. A graph named a moment ago at the
  // prompt has no address yet, and the server deriving one from the name is
  // exactly what it wants.
  it('sends NO `file` for an unbound tab', async () => {
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('brand-new');

    await saveActiveGraph();

    expect('file' in lastSaveBody()).toBe(false);
    expect(lastSaveBody().name).toBe('brand-new');
  });

  it('sends NO `file` for Save As, and still runs the collision guard', async () => {
    vi.mocked(listGraphs).mockResolvedValue([{ name: 'Taken', file: 'Taken' }]);
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('Taken');
    useTabStore.getState().setCurrentGraphFile('My_Graph', 'My Graph');

    await saveActiveGraph({ saveAs: true });

    // Save As is how a graph is deliberately written somewhere else, so it is
    // the one path where the typed name still chooses the address -- and
    // therefore the one path that must ask before landing on somebody else's
    // file.
    expect(listGraphs).toHaveBeenCalledTimes(1);
    expect(confirm).toHaveBeenCalledTimes(1);
    expect('file' in lastSaveBody()).toBe(false);
    expect(lastSaveBody().name).toBe('Taken');
  });
});

/**
 * A save writes the tab it STARTED from.
 *
 * The prompted path has always awaited `prompt()` and then `listGraphs()`
 * before reading the canvas, and it read the canvas through the ACTIVE tab --
 * so switching tabs while the name dialog was open serialized the other tab's
 * graph and wrote it to disk under the name typed for this one, with a success
 * toast and nothing on screen to say otherwise. Verified against the code
 * before this change: a two-tab probe saved 'B-node' under the name typed
 * while 'A-node' was in front of the user.
 */
describe('saveActiveGraph writes the tab the save started from', () => {
  it('serializes the starting tab, with the edits made to it while the dialog was open', async () => {
    useTabStore.getState().setNodes([node('A-node')]);
    const alpha = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('beta');
    const beta = useTabStore.getState().activeTabId;
    useTabStore.getState().setNodes([node('B-node')]);
    useTabStore.getState().setActiveTab(alpha);

    (prompt as unknown as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      // The seconds the dialog costs are seconds the canvas is still live:
      // this edit is part of what the user asked to save.
      useTabStore.getState().setNodes([node('A-node'), node('A-node-2')]);
      useTabStore.getState().setActiveTab(beta);
      return 'typed-name';
    });

    await saveActiveGraph();

    const body = vi.mocked(saveGraph).mock.calls[0][0] as { nodes: { id: string }[] };
    expect(body.nodes.map((n) => n.id)).toEqual(['A-node', 'A-node-2']);
  });

  it('binds and stamps the starting tab, leaving the tab now in front alone', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    const alpha = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('beta');
    const beta = useTabStore.getState().activeTabId;
    useTabStore.getState().setActiveTab(alpha);

    (prompt as unknown as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      useTabStore.getState().setActiveTab(beta);
      return 'typed-name';
    });

    await saveActiveGraph();

    const tabOf = (id: string) => useTabStore.getState().tabs.find((tb) => tb.id === id)!;
    expect(tabOf(alpha).currentGraphFile).toBe('typed-name');
    expect(tabOf(alpha).currentGraphName).toBe('typed-name');
    expect(tabOf(alpha).projectOrigin).toBe('/proj');
    // The tab the user moved to saved nothing, so it owns nothing.
    expect(tabOf(beta).currentGraphFile).toBeNull();
    expect(tabOf(beta).currentGraphName).toBeNull();
    expect(tabOf(beta).projectOrigin).toBeNull();
  });

  it('writes nothing at all when the starting tab is closed while the dialog is open', async () => {
    useToastStore.setState({ toasts: [] });
    const heard = vi.fn(() => undefined);
    setGraphsWriteListener(heard);
    useTabStore.getState().setNodes([node('A-node')]);
    const alpha = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('beta');
    useTabStore.getState().setNodes([node('B-node')]);
    useTabStore.getState().setActiveTab(alpha);

    (prompt as unknown as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      useTabStore.getState().removeTab(alpha);
      return 'typed-name';
    });

    await saveActiveGraph();

    // The graph the user asked to save no longer exists. The one that does
    // must not be written to disk under its name.
    expect(saveGraph).not.toHaveBeenCalled();
    expect(heard).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toEqual([]);
    setGraphsWriteListener(null);
  });
});

/**
 * A confirmed Save As is allowed to write over a graph another tab is showing.
 * That tab's binding then names a file holding somebody else's graph: its next
 * Save is promptless and in place, so it would put its own older graph back
 * over the work just saved -- and the Graphs panel's "already open" branch
 * would raise it, not the tab that owns the file now.
 */
describe('saveActiveGraph Save As over a file another tab is bound to', () => {
  it('leaves only the active tab bound to it', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    const stale = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('second');
    const saver = useTabStore.getState().activeTabId;
    vi.mocked(listGraphs).mockResolvedValueOnce([{ name: 'alpha', file: 'alpha' }]);
    (prompt as unknown as ReturnType<typeof vi.fn>).mockResolvedValue('alpha');

    await saveActiveGraph({ saveAs: true });

    expect(saveGraph).toHaveBeenCalledWith(expect.objectContaining({ name: 'alpha' }));
    const fileOf = (id: string) =>
      useTabStore.getState().tabs.find((t) => t.id === id)!.currentGraphFile;
    expect(fileOf(saver)).toBe('alpha');
    expect(fileOf(stale)).toBeNull();
  });
});

/**
 * The app warns that two names are one file, and then treats them as two.
 *
 * `findGraphNameCollision` folds case on purpose -- NTFS and APFS are
 * case-INSENSITIVE, and a silent overwrite on the majority platform is the
 * worse failure. `rebindGraphFile` compares stems with `===`, also on purpose
 * -- on a case-SENSITIVE filesystem `My_Graph` and `my_graph` really are two
 * different files, and folding them there would raise the wrong tab from the
 * Graphs panel and mark the wrong row Current. Each is right on its own; the
 * gap between them is a tab that still believes it owns a file somebody else
 * has just written over:
 *
 *   1. tab A saves "My Graph", the server writes `My_Graph.json`, tab A binds
 *      `My_Graph`
 *   2. tab B's Save As of "my graph" matches that row case-insensitively, the
 *      "will be overwritten" confirm fires, the user accepts
 *   3. the save is an `os.replace`, so the directory ENTRY is re-spelled to
 *      `my_graph.json`; tab B binds `my_graph`, and
 *      `rebindGraphFile('my_graph', null, B)` never matches tab A's `My_Graph`
 *   4. tab A's next Save is promptless, sends `file: "My_Graph"`, and on NTFS
 *      lands on that same physical file -- destroying tab B's graph with no
 *      dialog and no warning
 *
 * Reproduced end to end over real HTTP by an adversarial reviewer, in both
 * non-project and project mode.
 */
describe('saveActiveGraph after a CONFIRMED overwrite of another tab\'s file', () => {
  const tabOf = (id: string) => useTabStore.getState().tabs.find((tb) => tb.id === id)!;

  /** Tab A bound to `My_Graph`, with an unbound tab B in front of the user. */
  function twoTabs(): { a: string; b: string } {
    useTabStore.getState().setCurrentGraphFile('My_Graph', 'My Graph');
    const a = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('second');
    return { a, b: useTabStore.getState().activeTabId };
  }

  /** The reviewer's step 2-3: the guard matches, and the entry is re-spelled. */
  function caseSkewedOverwrite(): void {
    vi.mocked(listGraphs).mockResolvedValue([{ name: 'My Graph', file: 'My_Graph' }]);
    vi.mocked(prompt).mockResolvedValue('my graph');
    vi.mocked(saveGraph).mockResolvedValueOnce({
      message: 'Graph saved', path: '/graphs/my_graph.json', file: 'my_graph',
    });
  }

  it('clears the binding of the tab holding the file the guard actually matched', async () => {
    const { a, b } = twoTabs();
    caseSkewedOverwrite();

    await saveActiveGraph({ saveAs: true });

    // The user was asked, and said yes -- so this really is tab B's file now.
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(tabOf(b).currentGraphFile).toBe('my_graph');
    // And tab A, whose stem differs from the saved one only in case, is the
    // tab `rebindGraphFile(savedFile, ...)` cannot see.
    expect(tabOf(a).currentGraphFile).toBeNull();
    expect(tabOf(a).currentGraphName).toBeNull();
  });

  it('leaves that tab asking for a name on its next Save, instead of writing', async () => {
    const { a } = twoTabs();
    caseSkewedOverwrite();

    await saveActiveGraph({ saveAs: true });

    useTabStore.getState().setActiveTab(a);
    vi.mocked(prompt).mockClear();
    vi.mocked(saveGraph).mockClear();
    // Cancelled, so the assertion below is about the QUESTION being asked at
    // all -- which is the whole difference between this and the data loss.
    vi.mocked(prompt).mockResolvedValue(null);

    await saveActiveGraph();

    // Unbound, so the one-click Save is a question. Before the fix it was a
    // promptless `{file: "My_Graph", name: "My Graph"}` -- the same physical
    // file on NTFS, now holding tab B's graph.
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(saveGraph).not.toHaveBeenCalled();
  });

  // The ordinary case, which `rebindGraphFile(savedFile, ...)` has always
  // covered: both stems are spelled the same, so the `===` matches. Pinned
  // here so the case-skewed clear above cannot be mistaken for the only one.
  it('still clears a tab whose stem matches the saved one exactly', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    const a = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('second');
    const b = useTabStore.getState().activeTabId;
    vi.mocked(listGraphs).mockResolvedValue([{ name: 'alpha', file: 'alpha' }]);
    vi.mocked(prompt).mockResolvedValue('alpha');
    vi.mocked(saveGraph).mockResolvedValueOnce({
      message: 'Graph saved', path: '/graphs/alpha.json', file: 'alpha',
    });

    await saveActiveGraph({ saveAs: true });

    expect(confirm).toHaveBeenCalledTimes(1);
    expect(tabOf(b).currentGraphFile).toBe('alpha');
    expect(tabOf(a).currentGraphFile).toBeNull();
  });

  // The other side of it: clearing bindings is destructive to the tab it is
  // done to -- it turns a one-click Save into a dialog -- so a save that
  // overwrote nothing must leave every other tab exactly as it found it.
  it('clears nothing when the Save As collided with no saved graph', async () => {
    const { a, b } = twoTabs();
    vi.mocked(listGraphs).mockResolvedValue([{ name: 'My Graph', file: 'My_Graph' }]);
    vi.mocked(prompt).mockResolvedValue('brand new');
    vi.mocked(saveGraph).mockResolvedValueOnce({
      message: 'Graph saved', path: '/graphs/brand_new.json', file: 'brand_new',
    });

    await saveActiveGraph({ saveAs: true });

    expect(confirm).not.toHaveBeenCalled();
    expect(tabOf(a).currentGraphFile).toBe('My_Graph');
    expect(tabOf(a).currentGraphName).toBe('My Graph');
    expect(tabOf(b).currentGraphFile).toBe('brand_new');
  });
});

/**
 * WHICH STEM the tab is bound to after a save.
 *
 * It has to be the one the server reports, because the two sanitizers do not
 * agree: the frontend's `sanitizeGraphName` tests `[\p{L}\p{N}]` against
 * Node's ICU tables, the backend's `_sanitize_name` uses CPython's
 * `str.isalnum()`, and the two carry different Unicode versions -- they
 * differ on 16 BMP code points, plus astral ones. The write TARGET is safe
 * either way, since the server re-sanitizes whatever `file` it is handed; what
 * breaks is the binding, which ends up naming a stem `GET /api/graph/list`
 * will never report. Every exact-match comparison in the app then misses it,
 * the collision clear above included.
 */
describe('saveActiveGraph binds to the stem the server reports', () => {
  it('uses the returned `file`, not the stem re-derived from the typed name', async () => {
    vi.mocked(prompt).mockResolvedValue('My Graph');
    // Nothing like the local answer, so the assertion cannot pass by
    // coincidence the way a real one-codepoint divergence might.
    vi.mocked(saveGraph).mockResolvedValueOnce({
      message: 'Graph saved', path: '/graphs/server_spelling.json', file: 'server_spelling',
    });

    await saveActiveGraph();

    const tab = useTabStore.getState().getActiveTab();
    expect(tab.currentGraphFile).toBe('server_spelling');
    expect(tab.currentGraphFile).not.toBe(sanitizeGraphName('My Graph'));
    // Only the address came from the server. The title is still what was
    // typed -- the server never derives one from the other.
    expect(tab.currentGraphName).toBe('My Graph');
  });

  it('falls back to the local stem when the backend reports none', async () => {
    vi.mocked(prompt).mockResolvedValue('My Graph');
    // A frontend built from source can meet a backend older than the `file`
    // field, which is the same reason the Graphs panel's rename carries a
    // fallback. A missing address is a degraded binding, not a crash.
    vi.mocked(saveGraph).mockResolvedValueOnce({
      message: 'Graph saved', path: '/graphs/My_Graph.json',
    });

    await saveActiveGraph();

    expect(useTabStore.getState().getActiveTab().currentGraphFile).toBe('My_Graph');
  });

  it('falls back to the BOUND file when an older backend answers an in-place save', async () => {
    useTabStore.getState().setCurrentGraphFile('Beta', 'Alpha');
    vi.mocked(saveGraph).mockResolvedValueOnce({ message: 'Graph saved', path: '/graphs/Beta.json' });

    await saveActiveGraph();

    // Never `sanitizeGraphName('Alpha')`: a binding that followed the title
    // is the bug that put `file` on the request in the first place.
    expect(useTabStore.getState().getActiveTab().currentGraphFile).toBe('Beta');
  });
});
