/**
 * `TabState.savedRevision` -- the revision at which a tab's document last
 * matched a file (#596).
 *
 * The tab's close button asks before it discards a graph (#331), and it asked
 * about every tab holding one, because nothing recorded whether the graph on
 * screen was still the one that had been opened or saved. A student who saved
 * and then closed was told the work "cannot be undone", which reads as "the
 * save did not work".
 *
 * The record rides on the content-based `revision` (#341): every install of a
 * document and every successful save writes down the revision the tab is at,
 * and anything that moves the revision afterwards makes the tab ask again.
 * These tests pin both halves -- what keeps a tab matched, and what does not --
 * through the public actions, the way the editor drives them.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';
import {
  useTabStore,
  tabHasUnsavedWork,
  _buildPersistedTabForTesting,
  _tabFromPersistedForTesting,
} from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import type { NodeData, NodeDefinition } from '../types';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

/** A node type with one param the node list has a default for. */
function def(name: string): NodeDefinition {
  return {
    node_name: name, category: 'Layer', description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [
      { name: 'size', param_type: 'int', default: 8, description: '', options: [], min_value: 1, max_value: 64 },
    ],
  };
}

/** A node the way a saved file can store it: without `size`. */
function stored(id: string, x = 0): Node<NodeData> {
  return {
    id, type: 'baseNode', position: { x, y: 0 },
    data: { label: id, type: 'A', params: {} },
  } as Node<NodeData>;
}

/** Open a saved graph into the active tab, as the Graphs panel does. */
function openSaved(nodes: Node<NodeData>[] = [stored('a'), stored('b', 200)]): void {
  store().loadGraphDocumentInto(tab().id, {
    nodes, edges: [], boundFile: 'Exam', boundName: 'Exam',
  });
}

beforeEach(() => {
  // No node list until a test hands one over, as at boot before /api/nodes
  // answers. Reset first: a change to the list fills the open tabs.
  useNodeDefStore.setState({ definitions: [] } as never);
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
});

describe('a tab matched to a file (#596)', () => {
  it('a new tab has matched no file, so a node dropped into it is unsaved work', () => {
    expect(tab().savedRevision).toBeNull();
    store().setNodes([stored('a')]);
    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('an empty tab has nothing to lose', () => {
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('opening a graph records the revision the open leaves the tab at, in the open\'s one emission', () => {
    store().setNodes([stored('old')]);
    const before = tab().revision;
    let emissions = 0;
    const unsubscribe = useTabStore.subscribe(() => {
      emissions += 1;
    });

    openSaved();
    unsubscribe();

    // Recorded inside the install's own commit: a second `set` would show
    // subscribers the new graph as "unsaved" for one emission, and the
    // install is one emission by contract (`tabStore.loadDocument.test.ts`).
    expect(emissions).toBe(1);
    expect(tab().revision).toBe(before + 1);
    expect(tab().savedRevision).toBe(tab().revision);
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('a saved graph that leaves params out still matches what was opened once the open fills them', () => {
    // The report #596 started from: the defaults the open writes in are not
    // something the user did.
    useNodeDefStore.setState({ definitions: [def('A')] } as never);

    openSaved();

    expect(tab().nodes[0].data.params).toEqual({ size: 8 });
    expect(tab().savedRevision).toBe(tab().revision);
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('a re-install that changes nothing moves no revision and still records the match', () => {
    // A Source Control reload of a file identical to the canvas, say. The
    // record has to be the revision the commit leaves, not "one more".
    store().setNodes([stored('a')]);
    const revision = tab().revision;
    expect(tabHasUnsavedWork(tab())).toBe(true);

    openSaved(tab().nodes);

    expect(tab().revision).toBe(revision);
    expect(tab().savedRevision).toBe(revision);
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('records only the tab the graph went into, a background one included', () => {
    store().setNodes([stored('mine')]);
    const front = tab().id;
    // The plugin API and the workspace importer open into tabs they did not
    // activate.
    const back = store().createTab({ title: 'Background', activate: false });
    store().loadGraphDocumentInto(back, { nodes: [stored('a')], edges: [], boundFile: null });

    const backTab = store().getTab(back)!;
    expect(backTab.savedRevision).toBe(backTab.revision);
    expect(store().getTab(front)!.savedRevision).toBeNull();
  });

  it('defaults the node list fills in after the open are not an edit either', () => {
    // The list can answer after a tab is open (#556); the open then resolved
    // its nodes against an empty list, and the list's arrival fills them.
    openSaved();
    expect(tab().nodes[0].data.params).toEqual({});
    const opened = tab().revision;

    useNodeDefStore.setState({ definitions: [def('A')] } as never);

    expect(tab().nodes[0].data.params).toEqual({ size: 8 });
    // The document changed, so a plugin holding the old revision hears of it,
    expect(tab().revision).toBe(opened + 1);
    // but nobody edited it.
    expect(tab().savedRevision).toBe(tab().revision);
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('defaults filled into a tab already edited since the open leave it unsaved', () => {
    openSaved();
    store().onNodesChange([{ id: 'a', type: 'position', position: { x: 40, y: 0 }, dragging: false }]);
    expect(tabHasUnsavedWork(tab())).toBe(true);

    useNodeDefStore.setState({ definitions: [def('A')] } as never);

    expect(tab().nodes[0].data.params).toEqual({ size: 8 });
    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('markTabSaved records the revision it is given and moves none', () => {
    store().setNodes([stored('a')]);
    const revision = tab().revision;

    store().markTabSaved(tab().id, revision);

    expect(tab().savedRevision).toBe(revision);
    // No revision move, so no plugin is told the graph changed.
    expect(tab().revision).toBe(revision);
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('marked at an older revision -- an edit landed during the save -- the tab stays unsaved', () => {
    store().setNodes([stored('a')]);
    const serialized = tab().revision;
    store().setNodes([stored('a'), stored('b', 200)]);

    store().markTabSaved(tab().id, serialized);

    expect(tab().savedRevision).toBe(serialized);
    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('a match made while bound lapses when the binding is dropped: the file no longer holds the graph', () => {
    // The Graphs panel's delete, a rename onto another graph's file, and a
    // Save As in another tab that wrote over the file all unbind the tab
    // (`rebindGraphFile(file, null)`) and leave its graph nowhere on disk.
    openSaved();
    expect(tabHasUnsavedWork(tab())).toBe(false);

    store().rebindGraphFile('Exam', null);

    expect(tab().currentGraphFile).toBeNull();
    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('a rename keeps the match: the file still holds the graph, under its new name', () => {
    openSaved();

    store().rebindGraphFile('Exam', { file: 'Exam_v2', name: 'Exam v2' });

    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('a graph opened bound to nothing (an import, an example) stays matched while unbound', () => {
    store().loadGraphDocumentInto(tab().id, { nodes: [stored('a')], edges: [], boundFile: null });

    expect(tab().currentGraphFile).toBeNull();
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('a save records the file it saved to, so losing that binding voids the match too', () => {
    store().setNodes([stored('a')]);
    // What `saveActiveGraph` does on success: bind, then mark.
    store().setTabGraphFile(tab().id, 'Exam', 'Exam');
    store().markTabSaved(tab().id, tab().revision);
    expect(tabHasUnsavedWork(tab())).toBe(false);

    store().rebindGraphFile('Exam', null);

    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('forgetTabMatch makes the listed tabs ask again, until a reload records the match anew', () => {
    openSaved();
    const id = tab().id;
    const other = store().createTab({ title: 'Other', activate: false });
    store().loadGraphDocumentInto(other, {
      nodes: [stored('x')], edges: [], boundFile: 'Other', boundName: 'Other',
    });
    const revision = tab().revision;

    store().forgetTabMatch([id]);

    expect(tab().savedRevision).toBeNull();
    expect(tab().savedFile).toBeNull();
    // No revision move, so no plugin is told the graph changed.
    expect(tab().revision).toBe(revision);
    expect(tabHasUnsavedWork(tab())).toBe(true);
    // Only the tabs it names.
    expect(tabHasUnsavedWork(store().getTab(other)!)).toBe(false);

    // What the Source Control reload does: install the file again.
    openSaved();
    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('forgetTabMatch writes nothing when no listed tab holds a match', () => {
    store().setNodes([stored('a')]);
    let emissions = 0;
    const unsubscribe = useTabStore.subscribe(() => {
      emissions += 1;
    });

    store().forgetTabMatch([tab().id, 'no-such-tab']);
    unsubscribe();

    expect(emissions).toBe(0);
  });

  it('any edit after the open counts, and undoing it does not make the tab match again', () => {
    openSaved();
    store().updateNodeParams('a', { size: 3 });
    expect(tabHasUnsavedWork(tab())).toBe(true);

    store().undo();

    // The file's content again, but the store cannot tell that from a new
    // graph, and asking is the answer that loses nothing.
    expect(tab().nodes[0].data.params).toEqual({});
    expect(tabHasUnsavedWork(tab())).toBe(true);
  });

  it('selecting, measuring, renaming the tab and running it are not edits', () => {
    openSaved();
    const id = tab().id;

    store().selectNodeExclusively('a');
    store().onNodesChange([
      { id: 'b', type: 'dimensions', dimensions: { width: 214, height: 88 }, setAttributes: true },
    ]);
    store().renameTab(id, 'Renamed');
    store().setTabStatus(id, 'running');
    store().setNodeExecutionStatus('a', 'completed');
    store().setTabStatus(id, 'completed');

    expect(tabHasUnsavedWork(tab())).toBe(false);
  });

  it('is never persisted, and a restored tab has matched no file, even rebuilt over one that had', () => {
    openSaved();
    const live = tab();
    expect(live.savedRevision).toBe(live.revision);

    const record = _buildPersistedTabForTesting(live);
    expect('savedRevision' in record).toBe(false);

    // Hydration rebuilds a tab over the live one with the same id, and the
    // record can carry the very revision the live tab was matched at -- yet
    // its graph is the record's, which no file is known to hold.
    const restored = _tabFromPersistedForTesting(record, live);
    expect(restored.revision).toBe(live.revision);
    expect(restored.savedRevision).toBeNull();
    expect(restored.savedFile).toBeNull();
    expect(tabHasUnsavedWork(restored)).toBe(true);
    expect('savedFile' in record).toBe(false);
  });
});
