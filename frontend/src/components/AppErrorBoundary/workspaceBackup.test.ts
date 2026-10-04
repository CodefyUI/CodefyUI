import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';
import type { Edge, Node } from '@xyflow/react';
import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useProjectStore } from '../../store/projectStore';
import {
  _resetTabPersistenceForTests,
  readSnapshot,
  writeSnapshot,
} from '../../store/tabPersistence';
import { autosaveScope, useTabStore, type PersistedTab, type TabState } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import type { NodeData } from '../../types';
import { buildNoteNode } from '../../utils';
import { rememberAppVersion } from '../../utils/appVersion';
import { collectWorkspaceFile } from '../../utils/exportWorkspace';
import { _resetIdbForTests, idbGetKeysByPrefix } from '../../utils/idb';
import { importWorkspaceFile } from '../../utils/importWorkspaceFile';
import { parseWorkspaceFile } from '../../utils/workspaceFile';
import { buildWorkspaceBackup, readAutosavedTabs, startEmptyWorkspace } from './workspaceBackup';

// Passed through, so a case can make one write resolve without writing: what
// a page that does not hold the editing lock gets (#554).
vi.mock('../../store/tabPersistence', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../store/tabPersistence')>();
  return {
    ...actual,
    readSnapshot: vi.fn(actual.readSnapshot),
    writeSnapshot: vi.fn(actual.writeSnapshot),
  };
});
const realReadSnapshot = vi.mocked(readSnapshot).getMockImplementation()!;
const realWriteSnapshot = vi.mocked(writeSnapshot).getMockImplementation()!;

/**
 * The recovery screen's storage work: the backup, read back from storage and
 * written as a `.cduiworkspace` file, and the empty workspace a crash loop is
 * left by.
 */

const BASE_SCOPE = 'codefyui-tabs';
const NOW = new Date('2026-10-03T08:00:00.000Z');
const ASIDE = `${BASE_SCOPE}-aside::${NOW.getTime()}`;
const EMPTY_BLOB = { activeTabId: '', tabs: [] };
const store = () => useTabStore.getState();

function flowNode(id: string, type = 'Add', params: Record<string, unknown> = {}): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 40, y: 80 }, data: { label: id, type, params } };
}

/** A record with only the fields every build has written. */
function record(id: string, name: string, over: Partial<PersistedTab> = {}): PersistedTab {
  return { id, name, nodes: [flowNode(`${id}-n1`)], edges: [], ...over };
}

/** The localStorage tier: one JSON blob per scope. Returns what it wrote. */
function seedLocalStorage(scope: string, tabs: unknown[], activeTabId: string): string {
  const blob = JSON.stringify({ activeTabId, tabs });
  localStorage.setItem(scope, blob);
  return blob;
}

/** A tab holding a graph, through the store's own load path. Returns its id. */
function tabWith(title: string, nodes: Node<NodeData>[], edges: Edge[] = []): string {
  const id = store().createTab({ title, activate: false });
  store().loadGraphDocumentInto(id, {
    nodes,
    edges,
    boundFile: null,
    description: `${title} notes`,
    device: 'cpu',
  });
  return id;
}

/** Give this page an IndexedDB: an empty one, or a stand-in that fails. */
function useIndexedDb(factory: unknown = new IDBFactory()): void {
  vi.stubGlobal('indexedDB', factory);
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
  _resetIdbForTests();
  _resetTabPersistenceForTests();
}

const BROKEN_IDB = {
  open: () => {
    throw new Error('IndexedDB is broken');
  },
};

/** What the screen's Download backup builds, minus the download. */
async function backup() {
  return buildWorkspaceBackup(await readAutosavedTabs(), NOW);
}

function titles(built: ReturnType<typeof buildWorkspaceBackup>): unknown[] {
  return (built.file?.tabs ?? []).map((tab) => tab.title);
}

async function names(scope: string): Promise<string[] | null> {
  const snapshot = await readSnapshot(scope);
  return snapshot === null ? null : snapshot.tabs.map((tab) => tab.name);
}

beforeEach(() => {
  // Autosave is a 250 ms debounce on setTimeout. Frozen, so nothing writes to
  // storage unless a test moves the clock to let the real autosave run.
  // IndexedDB (fake-indexeddb) and FileReader run on setImmediate instead.
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  localStorage.clear();
  rememberAppVersion(null);
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useProjectStore.setState({ projectDir: null });
  useTabStore.setState({ tabs: [], activeTabId: '' });
});

afterEach(() => {
  vi.mocked(readSnapshot).mockImplementation(realReadSnapshot);
  vi.mocked(writeSnapshot).mockImplementation(realWriteSnapshot);
  vi.restoreAllMocks();
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  _resetIdbForTests();
  _resetTabPersistenceForTests();
  localStorage.clear();
});

describe('buildWorkspaceBackup', () => {
  it('holds what autosave wrote, serialized the way Export Workspace serializes those tabs', async () => {
    const model = tabWith(
      'Model',
      [
        flowNode('n1', 'Linear', { in_features: 4 }),
        flowNode('n2', 'ReLU'),
        buildNoteNode({ id: 'note1', text: 'why ReLU', position: { x: 0, y: 200 } }),
      ],
      [{ id: 'e1', source: 'n1', target: 'n2', sourceHandle: 'output', targetHandle: 'input' }],
    );
    store().setTabRunSettings(model, { seed: 7, deterministic: true });
    tabWith('Data', [flowNode('d1', 'Dataset')]);
    store().setActiveTab(model);
    // The real autosave, not a hand-made record.
    vi.advanceTimersByTime(250);
    expect(localStorage.getItem(BASE_SCOPE)).not.toBeNull();

    const built = await backup();

    const exported = collectWorkspaceFile(NOW).file!;
    expect(built.file!.tabs).toEqual(exported.tabs);
    expect(built.file!.active).toBe(exported.active);
    expect(built.file!.exported_at).toBe(NOW.toISOString());
    expect(built).toMatchObject({ skippedReadOnly: 0, unreadable: 0 });
    expect(parseWorkspaceFile(JSON.parse(JSON.stringify(built.file))).ok).toBe(true);
  });

  it('reads the scope autosave writes to, a project scope included', async () => {
    useProjectStore.setState({ projectDir: 'D:/courses/lab3' });
    tabWith('Project graph', [flowNode('p1')]);
    vi.advanceTimersByTime(250);
    // The base scope belongs to another session, not to this one.
    seedLocalStorage(BASE_SCOPE, [record('other', 'Base graph')], 'other');

    expect(localStorage.getItem(autosaveScope())).not.toBeNull();
    expect(titles(await backup())).toEqual(['Project graph']);
  });

  it('reads storage, never the tab store: the store may be what threw', async () => {
    seedLocalStorage(BASE_SCOPE, [record('p1', 'Saved')], 'p1');
    // Serializing this would throw, and the backup must not try.
    useTabStore.setState({
      tabs: [{ id: 'broken' } as unknown as TabState],
      activeTabId: 'broken',
    });

    const built = await backup();

    expect(titles(built)).toEqual(['Saved']);
    expect(built.file!.active).toBe(0);
  });

  it('reads the IndexedDB tier in tab-strip order, ahead of an older localStorage copy', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('b', 'Second'), record('a', 'First')], 'a');
    // What localStorage held on migration day; a reload ignores it too.
    seedLocalStorage(BASE_SCOPE, [record('old', 'Migration day')], 'old');

    const built = await backup();

    expect(titles(built)).toEqual(['Second', 'First']);
    expect(built.file!.active).toBe(1);
  });

  it('treats an empty workspace in IndexedDB as the answer, not as a miss', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [], '');
    seedLocalStorage(BASE_SCOPE, [record('old', 'Migration day')], 'old');

    expect(await backup()).toEqual({ file: null, skippedReadOnly: 0, unreadable: 0 });
  });

  it('falls back to localStorage when IndexedDB holds nothing for this scope', async () => {
    useIndexedDb();
    seedLocalStorage(BASE_SCOPE, [record('p1', 'Not migrated yet')], 'p1');

    expect(titles(await backup())).toEqual(['Not migrated yet']);
  });

  it('falls back to localStorage when IndexedDB cannot be read, since autosave then writes there', async () => {
    useIndexedDb(BROKEN_IDB);
    seedLocalStorage(BASE_SCOPE, [record('p1', 'Fallback tier')], 'p1');

    expect(titles(await backup())).toEqual(['Fallback tier']);
  });

  it('rejects with the reason when no tier can be read', async () => {
    useIndexedDb(BROKEN_IDB);
    await expect(readAutosavedTabs()).rejects.toThrow('IndexedDB is broken');

    vi.unstubAllGlobals();
    localStorage.setItem(BASE_SCOPE, '{not json');
    await expect(readAutosavedTabs()).rejects.toThrow(SyntaxError);
  });

  it('answers with no file when no autosaved tab holds a graph', async () => {
    const nothing = { file: null, skippedReadOnly: 0, unreadable: 0 };
    expect(await backup()).toEqual(nothing);

    seedLocalStorage(BASE_SCOPE, [record('e', 'Empty', { nodes: [] })], 'e');
    expect(await backup()).toEqual(nothing);
  });

  it('leaves out read-only tabs and records it cannot read, counts both, and keeps the rest', async () => {
    seedLocalStorage(
      BASE_SCOPE,
      [
        record('ro', 'Newer format', { readOnly: true }),
        null,
        record('bad', 'Damaged', { nodes: [null] as unknown as Node<NodeData>[] }),
        record('ok', 'Kept'),
      ],
      'ok',
    );

    const built = await backup();

    expect(titles(built)).toEqual(['Kept']);
    // Re-indexed: an index into what is IN the file.
    expect(built.file!.active).toBe(0);
    expect(built).toMatchObject({ skippedReadOnly: 1, unreadable: 2 });
  });

  it('counts a record JSON cannot write as unreadable instead of losing the whole file', async () => {
    // IndexedDB stores structured clones, which hold what JSON cannot.
    useIndexedDb();
    const cyclic: Record<string, unknown> = { a: 1 };
    cyclic.self = cyclic;
    await writeSnapshot(
      BASE_SCOPE,
      [
        record('g', 'Good'),
        record('c', 'Cyclic', { nodes: [flowNode('c1', 'Add', { weird: cyclic })] }),
        record('n', 'BigInt', { nodes: [flowNode('n1', 'Add', { big: BigInt(7) })] }),
      ],
      'g',
    );

    const built = await backup();

    expect(titles(built)).toEqual(['Good']);
    expect(built.unreadable).toBe(2);
    expect(() => JSON.stringify(built.file)).not.toThrow();
  });

  it('writes text where a damaged record holds something else, and keeps its graph', async () => {
    // A name the tab strip cannot render is one way a stored workspace
    // crashes every load; the file must not carry it on.
    seedLocalStorage(
      BASE_SCOPE,
      [
        record('bad', { broken: true } as unknown as string, {
          description: 42 as unknown as string,
        }),
      ],
      'bad',
    );

    const [entry] = (await backup()).file!.tabs;

    // A blank title is the importer's cue for the default tab name.
    expect(entry.title).toBe('');
    expect(entry.graph.name).toBe('graph');
    expect(entry.graph.description).toBe('');
    expect(entry.graph.nodes).toHaveLength(1);
  });

  it('writes a file Import opens with the node types and parameters intact', async () => {
    seedLocalStorage(
      BASE_SCOPE,
      [record('p1', 'Lab 3', { nodes: [flowNode('n1', 'Linear', { in_features: 4 })] })],
      'p1',
    );
    const built = await backup();

    const parsed = parseWorkspaceFile(JSON.parse(JSON.stringify(built.file)));
    if (!parsed.ok) throw new Error(`refused: ${parsed.reason}`);
    const { results } = await importWorkspaceFile(parsed.workspace);

    const opened = store().getTab((results[0] as { tabId: string }).tabId)!;
    expect(opened.name).toBe('Lab 3');
    expect(opened.nodes.map((n) => [n.data.type, n.data.params])).toEqual([
      ['Linear', { in_features: 4 }],
    ]);
  });
});

describe('startEmptyWorkspace', () => {
  const NOW2 = new Date('2026-10-03T09:00:00.000Z');
  const NOW3 = new Date('2026-10-03T10:00:00.000Z');
  const ASIDE2 = `${BASE_SCOPE}-aside::${NOW2.getTime()}`;
  const ASIDE3 = `${BASE_SCOPE}-aside::${NOW3.getTime()}`;
  const OLD_ASIDE = `${BASE_SCOPE}-aside::1000`;

  /** A workspace blob over half of jsdom's 5,000,000-unit localStorage quota. */
  function seedBigWorkspace(): string {
    const note = buildNoteNode({ id: 'big', text: 'x'.repeat(2_600_000), position: { x: 0, y: 0 } });
    return seedLocalStorage(BASE_SCOPE, [record('a', 'Lab 1', { nodes: [note] })], 'a');
  }

  const asideKeys = () => Object.keys(localStorage).filter((key) => key.includes('-aside::'));

  it('keeps a copy in IndexedDB, empties both tiers, and checks what a reload would restore', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('a', 'Lab 1'), record('b', 'Lab 2')], 'b');
    // What localStorage held on migration day: not the work, so not copied.
    seedLocalStorage(BASE_SCOPE, [record('old', 'Migration day')], 'old');

    expect(await startEmptyWorkspace(NOW)).toBe('reset');

    // The copy, under a scope of its own: nothing the user had is deleted.
    expect(await names(ASIDE)).toEqual(['Lab 1', 'Lab 2']);
    expect((await readSnapshot(ASIDE))!.activeTabId).toBe('b');
    expect(asideKeys()).toEqual([]);
    // The workspace itself is empty in both tiers...
    expect(await readSnapshot(BASE_SCOPE)).toEqual({ tabs: [], activeTabId: '' });
    expect(JSON.parse(localStorage.getItem(BASE_SCOPE)!)).toEqual(EMPTY_BLOB);
    // ...which is what the next load restores.
    expect((await readAutosavedTabs()).records).toEqual([]);
  });

  it('keeps the oldest copy and the newest in IndexedDB, and deletes those in between', async () => {
    useIndexedDb();
    for (const [now, name] of [[NOW, 'Lab 1'], [NOW2, 'Lab 2'], [NOW3, 'Lab 3']] as const) {
      await writeSnapshot(BASE_SCOPE, [record(name, name)], name);
      expect(await startEmptyWorkspace(now)).toBe('reset');
    }

    expect(await names(ASIDE)).toEqual(['Lab 1']);
    expect(await idbGetKeysByPrefix(`${ASIDE2}|`)).toEqual([]);
    expect(await names(ASIDE3)).toEqual(['Lab 3']);
  });

  it('keeps the oldest copy: after an Import it alone holds the tabs the backup left out', async () => {
    useIndexedDb();
    // The original workspace: one tab a backup carries, one it leaves out.
    await writeSnapshot(
      BASE_SCOPE,
      [record('a', 'Lab'), record('r', 'Reference', { readOnly: true })],
      'a',
    );
    expect(await startEmptyWorkspace(NOW)).toBe('reset');
    // The backup is imported (new ids, no read-only tab), and the crash comes back.
    await writeSnapshot(BASE_SCOPE, [record('a2', 'Lab')], 'a2');

    expect(await startEmptyWorkspace(NOW2)).toBe('reset');

    expect(await names(ASIDE)).toEqual(['Lab', 'Reference']);
    expect(await names(ASIDE2)).toEqual(['Lab']);
  });

  it('needs its IndexedDB copy to read back even when a complete backup was downloaded', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    vi.mocked(writeSnapshot).mockImplementation(async (scope, records, activeTabId) =>
      scope === BASE_SCOPE ? realWriteSnapshot(scope, records, activeTabId) : undefined,
    );

    expect(await startEmptyWorkspace(NOW, { completeBackup: true })).toBe('not_changed');

    expect(await names(BASE_SCOPE)).toEqual(['Lab 1']);
  });

  it('keeps its copy, and fails, when reading back the emptied workspace fails', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('a', 'Lab')], 'a');
    // The empty write lands; the read that would confirm it does not.
    let scopeReads = 0;
    vi.mocked(readSnapshot).mockImplementation(async (scope) => {
      if (scope === BASE_SCOPE && ++scopeReads === 2) throw new Error('transient read failure');
      return realReadSnapshot(scope);
    });

    await expect(startEmptyWorkspace(NOW)).rejects.toThrow('transient read failure');

    vi.mocked(readSnapshot).mockImplementation(realReadSnapshot);
    expect(await names(ASIDE)).toEqual(['Lab']);
  });

  it('keeps its localStorage copy when IndexedDB, unreadable at the start, answers at the check', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('i', 'Older in IndexedDB')], 'i');
    const blob = seedLocalStorage(BASE_SCOPE, [record('l', 'Newer in localStorage')], 'l');
    let scopeReads = 0;
    vi.mocked(readSnapshot).mockImplementation(async (scope) => {
      if (scope === BASE_SCOPE && ++scopeReads === 1) throw new Error('transient read failure');
      return realReadSnapshot(scope);
    });

    // localStorage is emptied, then IndexedDB answers with tabs: what a
    // reload restores did not change, but localStorage did.
    expect(await startEmptyWorkspace(NOW)).toBe('not_changed');

    expect(localStorage.getItem(ASIDE)).toBe(blob);
  });

  it('deletes no copy when emptying throws part way, since the workspace may be empty already', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    // IndexedDB is emptied and passes its own check; the final read of what a
    // reload would restore then fails in both tiers.
    let emptyReads = 0;
    vi.mocked(readSnapshot).mockImplementation(async (scope) => {
      const snapshot = await realReadSnapshot(scope);
      if (scope === BASE_SCOPE && snapshot?.tabs.length === 0 && ++emptyReads > 1) {
        throw new Error('IndexedDB went away');
      }
      return snapshot;
    });
    const getItem = Storage.prototype.getItem;
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(function (this: Storage, key) {
      if (key === BASE_SCOPE) throw new DOMException('denied', 'SecurityError');
      return getItem.call(this, key);
    });

    await expect(startEmptyWorkspace(NOW)).rejects.toThrow('denied');

    vi.mocked(readSnapshot).mockImplementation(realReadSnapshot);
    expect(await names(ASIDE)).toEqual(['Lab 1']);
  });

  it('changes nothing, older copies included, when the copy does not land even if the workspace could be written', async () => {
    useIndexedDb();
    await writeSnapshot(OLD_ASIDE, [record('o', 'Older copy')], 'o');
    await writeSnapshot(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    const blob = seedLocalStorage(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    vi.mocked(writeSnapshot).mockImplementation(async (scope, records, activeTabId) =>
      scope === BASE_SCOPE ? realWriteSnapshot(scope, records, activeTabId) : undefined,
    );

    expect(await startEmptyWorkspace(NOW)).toBe('not_changed');

    expect(await names(BASE_SCOPE)).toEqual(['Lab 1']);
    expect(await names(OLD_ASIDE)).toEqual(['Older copy']);
    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
  });

  it('changes nothing in a page whose writes all resolve without writing', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    const blob = seedLocalStorage(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    vi.mocked(writeSnapshot).mockImplementation(async () => {});

    expect(await startEmptyWorkspace(NOW)).toBe('not_changed');

    expect(await names(BASE_SCOPE)).toEqual(['Lab 1']);
    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
  });

  it('answers not_changed, leaves localStorage alone and takes its copy back, when the empty workspace does not land', async () => {
    useIndexedDb();
    await writeSnapshot(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    const blob = seedLocalStorage(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    // The copy aside lands; the write over the workspace itself does not.
    vi.mocked(writeSnapshot).mockImplementation(async (scope, records, activeTabId) =>
      scope === BASE_SCOPE ? undefined : realWriteSnapshot(scope, records, activeTabId),
    );

    expect(await startEmptyWorkspace(NOW)).toBe('not_changed');

    expect(await idbGetKeysByPrefix(`${BASE_SCOPE}-aside::`)).toEqual([]);
    expect(await names(BASE_SCOPE)).toEqual(['Lab 1']);
    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
  });

  it('works on the localStorage tier alone', async () => {
    const blob = seedLocalStorage(BASE_SCOPE, [record('a', 'Lab 1')], 'a');

    expect(await startEmptyWorkspace(NOW)).toBe('reset');

    expect(localStorage.getItem(ASIDE)).toBe(blob);
    expect(JSON.parse(localStorage.getItem(BASE_SCOPE)!)).toEqual(EMPTY_BLOB);
  });

  it('keeps the oldest copy and the newest in localStorage too', async () => {
    const first = seedLocalStorage(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    expect(await startEmptyWorkspace(NOW)).toBe('reset');
    seedLocalStorage(BASE_SCOPE, [record('b', 'Lab 2')], 'b');
    expect(await startEmptyWorkspace(NOW2)).toBe('reset');
    const third = seedLocalStorage(BASE_SCOPE, [record('c', 'Lab 3')], 'c');

    expect(await startEmptyWorkspace(NOW3)).toBe('reset');

    expect(asideKeys().sort()).toEqual([ASIDE, ASIDE3].sort());
    expect(localStorage.getItem(ASIDE)).toBe(first);
    expect(localStorage.getItem(ASIDE3)).toBe(third);
  });

  it('changes nothing when a workspace over half the localStorage quota leaves no room for a copy', async () => {
    const blob = seedBigWorkspace();

    expect(await startEmptyWorkspace(NOW)).toBe('no_copy');

    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
    expect(asideKeys()).toEqual([]);
  });

  it('goes on without a copy when a complete backup file was downloaded', async () => {
    seedBigWorkspace();

    expect(await startEmptyWorkspace(NOW, { completeBackup: true })).toBe('reset');

    expect(JSON.parse(localStorage.getItem(BASE_SCOPE)!)).toEqual(EMPTY_BLOB);
    expect(asideKeys()).toEqual([]);
  });

  it('holds localStorage to the same rule when IndexedDB is there but holds nothing for the scope', async () => {
    useIndexedDb();
    const blob = seedBigWorkspace();

    expect(await startEmptyWorkspace(NOW)).toBe('no_copy');

    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
    expect(await readSnapshot(BASE_SCOPE)).toBeNull();
  });

  it('answers not_changed, and takes its copy back, when localStorage is the only tier and cannot be written', async () => {
    const blob = seedLocalStorage(BASE_SCOPE, [record('a', 'Lab 1')], 'a');
    const setItem = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (this: Storage, key, value) {
      if (key === BASE_SCOPE) throw new DOMException('denied', 'SecurityError');
      setItem.call(this, key, value);
    });

    expect(await startEmptyWorkspace(NOW)).toBe('not_changed');

    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
    expect(asideKeys()).toEqual([]);
  });

  it('empties the project scope autosave writes, not the base one', async () => {
    useProjectStore.setState({ projectDir: 'D:/courses/lab3' });
    const projectScope = autosaveScope();
    seedLocalStorage(projectScope, [record('p', 'Project tab')], 'p');
    const base = seedLocalStorage(BASE_SCOPE, [record('b', 'Base tab')], 'b');

    expect(await startEmptyWorkspace(NOW)).toBe('reset');

    expect(JSON.parse(localStorage.getItem(projectScope)!)).toEqual(EMPTY_BLOB);
    expect(localStorage.getItem(`${projectScope}-aside::${NOW.getTime()}`)).not.toBeNull();
    expect(localStorage.getItem(BASE_SCOPE)).toBe(base);
  });
});
