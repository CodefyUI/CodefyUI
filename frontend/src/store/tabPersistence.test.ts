import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';

import { idbDelete, idbGet, idbGetKeysByPrefix, idbSet, _resetIdbForTests } from '../utils/idb';
import {
  WORKSPACE_EDITOR_KEY,
  startWorkspaceLock,
  _resetWorkspaceLockForTests,
  type LockManagerLike,
} from '../utils/workspaceLock';
import {
  tabMetaKey,
  tabRecordKey,
  readSnapshot,
  writeSnapshot,
  snapshotWritesInFlight,
  whenSnapshotWritesSettle,
  _resetTabPersistenceForTests,
} from './tabPersistence';
import type { PersistedTab } from './tabStore';

// The whole point of this layer is that a graph too big for localStorage's
// ~5MB quota still round-trips, so it is exercised against fake-indexeddb's
// real IDB state machine rather than a hand-written mock.
function installFakeIndexedDb() {
  vi.stubGlobal('indexedDB', new IDBFactory());
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
  _resetIdbForTests();
  _resetTabPersistenceForTests();
}

function record(id: string, over: Partial<PersistedTab> = {}): PersistedTab {
  return {
    id,
    name: id.toUpperCase(),
    description: '',
    currentGraphFile: null,
    nodes: [],
    edges: [],
    segmentGroups: [],
    recordOutputs: true,
    verboseMode: false,
    graphId: `gid-${id}`,
    keepWeights: true,
    backwardMode: false,
    autoBackward: false,
    ...over,
  };
}

const SCOPE = 'codefyui-tabs';

beforeEach(() => {
  installFakeIndexedDb();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  _resetWorkspaceLockForTests();
  _resetIdbForTests();
  _resetTabPersistenceForTests();
});

describe('tabPersistence key layout', () => {
  it('keys the meta record and each tab record under the storage scope', () => {
    expect(tabMetaKey(SCOPE)).toBe('codefyui-tabs|meta');
    expect(tabRecordKey(SCOPE, 'a')).toBe('codefyui-tabs|tab|a');
  });

  it('keeps a project scope disjoint from the base scope', () => {
    // `::` vs `|` is what stops a base-scope prefix scan from sweeping up a
    // project's records: `codefyui-tabs|tab|` cannot prefix `codefyui-tabs::`.
    expect(tabRecordKey('codefyui-tabs::/proj', 'a').startsWith(`${SCOPE}|tab|`)).toBe(false);
  });
});

describe('tabPersistence round-trip', () => {
  it('returns null when nothing has been written for the scope', async () => {
    expect(await readSnapshot(SCOPE)).toBeNull();
  });

  it('writes one record per tab plus a meta record', async () => {
    await writeSnapshot(SCOPE, [record('a'), record('b')], 'b');
    const keys = await idbGetKeysByPrefix('codefyui-tabs');
    expect(keys.sort()).toEqual([
      'codefyui-tabs|meta',
      'codefyui-tabs|tab|a',
      'codefyui-tabs|tab|b',
    ]);
    expect(await idbGet(tabMetaKey(SCOPE))).toEqual({
      activeTabId: 'b',
      tabIds: ['a', 'b'],
    });
  });

  it('reads the tabs back in the order meta recorded', async () => {
    await writeSnapshot(SCOPE, [record('a'), record('b'), record('c')], 'c');
    _resetTabPersistenceForTests(); // simulate a fresh page load
    const snapshot = await readSnapshot(SCOPE);
    expect(snapshot!.activeTabId).toBe('c');
    expect(snapshot!.tabs.map((t) => t.id)).toEqual(['a', 'b', 'c']);
  });

  it('round-trips a graph far larger than the localStorage quota', async () => {
    // ~8MB of node params: localStorage.setItem would throw QuotaExceededError
    // on this, which is the ceiling #125 is removing.
    const blob = 'x'.repeat(1024 * 1024);
    const nodes = Array.from({ length: 8 }, (_, i) => ({
      id: `n${i}`,
      type: 'baseNode',
      position: { x: i, y: i },
      data: { label: `n${i}`, type: 'Big', params: { blob } },
    }));
    await writeSnapshot(SCOPE, [record('big', { nodes: nodes as never })], 'big');
    _resetTabPersistenceForTests();
    const snapshot = await readSnapshot(SCOPE);
    expect(snapshot!.tabs[0].nodes).toHaveLength(8);
    expect(snapshot!.tabs[0].nodes[0].data.params.blob).toHaveLength(1024 * 1024);
  });

  it('stores values structurally, so a later mutation cannot reach them', async () => {
    const rec = record('a', { name: 'Before' });
    await writeSnapshot(SCOPE, [rec], 'a');
    rec.name = 'After';
    _resetTabPersistenceForTests();
    expect((await readSnapshot(SCOPE))!.tabs[0].name).toBe('Before');
  });
});

describe('tabPersistence incremental writes', () => {
  it('rewrites only the records whose object identity changed', async () => {
    const a = record('a');
    const b = record('b');
    await writeSnapshot(SCOPE, [a, b], 'a');

    // A second save where only `b` was rebuilt. `a` is the very object we
    // already made durable, so it must not be serialized again -- that is
    // what stops one dirty tab from re-serializing every other tab.
    const b2 = record('b', { name: 'B-EDITED' });
    const puts: string[] = [];
    const realOpen = indexedDB.open.bind(indexedDB);
    vi.spyOn(indexedDB, 'open').mockImplementation((...args: unknown[]) => {
      const request = realOpen(...(args as Parameters<typeof realOpen>));
      request.addEventListener('success', () => {
        const db = request.result;
        const realTransaction = db.transaction.bind(db);
        db.transaction = ((...targs: unknown[]) => {
          const tx = realTransaction(...(targs as Parameters<typeof realTransaction>));
          const realObjectStore = tx.objectStore.bind(tx);
          tx.objectStore = ((name: string) => {
            const store = realObjectStore(name);
            const realPut = store.put.bind(store);
            store.put = ((value: unknown, key: IDBValidKey) => {
              puts.push(String(key));
              return realPut(value, key);
            }) as typeof store.put;
            return store;
          }) as typeof tx.objectStore;
          return tx;
        }) as typeof db.transaction;
      });
      return request;
    });
    _resetIdbForTests(); // force a reopen so the spy is in play

    await writeSnapshot(SCOPE, [a, b2], 'a');
    expect(puts).toEqual([tabRecordKey(SCOPE, 'b')]);
  });

  it('rewrites meta when the active tab changes but no tab did', async () => {
    const a = record('a');
    const b = record('b');
    await writeSnapshot(SCOPE, [a, b], 'a');
    await writeSnapshot(SCOPE, [a, b], 'b');
    expect(await idbGet(tabMetaKey(SCOPE))).toEqual({
      activeTabId: 'b',
      tabIds: ['a', 'b'],
    });
  });

  it('rewrites meta when the tab ORDER changes', async () => {
    const a = record('a');
    const b = record('b');
    await writeSnapshot(SCOPE, [a, b], 'a');
    await writeSnapshot(SCOPE, [b, a], 'a');
    expect(await idbGet(tabMetaKey(SCOPE))).toEqual({
      activeTabId: 'a',
      tabIds: ['b', 'a'],
    });
  });

  it('deletes the record of a tab that was closed', async () => {
    const a = record('a');
    const b = record('b');
    await writeSnapshot(SCOPE, [a, b], 'a');
    await writeSnapshot(SCOPE, [a], 'a');
    expect(await idbGet(tabRecordKey(SCOPE, 'b'))).toBeUndefined();
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
  });

  it('a save with nothing to change touches nothing', async () => {
    const a = record('a');
    await writeSnapshot(SCOPE, [a], 'a');
    // No throw, no rewrite -- the second call short-circuits before opening a
    // transaction, which is the common case during an idle editor session.
    await expect(writeSnapshot(SCOPE, [a], 'a')).resolves.toBeUndefined();
    expect(await idbGet(tabMetaKey(SCOPE))).toEqual({ activeTabId: 'a', tabIds: ['a'] });
  });

  it('keeps scopes independent', async () => {
    await writeSnapshot(SCOPE, [record('base')], 'base');
    await writeSnapshot('codefyui-tabs::/proj', [record('proj')], 'proj');
    expect((await readSnapshot(SCOPE))!.tabs.map((t) => t.id)).toEqual(['base']);
    expect((await readSnapshot('codefyui-tabs::/proj'))!.tabs.map((t) => t.id)).toEqual([
      'proj',
    ]);
  });

  it('does not mark records durable when the write fails', async () => {
    const a = record('a');
    vi.stubGlobal('indexedDB', undefined);
    _resetIdbForTests();
    await expect(writeSnapshot(SCOPE, [a], 'a')).rejects.toThrow();

    // Recover: the retry must write `a` again rather than believing the
    // failed attempt already made it durable.
    vi.stubGlobal('indexedDB', new IDBFactory());
    _resetIdbForTests();
    await writeSnapshot(SCOPE, [a], 'a');
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
  });
});

// #554: only the page that edits the workspace writes it. `workspaceLock`
// decides which page that is; these pin what the write path does with the
// answer. Each case starts this page's claim against a lock manager that
// answers the one way the case needs.
describe('tabPersistence write gate', () => {
  /** Another page already holds the editing lock. */
  const heldElsewhere: LockManagerLike = {
    request: (_name, _options, callback) => Promise.resolve(callback(null)),
  };

  /** Answers when the test says so, the way the browser answers later. */
  function answerLater(): { locks: LockManagerLike; answer: (edits: boolean) => void } {
    let answer: (edits: boolean) => void = () => {};
    return {
      locks: {
        request: (_name, _options, callback) =>
          new Promise((resolve) => {
            answer = (edits) => resolve(callback(edits ? { name: 'workspace' } : null));
          }),
      },
      answer: (edits) => answer(edits),
    };
  }

  it('skips the write in a read-only page and says so once', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await startWorkspaceLock({ locks: heldElsewhere, openChannel: null }).decided();

    await writeSnapshot(SCOPE, [record('a')], 'a');
    await writeSnapshot(SCOPE, [record('a'), record('b')], 'b');

    expect(await idbGetKeysByPrefix(SCOPE)).toEqual([]);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it('holds a write until the page knows it edits, then writes it', async () => {
    const lock = answerLater();
    startWorkspaceLock({ locks: lock.locks, openChannel: null });

    const write = writeSnapshot(SCOPE, [record('a')], 'a');
    await Promise.resolve();
    expect(snapshotWritesInFlight()).toBe(0);

    lock.answer(true);
    await write;
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
  });

  it('drops a held write when another page turns out to be editing', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const lock = answerLater();
    startWorkspaceLock({ locks: lock.locks, openChannel: null });

    const write = writeSnapshot(SCOPE, [record('a')], 'a');
    lock.answer(false);
    await write;
    expect(await idbGetKeysByPrefix(SCOPE)).toEqual([]);
  });

  it('writes the record a skipped write left out once this page edits', async () => {
    // A skipped write must not count as durable: the bookkeeping that lets an
    // unchanged record be skipped would otherwise never write it at all.
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const a = record('a');
    const lock = answerLater();
    startWorkspaceLock({ locks: lock.locks, openChannel: null });
    const skipped = writeSnapshot(SCOPE, [a], 'a');
    lock.answer(false);
    await skipped;

    _resetWorkspaceLockForTests();
    await writeSnapshot(SCOPE, [a], 'a');
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
  });

  it('counts writes in flight until they are durable', async () => {
    const write = writeSnapshot(SCOPE, [record('a')], 'a');
    expect(snapshotWritesInFlight()).toBe(1);

    await whenSnapshotWritesSettle();
    expect(snapshotWritesInFlight()).toBe(0);
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
    await write;
  });

  it('settles at once when nothing is being written', async () => {
    await expect(whenSnapshotWritesSettle()).resolves.toBeUndefined();
  });

  it('settles after a write that fails, too', async () => {
    vi.stubGlobal('indexedDB', undefined);
    _resetIdbForTests();
    const write = writeSnapshot(SCOPE, [record('a')], 'a');
    await whenSnapshotWritesSettle();
    expect(snapshotWritesInFlight()).toBe(0);
    await expect(write).rejects.toThrow();
  });
});

// #554 review: a page frozen for a while wakes with its autosave due and its
// role still 'editor'. The role gate cannot know better; the editor record
// does. These run the real claim (utils/idb) and the real fenced write
// against fake-indexeddb, and read back through utils/idb -- which also pins
// that the fenced write opens the same database and store.
describe('tabPersistence editor record', () => {
  const grant: LockManagerLike = {
    request: (_name, _options, callback) =>
      Promise.resolve().then(() => callback({ name: 'workspace' })),
  };

  async function startEditing() {
    const page = startWorkspaceLock({ locks: grant, openChannel: null });
    expect(await page.decided()).toBe('editor');
    return page;
  }

  it('is claimed when the page starts editing, and the page writes', async () => {
    const page = await startEditing();
    expect(page.getEpoch()).not.toBeNull();
    expect(await idbGet(WORKSPACE_EDITOR_KEY)).toBe(page.getEpoch());

    await writeSnapshot(SCOPE, [record('a')], 'a');
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
  });

  it('keeps out a write made after another page claimed it, though this page still thinks it edits', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = await startEditing();
    await writeSnapshot(SCOPE, [record('a')], 'a');

    // The page that took over claimed the record; this one has not heard.
    await idbSet(WORKSPACE_EDITOR_KEY, 'the-page-that-took-over');
    expect(page.getRole()).toBe('editor');

    await writeSnapshot(SCOPE, [record('a', { name: 'STALE' }), record('b')], 'b');
    expect((await idbGet<PersistedTab>(tabRecordKey(SCOPE, 'a')))!.name).toBe('A');
    expect(await idbGet(tabRecordKey(SCOPE, 'b'))).toBeUndefined();
    expect(await idbGet(tabMetaKey(SCOPE))).toEqual({ activeTabId: 'a', tabIds: ['a'] });
    expect(await idbGet(WORKSPACE_EDITOR_KEY)).toBe('the-page-that-took-over');
    // It learns from the refusal, says so once, and the count is not left behind.
    expect(page.getRole()).toBe('displaced');
    expect(warn).toHaveBeenCalledTimes(1);
    expect(snapshotWritesInFlight()).toBe(0);
  });

  it('lets a reloaded page write again: it claims the record afresh', async () => {
    const first = await startEditing();
    await writeSnapshot(SCOPE, [record('a')], 'a');

    // The page reloads: its claim and its module state start over.
    _resetWorkspaceLockForTests();
    _resetTabPersistenceForTests();
    const second = await startEditing();
    expect(second.getEpoch()).not.toBe(first.getEpoch());

    await writeSnapshot(SCOPE, [record('a', { name: 'AFTER RELOAD' })], 'a');
    expect((await idbGet<PersistedTab>(tabRecordKey(SCOPE, 'a')))!.name).toBe('AFTER RELOAD');
  });

  it('is put back when it went missing, and the write goes ahead', async () => {
    // Site data cleared under an open page must not lock its editor out.
    const page = await startEditing();
    await idbDelete(WORKSPACE_EDITOR_KEY);

    await writeSnapshot(SCOPE, [record('a')], 'a');
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
    expect(await idbGet(WORKSPACE_EDITOR_KEY)).toBe(page.getEpoch());
  });

  it('lets the editing page write a scope nobody claimed, such as a copy set aside', async () => {
    // One record for the origin, not one per scope, so a scope made later --
    // a workspace moved aside under a new name -- is written like any other.
    await startEditing();
    const aside = `${SCOPE}-aside::1759400000000`;

    await writeSnapshot(aside, [record('a')], 'a');
    expect((await readSnapshot(aside))!.tabs.map((t) => t.id)).toEqual(['a']);
  });

  it("refuses a displaced page's write to the main scope", async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const page = await startEditing();
    await writeSnapshot(SCOPE, [record('a')], 'a');
    page.supersede();
    expect(page.getRole()).toBe('displaced');

    await writeSnapshot(SCOPE, [record('a', { name: 'LATE' })], 'a');
    expect((await idbGet<PersistedTab>(tabRecordKey(SCOPE, 'a')))!.name).toBe('A');
  });

  it('is neither checked nor created when no page claimed editing', async () => {
    // Tests, scripts and any caller that never mounts the app.
    await idbSet(WORKSPACE_EDITOR_KEY, 'some-page');
    await writeSnapshot(SCOPE, [record('a')], 'a');
    expect(await idbGet(tabRecordKey(SCOPE, 'a'))).toBeTruthy();
    expect(await idbGet(WORKSPACE_EDITOR_KEY)).toBe('some-page');
  });
});

describe('tabPersistence damaged data', () => {
  it('reads meta naming no tabs as an empty workspace, not as missing data', async () => {
    // Not damaged data at all any more: since the last tab became closable,
    // "no tabs open" is a state the user can save, and it has to survive a
    // reload. `null` would mean "nothing here" and send the caller off to
    // migrate whatever localStorage holds -- which is how the freshly-seeded
    // `Tab 1` used to be written straight back over the empty workspace.
    await idbGetKeysByPrefix(SCOPE); // ensure the db exists
    const { idbSet } = await import('../utils/idb');
    await idbSet(tabMetaKey(SCOPE), { activeTabId: 'a', tabIds: [] });
    expect(await readSnapshot(SCOPE)).toEqual({ tabs: [], activeTabId: '' });
  });

  it('returns null when meta names tabs whose records are gone', async () => {
    const { idbSet } = await import('../utils/idb');
    await idbSet(tabMetaKey(SCOPE), { activeTabId: 'a', tabIds: ['a'] });
    expect(await readSnapshot(SCOPE)).toBeNull();
  });

  it('skips a tab id in meta that has no record', async () => {
    await writeSnapshot(SCOPE, [record('a')], 'a');
    const { idbSet } = await import('../utils/idb');
    await idbSet(tabMetaKey(SCOPE), { activeTabId: 'a', tabIds: ['a', 'ghost'] });
    _resetTabPersistenceForTests();
    expect((await readSnapshot(SCOPE))!.tabs.map((t) => t.id)).toEqual(['a']);
  });

  it('returns null when meta is not the shape we wrote', async () => {
    const { idbSet } = await import('../utils/idb');
    await idbSet(tabMetaKey(SCOPE), 'not-an-object');
    expect(await readSnapshot(SCOPE)).toBeNull();
  });

  it('falls back to the first record when meta names an unknown active tab', async () => {
    await writeSnapshot(SCOPE, [record('a'), record('b')], 'a');
    const { idbSet } = await import('../utils/idb');
    await idbSet(tabMetaKey(SCOPE), { activeTabId: 'gone', tabIds: ['a', 'b'] });
    _resetTabPersistenceForTests();
    expect((await readSnapshot(SCOPE))!.activeTabId).toBe('a');
  });

  it('propagates a read failure rather than pretending the store is empty', async () => {
    vi.stubGlobal('indexedDB', undefined);
    _resetIdbForTests();
    await expect(readSnapshot(SCOPE)).rejects.toThrow();
  });
});
