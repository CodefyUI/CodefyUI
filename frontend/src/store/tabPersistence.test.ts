import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';

import { idbGet, idbGetKeysByPrefix, _resetIdbForTests } from '../utils/idb';
import {
  tabMetaKey,
  tabRecordKey,
  readSnapshot,
  writeSnapshot,
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
    weightsPersistent: true,
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

describe('tabPersistence damaged data', () => {
  it('returns null when meta is present but names no tabs', async () => {
    await idbGetKeysByPrefix(SCOPE); // ensure the db exists
    const { idbSet } = await import('../utils/idb');
    await idbSet(tabMetaKey(SCOPE), { activeTabId: 'a', tabIds: [] });
    expect(await readSnapshot(SCOPE)).toBeNull();
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
