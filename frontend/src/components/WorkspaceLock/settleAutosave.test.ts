import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';

/**
 * `settleAutosave` is what a page awaits before it hands editing to another
 * page (#554): whatever the tab store's debounced autosave still holds has
 * to reach IndexedDB first. These run the real store against
 * fake-indexeddb, the way `tabStore.idb.test.ts` does, with real timers --
 * the debounce under test is a real 250 ms one.
 */

type StoreModule = typeof import('../../store/tabStore');
type PersistenceModule = typeof import('../../store/tabPersistence');
type SettleModule = typeof import('./settleAutosave');

const SCOPE = 'codefyui-tabs';

let store: StoreModule;
let settle: SettleModule;
let observer: PersistenceModule;

beforeEach(async () => {
  localStorage.clear();
  vi.stubGlobal('indexedDB', new IDBFactory());
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
  // A reader with its own connection and bookkeeping, so looking at the
  // database does not change what the page under test thinks is durable.
  vi.resetModules();
  observer = await import('../../store/tabPersistence');
  vi.resetModules();
  store = await import('../../store/tabStore');
  settle = await import('./settleAutosave');
  await store.whenTabsHydrated();
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

async function savedTabNames(): Promise<string[]> {
  observer._resetTabPersistenceForTests();
  const snapshot = await observer.readSnapshot(SCOPE);
  return snapshot ? snapshot.tabs.map((t) => t.name) : [];
}

describe('settleAutosave', () => {
  it('waits for a pending autosave to reach IndexedDB', async () => {
    store.useTabStore.getState().addTab('pending');
    // Nothing is written yet: the change is inside the debounce.
    expect(await savedTabNames()).toEqual(['Tab 1']);

    await settle.settleAutosave();
    expect(await savedTabNames()).toEqual(['Tab 1', 'pending']);
  });

  it('waits again when the store changes while it waits', async () => {
    store.useTabStore.getState().addTab('first');
    const settled = settle.settleAutosave();
    await new Promise((resolve) => setTimeout(resolve, 200));
    store.useTabStore.getState().addTab('second');

    await settled;
    expect(await savedTabNames()).toEqual(['Tab 1', 'first', 'second']);
  });

  it('stops waiting for quiet after the cap, for a store that never goes quiet', async () => {
    // A visible page drawing a run's progress commits every frame, which
    // keeps re-arming the debounce. The handover must not wait for the run.
    const id = store.useTabStore.getState().activeTabId;
    let n = 0;
    const busy = setInterval(() => {
      n += 1;
      store.useTabStore.getState().renameTab(id, `busy ${n}`);
    }, 50);
    try {
      const started = Date.now();
      await settle.settleAutosave({ maxWaitMs: 300 });
      expect(Date.now() - started).toBeLessThan(1_000);
    } finally {
      clearInterval(busy);
    }
  });

  it('returns promptly when nothing is pending', async () => {
    const started = Date.now();
    await settle.settleAutosave();
    // One quiet window and no more.
    expect(Date.now() - started).toBeLessThan(1_000);
    expect(await savedTabNames()).toEqual(['Tab 1']);
  });
});
