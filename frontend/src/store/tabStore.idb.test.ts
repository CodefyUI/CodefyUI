import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange, IDBObjectStore } from 'fake-indexeddb';

/**
 * The IndexedDB tier of tab autosave (#125).
 *
 * Everything here re-imports `./tabStore` through a reset module registry,
 * because the module reads storage once at import (`loadTabs`) and kicks off
 * hydration from the same place — the two things under test. jsdom has no
 * IndexedDB of its own, so each case installs a fresh fake-indexeddb factory
 * first and the freshly-imported modules open against it.
 */

type TabStoreModule = typeof import('./tabStore');

const LEGACY_KEY = 'codefyui-tabs';

function installFakeIndexedDb() {
  vi.stubGlobal('indexedDB', new IDBFactory());
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
}

/** Re-import the store (and the idb helpers that share its registry). */
async function loadStore(): Promise<{
  store: TabStoreModule;
  idb: typeof import('../utils/idb');
  persistence: typeof import('./tabPersistence');
  toast: typeof import('./toastStore');
}> {
  vi.resetModules();
  const idb = await import('../utils/idb');
  const persistence = await import('./tabPersistence');
  const toast = await import('./toastStore');
  const store = await import('./tabStore');
  await store.whenTabsHydrated();
  return { store, idb, persistence, toast };
}

function legacyBlob(tabs: Array<Record<string, unknown>>, activeTabId: string) {
  return JSON.stringify({ activeTabId, tabs });
}

function node(id: string, params: Record<string, unknown> = {}) {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Test', params },
  };
}

beforeEach(() => {
  localStorage.clear();
  installFakeIndexedDb();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  localStorage.clear();
});

describe('tab autosave - IndexedDB migration on first load', () => {
  it('copies the localStorage tabs into IndexedDB when it holds nothing', async () => {
    localStorage.setItem(
      LEGACY_KEY,
      legacyBlob(
        [
          { id: 'a', name: 'Alpha', nodes: [node('n1')], edges: [] },
          { id: 'b', name: 'Beta', nodes: [], edges: [] },
        ],
        'b',
      ),
    );
    const { store, persistence } = await loadStore();

    expect(await store.whenTabsHydrated()).toBe('migrated');
    const snapshot = await persistence.readSnapshot(LEGACY_KEY);
    expect(snapshot!.activeTabId).toBe('b');
    expect(snapshot!.tabs.map((t) => t.name)).toEqual(['Alpha', 'Beta']);
    expect(snapshot!.tabs[0].nodes).toHaveLength(1);
  });

  it('leaves the legacy localStorage blob in place after migrating', async () => {
    const raw = legacyBlob([{ id: 'a', name: 'Alpha', nodes: [], edges: [] }], 'a');
    localStorage.setItem(LEGACY_KEY, raw);
    await loadStore();
    // A downgrade to a build without the IndexedDB tier must still find the
    // graph it last saw; the load path keeps reading it for one release.
    expect(localStorage.getItem(LEGACY_KEY)).toBe(raw);
  });

  it('reports a seed (not a migration) when neither tier had anything', async () => {
    const { store, persistence } = await loadStore();
    expect(await store.whenTabsHydrated()).toBe('seeded');
    // The default tab is written so the next load is a plain `loaded`.
    const snapshot = await persistence.readSnapshot(LEGACY_KEY);
    expect(snapshot!.tabs).toHaveLength(1);
    expect(snapshot!.tabs[0].name).toBe('Tab 1');
  });

  it('prefers IndexedDB over a stale localStorage blob on the next load', async () => {
    localStorage.setItem(
      LEGACY_KEY,
      legacyBlob([{ id: 'a', name: 'Stale', nodes: [], edges: [] }], 'a'),
    );
    const first = await loadStore();
    first.store.useTabStore.getState().renameTab('a', 'Current');
    await first.persistence.writeSnapshot(
      LEGACY_KEY,
      [{ id: 'a', name: 'Current', nodes: [], edges: [] }],
      'a',
    );

    const second = await loadStore();
    expect(await second.store.whenTabsHydrated()).toBe('loaded');
    expect(second.store.useTabStore.getState().tabs.map((t) => t.name)).toEqual(['Current']);
  });

  it('restores tab order and the active tab from IndexedDB', async () => {
    const first = await loadStore();
    await first.persistence.writeSnapshot(
      LEGACY_KEY,
      [
        { id: 'x', name: 'X', nodes: [], edges: [] },
        { id: 'y', name: 'Y', nodes: [], edges: [] },
        { id: 'z', name: 'Z', nodes: [], edges: [] },
      ],
      'y',
    );
    const second = await loadStore();
    const state = second.store.useTabStore.getState();
    expect(state.tabs.map((t) => t.id)).toEqual(['x', 'y', 'z']);
    expect(state.activeTabId).toBe('y');
  });

  it('keeps the live socket of a tab the hydrate also names', async () => {
    localStorage.setItem(
      LEGACY_KEY,
      legacyBlob([{ id: 'a', name: 'Alpha', nodes: [], edges: [] }], 'a'),
    );
    const first = await loadStore();
    await first.persistence.writeSnapshot(
      LEGACY_KEY,
      [{ id: 'a', name: 'Alpha', nodes: [], edges: [] }],
      'a',
    );

    // Second load: localStorage produces tab `a`, then hydration produces
    // tab `a` again. Recreating its ExecutionWebSocket would strand the
    // handlers useGraphExecution attached to the first one.
    vi.resetModules();
    const idb = await import('../utils/idb');
    void idb;
    const store = (await import('./tabStore')) as TabStoreModule;
    const socketBefore = store.useTabStore.getState().tabs[0].ws;
    await store.whenTabsHydrated();
    expect(store.useTabStore.getState().tabs[0].ws).toBe(socketBefore);
  });
});

describe('tab autosave - IndexedDB is the write target', () => {
  it('writes the debounced save to IndexedDB and not to localStorage', async () => {
    const { store, persistence } = await loadStore();
    const setItem = vi.spyOn(Storage.prototype, 'setItem');

    store.useTabStore.getState().addTab('Persisted');
    await vi.waitFor(async () => {
      const snapshot = await persistence.readSnapshot(LEGACY_KEY);
      expect(snapshot!.tabs.some((t) => t.name === 'Persisted')).toBe(true);
    });
    expect(setItem).not.toHaveBeenCalled();
    setItem.mockRestore();
  });

  it('survives a reload with a graph far past the localStorage quota', async () => {
    const first = await loadStore();
    // ~6MB of params: the pre-#125 blob write would throw QuotaExceededError
    // here and the user's work would silently stop being saved.
    const blob = 'x'.repeat(512 * 1024);
    first.store.useTabStore
      .getState()
      .setNodes(Array.from({ length: 12 }, (_, i) => node(`n${i}`, { blob })) as never);
    await vi.waitFor(async () => {
      const snapshot = await first.persistence.readSnapshot(LEGACY_KEY);
      expect(snapshot!.tabs[0].nodes).toHaveLength(12);
    });

    const second = await loadStore();
    const nodes = second.store.useTabStore.getState().tabs[0].nodes;
    expect(nodes).toHaveLength(12);
    expect(nodes[0].data.params.blob).toHaveLength(512 * 1024);
  });

  it('rewrites only the record of the tab that changed', async () => {
    const { store } = await loadStore();
    const puts = spyOnPuts();
    store.useTabStore.setState({
      tabs: [
        { ...store.useTabStore.getState().tabs[0], id: 'a', name: 'A' },
        { ...store.useTabStore.getState().tabs[0], id: 'b', name: 'B' },
      ],
      activeTabId: 'a',
    });
    await vi.waitFor(() => {
      expect(puts.keys).toContain('codefyui-tabs|tab|b');
    });

    // Now touch ONE tab and watch which keys the next save writes.
    puts.keys.length = 0;
    store.useTabStore.getState().renameTab('a', 'A-EDITED');
    await vi.waitFor(() => {
      expect(puts.keys).toContain('codefyui-tabs|tab|a');
    });
    // Tab `b` was untouched, so its record object is reused and never
    // re-serialized -- one dirty tab must not cost every other tab. Nor is
    // meta rewritten: neither the order nor the active tab moved.
    expect(puts.keys).toEqual(['codefyui-tabs|tab|a']);
    puts.stop();
  });

  it('strips SECRET param values before they reach IndexedDB', async () => {
    // The "Session only" promise on a secret field is about the STORAGE tier,
    // whichever one it is. Moving persistence to IndexedDB must not quietly
    // start durably storing typed API keys.
    const { store, persistence } = await loadStore();
    const llmDef = {
      node_name: 'LLMChat',
      category: 'LLM',
      description: '',
      inputs: [],
      outputs: [],
      params: [
        {
          name: 'openai_api_key',
          param_type: 'secret',
          default: '',
          description: '',
          options: [],
          min_value: null,
          max_value: null,
        },
      ],
    };
    store.useTabStore.getState().setNodes([
      {
        id: 'n1',
        type: 'baseNode',
        position: { x: 0, y: 0 },
        data: {
          label: 'LLM',
          type: 'LLMChat',
          params: { openai_api_key: 'sk-must-not-persist', model: 'gpt-5.2' },
          definition: llmDef,
        },
      },
    ] as never);

    await vi.waitFor(async () => {
      const snapshot = await persistence.readSnapshot(LEGACY_KEY);
      expect(snapshot!.tabs[0].nodes).toHaveLength(1);
    });
    const snapshot = await persistence.readSnapshot(LEGACY_KEY);
    const params = snapshot!.tabs[0].nodes[0].data.params;
    expect(params.openai_api_key).toBe('');
    expect(params.model).toBe('gpt-5.2');
    expect(JSON.stringify(snapshot)).not.toContain('sk-must-not-persist');
  });

  it('falls back to localStorage and toasts once when the IndexedDB write fails (#164)', async () => {
    const { store, idb, toast } = await loadStore();
    const setItem = vi.spyOn(Storage.prototype, 'setItem');
    // Keep `idbAvailable()` true (so the IndexedDB branch is taken) while
    // every actual open fails -- a database revoked mid-session.
    vi.stubGlobal('indexedDB', {
      open: () => {
        const request: Record<string, unknown> = { error: new Error('gone') };
        queueMicrotask(() => (request.onerror as (() => void) | undefined)?.());
        return request;
      },
    });
    idb._resetIdbForTests(); // drop the connection opened during hydration

    store.useTabStore.getState().addTab('Fallback');
    await vi.waitFor(() => {
      expect(setItem).toHaveBeenCalled();
    });
    const raw = localStorage.getItem(LEGACY_KEY)!;
    expect(JSON.parse(raw).tabs.some((t: { name: string }) => t.name === 'Fallback')).toBe(true);
    // The downgrade from IndexedDB to localStorage must not be silent: it
    // drops the storage ceiling from generous to the 5MB localStorage cap
    // with no notice otherwise (#164).
    const toasts = toast.useToastStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].type).toBe('error');
    expect(toasts[0].message.length).toBeGreaterThan(0);
    setItem.mockRestore();
  });
});

describe('tab autosave - hydration failure is surfaced', () => {
  /** An IndexedDB that exists (so `idbAvailable()` is true) but never opens. */
  function installBrokenIndexedDb() {
    vi.stubGlobal('indexedDB', {
      open: () => {
        const request: Record<string, unknown> = { error: new Error('broken') };
        queueMicrotask(() => (request.onerror as (() => void) | undefined)?.());
        return request;
      },
    });
  }

  it('toasts when the hydrate read fails, so the user knows to export', async () => {
    // The dangerous case: a migrated workspace whose IndexedDB has since gone
    // bad. localStorage is frozen at migration-day content, so failing quietly
    // shows stale tabs, accepts edits against them, and loses them at the next
    // healthy load.
    localStorage.setItem(
      LEGACY_KEY,
      legacyBlob([{ id: 'a', name: 'MigrationDay', nodes: [], edges: [] }], 'a'),
    );
    vi.resetModules();
    installBrokenIndexedDb();
    const toastMod = await import('./toastStore');
    const store = (await import('./tabStore')) as TabStoreModule;

    expect(await store.whenTabsHydrated()).toBe('failed');
    const toasts = toastMod.useToastStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].type).toBe('error');
    expect(toasts[0].message.length).toBeGreaterThan(0);
  });

  it('warns once, not once per retry, while storage stays broken', async () => {
    vi.resetModules();
    installBrokenIndexedDb();
    const toastMod = await import('./toastStore');
    const store = (await import('./tabStore')) as TabStoreModule;
    await store.whenTabsHydrated();
    expect(toastMod.useToastStore.getState().toasts).toHaveLength(1);

    // A resolved project retries the hydrate against the same broken store.
    await store.hydrateTabsFromPersistence();
    await store.hydrateTabsFromPersistence();
    expect(toastMod.useToastStore.getState().toasts).toHaveLength(1);
  });

  it('says nothing when there is simply no IndexedDB to use', async () => {
    // A browser without IndexedDB at all is not degraded — localStorage stays
    // authoritative and autosave keeps working, exactly as before #125.
    vi.resetModules();
    vi.stubGlobal('indexedDB', undefined);
    const toastMod = await import('./toastStore');
    const store = (await import('./tabStore')) as TabStoreModule;
    expect(await store.whenTabsHydrated()).toBe('unavailable');
    expect(toastMod.useToastStore.getState().toasts).toHaveLength(0);
  });
});

describe('tab autosave - hydration race', () => {
  it('discards a base-scope read that a resolved project has superseded', async () => {
    // The base scope's hydration starts at import; opening a project switches
    // the storage scope as soon as /api/health answers, which can beat the
    // IndexedDB round-trip. Landing the base read afterwards would replace
    // the project's graphs with the base ones.
    const first = await loadStore();
    await first.persistence.writeSnapshot(
      LEGACY_KEY,
      [{ id: 'base', name: 'BaseTab', nodes: [], edges: [] }],
      'base',
    );
    await first.persistence.writeSnapshot(
      'codefyui-tabs::/proj',
      [{ id: 'proj', name: 'ProjectTab', nodes: [], edges: [] }],
      'proj',
    );

    vi.resetModules();
    const projectStore = await import('./projectStore');
    const store = (await import('./tabStore')) as TabStoreModule;
    const baseHydration = store.whenTabsHydrated();

    // Health resolves before the base read comes back.
    projectStore.useProjectStore.getState().setProject('/proj');
    store.useTabStore.getState().rehydrateForProject('/proj');
    const projectHydration = store.whenTabsHydrated();

    expect(await baseHydration).toBe('superseded');
    await projectHydration;
    expect(store.useTabStore.getState().tabs.map((t) => t.name)).toEqual(['ProjectTab']);
    projectStore.useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  });

  it('does not let a save scheduled before hydration overwrite it', async () => {
    const first = await loadStore();
    await first.persistence.writeSnapshot(
      LEGACY_KEY,
      [{ id: 'real', name: 'FromIndexedDb', nodes: [], edges: [] }],
      'real',
    );
    localStorage.setItem(
      LEGACY_KEY,
      legacyBlob([{ id: 'stale', name: 'FromLocalStorage', nodes: [], edges: [] }], 'stale'),
    );

    vi.resetModules();
    const store = (await import('./tabStore')) as TabStoreModule;
    // Touch the store while hydration is still in flight. Before the guard,
    // the debounce would fire against the localStorage-derived tabs and write
    // them straight over the records being read.
    store.useTabStore.getState().renameTab('stale', 'Touched');
    await store.whenTabsHydrated();

    expect(store.useTabStore.getState().tabs.map((t) => t.name)).toEqual(['FromIndexedDb']);
    await vi.waitFor(async () => {
      const { readSnapshot } = await import('./tabPersistence');
      const snapshot = await readSnapshot(LEGACY_KEY);
      expect(snapshot!.tabs.map((t) => t.name)).toEqual(['FromIndexedDb']);
    });
  });
});

// ── helpers ─────────────────────────────────────────────────────────────────

/** Record every key written through `IDBObjectStore.put` until stopped. */
function spyOnPuts(): { keys: string[]; stop: () => void } {
  const keys: string[] = [];
  const proto = IDBObjectStore.prototype;
  const realPut = proto.put;
  proto.put = function put(this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
    if (key !== undefined) keys.push(String(key));
    return realPut.call(this, value, key);
  } as typeof proto.put;
  return {
    keys,
    stop: () => {
      proto.put = realPut;
    },
  };
}
