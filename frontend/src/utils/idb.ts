/**
 * A very small promise wrapper over IndexedDB — one database, one key/value
 * object store, six operations. Deliberately not a dependency: the tab
 * autosave (#125) needs get / set / delete / prefix-scan and nothing else,
 * and a library for that would cost more bytes than the code below.
 *
 * Why IndexedDB at all: localStorage caps out around 5 MB per origin and
 * throws once a graph crosses it, which made the quota a hard ceiling on
 * graph size. IndexedDB has no comparable practical limit, stores structured
 * values (so no JSON.stringify on the save path), and lets each tab be its
 * own record — a dirty tab no longer forces every other tab to re-serialize.
 *
 * Every function rejects rather than throwing synchronously, so callers can
 * treat "IndexedDB is missing / blocked / broken" uniformly with a single
 * `.catch`. Use {@link idbAvailable} to pick a storage backend up front:
 * jsdom (and therefore the whole test suite, minus this file) has no
 * IndexedDB at all, and neither do some privacy modes.
 */

const DB_NAME = 'codefyui';
const DB_VERSION = 1;
const STORE = 'kv';

/**
 * Whether this environment exposes a usable IndexedDB.
 *
 * Reading the property can itself throw (a sandboxed iframe raises
 * SecurityError instead of returning undefined), so the probe is wrapped.
 */
export function idbAvailable(): boolean {
  try {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  } catch {
    return false;
  }
}

// One connection per page, opened lazily and shared. Held as the PROMISE, not
// the database, so concurrent callers during startup queue behind a single
// open request instead of racing several.
let _dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (_dbPromise) return _dbPromise;

  _dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    if (!idbAvailable()) {
      reject(new Error('IndexedDB is not available in this environment'));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        // Out-of-line keys: records are plain values (arrays, strings,
        // objects without a stable id field), and the caller owns the key.
        db.createObjectStore(STORE);
      }
    };
    request.onsuccess = () => {
      const db = request.result;
      // Another page wants to upgrade the schema. Holding the connection
      // open would block it indefinitely, so step aside and let the next
      // call reopen against the new version.
      db.onversionchange = () => {
        db.close();
        _dbPromise = null;
      };
      resolve(db);
    };
    request.onerror = () => reject(request.error ?? new Error('IndexedDB open failed'));
    request.onblocked = () => reject(new Error('IndexedDB open blocked by another connection'));
  }).catch((err) => {
    // Never cache a failed open: a transient failure (first call landing
    // before the environment provided IndexedDB, a blocked upgrade that
    // since resolved) would otherwise strand persistence for the whole
    // session.
    _dbPromise = null;
    throw err;
  });

  return _dbPromise;
}

/** Run `body` inside one transaction, resolving with whatever it produces. */
function withStore<T>(
  mode: IDBTransactionMode,
  body: (store: IDBObjectStore, resolve: (value: T) => void) => void,
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        let result: T = undefined as T;
        const tx = db.transaction(STORE, mode);
        // Resolve on `oncomplete`, not on the request callback: for writes
        // that is the only point at which the data is actually durable.
        tx.oncomplete = () => resolve(result);
        tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
        tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
        body(tx.objectStore(STORE), (value) => {
          result = value;
        });
      }),
  );
}

/** Read one value. Resolves `undefined` when the key is absent. */
export function idbGet<T>(key: string): Promise<T | undefined> {
  return withStore<T | undefined>('readonly', (store, resolve) => {
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result as T | undefined);
  });
}

/** Write one value, replacing any existing record under `key`. */
export function idbSet(key: string, value: unknown): Promise<void> {
  return withStore<void>('readwrite', (store) => {
    store.put(value, key);
  });
}

/** Write many values in a single transaction (all-or-nothing). */
export function idbSetMany(entries: Array<[string, unknown]>): Promise<void> {
  if (entries.length === 0) return Promise.resolve();
  return withStore<void>('readwrite', (store) => {
    for (const [key, value] of entries) store.put(value, key);
  });
}

/** Delete one key. Deleting a key that does not exist is not an error. */
export function idbDelete(key: string): Promise<void> {
  return withStore<void>('readwrite', (store) => {
    store.delete(key);
  });
}

/** Delete many keys in a single transaction. */
export function idbDeleteMany(keys: string[]): Promise<void> {
  if (keys.length === 0) return Promise.resolve();
  return withStore<void>('readwrite', (store) => {
    for (const key of keys) store.delete(key);
  });
}

// U+FFFF sorts after every character that can appear in a key. Built from its
// code point rather than typed inline so this file stays pure ASCII on disk.
const LAST_CODE_UNIT = String.fromCharCode(0xffff);

/**
 * Half-open key range covering every key that starts with `prefix`.
 *
 * `[prefix, prefix + U+FFFF)` selects exactly the prefix's descendants under
 * IndexedDB's string ordering.
 */
function prefixRange(prefix: string): IDBKeyRange {
  return IDBKeyRange.bound(prefix, `${prefix}${LAST_CODE_UNIT}`, false, true);
}

/** All values whose key starts with `prefix`, in key order. */
export function idbGetByPrefix<T>(prefix: string): Promise<T[]> {
  return withStore<T[]>('readonly', (store, resolve) => {
    const req = store.getAll(prefixRange(prefix));
    req.onsuccess = () => resolve((req.result ?? []) as T[]);
  });
}

/** All keys starting with `prefix`, in key order. */
export function idbGetKeysByPrefix(prefix: string): Promise<string[]> {
  return withStore<string[]>('readonly', (store, resolve) => {
    const req = store.getAllKeys(prefixRange(prefix));
    req.onsuccess = () => resolve((req.result ?? []) as string[]);
  });
}

/**
 * Drop the cached connection. Tests swap `globalThis.indexedDB` between
 * cases, and a connection held against the previous factory would keep
 * answering with the previous database's contents.
 */
export function _resetIdbForTests(): void {
  _dbPromise = null;
}
