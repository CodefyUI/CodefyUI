import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';

import {
  idbAvailable,
  idbGet,
  idbSet,
  idbSetMany,
  idbDelete,
  idbDeleteMany,
  idbGetByPrefix,
  idbGetKeysByPrefix,
  _resetIdbForTests,
} from './idb';

// jsdom ships no IndexedDB at all, which is exactly why the tab autosave has
// to keep a localStorage path (see tabStore). These tests install
// fake-indexeddb's in-memory implementation on `globalThis` for the duration
// of the file, so the adapter is exercised against a real IDB state machine —
// transactions, key ranges, version upgrades and all — rather than a mock of
// our own that would agree with whatever we wrote.
function installFakeIndexedDb() {
  vi.stubGlobal('indexedDB', new IDBFactory());
  // Browsers expose IDBKeyRange as a sibling global; the prefix scans need it.
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
  _resetIdbForTests();
}

beforeEach(() => {
  installFakeIndexedDb();
});

afterEach(() => {
  vi.unstubAllGlobals();
  _resetIdbForTests();
});

describe('idbAvailable', () => {
  it('is true when the environment provides indexedDB', () => {
    expect(idbAvailable()).toBe(true);
  });

  it('is false when indexedDB is missing (jsdom, private mode, blocked)', () => {
    vi.stubGlobal('indexedDB', undefined);
    expect(idbAvailable()).toBe(false);
  });

  it('is false when reading indexedDB throws (sandboxed iframe)', () => {
    // Some browsers throw a SecurityError on property access rather than
    // exposing `undefined`, so availability has to be probed inside try/catch.
    Object.defineProperty(globalThis, 'indexedDB', {
      configurable: true,
      get() {
        throw new Error('SecurityError');
      },
    });
    expect(idbAvailable()).toBe(false);
    // Restore a plain data property so unstubAllGlobals can clean up.
    Object.defineProperty(globalThis, 'indexedDB', {
      configurable: true,
      writable: true,
      value: undefined,
    });
  });
});

describe('idb get/set', () => {
  it('round-trips a structured value', async () => {
    await idbSet('a', { hello: 'world', n: [1, 2, 3] });
    expect(await idbGet('a')).toEqual({ hello: 'world', n: [1, 2, 3] });
  });

  it('returns undefined for a missing key', async () => {
    expect(await idbGet('nope')).toBeUndefined();
  });

  it('overwrites an existing key', async () => {
    await idbSet('a', 1);
    await idbSet('a', 2);
    expect(await idbGet('a')).toBe(2);
  });

  it('stores values structurally, not by reference', async () => {
    const value = { nested: { count: 1 } };
    await idbSet('a', value);
    value.nested.count = 99;
    expect(await idbGet<typeof value>('a')).toEqual({ nested: { count: 1 } });
  });

  it('deletes a key', async () => {
    await idbSet('a', 1);
    await idbDelete('a');
    expect(await idbGet('a')).toBeUndefined();
  });

  it('deleting a missing key is a no-op', async () => {
    await expect(idbDelete('ghost')).resolves.toBeUndefined();
  });
});

describe('idb batch operations', () => {
  it('writes many entries in one transaction', async () => {
    await idbSetMany([
      ['x|1', 'one'],
      ['x|2', 'two'],
    ]);
    expect(await idbGet('x|1')).toBe('one');
    expect(await idbGet('x|2')).toBe('two');
  });

  it('writing an empty batch is a no-op', async () => {
    await expect(idbSetMany([])).resolves.toBeUndefined();
  });

  it('deletes many keys in one transaction', async () => {
    await idbSetMany([
      ['x|1', 'one'],
      ['x|2', 'two'],
      ['y|1', 'keep'],
    ]);
    await idbDeleteMany(['x|1', 'x|2']);
    expect(await idbGet('x|1')).toBeUndefined();
    expect(await idbGet('y|1')).toBe('keep');
  });

  it('deleting an empty batch is a no-op', async () => {
    await expect(idbDeleteMany([])).resolves.toBeUndefined();
  });
});

describe('idb prefix queries', () => {
  beforeEach(async () => {
    await idbSetMany([
      ['scope|tab|a', { id: 'a' }],
      ['scope|tab|b', { id: 'b' }],
      ['scope|meta', { activeTabId: 'a' }],
      ['other|tab|c', { id: 'c' }],
    ]);
  });

  it('returns only values whose key starts with the prefix', async () => {
    const values = await idbGetByPrefix<{ id: string }>('scope|tab|');
    expect(values.map((v) => v.id).sort()).toEqual(['a', 'b']);
  });

  it('returns keys for a prefix', async () => {
    const keys = await idbGetKeysByPrefix('scope|tab|');
    expect(keys.sort()).toEqual(['scope|tab|a', 'scope|tab|b']);
  });

  it('does not leak across scopes that share a prefix boundary', async () => {
    const values = await idbGetByPrefix<{ id: string }>('other|');
    expect(values.map((v) => v.id)).toEqual(['c']);
  });

  it('returns an empty list for an unmatched prefix', async () => {
    expect(await idbGetByPrefix('missing|')).toEqual([]);
    expect(await idbGetKeysByPrefix('missing|')).toEqual([]);
  });

  it('distinguishes a scoped project key from the bare base key', async () => {
    // The tab store keys records as `<storageKey>|tab|<id>`, and the
    // project-scoped storage key is a strict superstring of the base one
    // (`codefyui-tabs` vs `codefyui-tabs::/proj`). The `|` separator is what
    // keeps a base-scope prefix query from sweeping up project records.
    await idbSetMany([
      ['codefyui-tabs|tab|base', { id: 'base' }],
      ['codefyui-tabs::/proj|tab|proj', { id: 'proj' }],
    ]);
    const base = await idbGetByPrefix<{ id: string }>('codefyui-tabs|tab|');
    expect(base.map((v) => v.id)).toEqual(['base']);
  });
});

describe('idb failure handling', () => {
  it('rejects when indexedDB is unavailable', async () => {
    vi.stubGlobal('indexedDB', undefined);
    _resetIdbForTests();
    await expect(idbGet('a')).rejects.toThrow(/IndexedDB/i);
  });

  it('rejects when opening the database fails', async () => {
    const failing = {
      open: () => {
        const request: Record<string, unknown> = {
          error: new Error('boom'),
          onerror: null,
          onsuccess: null,
          onupgradeneeded: null,
          onblocked: null,
        };
        queueMicrotask(() => (request.onerror as (() => void) | null)?.());
        return request;
      },
    };
    vi.stubGlobal('indexedDB', failing);
    _resetIdbForTests();
    await expect(idbSet('a', 1)).rejects.toBeTruthy();
  });

  it('reopens after a failed open instead of caching the rejection', async () => {
    vi.stubGlobal('indexedDB', undefined);
    _resetIdbForTests();
    await expect(idbGet('a')).rejects.toThrow();
    // A later call with a working factory must succeed: caching the failed
    // open promise would strand persistence for the rest of the session.
    vi.stubGlobal('indexedDB', new IDBFactory());
    await idbSet('a', 'recovered');
    expect(await idbGet('a')).toBe('recovered');
  });
});
