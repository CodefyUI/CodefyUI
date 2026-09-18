import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';
import type { Node } from '@xyflow/react';
import type { NodeData } from '../types';
import type { ParsedWorkspace } from './workspaceFile';

/**
 * The importer against the IndexedDB tier of tab autosave.
 *
 * Like `store/tabStore.idb.test.ts`, everything is imported through a reset
 * module registry, because `tabStore` starts hydrating the moment it is
 * imported -- and that start is what this file is about.
 */

const SCOPE = 'codefyui-tabs';

const WORKSPACE: ParsedWorkspace = {
  version: 1,
  appVersion: null,
  exportedAt: null,
  active: 0,
  preferences: {},
  tabs: [
    {
      title: 'Imported',
      graph: {
        nodes: [{ id: 'n1', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
        edges: [],
      },
      run: {},
    },
  ],
};

function flowNode(id: string): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: 'Add', params: {} } };
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('indexedDB', new IDBFactory());
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
});

afterEach(() => {
  vi.doUnmock('../store/tabPersistence');
  vi.resetModules();
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe('importWorkspaceFile and tab hydration', () => {
  it('waits for hydration, so the restored tabs cannot overwrite the imported ones', async () => {
    // A previous session left one tab, holding a graph, in IndexedDB.
    vi.resetModules();
    const seed = await import('../store/tabPersistence');
    await seed.writeSnapshot(
      SCOPE,
      [{ id: 'p1', name: 'Persisted', nodes: [flowNode('keep')], edges: [] }],
      'p1',
    );

    // This session: the read of that tab is held open until the test lets it through.
    vi.resetModules();
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    vi.doMock('../store/tabPersistence', async (importOriginal) => {
      const actual = await importOriginal<typeof import('../store/tabPersistence')>();
      return {
        ...actual,
        readSnapshot: async (scope: string) => {
          await gate;
          return actual.readSnapshot(scope);
        },
      };
    });
    const { useTabStore } = await import('../store/tabStore');
    const { importWorkspaceFile } = await import('./importWorkspaceFile');
    const names = () => useTabStore.getState().tabs.map((t) => t.name);

    const pending = importWorkspaceFile(WORKSPACE);
    // One macrotask later, an importer that did not wait has run to the end.
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
    // Still only the placeholder tab the store boots with.
    expect(names()).toEqual(['Tab 1']);

    release();
    const result = await pending;
    expect(result.imported).toBe(1);
    // Hydration first, the import beside it -- and 'Persisted' holds a graph,
    // so it is not mistaken for a lone empty tab and closed.
    expect(names()).toEqual(['Persisted', 'Imported']);
  });
});
