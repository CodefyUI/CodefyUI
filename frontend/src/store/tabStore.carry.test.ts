import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';

/**
 * Phase 4A, R2: the tabs of the pre-project (flat) scope are COPIED into the
 * first project that opens with none of its own, once per browser.
 */

type TabStoreModule = typeof import('./tabStore');

const FLAT = 'codefyui-tabs';

function node(id: string) {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Test', params: {} },
  };
}

async function openProject(dir: string): Promise<{
  store: TabStoreModule;
  persistence: typeof import('./tabPersistence');
  toast: typeof import('./toastStore');
  outcome: string;
}> {
  vi.resetModules();
  const persistence = await import('./tabPersistence');
  const toast = await import('./toastStore');
  const projectStore = await import('./projectStore');
  const store = (await import('./tabStore')) as TabStoreModule;
  await store.whenTabsHydrated();
  projectStore.useProjectStore.getState().setProject(dir);
  store.useTabStore.getState().rehydrateForProject(dir);
  const outcome = await store.whenTabsHydrated();
  return { store, persistence, toast, outcome };
}

async function seedFlat(tabs: Array<Record<string, unknown>>, active: string) {
  vi.resetModules();
  const persistence = await import('./tabPersistence');
  await persistence.writeSnapshot(FLAT, tabs as never, active);
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('indexedDB', new IDBFactory());
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe('carrying the flat tabs into the first project (R2)', () => {
  it('copies them once and leaves the flat scope as it was', async () => {
    await seedFlat([
      { id: 'a', name: 'Unsaved work', nodes: [node('n1')], edges: [] },
      { id: 'b', name: 'Empty', nodes: [], edges: [] },
    ], 'a');

    const first = await openProject('/projects/workspace');
    expect(first.outcome).toBe('carried');
    expect(first.store.useTabStore.getState().tabs.map((t) => t.name))
      .toEqual(['Unsaved work', 'Empty']);
    const toasts = first.toast.useToastStore.getState().toasts.map((t) => t.message);
    expect(toasts.some((m) => m.includes('1 tab'))).toBe(true);

    // Written to the project's scope, and still in the flat one.
    const project = await first.persistence.readSnapshot('codefyui-tabs::/projects/workspace');
    expect(project?.tabs.map((t) => t.name)).toEqual(['Unsaved work', 'Empty']);
    const flat = await first.persistence.readSnapshot(FLAT);
    expect(flat?.tabs.map((t) => t.name)).toEqual(['Unsaved work', 'Empty']);

    // A project created later starts with its own tabs, not another copy.
    const second = await openProject('/projects/other');
    expect(second.outcome).not.toBe('carried');
    expect(second.store.useTabStore.getState().tabs.map((t) => t.name))
      .not.toContain('Unsaved work');
  });

  it('copies nothing when the flat tabs are empty', async () => {
    await seedFlat([{ id: 'x', name: 'Untitled', nodes: [], edges: [] }], 'x');
    const opened = await openProject('/projects/workspace');
    expect(opened.outcome).not.toBe('carried');
  });

  it('does not touch a project that already has tabs', async () => {
    await seedFlat([{ id: 'a', name: 'Flat work', nodes: [node('n1')], edges: [] }], 'a');
    vi.resetModules();
    const persistence = await import('./tabPersistence');
    await persistence.writeSnapshot('codefyui-tabs::/projects/mine',
      [{ id: 'p', name: 'Project work', nodes: [], edges: [] }] as never, 'p');
    const opened = await openProject('/projects/mine');
    expect(opened.outcome).toBe('loaded');
    expect(opened.store.useTabStore.getState().tabs.map((t) => t.name)).toEqual(['Project work']);
  });
});
