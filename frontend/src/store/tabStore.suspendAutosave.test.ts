import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { Node } from '@xyflow/react';
import type { NodeData } from '../types';

/**
 * `suspendAutosave` and `autosaveScope` (#555).
 *
 * A fresh module per case: a suspension lasts for the life of the page, which
 * here is the module registry, and nothing turns autosave back on.
 */

async function fresh() {
  vi.resetModules();
  const { useTabStore, suspendAutosave, autosaveScope } = await import('./tabStore');
  const { useProjectStore } = await import('./projectStore');
  return { useTabStore, suspendAutosave, autosaveScope, useProjectStore };
}

function flowNode(id: string): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: 'Add', params: {} } };
}

beforeEach(() => {
  // Autosave is a 250 ms debounce on setTimeout; jsdom has no IndexedDB, so
  // the save lands in localStorage, synchronously, when the clock moves.
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  localStorage.clear();
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  localStorage.clear();
});

describe('suspendAutosave', () => {
  it('writes nothing more once called, a save already pending included', async () => {
    const { useTabStore, suspendAutosave, autosaveScope } = await fresh();
    const store = () => useTabStore.getState();
    const saved = () => localStorage.getItem(autosaveScope());

    const id = store().createTab({ title: 'Good', activate: true });
    store().loadGraphDocumentInto(id, { nodes: [flowNode('n1')], edges: [], boundFile: null });
    vi.advanceTimersByTime(250);
    const good = saved();
    expect(good).toContain('"Good"');

    // A change whose save is still pending when the suspension comes...
    store().renameTab(id, 'Pending');
    suspendAutosave();
    vi.advanceTimersByTime(1000);
    expect(saved()).toBe(good);

    // ...and one made after it.
    store().renameTab(id, 'After');
    vi.advanceTimersByTime(1000);
    expect(saved()).toBe(good);
  });
});

describe('autosaveScope', () => {
  it('names the key autosave writes, a project scope included', async () => {
    const { useTabStore, autosaveScope, useProjectStore } = await fresh();
    expect(autosaveScope()).toBe('codefyui-tabs');

    useProjectStore.getState().setProject('D:/courses/lab3');
    expect(autosaveScope()).toBe('codefyui-tabs::D:/courses/lab3');
    useTabStore.getState().createTab({ title: 'Lab 3', activate: true });
    vi.advanceTimersByTime(250);
    expect(localStorage.getItem(autosaveScope())).toContain('"Lab 3"');
  });
});
