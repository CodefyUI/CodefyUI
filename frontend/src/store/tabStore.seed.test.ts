/**
 * The run seed travels with the graph (2.8.9).
 *
 * The seed used to live on the tab alone: Save never wrote it, Open and
 * Import never read it, and a tab opened for an Import into a non-empty tab
 * started unseeded -- so a student who set seed 0 and then imported a starter
 * got a canvas result the exported script (which bakes the seed in) did not
 * print. These pin the four store halves of the fix: the serializer writes
 * `settings.seed`, a load installs a seed the file carries and keeps the tab's
 * when it carries none, a new tab starts with the active tab's seed, and a
 * seed change is a document change.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { Node } from '@xyflow/react';

import { NO_ACTIVE_TAB, documentChanged, useTabStore } from './tabStore';
import type { NodeData } from '../types';

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();
const serialized = () => useTabStore.getState().getSerializedGraph();

function node(id: string): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: 'Add', params: {} },
  } as Node<NodeData>;
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
  localStorage.clear();
});

describe('the serializer writes the seed as settings.seed', () => {
  it('writes settings: {seed} and no device key for a seeded tab', () => {
    store().setSeed(7);
    expect(serialized().settings).toEqual({ seed: 7 });
  });

  it('writes both when the graph has a device and a seed', () => {
    store().setGraphDevice('cuda:1');
    store().setSeed(7);
    expect(serialized().settings).toEqual({ device: 'cuda:1', seed: 7 });
  });

  it('writes seed 0, which is a seed', () => {
    store().setSeed(0);
    expect(serialized().settings).toEqual({ seed: 0 });
  });

  it('writes no settings key with neither, so such a graph keeps its bytes', () => {
    expect('settings' in serialized()).toBe(false);
    store().setSeed(7);
    store().setSeed(null);
    expect('settings' in serialized()).toBe(false);
  });

  it('writes nothing for a seed the backend would refuse', () => {
    // The field is a free number input; the server answers 422 to a seed
    // outside 0..2**32-1, and a Save must not fail over a run setting.
    store().setSeed(-1);
    expect('settings' in serialized()).toBe(false);
    store().setSeed(2 ** 32);
    expect('settings' in serialized()).toBe(false);
    store().setGraphDevice('mps');
    expect(serialized().settings).toEqual({ device: 'mps' });
  });
});

describe('loadGraphDocumentInto and the seed', () => {
  it('installs the seed the document carries, in the same emission as the graph', () => {
    store().setSeed(9);
    let emissions = 0;
    const unsubscribe = useTabStore.subscribe(() => {
      emissions += 1;
    });
    store().loadGraphDocument({ nodes: [node('a')], edges: [], boundFile: null, seed: 42 });
    unsubscribe();

    expect(emissions).toBe(1);
    expect(tab().seed).toBe(42);
  });

  it('installs seed 0', () => {
    store().setSeed(9);
    store().loadGraphDocument({ nodes: [], edges: [], boundFile: null, seed: 0 });
    expect(tab().seed).toBe(0);
  });

  it('keeps the tab seed for a document that carries none (a file written before 2.8.9)', () => {
    store().setSeed(9);
    store().loadGraphDocument({ nodes: [node('a')], edges: [], boundFile: null });
    expect(tab().seed).toBe(9);
    store().loadGraphDocument({ nodes: [node('b')], edges: [], boundFile: null, seed: null });
    expect(tab().seed).toBe(9);
  });

  it('addresses the tab it names, not the active one', () => {
    const background = store().createTab({ title: 'Background', activate: false });
    store().loadGraphDocumentInto(background, { nodes: [], edges: [], boundFile: null, seed: 5 });
    expect(store().getTab(background)!.seed).toBe(5);
    expect(tab().seed).toBeNull();
  });
});

describe('a new tab starts with the seed of the tab that was active', () => {
  it('copies the active tab seed', () => {
    store().setSeed(5);
    const id = store().createTab();
    expect(store().getTab(id)!.seed).toBe(5);
  });

  it('copies seed 0', () => {
    store().setSeed(0);
    const id = store().createTab();
    expect(store().getTab(id)!.seed).toBe(0);
  });

  it('copies it into a tab opened in the background too', () => {
    store().setSeed(11);
    const id = store().createTab({ title: 'Plugin tab', activate: false });
    expect(store().getTab(id)!.seed).toBe(11);
  });

  it('starts unseeded when the active tab has no seed', () => {
    const id = store().createTab();
    expect(store().getTab(id)!.seed).toBeNull();
  });

  it('starts unseeded when no tab is open', () => {
    useTabStore.setState({ tabs: [], activeTabId: NO_ACTIVE_TAB });
    const id = store().createTab();
    expect(store().getTab(id)!.seed).toBeNull();
  });

  it('copies the value, so the two tabs change independently afterwards', () => {
    store().setSeed(5);
    const first = tab().id;
    const second = store().createTab();
    store().setSeed(6);
    expect(store().getTab(first)!.seed).toBe(5);
    expect(store().getTab(second)!.seed).toBe(6);
  });
});

describe('each tab keeps its own seed', () => {
  it('switching tabs shows each tab its own seed', () => {
    const first = tab().id;
    store().setSeed(1);
    const second = store().createTab();
    store().setSeed(2);

    store().setActiveTab(first);
    expect(tab().seed).toBe(1);
    expect(serialized().settings).toEqual({ seed: 1 });
    store().setActiveTab(second);
    expect(tab().seed).toBe(2);
    expect(serialized().settings).toEqual({ seed: 2 });
  });
});

describe('a seed change is a document change', () => {
  it('documentChanged answers true for a seed change alone', () => {
    const before = tab();
    store().setSeed(3);
    expect(documentChanged(before, tab())).toBe(true);
  });

  it('advances the revision, as the serialized graph now differs', () => {
    const before = tab().revision;
    store().setSeed(3);
    expect(tab().revision).toBe(before + 1);
    // The same value again writes nothing new.
    store().setSeed(3);
    expect(tab().revision).toBe(before + 1);
  });
});
