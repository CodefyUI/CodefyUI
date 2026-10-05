/**
 * A graph a plugin opens with `api.workspace.openGraphs` keeps asking before
 * its tab closes (#596).
 *
 * Every install of a document records that the tab matches a file
 * (`savedRevision`), which is what lets an unchanged graph opened from the
 * Graphs panel close without the "cannot be undone" warning. A plugin's graph
 * is not known to be in any file -- an agent's candidate can exist only in
 * the plugin's memory -- so this caller takes the match back
 * (`forgetTabMatch`).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useTabStore, tabHasUnsavedWork } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useProjectStore } from '../store/projectStore';
import { buildPluginAPI } from './api';
import type { NodeDefinition } from '../types';

vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();

const DEFS: NodeDefinition[] = [
  {
    node_name: 'Source', category: 'Layer', description: '',
    inputs: [],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  },
];

beforeEach(() => {
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('live');
  useNodeDefStore.setState({ definitions: DEFS, presets: [] } as never);
  window.localStorage.clear();
});

describe('workspace.openGraphs and the close confirm (#596)', () => {
  it('a graph a plugin opened asks before its tab closes, changed or not', () => {
    const api = buildPluginAPI('test-plugin', () => document.createElement('div'));

    const [result] = api.workspace.openGraphs(
      [{
        title: 'Variant A',
        graph: {
          name: 'candidate', description: '',
          nodes: [{ id: 'a', type: 'Source', position: { x: 0, y: 0 }, data: { params: {} } }],
          edges: [],
        },
        persist: true,
      }],
      { activate: 'none' },
    );

    expect('tabId' in result).toBe(true);
    if (!('tabId' in result)) return;
    const tab = store().getTab(result.tabId)!;
    expect(tab.savedRevision).toBeNull();
    expect(tabHasUnsavedWork(tab)).toBe(true);
    // Taking the match back moves no revision: the number handed to the
    // plugin is still the tab's.
    expect(result.revision).toBe(tab.revision);
  });
});
