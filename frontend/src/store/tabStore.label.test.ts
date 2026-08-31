/**
 * `data.label` survives serialization (#342).
 *
 * The deserializer has always read it (`utils/index.ts:505`,
 * `label: raw.data?.label ?? nodeType`) and the serializer never wrote it, so
 * a node renamed in the editor lost its name on save and `getGraph()` never
 * showed it. That asymmetry is what #342's round-trip criterion runs into.
 *
 * The other half of the fix is the one that keeps it cheap: a node nobody has
 * renamed serializes byte-identically to before, so this does not rewrite
 * every graph file in the repository.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import { resolveSerializedNodes } from '../utils';
import type { NodeDefinition } from '../types';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();

const DEF: NodeDefinition = {
  node_name: 'Dataset', category: 'data', description: '',
  inputs: [], outputs: [], params: [],
};

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  useNodeDefStore.setState({ definitions: [DEF], presets: [] } as never);
});

describe('label serialization', () => {
  it('omits label on a node nobody renamed', () => {
    store().addNode(DEF, { x: 0, y: 0 });
    const [node] = store().getSerializedGraph().nodes;
    expect(node.data.label).toBeUndefined();
    expect(Object.keys(node.data)).toEqual(['params']);
  });

  it('emits label once the node is renamed', () => {
    store().addNode(DEF, { x: 0, y: 0 });
    const id = store().getActiveTab().nodes[0].id;
    store().renameNode(id, 'Training set');
    const [node] = store().getSerializedGraph().nodes;
    expect(node.data.label).toBe('Training set');
  });

  it('round-trips: rename, serialize, resolve, still named', () => {
    store().addNode(DEF, { x: 0, y: 0 });
    const id = store().getActiveTab().nodes[0].id;
    store().renameNode(id, 'Training set');
    const serialized = store().getSerializedGraph();
    const resolved = resolveSerializedNodes(serialized.nodes, [DEF], []);
    expect(resolved[0].data.label).toBe('Training set');
  });

  it('a note carries no label field of its own', () => {
    store().addNote('text', { x: 0, y: 0 });
    const [note] = store().getSerializedGraph().nodes;
    expect(note.type).toBe('note');
    expect(note.data.label).toBeUndefined();
  });
});

/**
 * Start was the one node class the reader ignored a saved label on (review I1).
 *
 * `resolveSerializedNodes` has a dedicated Start branch, and it hardcoded
 * `label: 'Start'` instead of reading `raw.data?.label`. So a renamed Start --
 * by the canvas context menu, or by an agent's `set_node_meta`, which both
 * write the same `data.label` -- serialized its new name, came back as 'Start'
 * on the next load, and lost the key entirely on the save after that. The
 * rename reported success at every step and reverted anyway.
 */
describe('label serialization — the Start node', () => {
  const START: NodeDefinition = {
    node_name: 'Start', category: 'Control', description: '',
    inputs: [],
    outputs: [{ name: 'trigger', data_type: 'TRIGGER', description: '', optional: false }],
    params: [],
  };

  it('round-trips a renamed Start', () => {
    store().addNode(START, { x: 0, y: 0 });
    const id = store().getActiveTab().nodes[0].id;
    store().renameNode(id, 'Entry');

    const serialized = store().getSerializedGraph();
    expect(serialized.nodes[0].type).toBe('Start');
    expect(serialized.nodes[0].data.label).toBe('Entry');

    const resolved = resolveSerializedNodes(serialized.nodes, [START], []);
    expect(resolved[0].data.label).toBe('Entry');
  });

  it('falls back to Start when the file carries no label', () => {
    store().addNode(START, { x: 0, y: 0 });
    const serialized = store().getSerializedGraph();
    expect(serialized.nodes[0].data.label).toBeUndefined();

    const resolved = resolveSerializedNodes(serialized.nodes, [START], []);
    expect(resolved[0].data.label).toBe('Start');
    // The rest of the Start branch is untouched by the fix.
    expect(resolved[0].type).toBe('start');
    expect(resolved[0].data.type).toBe('Start');
  });
});
