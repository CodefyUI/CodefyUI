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
