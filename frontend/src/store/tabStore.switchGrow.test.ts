/**
 * A wire on a Switch's last empty input adds another (#655), in the same
 * undo step as the wire.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useTabStore } from './tabStore';
import type { NodeDefinition } from '../types';

vi.mock('./tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();

const SWITCH: NodeDefinition = {
  node_name: 'Switch', category: 'Data Flow', description: '',
  inputs: [], outputs: [{ name: 'output', data_type: 'ANY', description: '', optional: false }],
  params: [
    { name: 'selector', param_type: 'int', default: 0, description: '', options: [], min_value: 0, max_value: 15 },
    { name: 'inputs', param_type: 'int', default: 4, description: '', options: [], min_value: 2, max_value: 16 },
  ],
};
const SOURCE: NodeDefinition = {
  node_name: 'TensorCreate', category: 'Tensor', description: '',
  inputs: [], outputs: [{ name: 'tensor', data_type: 'TENSOR', description: '', optional: false }],
  params: [],
};

function setUp() {
  store().addNode(SOURCE, { x: 0, y: 0 });
  store().addNode(SWITCH, { x: 300, y: 0 });
  const [source, sw] = store().getActiveTab().nodes;
  return { source: source.id, sw: sw.id };
}

const switchParams = (id: string) => store().getActiveTab().nodes.find((n) => n.id === id)!.data.params;

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
});

describe('a Switch grows as it is wired', () => {
  it('adds an input when the last one is wired', () => {
    const { source, sw } = setUp();
    store().onConnect({ source, sourceHandle: 'tensor', target: sw, targetHandle: 'input_3' });
    expect(switchParams(sw).inputs).toBe(5);
    expect(store().getActiveTab().edges).toHaveLength(1);
  });

  it('leaves the count alone for another input or the selector', () => {
    const { source, sw } = setUp();
    store().onConnect({ source, sourceHandle: 'tensor', target: sw, targetHandle: 'input_1' });
    store().onConnect({ source, sourceHandle: 'tensor', target: sw, targetHandle: 'selector' });
    expect(switchParams(sw).inputs).toBe(4);
  });

  it('undoes the wire and the new input in one step', () => {
    const { source, sw } = setUp();
    store().onConnect({ source, sourceHandle: 'tensor', target: sw, targetHandle: 'input_3' });
    store().undo();
    expect(switchParams(sw).inputs).toBe(4);
    expect(store().getActiveTab().edges).toHaveLength(0);
  });
});
