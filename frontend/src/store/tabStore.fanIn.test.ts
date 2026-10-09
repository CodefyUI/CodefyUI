/**
 * The quick-fix for a graph saved with several wires into one input (#658),
 * and the notice a load of one raises.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { useTabStore } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import { useToastStore } from './toastStore';
import { useI18n } from '../i18n';
import type { NodeData, NodeDefinition } from '../types';

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

function card(id: string): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: 'X', params: {} } };
}
const wire = (id: string, source: string, target: string, targetHandle = 'value'): Edge => ({
  id, source, sourceHandle: 'out', target, targetHandle,
});

const NODES = [card('a'), card('b'), card('sink'), card('other')];
const EDGES = [
  wire('e1', 'a', 'sink'), wire('e2', 'b', 'sink'),
  wire('e3', 'a', 'other', 'x'), wire('e4', 'b', 'other', 'x'),
];

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('test');
  useNodeDefStore.setState({ definitions: [SWITCH], presets: [] } as never);
  useToastStore.setState({ toasts: [] });
});

const tab = () => store().getActiveTab();

describe('insertSwitchesForFanIn', () => {
  beforeEach(() => {
    useTabStore.setState((s) => ({
      tabs: s.tabs.map((t) => ({ ...t, nodes: NODES, edges: EDGES })),
    }));
  });

  it('fixes every such input in one undo step and selects the last Switch', () => {
    expect(store().insertSwitchesForFanIn()).toBe(2);
    const switches = tab().nodes.filter((n) => n.data.type === 'Switch');
    expect(switches).toHaveLength(2);
    expect(tab().selectedNodeId).toBe(switches[1].id);
    for (const target of ['sink', 'other']) {
      expect(tab().edges.filter((e) => e.target === target)).toHaveLength(1);
    }
    store().undo();
    expect(tab().nodes).toHaveLength(4);
    expect(tab().edges.map((e) => e.id)).toEqual(['e1', 'e2', 'e3', 'e4']);
  });

  it('fixes only the inputs it is given', () => {
    expect(store().insertSwitchesForFanIn([{ nodeId: 'other', port: 'x' }])).toBe(1);
    expect(tab().edges.filter((e) => e.target === 'sink')).toHaveLength(2);
  });

  it('does nothing without a Switch in the node list, or with nothing to fix', () => {
    useNodeDefStore.setState({ definitions: [] } as never);
    expect(store().insertSwitchesForFanIn()).toBe(0);
    useNodeDefStore.setState({ definitions: [SWITCH] } as never);
    expect(store().insertSwitchesForFanIn([{ nodeId: 'nope', port: 'value' }])).toBe(0);
    expect(tab().undoStack).toHaveLength(0);
  });

  it('skips an input with more wires than a Switch takes', () => {
    const many = Array.from({ length: 17 }, (_, i) => wire(`m${i}`, 'a', 'sink'));
    useTabStore.setState((s) => ({ tabs: s.tabs.map((t) => ({ ...t, edges: many })) }));
    expect(store().insertSwitchesForFanIn()).toBe(0);
  });
});

describe('opening a graph with several wires into one input', () => {
  it('says so and offers the fix', () => {
    store().loadGraphDocument({ nodes: NODES, edges: EDGES, boundFile: null });
    const [toast] = useToastStore.getState().toasts;
    expect(toast.type).toBe('warning');
    expect(toast.message).toContain('2 input(s)');
    toast.action!.onClick();
    expect(tab().nodes.filter((n) => n.data.type === 'Switch')).toHaveLength(2);
  });

  it('does not fix a tab that is no longer on screen', () => {
    store().loadGraphDocument({ nodes: NODES, edges: EDGES, boundFile: null });
    const [toast] = useToastStore.getState().toasts;
    const first = store().activeTabId;
    store().addTab('another');
    toast.action!.onClick();
    expect(store().tabs.find((t) => t.id === first)!.nodes).toHaveLength(4);
  });

  it('says nothing for a graph with one wire per input', () => {
    store().loadGraphDocument({ nodes: NODES, edges: [EDGES[0], EDGES[2]], boundFile: null });
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });
});
