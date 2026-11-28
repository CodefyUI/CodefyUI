/**
 * Phase 4C: a graph that uses `project:` nodes while the project's mods are
 * NOT loaded. The nodes carry a placeholder definition (no ports), and a save
 * of that tab must write them back exactly as they were: type, params, label
 * and every wire to them. Then, once the mods are trusted, the same nodes
 * get their real definition without anything else changing.
 */
import { beforeEach, describe, expect, it } from 'vitest';

import { useNodeDefStore } from './nodeDefStore';
import { useTabStore } from './tabStore';
import type { NodeDefinition } from '../types';
import { resolveNodeComponentType, resolveSerializedNodes } from '../utils';

const store = () => useTabStore.getState();

const CONV: NodeDefinition = {
  node_name: 'Conv2d', category: 'CNN', description: '',
  inputs: [{ name: 'tensor', data_type: 'TENSOR', description: '', optional: false }],
  outputs: [{ name: 'tensor', data_type: 'TENSOR', description: '', optional: false }],
  params: [{ name: 'out_channels', param_type: 'int', default: 8, description: '' }],
} as unknown as NodeDefinition;

const FOCAL: NodeDefinition = {
  node_name: 'project:FocalLoss', category: 'Loss', description: 'from mods',
  inputs: [{ name: 'logits', data_type: 'TENSOR', description: '', optional: false }],
  outputs: [{ name: 'loss', data_type: 'TENSOR', description: '', optional: false }],
  params: [{ name: 'gamma', param_type: 'float', default: 2, description: '' }],
  provider: 'project',
} as unknown as NodeDefinition;

const SAVED = {
  nodes: [
    { id: 'c', type: 'Conv2d', position: { x: 0, y: 0 }, data: { params: { out_channels: 8 } } },
    { id: 'f', type: 'project:FocalLoss', position: { x: 300, y: 0 },
      data: { params: { gamma: 3.5, alpha: 0.25 }, label: 'My focal' } },
  ],
  edges: [
    { id: 'e1', source: 'c', sourceHandle: 'tensor', target: 'f', targetHandle: 'logits' },
  ],
};

function open(definitions: NodeDefinition[]) {
  useNodeDefStore.setState({ definitions });
  store().loadGraphDocument({
    nodes: resolveSerializedNodes(SAVED.nodes, definitions, []),
    edges: SAVED.edges as never,
    boundFile: 'demo',
  });
}

function projectNode(graph: unknown) {
  return (graph as { nodes: { id: string; type: string; data: Record<string, unknown> }[] })
    .nodes.find((n) => n.id === 'f')!;
}

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
  localStorage.clear();
});

describe('project: nodes while mods are not loaded', () => {
  it('never routes a project node to a viz renderer or the plugin bridge', () => {
    expect(resolveNodeComponentType('project:AttentionHeatmap')).toBe('baseNode');
    expect(resolveNodeComponentType('project:Anything')).toBe('baseNode');
    expect(resolveNodeComponentType('edu:Anything')).toBe('pluginNode');
  });

  it('decline, open, edit something else, save, trust: the nodes come back intact', () => {
    // Declined: the server lists no project:FocalLoss.
    open([CONV]);
    const placeholder = store().getActiveTab().nodes.find((n) => n.id === 'f')!;
    expect(placeholder.data.definition?.inputs).toEqual([]);

    // Edit something else, then save.
    store().updateNodeParams('c', { out_channels: 16 });
    const saved = store().getSerializedGraph() as unknown as {
      nodes: { id: string; type: string; data: Record<string, unknown> }[];
      edges: { source: string; target: string; sourceHandle?: string; targetHandle?: string }[];
    };
    const kept = saved.nodes.find((n) => n.id === 'f')!;
    expect(kept.type).toBe('project:FocalLoss');
    expect((kept.data as { params: unknown }).params).toEqual({ gamma: 3.5, alpha: 0.25 });
    expect((kept.data as { label?: string }).label).toBe('My focal');
    expect(saved.edges).toEqual([expect.objectContaining({
      source: 'c', sourceHandle: 'tensor', target: 'f', targetHandle: 'logits',
    })]);
    expect((saved.nodes.find((n) => n.id === 'c')!.data as { params: { out_channels: number } })
      .params.out_channels).toBe(16);

    // Trust: the node list now has the type; the open card gets it.
    useNodeDefStore.setState({ definitions: [CONV, FOCAL] });
    store().refreshNodeDefinitions('project:');
    const live = store().getActiveTab().nodes.find((n) => n.id === 'f')!;
    expect(live.data.definition).toBe(FOCAL);
    const after = store().getSerializedGraph();
    expect(projectNode(after).type).toBe('project:FocalLoss');
    expect((projectNode(after).data as { params: unknown }).params)
      .toEqual({ gamma: 3.5, alpha: 0.25 });
  });

  it('refreshing leaves every other node and tab untouched', () => {
    open([CONV]);
    const before = store().getActiveTab();
    store().refreshNodeDefinitions('project:');   // nothing new to give
    expect(store().getActiveTab()).toBe(before);
  });
});
