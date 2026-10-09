/**
 * Show on a toast raised while a block is open (found during the browser e2e
 * of the fixes for #618-#625): a Run or an Export refused from inside a block
 * names a node the canvas on screen shows, and offered no way to it.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { Edge, Node } from '@xyflow/react';

import { dismissValidationToasts, showValidationIssues } from './validationToasts';
import { buildInstanceNode, subgraphIdOf } from './subgraph';
import type { ValidationIssue } from '../api/rest';
import { useI18n } from '../i18n';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';
import type { NodeData, NodeDefinition, SubgraphDefinition } from '../types';

vi.mock('../store/tabPersistence', () => ({
  readSnapshot: vi.fn(async () => null),
  writeSnapshot: vi.fn(async () => {}),
}));

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();
const toasts = () => useToastStore.getState().toasts;

function def(name: string): NodeDefinition {
  return {
    node_name: name,
    category: 'x',
    description: '',
    inputs: [{ name: 'in', data_type: 'TENSOR', description: '', optional: false }],
    outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function node(id: string, type: string, x: number): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x, y: 0 },
    data: { label: id, type, params: {}, definition: def(type), executionStatus: 'idle' },
  };
}

function dataEdge(id: string, source: string, target: string): Edge {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

/** a -> [b -> c] with b and c collapsed into "Block"; returns the card id. */
function graphWithABlock(): string {
  store().setNodes([node('a', 'A', 0), node('b', 'B', 100), node('c', 'C', 200)]);
  store().setEdges([dataEdge('e1', 'a', 'b'), dataEdge('e2', 'b', 'c')]);
  store().setNodes(tab().nodes.map((n) => ({ ...n, selected: n.id !== 'a' })));
  expect(store().collapseSelectionToSubgraph('Block').ok).toBe(true);
  return tab().nodes.find((n) => subgraphIdOf(n.data.type))!.id;
}

function missingInput(nodeId: string): ValidationIssue {
  return {
    message: `Missing required input 'in' on node ${nodeId}`,
    code: 'missing_input',
    node_id: nodeId,
    params: { port: 'in', type: 'B' },
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({ layoutFitRequests: {} });
  dismissValidationToasts();
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  store().addTab('test');
  useNodeDefStore.setState({
    definitions: [def('A'), def('B'), def('C'), def('Mul')],
    presets: [],
    categorized: {},
    loading: false,
    error: null,
  } as never);
});

describe('Show while a block is open', () => {
  it('selects the node of the open block the finding names', () => {
    const card = graphWithABlock();
    store().enterSubgraph(card);

    showValidationIssues(tab().id, [missingInput(`${card}/b`)]);

    expect(toasts().map((toast) => toast.message)).toEqual(['Block: input "in" is not connected']);
    expect(toasts()[0].action?.label).toBe('Show');
    toasts()[0].action!.onClick();
    expect(tab().selectedNodeId).toBe('b');
    expect(useUIStore.getState().layoutFitRequests[tab().id]).toBeDefined();
  });

  it('selects the card of a block inside the open block', () => {
    const inner: SubgraphDefinition = {
      id: 'inner', name: 'Inner', description: '',
      nodes: [{ id: 'mul2', type: 'Mul', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [], interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    const outer: SubgraphDefinition = {
      id: 'outer', name: 'Outer', description: '',
      nodes: [
        { id: 'mul', type: 'Mul', position: { x: 0, y: 0 }, data: { params: {} } },
        { id: 'nest', type: 'subgraph:inner', position: { x: 200, y: 0 }, data: { params: {} } },
      ],
      edges: [], interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    store().setSubgraphs([outer, inner]);
    store().setNodes([buildInstanceNode(outer, { x: 0, y: 0 }, 'blk')]);
    store().enterSubgraph('blk');

    showValidationIssues(tab().id, [missingInput('blk/nest/mul2')]);

    toasts()[0].action!.onClick();
    expect(tab().selectedNodeId).toBe('nest');
  });

  it('offers no Show for a node outside the open block', () => {
    const card = graphWithABlock();
    store().enterSubgraph(card);

    showValidationIssues(tab().id, [missingInput('a')]);

    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].action).toBeUndefined();
  });

  it('does nothing once the canvas on screen is another level', () => {
    // The same canvas id can name another node there.
    const card = graphWithABlock();
    store().enterSubgraph(card);
    showValidationIssues(tab().id, [missingInput(`${card}/b`)]);
    store().exitSubgraph();
    store().setNodes([...tab().nodes, node('b', 'B', 500)]);
    const selected = tab().selectedNodeId;

    toasts()[0].action!.onClick();

    expect(tab().selectedNodeId).toBe(selected);
    expect(useUIStore.getState().layoutFitRequests).toEqual({});
  });
});
