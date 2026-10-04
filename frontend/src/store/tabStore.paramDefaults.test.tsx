/**
 * #556: a node that comes in with a document behaves like the same node
 * dropped from the palette.
 *
 * AI-Algo-textbook's I1-2 graph stores each `edu:SlidingWindow2D` with only
 * `preset` and `padding`. Opened as stored, switching one to Custom showed a
 * locked grid (the grid takes its shape from `kernel_size`, which was not
 * there) and Run failed with "preset=Custom requires `weights`". A drop fills
 * every param from the definition's defaults; a load now does too, at every
 * door the tab store has:
 *
 *  - `loadGraphDocument` / `loadGraphDocumentInto` (Import, a saved graph, an
 *    example, a workspace file, the plugin API), in the same single commit
 *    and with no undo entry or dirty mark;
 *  - the restore from autosave, and the node list arriving after it;
 *  - paste and template insert, inside their own undo step.
 *
 * A stored value is never replaced, a SECRET param is never filled, and a type
 * the node list does not have is left as stored.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { Node } from '@xyflow/react';

import { useTabStore, _tabFromPersistedForTesting, type PersistedTab } from './tabStore';
import { useNodeDefStore } from './nodeDefStore';
import { buildFlowNode } from '../utils';
import { resolveUnboundDocument } from '../utils/openExample';
import { NodeParamList } from '../components/shared/NodeParamList';
import { useI18n } from '../i18n';
import type { NodeData, NodeDefinition, ParamDefinition, PresetDefinition } from '../types';

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();
const nodeOf = (id: string) => tab().nodes.find((n) => n.id === id)!;

function param(
  name: string,
  paramType: ParamDefinition['param_type'],
  fallback: unknown,
  over: Partial<ParamDefinition> = {},
): ParamDefinition {
  return {
    name, param_type: paramType, default: fallback, description: '',
    options: [], min_value: null, max_value: null, ...over,
  };
}

/** `plugins/edu/nodes/edu_sliding_window_2d_node.py`, as `/api/nodes` reports it. */
const WINDOW: NodeDefinition = {
  node_name: 'edu:SlidingWindow2D',
  category: 'EDU',
  description: 'Slide a kernel over an image, weighted sum per window',
  inputs: [{ name: 'image', data_type: 'TENSOR', description: '', optional: false }],
  outputs: [{ name: 'image', data_type: 'TENSOR', description: '', optional: false }],
  params: [
    param('preset', 'select', 'Blur3x3', {
      options: ['Blur3x3', 'EdgeDetection3x3', 'Sharpen3x3', 'VerticalEdge3x3', 'Custom'],
    }),
    param('kernel_size', 'int', 3, {
      min_value: 1, max_value: 15, visible_when: { preset: 'Custom' },
    }),
    param('weights', 'tensor_grid', [[0, 0, 0], [0, 1, 0], [0, 0, 0]], {
      visible_when: { preset: 'Custom' },
    }),
    param('padding', 'int', 0, { min_value: 0 }),
  ],
};

const SPLIT: NodeDefinition = {
  node_name: 'Split', category: 'Tensor', description: '',
  inputs: [{ name: 'tensor', data_type: 'TENSOR', description: '', optional: false }],
  outputs: [{ name: 'chunk_0', data_type: 'TENSOR', description: '', optional: false }],
  params: [param('chunks', 'int', 2), param('dim', 'int', 0)],
};

/** A key beside a plain setting. */
const CHAT: NodeDefinition = {
  node_name: 'LLMChat', category: 'LLM', description: '', inputs: [], outputs: [],
  params: [param('model', 'string', 'gpt-5.2'), param('openai_api_key', 'secret', '')],
};

const CATALOG = [WINDOW, SPLIT, CHAT];

/** What a palette drop of `definition` starts `name` at. */
function dropped(definition: NodeDefinition, name: string): unknown {
  return buildFlowNode(definition, { x: 0, y: 0 }).data.params[name];
}

/** A serialized node, the shape a graph file stores. */
function raw(id: string, type: string, params: Record<string, unknown>) {
  return { id, type, position: { x: 0, y: 0 }, data: { params } };
}

/** Read a serialized graph the way a reader does, into a document. */
function documentOf(graph: Record<string, unknown>) {
  return resolveUnboundDocument({ edges: [], ...graph });
}

/** What a drop gives a SlidingWindow2D stored with `preset: 'Blur3x3'` only. */
const BLUR_FILLED = {
  preset: 'Blur3x3', kernel_size: 3, weights: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], padding: 0,
};

/** A block definition holding one SlidingWindow2D stored with `preset` only. */
function filterBlock() {
  return {
    id: 'blk', name: 'Filters', description: '', edges: [],
    interface: { inputs: [], outputs: [], triggerTargets: [] },
    nodes: [raw('w', 'edu:SlidingWindow2D', { preset: 'Blur3x3' })],
  };
}

/** The serialized instance of `filterBlock`. */
const INSTANCE = { id: 'inst', type: 'subgraph:blk', position: { x: 0, y: 0 }, data: { params: {} } };

/** The params of the one node inside the active tab's block. */
const blockParams = () => tab().subgraphs[0].nodes[0].data.params;

/** A canvas node as the reader resolved it while the node list was there. */
function windowNode(id: string, params: Record<string, unknown>): Node<NodeData> {
  return {
    id, type: 'pluginNode', position: { x: 0, y: 0 },
    data: {
      label: 'edu:SlidingWindow2D', type: 'edu:SlidingWindow2D', params,
      definition: WINDOW, executionStatus: 'idle',
    },
  };
}

/**
 * The I1-2 graph's nodes and wires, trimmed to one of each kind, every stored
 * param as the textbook ships it. `ImageReader` and `Visualize` are left out
 * of the node list here, so they also stand for a type it does not have.
 */
const I1_2 = {
  name: 'I1-2 complete',
  nodes: [
    raw('start', 'Start', {}),
    raw('img', 'ImageReader', { path: 'sample.png', mode: 'RGB', resize: 0 }),
    raw('win_blur', 'edu:SlidingWindow2D', { preset: 'Blur3x3', padding: 0 }),
    raw('win_edge', 'edu:SlidingWindow2D', { preset: 'EdgeDetection3x3', padding: 0 }),
    raw('viz_blur', 'Visualize', { title: 'Blur', plot_type: 'image' }),
    raw('split', 'Split', { chunks: 3, dim: 0 }),
  ],
  edges: [
    { id: 'e_trig', source: 'start', target: 'img', sourceHandle: 'trigger', targetHandle: '__trigger', type: 'trigger' },
    { id: 'e_img_blur', source: 'img', target: 'win_blur', sourceHandle: 'tensor', targetHandle: 'image' },
    { id: 'e_img_edge', source: 'img', target: 'win_edge', sourceHandle: 'tensor', targetHandle: 'image' },
    { id: 'e_blur_viz', source: 'win_blur', target: 'viz_blur', sourceHandle: 'image', targetHandle: 'data' },
    { id: 'e_img_split', source: 'img', target: 'split', sourceHandle: 'tensor', targetHandle: 'tensor' },
  ],
  presets: [],
  segmentGroups: [],
  subgraphs: [],
};

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [...CATALOG], presets: [] });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
  localStorage.clear();
});

describe('opening a document (#556)', () => {
  it('gives a node every param its definition has a default for, at the palette default', () => {
    store().loadGraphDocument(documentOf(I1_2));

    const params = nodeOf('win_blur').data.params;
    expect(params.kernel_size).toBe(dropped(WINDOW, 'kernel_size'));
    expect(params.weights).toEqual(dropped(WINDOW, 'weights'));
    // The stored values are the document's.
    expect(params.preset).toBe('Blur3x3');
    expect(params.padding).toBe(0);
    expect(nodeOf('win_edge').data.params.preset).toBe('EdgeDetection3x3');
    expect(nodeOf('win_edge').data.params.weights).toEqual(dropped(WINDOW, 'weights'));
  });

  it('never replaces a stored value', () => {
    store().loadGraphDocument(documentOf({
      nodes: [raw('w', 'edu:SlidingWindow2D', {
        preset: 'Custom', kernel_size: 2, weights: [[1, 2], [3, 4]], padding: null,
      })],
    }));
    expect(nodeOf('w').data.params).toEqual({
      preset: 'Custom', kernel_size: 2, weights: [[1, 2], [3, 4]], padding: null,
    });
  });

  it('leaves a node whose type the node list does not have as stored', () => {
    store().loadGraphDocument(documentOf(I1_2));
    expect(nodeOf('img').data.params).toEqual({ path: 'sample.png', mode: 'RGB', resize: 0 });
    expect(nodeOf('viz_blur').data.params).toEqual({ title: 'Blur', plot_type: 'image' });
    // A type the list has, with nothing missing, is not copied either.
    const doc = documentOf(I1_2);
    const split = doc.nodes.find((n) => n.id === 'split')!;
    store().loadGraphDocument(doc);
    expect(nodeOf('split')).toBe(split);
  });

  it('is part of the one install: one emission, one revision step, no undo entry, nothing dirty', () => {
    const before = tab().revision;
    let emissions = 0;
    const unsubscribe = useTabStore.subscribe(() => {
      emissions += 1;
    });
    store().loadGraphDocument(documentOf(I1_2));
    unsubscribe();

    expect(emissions).toBe(1);
    expect(tab().revision).toBe(before + 1);
    expect(tab().undoStack).toEqual([]);
    expect(tab().redoStack).toEqual([]);
    expect(tab().dirtyNodeIds.size).toBe(0);
    expect(nodeOf('win_blur').data.params.weights).toBeDefined();
  });

  it('fills a background tab a workspace file or a plugin opens', () => {
    const background = store().createTab({ title: 'Lab', activate: false });
    store().loadGraphDocumentInto(background, documentOf(I1_2));
    const opened = store().getTab(background)!;
    const node = opened.nodes.find((n) => n.id === 'win_blur')!;
    expect(node.data.params.kernel_size).toBe(3);
    expect(opened.undoStack).toEqual([]);
    expect(opened.dirtyNodeIds.size).toBe(0);
  });

  it('never fills a SECRET param, and leaves a preset card as stored', () => {
    const card: PresetDefinition = {
      preset_name: 'Chatter', category: 'Portable', description: '', tags: [],
      nodes: [{ id: 'inner', type: 'LLMChat', params: {} }], edges: [],
      exposed_inputs: [], exposed_outputs: [], exposed_params: [],
    };
    const internalParams = { inner: {} };
    store().loadGraphDocument(documentOf({
      nodes: [
        raw('chat', 'LLMChat', {}),
        { id: 'card', type: 'preset:Chatter', position: { x: 0, y: 0 }, data: { params: {}, internalParams } },
      ],
      presets: [card],
    }));
    expect(nodeOf('chat').data.params).toEqual({ model: 'gpt-5.2' });
    expect(nodeOf('card').data.params).toEqual({});
    expect(nodeOf('card').data.internalParams).toBe(internalParams);
  });

  it('fills the nodes inside a block the document brings, and looking inside costs no undo step', () => {
    store().loadGraphDocument(documentOf({ nodes: [INSTANCE], subgraphs: [filterBlock()] }));
    expect(blockParams()).toEqual(BLUR_FILLED);

    expect(store().enterSubgraph('inst')).toBe(true);
    expect(nodeOf('w').data.params).toEqual(BLUR_FILLED);
    store().exitSubgraph();
    expect(tab().undoStack).toEqual([]);
  });
});

describe('restoring from autosave (#556)', () => {
  /** A record 2.8.7 wrote for a tab holding an imported, unfilled I1-2 node. */
  function record(): PersistedTab {
    return {
      id: 'restored', name: 'I1-2', revision: 7,
      nodes: [windowNode('w', { preset: 'Blur3x3', padding: 0 })],
      edges: [],
    };
  }

  it('fills a restored node when the node list is already there', () => {
    const restored = _tabFromPersistedForTesting(record(), tab());
    expect(restored.nodes[0].data.params).toEqual({
      preset: 'Blur3x3', padding: 0, kernel_size: 3,
      weights: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    });
    // The record's number stands: the loader states it.
    expect(restored.revision).toBe(7);
  });

  it('fills the nodes inside a restored block', () => {
    const restored = _tabFromPersistedForTesting(
      { ...record(), subgraphs: [filterBlock()] },
      tab(),
    );
    expect(restored.subgraphs[0].nodes[0].data.params).toEqual(BLUR_FILLED);
  });

  it('leaves it as stored while the list has not answered, and fills it once when it does', () => {
    useNodeDefStore.setState({ definitions: [] });
    // What hydration does: install the restored tabs with the raw setter.
    const restored = _tabFromPersistedForTesting(record(), tab());
    useTabStore.setState({ tabs: [restored], activeTabId: restored.id });
    expect(tab().nodes[0].data.params).toEqual({ preset: 'Blur3x3', padding: 0 });

    useNodeDefStore.setState({ definitions: [...CATALOG] });

    expect(tab().nodes[0].data.params).toEqual({
      preset: 'Blur3x3', padding: 0, kernel_size: 3,
      weights: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    });
    expect(tab().undoStack).toEqual([]);
    expect(tab().dirtyNodeIds.size).toBe(0);
    // The document changed, so a plugin holding the old number must be told.
    expect(tab().revision).toBe(8);

    // A refetch with nothing new for this tab touches nothing.
    const settled = useTabStore.getState().tabs;
    useNodeDefStore.setState({ definitions: [...CATALOG] });
    expect(useTabStore.getState().tabs).toBe(settled);
  });
});

describe('a node list that arrives or changes later (#556)', () => {
  it('fills every open tab, leaving unknown types and stored values alone', () => {
    useNodeDefStore.setState({ definitions: [] });
    store().loadGraphDocument(documentOf(I1_2));
    const second = store().createTab({ title: 'Second', activate: false });
    store().loadGraphDocumentInto(second, documentOf({
      nodes: [raw('w2', 'edu:SlidingWindow2D', { preset: 'Custom', kernel_size: 5 })],
    }));
    // Opened before the list answered: stubs, as stored.
    expect(nodeOf('win_blur').data.params).toEqual({ preset: 'Blur3x3', padding: 0 });

    useNodeDefStore.setState({ definitions: [...CATALOG] });

    expect(nodeOf('win_blur').data.params.weights).toEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
    expect(nodeOf('img').data.params).toEqual({ path: 'sample.png', mode: 'RGB', resize: 0 });
    const w2 = store().getTab(second)!.nodes[0];
    expect(w2.data.params).toEqual({
      preset: 'Custom', kernel_size: 5, weights: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], padding: 0,
    });
    expect(store().getTab(second)!.undoStack).toEqual([]);
  });

  it('fills the nodes inside a block too', () => {
    useNodeDefStore.setState({ definitions: [] });
    store().loadGraphDocument(documentOf({ nodes: [INSTANCE], subgraphs: [filterBlock()] }));
    expect(blockParams()).toEqual({ preset: 'Blur3x3' });

    useNodeDefStore.setState({ definitions: [...CATALOG] });

    expect(blockParams()).toEqual(BLUR_FILLED);
    expect(tab().undoStack).toEqual([]);
  });

  it('leaves a tab alone while a block is open in it', () => {
    useNodeDefStore.setState({ definitions: [] });
    store().loadGraphDocument(documentOf({
      nodes: [INSTANCE, raw('top', 'edu:SlidingWindow2D', { preset: 'Blur3x3' })],
      subgraphs: [filterBlock()],
    }));
    expect(store().enterSubgraph('inst')).toBe(true);
    const inside = useTabStore.getState().tabs;

    useNodeDefStore.setState({ definitions: [...CATALOG] });

    // Folding the block back is compared against the copy taken on entry, so
    // filling either side now would charge the user an undo step for the visit.
    expect(useTabStore.getState().tabs).toBe(inside);
    store().exitSubgraph();
    expect(tab().undoStack).toEqual([]);
  });
});

describe('paste and insert (#556)', () => {
  it('insertGraph fills the inserted nodes inside its own single undo step', () => {
    store().insertGraph([windowNode('w', { preset: 'Blur3x3' })], []);
    const inserted = tab().nodes[0];
    expect(inserted.data.params.kernel_size).toBe(3);
    expect(inserted.data.params.weights).toEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
    expect(tab().undoStack).toHaveLength(1);
  });

  it('pasteNodes fills a clipboard copied before the node list answered', () => {
    useTabStore.setState({
      clipboard: { nodes: [windowNode('w', { preset: 'Sharpen3x3' })], edges: [] },
    });
    store().pasteNodes();
    const pasted = tab().nodes[0];
    expect(pasted.data.params.preset).toBe('Sharpen3x3');
    expect(pasted.data.params.weights).toEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
    expect(tab().undoStack).toHaveLength(1);
  });

  it('both fill the nodes inside a block they bring', () => {
    const template = documentOf({ nodes: [INSTANCE], subgraphs: [filterBlock()] });
    store().insertGraph(template.nodes, template.edges, template.subgraphs);
    expect(blockParams()).toEqual(BLUR_FILLED);

    store().addTab('Paste target');
    useTabStore.setState({
      clipboard: { nodes: template.nodes, edges: [], subgraphs: [filterBlock()] },
    });
    store().pasteNodes();
    expect(blockParams()).toEqual(BLUR_FILLED);
  });
});

describe('the I1-2 graph, end to end (#556)', () => {
  it('Custom opens an editable grid, and Run sends `weights`', () => {
    store().loadGraphDocument(documentOf(I1_2));
    // What the panel's preset select does.
    store().updateNodeParams('win_blur', { preset: 'Custom' });
    const node = nodeOf('win_blur');

    render(
      <NodeParamList
        nodeId={node.id}
        definition={node.data.definition}
        params={node.data.params}
      />,
    );
    // The locked grid said "Set value_mode to explicit to edit these values."
    expect(screen.queryByText(/to edit these values/)).toBeNull();
    expect(screen.getByText(/\[3, 3\] · 9 cells/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Fill 0' })).toBeInTheDocument();

    const graph = store().getSerializedGraph({ keepSecrets: true });
    const sent = graph.nodes.find((n) => n.id === 'win_blur')!;
    expect(sent.data.params).toEqual({
      preset: 'Custom', padding: 0, kernel_size: 3,
      weights: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    });
    const edge = graph.nodes.find((n) => n.id === 'win_edge')!;
    expect(edge.data.params.weights).toEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
  });
});
