/**
 * #556: a node a document brought gets every param its definition has a
 * default for, the value a palette drop gives it.
 *
 * Pinned here, on the two pure functions every door of the tab store calls:
 *  - a missing param gets exactly what `buildFlowNode` puts there;
 *  - a stored value is never replaced, `null` and falsy values included;
 *  - a SECRET param is never filled;
 *  - a type the node list does not have is left exactly as stored;
 *  - nothing to fill hands back the same array, so the persistence record
 *    cache and the revision counter see no change.
 */
import { describe, it, expect } from 'vitest';
import type { Node } from '@xyflow/react';

import { buildFlowNode } from '.';
import { withParamDefaults, withSubgraphParamDefaults } from './paramDefaults';
import type { NodeData, NodeDefinition, ParamDefinition, SubgraphDefinition } from '../types';

function param(
  name: string,
  paramType: ParamDefinition['param_type'],
  fallback: unknown,
): ParamDefinition {
  return {
    name, param_type: paramType, default: fallback, description: '',
    options: [], min_value: null, max_value: null,
  };
}

/** Shaped like `edu:SlidingWindow2D`, the node #556 was found on. */
const WINDOW: NodeDefinition = {
  node_name: 'edu:SlidingWindow2D', category: 'EDU', description: '',
  inputs: [{ name: 'image', data_type: 'TENSOR', description: '', optional: false }],
  outputs: [{ name: 'image', data_type: 'TENSOR', description: '', optional: false }],
  params: [
    { ...param('preset', 'select', 'Blur3x3'), options: ['Blur3x3', 'Custom'] },
    param('kernel_size', 'int', 3),
    param('weights', 'tensor_grid', [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    param('padding', 'int', 0),
  ],
};

/** A type with a key beside a plain setting, and a param with no default. */
const CHAT: NodeDefinition = {
  node_name: 'LLMChat', category: 'LLM', description: '', inputs: [], outputs: [],
  params: [
    param('model', 'string', 'gpt-5.2'),
    param('api_key', 'secret', ''),
    param('values', 'tensor_grid', null),
    param('unset', 'string', undefined),
  ],
};

const CATALOG = [WINDOW, CHAT];

function canvasNode(
  id: string,
  type: string,
  params: unknown,
  extra: Partial<NodeData> = {},
): Node<NodeData> {
  return {
    id, type: 'baseNode', position: { x: 0, y: 0 },
    data: { label: id, type, params, executionStatus: 'idle', ...extra } as NodeData,
  };
}

/** What a palette drop of `definition` starts `name` at. */
function dropped(definition: NodeDefinition, name: string): unknown {
  return buildFlowNode(definition, { x: 0, y: 0 }).data.params[name];
}

describe('withParamDefaults', () => {
  it('fills every missing param with what a palette drop gives it', () => {
    const [node] = withParamDefaults(
      [canvasNode('w', 'edu:SlidingWindow2D', { preset: 'Blur3x3', padding: 0 })],
      CATALOG,
    );
    expect(node.data.params).toEqual({
      preset: 'Blur3x3',
      padding: 0,
      kernel_size: dropped(WINDOW, 'kernel_size'),
      weights: dropped(WINDOW, 'weights'),
    });
    expect(node.data.params.kernel_size).toBe(3);
    expect(node.data.params.weights).toEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
  });

  it('never replaces a stored value, null and falsy ones included', () => {
    const stored = { preset: 'Custom', kernel_size: 0, weights: null, padding: '' };
    const nodes = [canvasNode('w', 'edu:SlidingWindow2D', stored)];
    const out = withParamDefaults(nodes, CATALOG);
    expect(out).toBe(nodes);
    expect(out[0].data.params).toBe(stored);
  });

  it('fills a param whose stored value is undefined, which no file can hold', () => {
    const [node] = withParamDefaults(
      [canvasNode('w', 'edu:SlidingWindow2D', {
        preset: 'Blur3x3', kernel_size: undefined, weights: [[1]], padding: 1,
      })],
      CATALOG,
    );
    expect(node.data.params.kernel_size).toBe(3);
    expect(node.data.params.weights).toEqual([[1]]);
  });

  it('never fills a SECRET param, and fills a null default the way a drop does', () => {
    const [node] = withParamDefaults([canvasNode('c', 'LLMChat', {})], CATALOG);
    expect(node.data.params).toEqual({ model: 'gpt-5.2', values: null });
    expect('api_key' in node.data.params).toBe(false);
    // A default the definition does not state is no default at all.
    expect('unset' in node.data.params).toBe(false);
  });

  it('leaves a node whose type the list does not have exactly as stored', () => {
    const stub: NodeDefinition = {
      node_name: 'edu:Missing', category: 'Utility', description: '',
      inputs: [], outputs: [], params: [],
    };
    const unknown = canvasNode('u', 'edu:Missing', { a: 1 }, { definition: stub });
    // The palette resolves exact names only; a bare name a plugin type would
    // answer to on the server is a stub on the canvas, and stays one here.
    const bare = canvasNode('b', 'SlidingWindow2D', { preset: 'Blur3x3' });
    const nodes = [unknown, bare];
    const out = withParamDefaults(nodes, CATALOG);
    expect(out).toBe(nodes);
    expect(out[0]).toBe(unknown);
    expect(out[1]).toBe(bare);
  });

  it('leaves notes, preset cards and block instances alone', () => {
    const note = canvasNode('n', 'note', {}, { noteKind: 'text', noteContent: 'hi' });
    const card = canvasNode('p', 'preset:Encoder', {}, {
      isPreset: true, internalParams: { inner: {} },
    });
    const block = canvasNode('s', 'subgraph:blk', {});
    const nodes = [note, card, block];
    expect(withParamDefaults(nodes, CATALOG)).toBe(nodes);
  });

  it('copies only the node it fills', () => {
    const filled = canvasNode('w', 'edu:SlidingWindow2D', { preset: 'Blur3x3' });
    const untouched = canvasNode('x', 'Other', { a: 1 });
    const out = withParamDefaults([filled, untouched], CATALOG);
    expect(out[0]).not.toBe(filled);
    expect(out[0].data).not.toBe(filled.data);
    expect(out[0].data.label).toBe('w');
    expect(out[1]).toBe(untouched);
    // The input is not written through.
    expect(filled.data.params).toEqual({ preset: 'Blur3x3' });
  });

  it('gives a node with no params map the defaults, and leaves a malformed one', () => {
    const [missing] = withParamDefaults(
      [canvasNode('w', 'edu:SlidingWindow2D', undefined)],
      CATALOG,
    );
    expect(missing.data.params).toEqual({
      preset: 'Blur3x3', kernel_size: 3, weights: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], padding: 0,
    });
    const odd = [canvasNode('w', 'edu:SlidingWindow2D', ['not', 'a', 'map'])];
    expect(withParamDefaults(odd, CATALOG)).toBe(odd);
  });

  it('returns the same array for an empty list or an empty node list', () => {
    const nodes = [canvasNode('w', 'edu:SlidingWindow2D', {})];
    expect(withParamDefaults(nodes, [])).toBe(nodes);
    const none: Node<NodeData>[] = [];
    expect(withParamDefaults(none, CATALOG)).toBe(none);
  });

  it('fills nothing from a definition the palette could not drop, and does not throw', () => {
    const broken = [
      { node_name: 'NoList', category: '', description: '', inputs: [], outputs: [], params: null },
      { node_name: 'NullEntry', category: '', description: '', inputs: [], outputs: [], params: [null] },
    ] as unknown as NodeDefinition[];
    const nodes = [canvasNode('a', 'NoList', {}), canvasNode('b', 'NullEntry', {})];
    expect(withParamDefaults(nodes, broken)).toBe(nodes);
  });

  it('tolerates a node with no data', () => {
    const nodes = [{ id: 'x', position: { x: 0, y: 0 } } as unknown as Node<NodeData>];
    expect(withParamDefaults(nodes, CATALOG)).toBe(nodes);
  });

  it('hands back whatever it was given when either list is not a list', () => {
    // The store runs this over every open tab when the node list changes, from
    // inside that list's own update: a throw there would land in the fetch.
    const missing = undefined as unknown as Node<NodeData>[];
    expect(withParamDefaults(missing, CATALOG)).toBe(missing);
    const nodes = [canvasNode('w', 'edu:SlidingWindow2D', {})];
    expect(withParamDefaults(nodes, undefined as unknown as NodeDefinition[])).toBe(nodes);
  });
});

describe('withSubgraphParamDefaults', () => {
  function block(nodes: unknown[]): SubgraphDefinition {
    return {
      id: 'blk', name: 'Block', description: '', nodes, edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
  }

  it('fills the serialized nodes inside a block definition by the same rule', () => {
    const inner = {
      id: 'w', type: 'edu:SlidingWindow2D', position: { x: 0, y: 0 },
      data: { params: { preset: 'Custom' }, label: 'Window' },
    };
    const [definition] = withSubgraphParamDefaults([block([inner])], CATALOG);
    expect(definition.nodes[0]).toEqual({
      ...inner,
      data: {
        params: {
          preset: 'Custom', kernel_size: 3,
          weights: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], padding: 0,
        },
        label: 'Window',
      },
    });
    expect(inner.data.params).toEqual({ preset: 'Custom' });
  });

  it('returns the same list, and the same definitions, when nothing is missing', () => {
    const complete = {
      id: 'w', type: 'edu:SlidingWindow2D',
      data: { params: { preset: 'Blur3x3', kernel_size: 5, weights: [[1]], padding: 2 } },
    };
    const chat = { id: 'c', type: 'LLMChat', data: { params: { model: 'm', values: null } } };
    const other = block([complete, chat, { id: 'u', type: 'Unknown', data: { params: {} } }]);
    const list = [other];
    const out = withSubgraphParamDefaults(list, CATALOG);
    expect(out).toBe(list);
    expect(out[0]).toBe(other);
  });

  it('leaves an inner entry that is not a readable node as it came', () => {
    const odd = [
      null,
      'text',
      { id: 'a', type: 'edu:SlidingWindow2D' },
      { id: 'b', type: 'edu:SlidingWindow2D', data: null },
      { id: 'c', type: 'edu:SlidingWindow2D', data: { params: [1, 2] } },
      { id: 'd', type: 42, data: { params: {} } },
    ];
    const list = [block(odd)];
    expect(withSubgraphParamDefaults(list, CATALOG)).toBe(list);
  });

  it('gives an inner node with no params map the defaults', () => {
    const [definition] = withSubgraphParamDefaults(
      [block([{ id: 'c', type: 'LLMChat', data: { label: 'Chat' } }])],
      CATALOG,
    );
    expect(definition.nodes[0].data).toEqual({
      label: 'Chat', params: { model: 'gpt-5.2', values: null },
    });
  });

  it('returns the same array for an empty catalog, and skips a definition without a node list', () => {
    const list = [block([{ id: 'w', type: 'edu:SlidingWindow2D', data: { params: {} } }])];
    expect(withSubgraphParamDefaults(list, [])).toBe(list);
    const listless = [{ ...block([]), nodes: undefined } as unknown as SubgraphDefinition];
    expect(withSubgraphParamDefaults(listless, CATALOG)).toBe(listless);
    const missing = undefined as unknown as SubgraphDefinition[];
    expect(withSubgraphParamDefaults(missing, CATALOG)).toBe(missing);
    expect(withSubgraphParamDefaults(list, null as unknown as NodeDefinition[])).toBe(list);
  });
});
