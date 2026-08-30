import { describe, expect, it } from 'vitest';
import {
  generateId,
  getPortColor,
  isParamVisible,
  isValidConnection,
  DATA_TYPE_COLORS,
  DATA_TYPE_COMPATIBILITY,
  SELECTABLE_DATA_TYPES,
  VIZ_NODE_TYPES,
  resolveSerializedNodes,
  resolveSerializedEdges,
  migrateStaleEdgeStrokes,
  STALE_TYPE_STROKES,
  resolveDynamicInputs,
  resolveDynamicOutputs,
  buildFlowNode,
  sanitizeGraphName,
  findGraphNameCollision,
} from './index';
import type { NodeDefinition, ParamDefinition, PresetDefinition } from '../types';

function buildParam(overrides: Partial<ParamDefinition>): ParamDefinition {
  return {
    name: 'p',
    param_type: 'int',
    default: 0,
    description: '',
    options: [],
    min_value: null,
    max_value: null,
    ...overrides,
  };
}

describe('isParamVisible', () => {
  it('returns true when visible_when is omitted', () => {
    const param = buildParam({ name: 'preset' });
    expect(isParamVisible(param, { preset: 'EdgeDetection3x3' })).toBe(true);
  });

  it('returns true when visible_when is null', () => {
    const param = buildParam({ name: 'preset', visible_when: null });
    expect(isParamVisible(param, { preset: 'EdgeDetection3x3' })).toBe(true);
  });

  it('returns true when every key in visible_when matches the live params', () => {
    const param = buildParam({ name: 'weights', visible_when: { preset: 'Custom' } });
    expect(isParamVisible(param, { preset: 'Custom' })).toBe(true);
  });

  it('returns false when a visible_when key does not match', () => {
    const param = buildParam({ name: 'weights', visible_when: { preset: 'Custom' } });
    expect(isParamVisible(param, { preset: 'EdgeDetection3x3' })).toBe(false);
  });

  it('returns false when the sibling param is absent from live params', () => {
    const param = buildParam({ name: 'weights', visible_when: { preset: 'Custom' } });
    expect(isParamVisible(param, {})).toBe(false);
  });

  it('requires every entry in visible_when to match (logical AND)', () => {
    const param = buildParam({
      name: 'advanced',
      visible_when: { preset: 'Custom', mode: 'expert' },
    });
    expect(isParamVisible(param, { preset: 'Custom', mode: 'expert' })).toBe(true);
    expect(isParamVisible(param, { preset: 'Custom', mode: 'beginner' })).toBe(false);
    expect(isParamVisible(param, { preset: 'Sharpen3x3', mode: 'expert' })).toBe(false);
  });

  it('coerces non-string values for comparison', () => {
    const param = buildParam({ name: 'kernel_size', visible_when: { use_custom: 'true' } });
    expect(isParamVisible(param, { use_custom: true })).toBe(true);
    expect(isParamVisible(param, { use_custom: false })).toBe(false);
    const numeric = buildParam({ visible_when: { kernel_size: 5 } });
    expect(isParamVisible(numeric, { kernel_size: 5 })).toBe(true);
    expect(isParamVisible(numeric, { kernel_size: '5' })).toBe(true);
  });

  // core#134: "any of" rules, for a param several sibling values share.
  it('accepts an array of expected values as "any of"', () => {
    const param = buildParam({
      name: 'betas',
      visible_when: { type: ['Adam', 'AdamW', 'NAdam', 'RAdam'] },
    });
    expect(isParamVisible(param, { type: 'Adam' })).toBe(true);
    expect(isParamVisible(param, { type: 'RAdam' })).toBe(true);
    expect(isParamVisible(param, { type: 'SGD' })).toBe(false);
    expect(isParamVisible(param, {})).toBe(false);
  });

  it('coerces inside an array rule too', () => {
    const param = buildParam({ visible_when: { kernel_size: [3, 5] } });
    expect(isParamVisible(param, { kernel_size: '5' })).toBe(true);
    expect(isParamVisible(param, { kernel_size: 7 })).toBe(false);
  });

  it('treats an empty array as "never visible"', () => {
    // Degenerate, but it must not silently mean "always": a rule listing no
    // acceptable value has nothing it can match.
    const param = buildParam({ visible_when: { type: [] } });
    expect(isParamVisible(param, { type: 'Adam' })).toBe(false);
  });

  it('treats undefined live params as no match', () => {
    const param = buildParam({ name: 'weights', visible_when: { preset: 'Custom' } });
    expect(isParamVisible(param, undefined)).toBe(false);
  });
});

describe('generateId', () => {
  it('returns a valid UUID string', () => {
    const id = generateId();
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
  });

  it('returns unique values', () => {
    const ids = new Set(Array.from({ length: 100 }, () => generateId()));
    expect(ids.size).toBe(100);
  });
});

describe('getPortColor', () => {
  it('returns the correct color for known types', () => {
    expect(getPortColor('TENSOR')).toBe('#4CAF50');
    expect(getPortColor('MODEL')).toBe('#2196F3');
    expect(getPortColor('DATASET')).toBe('#FF9800');
  });

  it('is case-insensitive', () => {
    expect(getPortColor('tensor')).toBe('#4CAF50');
    expect(getPortColor('Tensor')).toBe('#4CAF50');
  });

  it('returns ANY color for unknown types', () => {
    expect(getPortColor('UNKNOWN_TYPE')).toBe(DATA_TYPE_COLORS['ANY']);
  });

  it('keeps TRANSFORM and DATASET apart on the canvas', () => {
    // These two meet at every train_transform / eval_transform port, and as
    // two neighbouring Material ambers ('#FFC107' vs '#FF9800') they sat
    // 14.5 dE00 apart in normal vision but only 6.1 under simulated
    // deuteranopia, because most of the separation lived on the red-green
    // axis (#197 item 5). The fix was a LIGHTER amber, so this asserts the
    // lightness gap rather than the literal hex: a future edit may retune the
    // value, but pulling TRANSFORM back down next to DATASET would undo the
    // whole point. 12 L* is a floor, not the target — the shipped pair is
    // ~18 apart, where the old one was ~9.5.
    const lstar = (hex: string) => {
      const [r, g, b] = [0, 2, 4].map((i) => {
        const c = parseInt(hex.slice(1 + i, 3 + i), 16) / 255;
        return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
      });
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      return y <= 216 / 24389 ? (y * 24389) / 27 : Math.cbrt(y) * 116 - 16;
    };
    const gap = lstar(DATA_TYPE_COLORS['TRANSFORM']) - lstar(DATA_TYPE_COLORS['DATASET']);
    expect(gap).toBeGreaterThan(12);
  });

  it('gives TRANSFORM a colour of its own', () => {
    // Without an entry it would silently fall back to ANY grey, and every
    // transform-chain edge on the canvas would look like an untyped one.
    expect(getPortColor('TRANSFORM')).toBe(DATA_TYPE_COLORS['TRANSFORM']);
    expect(getPortColor('TRANSFORM')).not.toBe(DATA_TYPE_COLORS['ANY']);
  });
});

describe('isValidConnection', () => {
  it('allows same type connections', () => {
    expect(isValidConnection('TENSOR', 'TENSOR')).toBe(true);
    expect(isValidConnection('MODEL', 'MODEL')).toBe(true);
  });

  it('allows ANY on either side', () => {
    expect(isValidConnection('ANY', 'TENSOR')).toBe(true);
    expect(isValidConnection('TENSOR', 'ANY')).toBe(true);
    expect(isValidConnection('ANY', 'ANY')).toBe(true);
  });

  it('allows IMAGE to TENSOR', () => {
    expect(isValidConnection('IMAGE', 'TENSOR')).toBe(true);
  });

  it('rejects incompatible types', () => {
    expect(isValidConnection('TENSOR', 'MODEL')).toBe(false);
    expect(isValidConnection('MODEL', 'TENSOR')).toBe(false);
    expect(isValidConnection('OPTIMIZER', 'LOSS_FN')).toBe(false);
  });

  it('rejects any connection involving TRIGGER on either side', () => {
    expect(isValidConnection('TRIGGER', 'TRIGGER')).toBe(false);
    expect(isValidConnection('TRIGGER', 'TENSOR')).toBe(false);
    expect(isValidConnection('TENSOR', 'TRIGGER')).toBe(false);
    // TRIGGER is rejected before the ANY allowance is considered.
    expect(isValidConnection('TRIGGER', 'ANY')).toBe(false);
  });

  it('allows DATASET → DATALOADER via the compatibility map', () => {
    expect(isValidConnection('DATASET', 'DATALOADER')).toBe(true);
  });

  it('wires TRANSFORM only to itself and ANY, all via the early returns', () => {
    // Every line here is settled BEFORE the compatibility map is consulted:
    // same-type, ANY on either side, and the TRIGGER rejection are the three
    // early returns, and an unrelated pair falls through to a map lookup that
    // cannot match. So this pins the early returns, not TRANSFORM's row --
    // the row itself is pinned by the DATA_TYPE_COMPATIBILITY suite below.
    expect(isValidConnection('TRANSFORM', 'TRANSFORM')).toBe(true);
    expect(isValidConnection('TRANSFORM', 'ANY')).toBe(true);
    expect(isValidConnection('ANY', 'TRANSFORM')).toBe(true);
    expect(isValidConnection('TRANSFORM', 'DATASET')).toBe(false);
    expect(isValidConnection('DATASET', 'TRANSFORM')).toBe(false);
    expect(isValidConnection('TRANSFORM', 'TRIGGER')).toBe(false);
  });

  it('returns false for a source type absent from the compatibility map', () => {
    // SCALAR exists; but an unknown like "FOO" is not a key → `compatible` undefined.
    expect(isValidConnection('FOO', 'BAR')).toBe(false);
  });
});

describe('SELECTABLE_DATA_TYPES', () => {
  it('lists the types in the backend DataType declaration order', () => {
    // Transcribed by hand from `backend/app/core/node_base.py`'s `DataType`
    // enum — the frontend cannot read Python at test time, so this list is
    // the mirror and this test is what keeps it honest. TRIGGER is the one
    // member with no entry here: it is control flow, never a data port, so
    // the canvas has no colour for it and the per-port select must not offer
    // it. Adding a DataType means adding it HERE in the enum's position too
    // (#197 item 4); a mismatch shows up as this test failing, not as a
    // dropdown that quietly disagrees with the backend.
    expect(SELECTABLE_DATA_TYPES).toEqual([
      'TENSOR',
      'MODEL',
      'DATASET',
      'DATALOADER',
      'OPTIMIZER',
      'LOSS_FN',
      'SCALAR',
      'STRING',
      'IMAGE',
      'LIST',
      'ANY',
      'TRANSFORM',
    ]);
  });

  it('offers no TRIGGER entry', () => {
    // Membership is the part that must not change while the order does.
    expect(SELECTABLE_DATA_TYPES).not.toContain('TRIGGER');
    expect(SELECTABLE_DATA_TYPES).toEqual(Object.keys(DATA_TYPE_COLORS));
  });
});

describe('DATA_TYPE_COMPATIBILITY', () => {
  it('gives TRANSFORM a row of exactly TRANSFORM and ANY', () => {
    expect(DATA_TYPE_COMPATIBILITY['TRANSFORM']).toEqual(['TRANSFORM', 'ANY']);
  });

  it('covers every type the palette can select', () => {
    // A type added to DATA_TYPE_COLORS (which is what the per-port selects
    // offer) but forgotten here would silently connect to nothing but itself
    // and ANY. ANY needs no row: it short-circuits before the lookup.
    const missing = SELECTABLE_DATA_TYPES.filter(
      (type) => type !== 'ANY' && type !== 'TRIGGER' && !DATA_TYPE_COMPATIBILITY[type],
    );
    expect(missing).toEqual([]);
  });

  it('lets every row reach its own type and ANY', () => {
    for (const [source, targets] of Object.entries(DATA_TYPE_COMPATIBILITY)) {
      expect(targets).toContain(source);
      expect(targets).toContain('ANY');
    }
  });

  it('is what isValidConnection reads for a cross-type edge', () => {
    // DATASET -> DATALOADER is the one entry no early return can produce, so
    // it is the proof that the hoisted constant is still the live lookup.
    expect(DATA_TYPE_COMPATIBILITY['DATASET']).toContain('DATALOADER');
    expect(isValidConnection('DATASET', 'DATALOADER')).toBe(true);
    // ...and the relation is directional.
    expect(DATA_TYPE_COMPATIBILITY['DATALOADER']).not.toContain('DATASET');
    expect(isValidConnection('DATALOADER', 'DATASET')).toBe(false);
  });
});

describe('resolveSerializedEdges', () => {
  it('builds a data edge with default styling and no trigger handle', () => {
    const [edge] = resolveSerializedEdges([
      { id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' },
    ]);
    expect(edge).toMatchObject({
      id: 'e1',
      source: 'a',
      target: 'b',
      sourceHandle: 'out',
      targetHandle: 'in',
      animated: false,
      style: { stroke: '#555', strokeWidth: 2 },
    });
    expect(edge).not.toHaveProperty('type');
  });

  it('treats type==="trigger" as a trigger edge with __trigger target handle', () => {
    const [edge] = resolveSerializedEdges([
      { id: 'e2', source: 'a', target: 'b', type: 'trigger' },
    ]);
    expect(edge.type).toBe('triggerEdge');
    expect(edge.targetHandle).toBe('__trigger');
    expect(edge.data).toEqual({ type: 'trigger' });
  });

  it('treats sourceHandle==="trigger" as a trigger edge', () => {
    const [edge] = resolveSerializedEdges([
      { id: 'e3', source: 'a', target: 'b', sourceHandle: 'trigger' },
    ]);
    expect(edge.type).toBe('triggerEdge');
    expect(edge.targetHandle).toBe('__trigger');
    expect(edge.sourceHandle).toBe('trigger');
  });

  it('generates an id when one is missing', () => {
    const [edge] = resolveSerializedEdges([{ source: 'a', target: 'b' }]);
    expect(edge.id).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('coerces falsy source/target handles to undefined', () => {
    const [edge] = resolveSerializedEdges([
      { id: 'e4', source: 'a', target: 'b', sourceHandle: '', targetHandle: '' },
    ]);
    expect(edge.sourceHandle).toBeUndefined();
    expect(edge.targetHandle).toBeUndefined();
  });

  // ── per-data-type stroke restoration (nodes argument) ─────────────────────

  function edgeSourceDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
    return {
      node_name: 'Linear',
      category: 'Utility',
      description: '',
      inputs: [],
      outputs: [{ name: 'out', data_type: 'MODEL', description: '', optional: false }],
      params: [],
      ...overrides,
    };
  }

  function flowNode(id: string, definition: NodeDefinition, params: Record<string, unknown> = {}) {
    return {
      id,
      type: 'baseNode',
      position: { x: 0, y: 0 },
      data: { label: id, type: definition.node_name, params, definition },
    } as import('@xyflow/react').Node<import('../types').NodeData>;
  }

  it('colors a value edge by its source port data type when nodes are passed', () => {
    const [edge] = resolveSerializedEdges(
      [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' }],
      [flowNode('a', edgeSourceDef())],
    );
    // MODEL -> blue.
    expect(edge.style).toEqual({ stroke: '#2196F3', strokeWidth: 2 });
  });

  it('resolves dynamic outputs (Split chunk_N ports) for coloring', () => {
    const [edge] = resolveSerializedEdges(
      [{ id: 'e1', source: 's', target: 'b', sourceHandle: 'chunk_2', targetHandle: 'in' }],
      [flowNode('s', edgeSourceDef({ node_name: 'Split', outputs: [] }), { chunks: 3 })],
    );
    // Split chunks are TENSOR -> green.
    expect((edge.style as { stroke: string }).stroke).toBe('#4CAF50');
  });

  it('falls back to #555 when the source node is missing from the list', () => {
    const [edge] = resolveSerializedEdges(
      [{ id: 'e1', source: 'ghost', target: 'b', sourceHandle: 'out', targetHandle: 'in' }],
      [flowNode('a', edgeSourceDef())],
    );
    expect((edge.style as { stroke: string }).stroke).toBe('#555');
  });

  it('falls back to #555 when the definition has no output of that name', () => {
    const [edge] = resolveSerializedEdges(
      [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'nope', targetHandle: 'in' }],
      [flowNode('a', edgeSourceDef())],
    );
    expect((edge.style as { stroke: string }).stroke).toBe('#555');
  });

  it('falls back to #555 when the node carries no definition (e.g. unloaded plugin)', () => {
    const node = flowNode('a', edgeSourceDef());
    delete (node.data as { definition?: NodeDefinition }).definition;
    const [edge] = resolveSerializedEdges(
      [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' }],
      [node],
    );
    expect((edge.style as { stroke: string }).stroke).toBe('#555');
  });

  it('leaves trigger edges unstyled even when nodes are passed', () => {
    const [edge] = resolveSerializedEdges(
      [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'trigger' }],
      [flowNode('a', edgeSourceDef())],
    );
    expect(edge.type).toBe('triggerEdge');
    expect(edge.targetHandle).toBe('__trigger');
    expect(edge.style).toBeUndefined();
  });
});

describe('migrateStaleEdgeStrokes', () => {
  function def(dataType: string): NodeDefinition {
    return {
      node_name: 'Source',
      category: 'Utility',
      description: '',
      inputs: [],
      outputs: [{ name: 'out', data_type: dataType, description: '', optional: false }],
      params: [],
    };
  }

  function flowNode(id: string, definition: NodeDefinition) {
    return {
      id,
      type: 'baseNode',
      position: { x: 0, y: 0 },
      data: { label: id, type: definition.node_name, params: {}, definition },
    } as import('@xyflow/react').Node<import('../types').NodeData>;
  }

  function edge(stroke?: string) {
    return {
      id: 'e1',
      source: 'a',
      target: 'b',
      sourceHandle: 'out',
      targetHandle: 'in',
      ...(stroke ? { style: { stroke, strokeWidth: 2 } } : {}),
    } as import('@xyflow/react').Edge;
  }

  const strokeOf = (e: import('@xyflow/react').Edge) =>
    (e.style as { stroke?: string } | undefined)?.stroke;

  it('names the old amber as TRANSFORM', () => {
    // One entry today. The list is the record of every palette hex a saved
    // graph may still be carrying, so a future palette change appends here.
    expect(STALE_TYPE_STROKES).toContainEqual({ hex: '#FFC107', type: 'TRANSFORM' });
  });

  it('lets one retired hex name more than one type', () => {
    // Why it is a list and not a map keyed by hex: palettes get reshuffled,
    // and the same value can have belonged to two types at different times.
    // Keyed by hex, the second one could not be added without editing the
    // first -- and "append, never edit" is the whole safety property of this
    // record. The edge's CURRENT type picks which entry applies.
    STALE_TYPE_STROKES.push({ hex: '#FFC107', type: 'MODEL' });
    try {
      const [migrated] = migrateStaleEdgeStrokes(
        [edge('#FFC107')],
        [flowNode('a', def('MODEL'))],
      );
      expect(strokeOf(migrated)).toBe(DATA_TYPE_COLORS['MODEL']);
    } finally {
      STALE_TYPE_STROKES.pop();
    }
  });

  it('repaints a stale TRANSFORM stroke in the current amber', () => {
    const [migrated] = migrateStaleEdgeStrokes(
      [edge('#FFC107')],
      [flowNode('a', def('TRANSFORM'))],
    );
    expect(strokeOf(migrated)).toBe(DATA_TYPE_COLORS['TRANSFORM']);
    // Everything else about the edge survives.
    expect(migrated.style).toMatchObject({ strokeWidth: 2 });
    expect(migrated.id).toBe('e1');
  });

  it('leaves a stroke that is not a known stale hex alone', () => {
    // A colour the user or a plugin chose is not ours to overwrite.
    const [migrated] = migrateStaleEdgeStrokes(
      [edge('#123456')],
      [flowNode('a', def('TRANSFORM'))],
    );
    expect(strokeOf(migrated)).toBe('#123456');
  });

  it('leaves the stale hex alone on an edge of a different type', () => {
    // #FFC107 was TRANSFORM's colour, so on a MODEL wire it is somebody
    // else's choice that happens to collide.
    const [migrated] = migrateStaleEdgeStrokes(
      [edge('#FFC107')],
      [flowNode('a', def('MODEL'))],
    );
    expect(strokeOf(migrated)).toBe('#FFC107');
  });

  it('leaves an edge with no baked stroke alone', () => {
    // Committed examples persist no style at all and take the neutral wire.
    const [migrated] = migrateStaleEdgeStrokes(
      [edge()],
      [flowNode('a', def('TRANSFORM'))],
    );
    expect(migrated.style).toBeUndefined();
  });

  it('leaves the stroke alone when the type cannot be derived', () => {
    // An unloaded plugin's node is missing from the list; guessing that its
    // port is TRANSFORM because the wire is amber is exactly backwards.
    const [migrated] = migrateStaleEdgeStrokes([edge('#FFC107')], []);
    expect(strokeOf(migrated)).toBe('#FFC107');
  });

  it('returns the same array when nothing is stale', () => {
    // The autosave record cache compares edge arrays by reference; a restore
    // that rebuilt the array every time would rewrite storage for nothing.
    const edges = [edge('#123456')];
    expect(migrateStaleEdgeStrokes(edges, [flowNode('a', def('TRANSFORM'))])).toBe(edges);
  });
});

describe('resolveSerializedNodes', () => {
  const defs: NodeDefinition[] = [
    {
      node_name: 'Dataset',
      category: 'Data',
      description: 'a dataset',
      inputs: [],
      outputs: [],
      params: [],
    } as NodeDefinition,
    {
      node_name: 'Start',
      category: 'Control',
      description: 'start',
      inputs: [],
      outputs: [{ name: 'trigger', data_type: 'TRIGGER', description: '', optional: false }],
      params: [],
    } as NodeDefinition,
  ];

  const presets: PresetDefinition[] = [
    {
      preset_name: 'MyPreset',
      category: 'Preset',
      description: 'preset desc',
      exposed_inputs: [
        { name: 'in1', internal_node: 'n', internal_port: 'p', data_type: 'TENSOR', description: 'd' },
      ],
      exposed_outputs: [
        { name: 'out1', internal_node: 'n', internal_port: 'p', data_type: 'TENSOR', description: 'd' },
      ],
    } as PresetDefinition,
  ];

  it('resolves a note node with provided fields', () => {
    const [node] = resolveSerializedNodes(
      [
        {
          id: 'note1',
          type: 'note',
          position: { x: 5, y: 6 },
          data: {
            noteKind: 'markdown',
            noteContent: 'hi',
            noteColor: '#fff',
            boundToNodeId: 'x',
            boundOffset: { x: 1, y: 2 },
            noteWidth: 300,
            noteHeight: 150,
          },
        },
      ],
      defs,
      presets,
    );
    expect(node.type).toBe('noteNode');
    expect(node.data).toMatchObject({
      label: 'Note',
      type: 'note',
      noteKind: 'markdown',
      noteContent: 'hi',
      noteColor: '#fff',
      boundToNodeId: 'x',
      boundOffset: { x: 1, y: 2 },
      noteWidth: 300,
      noteHeight: 150,
    });
  });

  it('resolves a note node with all defaults when data is absent', () => {
    const [node] = resolveSerializedNodes([{ id: 'note2', type: 'note' }], defs, presets);
    expect(node.position).toEqual({ x: 0, y: 0 });
    expect(node.data).toMatchObject({
      noteKind: 'text',
      noteContent: '',
      noteColor: '#3d3d1a',
      boundToNodeId: null,
      boundOffset: null,
      noteWidth: 200,
    });
    expect((node.data as Record<string, unknown>).noteHeight).toBeUndefined();
  });

  it('resolves a preset node when the preset is found', () => {
    const [node] = resolveSerializedNodes(
      [{ id: 'p1', type: 'preset:MyPreset', data: { params: { a: 1 }, internalParams: { b: 2 } } }],
      defs,
      presets,
    );
    expect(node.type).toBe('presetNode');
    expect(node.data).toMatchObject({
      label: 'MyPreset',
      type: 'preset:MyPreset',
      isPreset: true,
      executionStatus: 'idle',
      internalParams: { b: 2 },
    });
    const def = (node.data as Record<string, NodeDefinition>).definition;
    expect(def.node_name).toBe('MyPreset');
    expect(def.inputs[0]).toMatchObject({ name: 'in1', data_type: 'TENSOR', optional: false });
    expect(def.outputs[0]).toMatchObject({ name: 'out1', data_type: 'TENSOR' });
  });

  it('resolves a preset node with a synthetic definition when the preset is missing', () => {
    const [node] = resolveSerializedNodes(
      [{ id: 'p2', type: 'preset:Unknown' }],
      defs,
      presets,
    );
    const def = (node.data as Record<string, NodeDefinition>).definition;
    expect(def).toMatchObject({ node_name: 'Unknown', category: 'Preset', inputs: [], outputs: [] });
    expect((node.data as Record<string, unknown>).internalParams).toEqual({});
  });

  it('resolves an instance whose definition the file left half-written', () => {
    // `{"id":"x"}` is a definition the SERVER accepts and runs as an empty
    // block (`id` is its only required field), and every reader of a graph
    // document resolves the nodes BEFORE `setSubgraphs` gets a chance to
    // coerce anything -- so an import died here, on `interface.inputs`,
    // before the store was ever involved.
    const [node] = resolveSerializedNodes(
      [{ id: 'inst', type: 'subgraph:x' }],
      defs,
      presets,
      [{ id: 'x' }] as never,
    );
    expect(node.type).toBe('subgraphNode');
    const def = (node.data as Record<string, NodeDefinition>).definition;
    expect(def).toMatchObject({ category: 'Subgraph', inputs: [], outputs: [] });
  });

  it('resolves a Start node using the provided definition', () => {
    const [node] = resolveSerializedNodes([{ id: 's1', type: 'Start' }], defs, presets);
    expect(node.type).toBe('start');
    const def = (node.data as Record<string, NodeDefinition>).definition;
    expect(def.node_name).toBe('Start');
    expect(def.outputs[0].data_type).toBe('TRIGGER');
  });

  it('resolves a Start node with a synthetic definition when none is provided', () => {
    const [node] = resolveSerializedNodes([{ id: 's2', type: 'Start' }], [], presets);
    const def = (node.data as Record<string, NodeDefinition>).definition;
    expect(def.node_name).toBe('Start');
    expect(def.outputs[0]).toMatchObject({ name: 'trigger', data_type: 'TRIGGER' });
  });

  it('resolves a regular node with a known definition (baseNode renderer)', () => {
    const [node] = resolveSerializedNodes(
      [{ id: 'd1', type: 'Dataset', position: { x: 1, y: 2 }, data: { params: { p: 1 } } }],
      defs,
      presets,
    );
    expect(node.type).toBe('baseNode');
    expect(node.position).toEqual({ x: 1, y: 2 });
    const def = (node.data as Record<string, NodeDefinition>).definition;
    expect(def.node_name).toBe('Dataset');
    expect((node.data as Record<string, unknown>).label).toBe('Dataset');
  });

  it('uses the custom viz renderer for a node listed in VIZ_NODE_TYPES', () => {
    const [node] = resolveSerializedNodes([{ id: 't1', type: 'Tokenizer' }], [], presets);
    expect(node.type).toBe(VIZ_NODE_TYPES.Tokenizer);
    expect(node.type).toBe('tokenizerNode');
  });

  it('strips a plugin namespace prefix before matching VIZ_NODE_TYPES', () => {
    const [node] = resolveSerializedNodes([{ id: 'k1', type: 'foundations:Edu-KNN' }], [], presets);
    // Bare type "Edu-KNN" maps to eduKNNNode even though the stored type is namespaced.
    expect(node.type).toBe('eduKNNNode');
    expect((node.data as Record<string, unknown>).type).toBe('foundations:Edu-KNN');
  });

  it('falls back to a synthetic Utility definition and label for an unknown node type', () => {
    const [node] = resolveSerializedNodes([{ id: 'u1', type: 'Mystery' }], [], presets);
    expect(node.type).toBe('baseNode');
    const def = (node.data as Record<string, NodeDefinition>).definition;
    expect(def).toMatchObject({ node_name: 'Mystery', category: 'Utility' });
    expect((node.data as Record<string, unknown>).label).toBe('Mystery');
  });

  it('uses an explicit label from data when present', () => {
    const [node] = resolveSerializedNodes(
      [{ id: 'l1', type: 'Mystery', data: { label: 'Custom Label' } }],
      [],
      presets,
    );
    expect((node.data as Record<string, unknown>).label).toBe('Custom Label');
  });

  it('defaults type/position/params when the raw node omits them', () => {
    // raw.type missing → nodeType '' → regular-node branch with empty type.
    const [node] = resolveSerializedNodes([{ id: 'empty1' }], [], presets);
    expect(node.position).toEqual({ x: 0, y: 0 });
    expect((node.data as Record<string, unknown>).params).toEqual({});
    expect(node.type).toBe('baseNode');
  });
});

describe('resolveDynamicOutputs', () => {
  function buildDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
    return {
      node_name: 'Split',
      category: 'Tensor Operations',
      description: '',
      inputs: [],
      outputs: [{ name: 'out', data_type: 'TENSOR', description: '', optional: false }],
      params: [],
      ...overrides,
    };
  }

  it('returns [] when the definition is undefined', () => {
    expect(resolveDynamicOutputs(undefined, {})).toEqual([]);
  });

  it('returns the static outputs verbatim for a non-Split node', () => {
    const def = buildDef({ node_name: 'Add' });
    expect(resolveDynamicOutputs(def, {})).toBe(def.outputs);
  });

  it('expands Split into N TENSOR chunk ports for a numeric chunks param', () => {
    const ports = resolveDynamicOutputs(buildDef(), { chunks: 3 });
    expect(ports.map((p) => p.name)).toEqual(['chunk_0', 'chunk_1', 'chunk_2']);
    expect(ports.every((p) => p.data_type === 'TENSOR' && !p.optional)).toBe(true);
    expect(ports[2].description).toBe('Chunk 2 of 3');
  });

  it('parses a string chunks param', () => {
    expect(resolveDynamicOutputs(buildDef(), { chunks: '4' })).toHaveLength(4);
  });

  it('defaults to 2 chunks when the param is missing or non-numeric', () => {
    expect(resolveDynamicOutputs(buildDef(), undefined)).toHaveLength(2);
    expect(resolveDynamicOutputs(buildDef(), { chunks: 'abc' })).toHaveLength(2);
  });

  it('clamps chunks to the [1, 32] range', () => {
    expect(resolveDynamicOutputs(buildDef(), { chunks: 99 })).toHaveLength(32);
    expect(resolveDynamicOutputs(buildDef(), { chunks: 0 })).toHaveLength(1);
    expect(resolveDynamicOutputs(buildDef(), { chunks: -5 })).toHaveLength(1);
  });

  it('strips a plugin prefix before matching the Split node name', () => {
    const ports = resolveDynamicOutputs(buildDef({ node_name: 'foundations:Split' }), { chunks: 2 });
    expect(ports.map((p) => p.name)).toEqual(['chunk_0', 'chunk_1']);
  });
});

describe('resolveDynamicOutputs / resolveDynamicInputs for PythonScript', () => {
  function scriptDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
    return {
      node_name: 'PythonScript',
      category: 'Utility',
      description: '',
      inputs: [{ name: 'in1', data_type: 'TENSOR', description: '', optional: true }],
      outputs: [{ name: 'out1', data_type: 'ANY', description: '', optional: false }],
      params: [],
      ...overrides,
    };
  }

  it('expands input ports to match input_ports', () => {
    const ports = resolveDynamicInputs(scriptDef(), { input_ports: 3 });
    expect(ports.map((p) => p.name)).toEqual(['in1', 'in2', 'in3']);
  });

  it('expands output ports to match output_ports', () => {
    const ports = resolveDynamicOutputs(scriptDef(), { output_ports: 2 });
    expect(ports.map((p) => p.name)).toEqual(['out1', 'out2']);
  });

  it('leaves every input optional so a partly wired script still validates', () => {
    const ports = resolveDynamicInputs(scriptDef(), { input_ports: 4 });
    expect(ports.every((p) => p.optional)).toBe(true);
  });

  it('clamps the port count to 1..8 and tolerates junk', () => {
    expect(resolveDynamicInputs(scriptDef(), { input_ports: 0 })).toHaveLength(1);
    expect(resolveDynamicInputs(scriptDef(), { input_ports: 99 })).toHaveLength(8);
    expect(resolveDynamicInputs(scriptDef(), { input_ports: 'many' })).toHaveLength(1);
    expect(resolveDynamicOutputs(scriptDef(), undefined)).toHaveLength(1);
  });

  it('reads a string port count, as the store holds it after a select edit', () => {
    expect(resolveDynamicOutputs(scriptDef(), { output_ports: '3' })).toHaveLength(3);
  });

  it('types each port from the comma-separated types param', () => {
    const ports = resolveDynamicInputs(scriptDef(), {
      input_ports: 3,
      input_types: 'TENSOR,STRING,SCALAR',
    });
    expect(ports.map((p) => p.data_type)).toEqual(['TENSOR', 'STRING', 'SCALAR']);
  });

  it('repeats the last declared type over ports added later', () => {
    const ports = resolveDynamicOutputs(scriptDef(), {
      output_ports: 4,
      output_types: 'TENSOR,STRING',
    });
    expect(ports.map((p) => p.data_type)).toEqual(['TENSOR', 'STRING', 'STRING', 'STRING']);
  });

  it('defaults inputs to TENSOR and outputs to ANY', () => {
    expect(resolveDynamicInputs(scriptDef(), {})[0].data_type).toBe('TENSOR');
    expect(resolveDynamicOutputs(scriptDef(), {})[0].data_type).toBe('ANY');
  });

  it('falls back to the default for an unknown type name or TRIGGER', () => {
    expect(resolveDynamicOutputs(scriptDef(), { output_types: 'NOPE' })[0].data_type).toBe('ANY');
    expect(resolveDynamicInputs(scriptDef(), { input_types: 'TRIGGER' })[0].data_type).toBe('TENSOR');
  });

  it('strips a plugin prefix before matching the node name', () => {
    const def = scriptDef({ node_name: 'somepack:PythonScript' });
    expect(resolveDynamicInputs(def, { input_ports: 2 })).toHaveLength(2);
  });

  it('returns the definition arrays verbatim for every other node', () => {
    const def = scriptDef({ node_name: 'Add' });
    // Identity, not equality: tabStore uses it as a "ports cannot have
    // changed" short-circuit on every keystroke.
    expect(resolveDynamicInputs(def, { input_ports: 5 })).toBe(def.inputs);
    expect(resolveDynamicOutputs(def, { output_ports: 5 })).toBe(def.outputs);
  });

  it('returns [] when the definition is undefined', () => {
    expect(resolveDynamicInputs(undefined, {})).toEqual([]);
  });
});

describe('buildFlowNode', () => {
  const def: NodeDefinition = {
    node_name: 'Conv2d',
    category: 'Layer',
    description: '',
    inputs: [],
    outputs: [],
    params: [
      { name: 'out_channels', param_type: 'int' as const, default: 32, description: '', options: [], min_value: 1, max_value: null },
    ],
  };

  it('builds a baseNode with default params and idle status', () => {
    const n = buildFlowNode(def, { x: 10, y: 20 });
    expect(n.type).toBe('baseNode');
    expect(n.position).toEqual({ x: 10, y: 20 });
    expect(n.data.type).toBe('Conv2d');
    expect(n.data.label).toBe('Conv2d');
    expect(n.data.params).toEqual({ out_channels: 32 });
    expect(n.data.definition).toBe(def);
    expect(n.data.executionStatus).toBe('idle');
    expect(n.id).toBeTruthy();
  });

  it('maps Start to the start renderer', () => {
    const n = buildFlowNode({ ...def, node_name: 'Start' }, { x: 0, y: 0 });
    expect(n.type).toBe('start');
  });

  it('routes a namespaced plugin node (no first-party viz) to the plugin bridge', () => {
    const n = buildFlowNode({ ...def, node_name: 'somepack:FancyNode' }, { x: 0, y: 0 });
    expect(n.type).toBe('pluginNode');
    expect(n.data.type).toBe('somepack:FancyNode');
  });

  it('still resolves a first-party viz when a namespaced node\'s bare name has one', () => {
    const n = buildFlowNode({ ...def, node_name: 'foundations:Edu-KNN' }, { x: 0, y: 0 });
    expect(n.type).toBe('eduKNNNode');
  });
});

describe('sanitizeGraphName', () => {
  it('keeps alphanumerics, hyphen and underscore verbatim (matches backend)', () => {
    expect(sanitizeGraphName('my-graph_2')).toBe('my-graph_2');
  });

  it('replaces every other char with an underscore', () => {
    // Mirrors the backend test: "a b.c/d" -> "a_b_c_d".
    expect(sanitizeGraphName('a b.c/d')).toBe('a_b_c_d');
  });

  it('preserves Unicode letters/digits (Python str.isalnum is Unicode-aware)', () => {
    // Chinese kept, the space between becomes '_'.
    expect(sanitizeGraphName('中文 graph')).toBe('中文_graph');
    // Accented Latin kept.
    expect(sanitizeGraphName('café')).toBe('café');
  });
});

describe('findGraphNameCollision', () => {
  const existing = [
    { name: 'My Graph', file: 'My_Graph' },
    { name: 'Other', file: 'Other' },
  ];

  it('returns the colliding graph display name when a different graph shares the sanitized file', () => {
    // "My.Graph" sanitizes to "My_Graph", colliding with the saved "My Graph".
    expect(findGraphNameCollision('My.Graph', existing, null)).toBe('My Graph');
  });

  it('returns null when no existing graph shares the sanitized file', () => {
    expect(findGraphNameCollision('brand-new', existing, null)).toBeNull();
  });

  it('returns null when the only match IS the currently-open graph (re-save)', () => {
    // Editing "My_Graph" and saving under a name that maps to it again.
    expect(findGraphNameCollision('My Graph', existing, 'My_Graph')).toBeNull();
  });

  it('still warns when the collision is a DIFFERENT graph than the open one', () => {
    // Open graph is "Other"; saving as "My Graph" collides with the other file.
    expect(findGraphNameCollision('My Graph', existing, 'Other')).toBe('My Graph');
  });

  it('flags a CASE-ONLY collision (NTFS/APFS are case-insensitive)', () => {
    // "my graph" sanitizes to "my_graph"; on Windows/macOS that overwrites the
    // saved "My_Graph", so we must warn despite the case difference.
    expect(findGraphNameCollision('my graph', existing, null)).toBe('My Graph');
  });

  it('does not warn on a case-only re-save of the currently-open graph', () => {
    // Open file stem "My_Graph"; saving "my graph" maps to the same file.
    expect(findGraphNameCollision('my graph', existing, 'My_Graph')).toBeNull();
  });
});

describe('resolveDynamicInputs for ComposeTransform', () => {
  function composeDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
    return {
      node_name: 'ComposeTransform',
      category: 'Data',
      description: '',
      inputs: [
        { name: 'step_1', data_type: 'TRANSFORM', description: '', optional: true },
        { name: 'step_2', data_type: 'TRANSFORM', description: '', optional: true },
      ],
      outputs: [{ name: 'transform', data_type: 'TRANSFORM', description: '', optional: false }],
      params: [],
      ...overrides,
    };
  }

  it('expands steps into that many optional TRANSFORM ports', () => {
    const ports = resolveDynamicInputs(composeDef(), { steps: 4 });
    expect(ports.map((p) => p.name)).toEqual(['step_1', 'step_2', 'step_3', 'step_4']);
    expect(ports.every((p) => p.data_type === 'TRANSFORM' && p.optional)).toBe(true);
  });

  it('parses a string steps param', () => {
    expect(resolveDynamicInputs(composeDef(), { steps: '5' })).toHaveLength(5);
  });

  it('clamps to the backend range of 2..8', () => {
    // Mirrors compose_transform_node._step_count; a frontend that allowed 1
    // or 9 would draw handles the backend has no port for.
    expect(resolveDynamicInputs(composeDef(), { steps: 1 })).toHaveLength(2);
    expect(resolveDynamicInputs(composeDef(), { steps: 0 })).toHaveLength(2);
    expect(resolveDynamicInputs(composeDef(), { steps: 99 })).toHaveLength(8);
    expect(resolveDynamicInputs(composeDef(), { steps: 'abc' })).toHaveLength(2);
    expect(resolveDynamicInputs(composeDef(), undefined)).toHaveLength(2);
  });

  it('strips a plugin prefix before matching the node name', () => {
    const ports = resolveDynamicInputs(composeDef({ node_name: 'pack:ComposeTransform' }), { steps: 3 });
    expect(ports).toHaveLength(3);
  });

  it('leaves every other node definition untouched by reference', () => {
    const def = composeDef({ node_name: 'Transform' });
    expect(resolveDynamicInputs(def, { steps: 5 })).toBe(def.inputs);
  });
});
