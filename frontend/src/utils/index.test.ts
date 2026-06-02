import { describe, expect, it } from 'vitest';
import {
  generateId,
  getPortColor,
  isValidConnection,
  DATA_TYPE_COLORS,
  VIZ_NODE_TYPES,
  resolveSerializedNodes,
  resolveSerializedEdges,
} from './index';
import type { NodeDefinition, PresetDefinition } from '../types';

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

  it('returns false for a source type absent from the compatibility map', () => {
    // SCALAR exists; but an unknown like "FOO" is not a key → `compatible` undefined.
    expect(isValidConnection('FOO', 'BAR')).toBe(false);
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
