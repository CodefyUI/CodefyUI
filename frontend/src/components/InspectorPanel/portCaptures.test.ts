import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import type { ExecutionStatus, NodeData, NodeDefinition } from '../../types';
import {
  fetchOutput,
  NoValueError,
  RunDataExpiredError,
} from '../../api/executionOutputs';
import { keyOf } from './PortGroup';
import {
  capturePhase,
  capturePhaseNoteKey,
  portDataType,
  resolveInputSources,
  resolveSingleNodePorts,
  takeDuePorts,
  usePortFetches,
} from './portCaptures';

vi.mock('../../api/executionOutputs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/executionOutputs')>();
  return { ...actual, fetchOutput: vi.fn() };
});

const mockOutput = vi.mocked(fetchOutput);

function def(outputs: { name: string; data_type: string }[]): NodeDefinition {
  return {
    node_name: 'N',
    category: 'c',
    description: '',
    inputs: [],
    outputs: outputs.map((o) => ({ ...o, description: '', optional: false })),
    params: [],
  };
}

function node(
  id: string,
  over: { label?: string; definition?: NodeDefinition } = {},
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: over.label ?? id,
      type: 'N',
      params: {},
      ...(over.definition !== undefined ? { definition: over.definition } : {}),
    },
  };
}

function edge(
  id: string,
  source: string,
  target: string,
  over: Partial<Edge> = {},
): Edge {
  return { id, source, target, sourceHandle: 'out', ...over } as Edge;
}

describe('portDataType', () => {
  it('reads the declared type off the source node definition', () => {
    const nodes = [node('a', { definition: def([{ name: 'out', data_type: 'TENSOR' }]) })];
    expect(portDataType(nodes, 'a', 'out')).toBe('TENSOR');
  });

  it('returns undefined for an unknown node, port, or a node with no definition', () => {
    const nodes = [
      node('a', { definition: def([{ name: 'out', data_type: 'TENSOR' }]) }),
      node('b'),
    ];
    expect(portDataType(nodes, 'ghost', 'out')).toBeUndefined();
    expect(portDataType(nodes, 'a', 'nope')).toBeUndefined();
    expect(portDataType(nodes, 'b', 'out')).toBeUndefined();
  });
});

describe('resolveInputSources', () => {
  it('collects the connected upstream (node, port) pairs', () => {
    const edges = [edge('e1', 's1', 'target', { sourceHandle: 'x' }), edge('e2', 's2', 'other')];
    expect(resolveInputSources('target', edges)).toEqual([{ nodeId: 's1', port: 'x' }]);
  });

  it('skips trigger edges in both shapes, and edges naming no source port', () => {
    const edges = [
      edge('e1', 's1', 'target', { type: 'triggerEdge' }),
      edge('e2', 's2', 'target', { data: { type: 'trigger' } }),
      edge('e3', 's3', 'target', { sourceHandle: null }),
    ];
    expect(resolveInputSources('target', edges)).toEqual([]);
  });
});

describe('resolveSingleNodePorts', () => {
  it('labels inputs with their provenance and lists the node’s own outputs', () => {
    const nodes = [
      node('n1', { definition: def([{ name: 'logits', data_type: 'TENSOR' }]) }),
      node('src', { label: 'Source', definition: def([{ name: 'y', data_type: 'SCALAR' }]) }),
    ];
    const edges = [edge('e1', 'src', 'n1', { sourceHandle: 'y' })];
    expect(resolveSingleNodePorts('n1', nodes, edges)).toEqual({
      inputs: [{ nodeId: 'src', port: 'y', displayName: 'Source.y', dataType: 'SCALAR' }],
      outputs: [{ nodeId: 'n1', port: 'logits', dataType: 'TENSOR' }],
    });
  });

  it('returns nothing for a node that is not on the canvas', () => {
    expect(resolveSingleNodePorts('ghost', [], [])).toEqual({ inputs: [], outputs: [] });
  });

  it('falls back to a truncated id when the upstream node has no usable label', () => {
    // Source is absent from `nodes` entirely, and its label would be blank
    // anyway — either way the row still names where the value came from.
    const nodes = [node('n1', { definition: def([]) })];
    const edges = [edge('e1', 'sourcenode123', 'n1', { sourceHandle: 'y' })];
    expect(resolveSingleNodePorts('n1', nodes, edges).inputs).toEqual([
      { nodeId: 'sourcenode123', port: 'y', displayName: 'source.y', dataType: undefined },
    ]);
  });

  it('treats a node with no definition as having no outputs', () => {
    const nodes = [node('n1')];
    expect(resolveSingleNodePorts('n1', nodes, [])).toEqual({ inputs: [], outputs: [] });
  });

  it('leaves out a trigger output, which has no value behind it', () => {
    // Start returns nothing for `trigger`, so a request for it could only 404.
    const nodes = [
      node('start', { definition: def([{ name: 'trigger', data_type: 'TRIGGER' }]) }),
    ];
    expect(resolveSingleNodePorts('start', nodes, [])).toEqual({ inputs: [], outputs: [] });
  });

  it('keeps the data outputs listed next to a trigger output', () => {
    const nodes = [
      node('n1', {
        definition: def([
          { name: 'trigger', data_type: 'TRIGGER' },
          { name: 'out', data_type: 'TENSOR' },
        ]),
      }),
    ];
    expect(resolveSingleNodePorts('n1', nodes, []).outputs).toEqual([
      { nodeId: 'n1', port: 'out', dataType: 'TENSOR' },
    ]);
  });
});

// ── A port the node left empty ──────────────────────────────────────────────
// The server answers 204 when the node ran and returned None for the port
// (TrainingLoop's grad_scaler_state outside fp16), and `fetchOutput` turns
// that into NoValueError. The row says so quietly; it is neither an error nor
// an expired run.

describe('usePortFetches', () => {
  const PORTS = [{ nodeId: 'train', port: 'grad_scaler_state' }];
  const KEY = keyOf('train', 'grad_scaler_state');

  beforeEach(() => {
    mockOutput.mockReset();
  });

  it('notes a port that produced no value instead of failing it', async () => {
    mockOutput.mockRejectedValue(new NoValueError('run1', 'train', 'grad_scaler_state'));
    const { result } = renderHook(() => usePortFetches('run1', PORTS));
    await waitFor(() =>
      expect(result.current[KEY]).toEqual({
        loading: false,
        error: null,
        errorKey: null,
        noteKey: 'inspector.noValue',
        data: null,
      }),
    );
    expect(mockOutput).toHaveBeenCalledTimes(1);
  });

  it('still reads a 404 as expired run data', async () => {
    mockOutput.mockRejectedValue(new RunDataExpiredError('run1'));
    const { result } = renderHook(() => usePortFetches('run1', PORTS));
    await waitFor(() =>
      expect(result.current[KEY]).toMatchObject({
        loading: false,
        error: null,
        errorKey: 'inspector.dataExpired',
        data: null,
      }),
    );
    expect(result.current[KEY].noteKey).toBeFalsy();
  });
});

// ── Captures exist only once a node has returned ────────────────────────────
// The engine writes a node's captures after the node returns and answers 404
// for anything not written yet, so a port read mid-run has three states, not
// two: nothing to read YET is different from nothing to read.

describe('capturePhase', () => {
  const TERMINAL: ExecutionStatus[] = ['completed', 'cached', 'error', 'skipped', 'interrupted'];

  it('is running while the node runs in a run that is in progress', () => {
    expect(capturePhase('running', true)).toBe('running');
  });

  it('is pending for a node the run has not reached, and never claims it is running', () => {
    expect(capturePhase('idle', true)).toBe('pending');
    expect(capturePhase(undefined, true)).toBe('pending');
  });

  it('is settled once the node reports any terminal status', () => {
    for (const status of TERMINAL) expect(capturePhase(status, true)).toBe('settled');
  });

  it('is settled for every status when no run is in progress', () => {
    const all: (ExecutionStatus | undefined)[] = [undefined, 'idle', 'running', ...TERMINAL];
    for (const status of all) expect(capturePhase(status, false)).toBe('settled');
  });
});

describe('capturePhaseNoteKey', () => {
  it('names a line for the two phases that have nothing to read yet, and none once settled', () => {
    expect(capturePhaseNoteKey('running')).toBe('inspector.nodeRunning');
    expect(capturePhaseNoteKey('pending')).toBe('inspector.nodePending');
    expect(capturePhaseNoteKey('settled')).toBeNull();
  });
});

describe('takeDuePorts', () => {
  const A = { nodeId: 'a', port: 'out' };
  const B = { nodeId: 'b', port: 'out' };

  it('hands out a settled port once, however often it is asked', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A, B], ['settled', 'settled'], asked)).toEqual([A, B]);
    expect(takeDuePorts([A, B], ['settled', 'settled'], asked)).toEqual([]);
  });

  it('holds back a port whose owner has not returned, and releases it when it has', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A, B], ['settled', 'running'], asked)).toEqual([A]);
    expect(takeDuePorts([A, B], ['settled', 'pending'], asked)).toEqual([]);
    // B finishing must not hand A out a second time.
    expect(takeDuePorts([A, B], ['settled', 'settled'], asked)).toEqual([B]);
  });

  it('hands a port out again after its owner ran a second time', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
    expect(takeDuePorts([A], ['running'], asked)).toEqual([]);
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
  });

  it('forgets a port that left the view, so coming back to it reads it afresh', () => {
    const asked = new Set<string>();
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
    expect(takeDuePorts([B], ['settled'], asked)).toEqual([B]);
    expect(takeDuePorts([A], ['settled'], asked)).toEqual([A]);
  });

  it('asks once for a port listed twice', () => {
    expect(takeDuePorts([A, { ...A }], ['settled', 'settled'], new Set())).toEqual([A]);
  });
});
