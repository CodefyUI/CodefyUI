import { describe, it, expect } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, PortDefinition } from '../types';
import { isValidConnection, resolveDynamicInputs, resolveDynamicOutputs } from '.';
import {
  SWITCH_DEFAULT_INPUTS,
  SWITCH_MAX_INPUTS,
  grownSwitchParams,
  isSwitchNode,
  liveOutputType,
  switchAcceptsType,
  switchInputCount,
  switchInputIndex,
  switchInputs,
  switchOutputType,
} from './switchNode';

const port = (name: string, data_type: string): PortDefinition => ({
  name,
  data_type,
  description: '',
  optional: false,
});

const def = (node_name: string, inputs: PortDefinition[], outputs: PortDefinition[]): NodeDefinition => ({
  node_name,
  category: 'Utility',
  description: '',
  inputs,
  outputs,
  params: [],
});

const SWITCH = def('Switch', [], [port('output', 'ANY')]);
const TENSOR_SRC = def('TensorCreate', [], [port('tensor', 'TENSOR')]);
const TEXT_SRC = def('Text', [], [port('text', 'STRING')]);
const ANY_SRC = def('Anything', [], [port('value', 'ANY')]);
const TEXT_SINK = def('TextSink', [port('text', 'STRING')], []);
const TENSOR_SINK = def('TensorSink', [port('tensor', 'TENSOR')], []);

function card(id: string, definition: NodeDefinition, params: Record<string, unknown> = {}): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: definition.node_name, params, definition } };
}

function wire(source: string, sourceHandle: string, target: string, targetHandle: string): Edge {
  return { id: `${source}.${sourceHandle}->${target}.${targetHandle}`, source, sourceHandle, target, targetHandle };
}

const outputsOf = (n: Node<NodeData>) => resolveDynamicOutputs(n.data.definition, n.data.params);
const inputsOf = (n: Node<NodeData>) => resolveDynamicInputs(n.data.definition, n.data.params);

describe('isSwitchNode', () => {
  it('reads the definition name, a plugin-qualified one, or the type', () => {
    expect(isSwitchNode(card('s', SWITCH))).toBe(true);
    expect(isSwitchNode(card('s', def('pack:Switch', [], [])))).toBe(true);
    const bare = card('s', SWITCH);
    delete bare.data.definition;
    expect(isSwitchNode(bare)).toBe(true);
    expect(isSwitchNode(card('t', TENSOR_SRC))).toBe(false);
    expect(isSwitchNode(undefined)).toBe(false);
    expect(isSwitchNode({ data: undefined } as unknown as Node<NodeData>)).toBe(false);
    expect(isSwitchNode({ data: {} } as unknown as Node<NodeData>)).toBe(false);
  });
});

describe('switchInputCount', () => {
  it('defaults to the four ports Switch had before the param', () => {
    expect(switchInputCount(undefined)).toBe(4);
    expect(switchInputCount({})).toBe(SWITCH_DEFAULT_INPUTS);
  });

  it('clamps, and reads a numeric prefix as the backend does', () => {
    expect(switchInputCount({ inputs: 1 })).toBe(2);
    expect(switchInputCount({ inputs: 99 })).toBe(SWITCH_MAX_INPUTS);
    expect(switchInputCount({ inputs: 6.9 })).toBe(6);
    expect(switchInputCount({ inputs: '7 slots' })).toBe(7);
    expect(switchInputCount({ inputs: 'abc' })).toBe(4);
    expect(switchInputCount({ inputs: true })).toBe(4);
    expect(switchInputCount({ inputs: Infinity })).toBe(4);
  });
});

describe('switchInputIndex', () => {
  it('reads input_N and nothing else', () => {
    expect(switchInputIndex('input_0')).toBe(0);
    expect(switchInputIndex('input_12')).toBe(12);
    expect(switchInputIndex('input_x')).toBeNull();
    expect(switchInputIndex('selector')).toBeNull();
    expect(switchInputIndex(null)).toBeNull();
    expect(switchInputIndex(undefined)).toBeNull();
  });
});

describe('switchInputs', () => {
  it('lists the options first and the selector last, all optional', () => {
    const ports = switchInputs({ inputs: 3 });
    expect(ports.map((p) => p.name)).toEqual(['input_0', 'input_1', 'input_2', 'selector']);
    expect(ports.every((p) => p.optional)).toBe(true);
    expect(ports[ports.length - 1].data_type).toBe('SCALAR');
  });

  it('is what resolveDynamicInputs answers for a Switch', () => {
    expect(resolveDynamicInputs(SWITCH, { inputs: 2 }).map((p) => p.name)).toEqual([
      'input_0', 'input_1', 'selector',
    ]);
  });
});

describe('grownSwitchParams', () => {
  it('adds an input when the wire lands on the last one', () => {
    expect(grownSwitchParams(card('s', SWITCH, { selector: 1 }), 'input_3')).toEqual({ selector: 1, inputs: 5 });
    expect(grownSwitchParams(card('s', SWITCH, { inputs: 2 }), 'input_1')).toEqual({ inputs: 3 });
  });

  it('leaves the Switch alone for any other input, the selector, or at the maximum', () => {
    expect(grownSwitchParams(card('s', SWITCH), 'input_2')).toBeNull();
    expect(grownSwitchParams(card('s', SWITCH), 'selector')).toBeNull();
    expect(grownSwitchParams(card('s', SWITCH, { inputs: SWITCH_MAX_INPUTS }), `input_${SWITCH_MAX_INPUTS - 1}`)).toBeNull();
  });

  it('leaves other nodes and missing nodes alone', () => {
    expect(grownSwitchParams(card('t', TENSOR_SRC), 'input_3')).toBeNull();
    expect(grownSwitchParams(undefined, 'input_3')).toBeNull();
  });

  it('starts from empty params when a Switch has none', () => {
    const s = card('s', SWITCH);
    delete (s.data as Partial<NodeData>).params;
    expect(grownSwitchParams(s, 'input_3')).toEqual({ inputs: 5 });
  });
});

describe('switchOutputType', () => {
  const t = card('t', TENSOR_SRC);
  const u = card('u', TENSOR_SRC);
  const x = card('x', TEXT_SRC);
  const a = card('a', ANY_SRC);
  const sw = card('sw', SWITCH);

  it('is the one type the inputs carry', () => {
    const edges = [wire('t', 'tensor', 'sw', 'input_0'), wire('u', 'tensor', 'sw', 'input_1')];
    expect(switchOutputType('sw', [t, u, sw], edges, outputsOf)).toBe('TENSOR');
  });

  it('ignores ANY inputs, the selector and other nodes', () => {
    const edges = [
      wire('a', 'value', 'sw', 'input_0'),
      wire('t', 'tensor', 'sw', 'input_1'),
      wire('x', 'text', 'sw', 'selector'),
      wire('x', 'text', 'other', 'input_0'),
    ];
    expect(switchOutputType('sw', [t, x, a, sw], edges, outputsOf)).toBe('TENSOR');
  });

  it('is ANY with no wired input, with inputs that disagree, or from an unknown source or port', () => {
    expect(switchOutputType('sw', [sw], [], outputsOf)).toBe('ANY');
    const disagree = [wire('t', 'tensor', 'sw', 'input_0'), wire('x', 'text', 'sw', 'input_1')];
    expect(switchOutputType('sw', [t, x, sw], disagree, outputsOf)).toBe('ANY');
    const unknown = [wire('ghost', 'tensor', 'sw', 'input_0'), wire('t', 'nope', 'sw', 'input_1')];
    expect(switchOutputType('sw', [t, sw], unknown, outputsOf)).toBe('ANY');
  });

  it('leaves out the wires into ignoreHandle', () => {
    const edges = [wire('t', 'tensor', 'sw', 'input_0'), wire('x', 'text', 'sw', 'input_1')];
    expect(switchOutputType('sw', [t, x, sw], edges, outputsOf, 'input_1')).toBe('TENSOR');
  });

  it('takes the type of a Switch that feeds it, and survives a loop', () => {
    const sw2 = card('sw2', SWITCH);
    const chain = [wire('t', 'tensor', 'sw', 'input_0'), wire('sw', 'output', 'sw2', 'input_0')];
    expect(switchOutputType('sw2', [t, sw, sw2], chain, outputsOf)).toBe('TENSOR');
    const loop = [wire('sw2', 'output', 'sw', 'input_0'), wire('sw', 'output', 'sw2', 'input_0')];
    expect(switchOutputType('sw', [sw, sw2], loop, outputsOf)).toBe('ANY');
  });
});

describe('liveOutputType', () => {
  it('is the declared type for an ordinary node, and the inferred one for a Switch', () => {
    const t = card('t', TENSOR_SRC);
    const sw = card('sw', SWITCH);
    const edges = [wire('t', 'tensor', 'sw', 'input_0')];
    expect(liveOutputType(t, 'tensor', [t, sw], edges, outputsOf)).toBe('TENSOR');
    expect(liveOutputType(sw, 'output', [t, sw], edges, outputsOf)).toBe('TENSOR');
    expect(liveOutputType(t, 'missing', [t, sw], edges, outputsOf)).toBe('ANY');
    expect(liveOutputType(undefined, 'output', [t, sw], edges, outputsOf)).toBe('ANY');
    expect(liveOutputType(sw, null, [t, sw], edges, outputsOf)).toBe('ANY');
  });
});

describe('switchAcceptsType', () => {
  const t = card('t', TENSOR_SRC);
  const x = card('x', TEXT_SRC);
  const sw = card('sw', SWITCH);
  const accepts = (
    target: Node<NodeData>,
    handle: string,
    type: string,
    nodes: Node<NodeData>[],
    edges: Edge[],
  ) => switchAcceptsType(target, handle, type, nodes, edges, outputsOf, inputsOf, isValidConnection);

  it('takes a type matching the other inputs and refuses another', () => {
    const edges = [wire('t', 'tensor', 'sw', 'input_0')];
    expect(accepts(sw, 'input_1', 'TENSOR', [t, x, sw], edges)).toBe(true);
    expect(accepts(sw, 'input_1', 'STRING', [t, x, sw], edges)).toBe(false);
  });

  it('lets a wire replace the only other input on its own handle', () => {
    const edges = [wire('t', 'tensor', 'sw', 'input_0')];
    expect(accepts(sw, 'input_0', 'STRING', [t, x, sw], edges)).toBe(true);
  });

  it('does not check the selector or an ANY source', () => {
    const edges = [wire('t', 'tensor', 'sw', 'input_0')];
    expect(accepts(sw, 'selector', 'STRING', [t, x, sw], edges)).toBe(true);
    expect(accepts(sw, 'input_1', 'ANY', [t, x, sw], edges)).toBe(true);
    expect(switchAcceptsType(sw, null, 'STRING', [t, x, sw], edges, outputsOf, inputsOf, isValidConnection)).toBe(true);
  });

  it('refuses a type a port the Switch feeds cannot take', () => {
    const sink = card('sink', TEXT_SINK);
    const edges = [wire('sw', 'output', 'sink', 'text')];
    expect(accepts(sw, 'input_0', 'TENSOR', [t, sw, sink], edges)).toBe(false);
    expect(accepts(sw, 'input_0', 'STRING', [x, sw, sink], edges)).toBe(true);
  });

  it('skips a fed node or port it cannot read, and a wire from no output', () => {
    const sink = card('sink', TENSOR_SINK);
    const edges = [
      wire('sw', 'output', 'ghost', 'tensor'),
      wire('sw', 'output', 'sink', 'nope'),
      { id: 'bare', source: 'sw', target: 'sink', targetHandle: 'tensor' },
    ];
    expect(accepts(sw, 'input_0', 'STRING', [x, sw, sink], edges)).toBe(true);
  });

  it('follows a Switch into the Switch it feeds, and survives a loop', () => {
    const sw2 = card('sw2', SWITCH);
    const sink = card('sink', TEXT_SINK);
    const chain = [wire('sw', 'output', 'sw2', 'input_0'), wire('sw2', 'output', 'sink', 'text')];
    expect(accepts(sw, 'input_0', 'TENSOR', [t, sw, sw2, sink], chain)).toBe(false);
    expect(accepts(sw, 'input_0', 'STRING', [x, sw, sw2, sink], chain)).toBe(true);
    const loop = [wire('sw', 'output', 'sw2', 'input_0'), wire('sw2', 'output', 'sw', 'input_1')];
    expect(accepts(sw, 'input_0', 'TENSOR', [t, sw, sw2], loop)).toBe(true);
  });
});
