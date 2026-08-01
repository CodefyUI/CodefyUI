import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import type { NodeData, NodeDefinition } from '../../types';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';

// The field is covered by ScriptCodeField.test.tsx; stub it so this file is
// about the port controls and does not need the validation endpoint.
vi.mock('../shared/ScriptCodeField', () => ({
  ScriptCodeField: ({ value, onChange, displayLabel }: any) => (
    <textarea
      aria-label={displayLabel}
      value={String(value ?? '')}
      onChange={(e) => onChange('code', e.target.value)}
    />
  ),
}));

import { CodeTab, hasCodeParam } from './CodeTab';
import type { NodeDetailTabContext } from './tabs';

const scriptDef: NodeDefinition = {
  node_name: 'PythonScript',
  category: 'Utility',
  description: '',
  inputs: [{ name: 'in1', data_type: 'TENSOR', description: '', optional: true }],
  outputs: [{ name: 'out1', data_type: 'ANY', description: '', optional: false }],
  params: [
    { name: 'code', param_type: 'code', default: 'def run(inputs, params):\n    return 1\n', description: '', options: [], min_value: null, max_value: null },
    { name: 'input_ports', param_type: 'int', default: 1, description: '', options: [], min_value: 1, max_value: 8 },
    { name: 'output_ports', param_type: 'int', default: 1, description: '', options: [], min_value: 1, max_value: 8 },
    { name: 'input_types', param_type: 'string', default: 'TENSOR', description: '', options: [], min_value: null, max_value: null },
    { name: 'output_types', param_type: 'string', default: 'ANY', description: '', options: [], min_value: null, max_value: null },
  ],
};

function makeNode(params: Record<string, unknown> = {}): Node<NodeData> {
  return {
    id: 'py1',
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: 'PythonScript',
      type: 'PythonScript',
      definition: scriptDef,
      params: {
        code: 'def run(inputs, params):\n    return 1\n',
        input_ports: 1,
        output_ports: 1,
        input_types: 'TENSOR',
        output_types: 'ANY',
        ...params,
      },
    },
  } as Node<NodeData>;
}

function ctxFor(node: Node<NodeData>): NodeDetailTabContext {
  return {
    nodeId: node.id,
    node,
    runId: null,
    nodes: [node],
    edges: [],
    recordOutputs: false,
    outputSummaries: {},
    focusPort: null,
  };
}

const updateNodeParams = vi.fn();

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  updateNodeParams.mockReset();
  useTabStore.setState({ updateNodeParams } as never);
});

describe('CodeTab', () => {
  it('lists one row per live port, typed', () => {
    render(<CodeTab ctx={ctxFor(makeNode({ input_ports: 2, input_types: 'TENSOR,STRING' }))} />);
    expect(screen.getByText('in1')).toBeTruthy();
    expect(screen.getByText('in2')).toBeTruthy();
    expect((screen.getByLabelText('in2 type') as HTMLSelectElement).value).toBe('STRING');
  });

  it('regenerates the port rows when the count changes', () => {
    const { rerender } = render(<CodeTab ctx={ctxFor(makeNode())} />);
    expect(screen.queryByText('in3')).toBeNull();

    fireEvent.change(screen.getByLabelText('Number of input ports'), {
      target: { value: '3' },
    });
    expect(updateNodeParams).toHaveBeenCalledWith('py1', {
      input_ports: 3,
      // The type list grows with the count, repeating the last entry, so the
      // new ports are typed rather than silently ANY.
      input_types: 'TENSOR,TENSOR,TENSOR',
    });

    // The store is the source of truth; re-render as the app would.
    rerender(<CodeTab ctx={ctxFor(makeNode({ input_ports: 3, input_types: 'TENSOR,TENSOR,TENSOR' }))} />);
    expect(screen.getByText('in3')).toBeTruthy();
  });

  it('writes one type per port when a single port type changes', () => {
    render(<CodeTab ctx={ctxFor(makeNode({ output_ports: 3, output_types: 'ANY' }))} />);
    fireEvent.change(screen.getByLabelText('out2 type'), { target: { value: 'TENSOR' } });
    expect(updateNodeParams).toHaveBeenCalledWith('py1', {
      output_types: 'ANY,TENSOR,ANY',
    });
  });

  it('offers every data type except TRIGGER', () => {
    render(<CodeTab ctx={ctxFor(makeNode())} />);
    const options = [...(screen.getByLabelText('out1 type') as HTMLSelectElement).options].map(
      (o) => o.value,
    );
    expect(options).toContain('TENSOR');
    expect(options).toContain('ANY');
    expect(options).not.toContain('TRIGGER');
  });

  it('writes code edits back to the same param', () => {
    render(<CodeTab ctx={ctxFor(makeNode())} />);
    fireEvent.change(screen.getByLabelText('Script'), { target: { value: 'x = 2' } });
    expect(updateNodeParams).toHaveBeenCalledWith('py1', { code: 'x = 2' });
  });

  it('states the security model in plain words', () => {
    render(<CodeTab ctx={ctxFor(makeNode())} />);
    expect(screen.getByText(/guardrail, not a sandbox/)).toBeTruthy();
  });

  it('says so rather than throwing when the node has no code param', () => {
    const node = makeNode();
    node.data.definition = { ...scriptDef, params: [] };
    render(<CodeTab ctx={ctxFor(node)} />);
    expect(screen.getByText('This node has no code parameter')).toBeTruthy();
  });
});

describe('hasCodeParam', () => {
  it('is true only for a node whose definition declares a code param', () => {
    expect(hasCodeParam(makeNode())).toBe(true);
    const plain = makeNode();
    plain.data.definition = { ...scriptDef, params: [] };
    expect(hasCodeParam(plain)).toBe(false);
    expect(hasCodeParam(undefined)).toBe(false);
  });
});
