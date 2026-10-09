import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, PortDefinition } from '../../types';

// #655. A Switch's output carries the type of the inputs it chooses between,
// so the canvas checks a wire from it, and a new option into it, against
// that type. The backend's validate_graph refuses the same wires.

type RFProps = Record<string, any>;
const captured: { rf: RFProps } = { rf: {} };

vi.mock('@xyflow/react', async (importActual) => {
  const actual = await importActual<typeof import('@xyflow/react')>();
  return {
    ...actual,
    ReactFlow: (props: RFProps) => {
      captured.rf = props;
      return <div data-testid="reactflow">{props.children}</div>;
    },
    MiniMap: () => null,
    Background: () => null,
    Controls: () => null,
  };
});

import { FlowCanvas } from './FlowCanvas';
import { renderWithFlow } from '../../test/utils';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useI18n } from '../../i18n';

const port = (name: string, data_type: string): PortDefinition => ({
  name, data_type, description: '', optional: false,
});
const def = (node_name: string, inputs: PortDefinition[], outputs: PortDefinition[]): NodeDefinition => ({
  node_name, category: 'Utility', description: '', inputs, outputs, params: [],
});

const SWITCH = def('Switch', [], [port('output', 'ANY')]);
const TENSOR_SRC = def('TensorCreate', [], [port('tensor', 'TENSOR')]);
const TEXT_SRC = def('Text', [], [port('text', 'STRING')]);
const TEXT_SINK = def('TextSink', [port('text', 'STRING')], []);
const TENSOR_SINK = def('TensorSink', [port('tensor', 'TENSOR')], []);

function card(id: string, definition: NodeDefinition): Node<NodeData> {
  return {
    id, type: 'baseNode', position: { x: 0, y: 0 },
    data: { label: id, type: definition.node_name, params: {}, definition },
  };
}
const wire = (source: string, sourceHandle: string, target: string, targetHandle: string): Edge => ({
  id: `${source}-${target}-${targetHandle}`, source, sourceHandle, target, targetHandle,
});

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;
const TAB_ID = 'tab-switch-test';

function mount(nodes: Node<NodeData>[], edges: Edge[]) {
  const base = ORIGINAL_TABS[0];
  useTabStore.setState({
    tabs: [{ ...base, id: TAB_ID, name: 'test', nodes, edges, outputSummaries: {} }],
    activeTabId: TAB_ID,
  });
  renderWithFlow(<FlowCanvas />);
}

const valid = (connection: Partial<Edge>) => captured.rf.isValidConnection(connection) as boolean;

const NODES = [
  card('t', TENSOR_SRC), card('x', TEXT_SRC), card('sw', SWITCH),
  card('textSink', TEXT_SINK), card('tensorSink', TENSOR_SINK),
];

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ gridSnapEnabled: false, draggingSourceType: null, reconnectingHandle: null });
  useNodeDefStore.setState({ definitions: [SWITCH, TENSOR_SRC, TEXT_SRC, TEXT_SINK, TENSOR_SINK], presets: [] });
  captured.rf = {};
});

afterEach(() => {
  cleanup();
  useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
});

describe('wires from and into a Switch', () => {
  it('checks a wire from a Switch against the type its inputs carry', () => {
    mount(NODES, [wire('t', 'tensor', 'sw', 'input_0')]);
    expect(valid({ source: 'sw', sourceHandle: 'output', target: 'tensorSink', targetHandle: 'tensor' })).toBe(true);
    expect(valid({ source: 'sw', sourceHandle: 'output', target: 'textSink', targetHandle: 'text' })).toBe(false);
  });

  it('lets an unwired Switch feed anything', () => {
    mount(NODES, []);
    expect(valid({ source: 'sw', sourceHandle: 'output', target: 'textSink', targetHandle: 'text' })).toBe(true);
  });

  it('refuses an option whose type differs from the other inputs', () => {
    mount(NODES, [wire('t', 'tensor', 'sw', 'input_0')]);
    expect(valid({ source: 't', sourceHandle: 'tensor', target: 'sw', targetHandle: 'input_1' })).toBe(true);
    expect(valid({ source: 'x', sourceHandle: 'text', target: 'sw', targetHandle: 'input_1' })).toBe(false);
  });

  it('refuses an option the ports the Switch feeds cannot take', () => {
    mount(NODES, [wire('sw', 'output', 'textSink', 'text')]);
    expect(valid({ source: 't', sourceHandle: 'tensor', target: 'sw', targetHandle: 'input_0' })).toBe(false);
    expect(valid({ source: 'x', sourceHandle: 'text', target: 'sw', targetHandle: 'input_0' })).toBe(true);
  });

  it('drags a Switch output in the type its inputs carry', () => {
    mount(NODES, [wire('t', 'tensor', 'sw', 'input_0')]);
    act(() => {
      captured.rf.onConnectStart({}, { nodeId: 'sw', handleId: 'output', handleType: 'source' });
    });
    expect(useUIStore.getState().draggingSourceType).toBe('TENSOR');
    act(() => {
      captured.rf.onConnectStart({}, { nodeId: 'x', handleId: 'text', handleType: 'source' });
    });
    expect(useUIStore.getState().draggingSourceType).toBe('STRING');
  });
});
