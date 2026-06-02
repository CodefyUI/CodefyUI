import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import { NodeConfigPanel } from './NodeConfigPanel';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import type { NodeData, NodeDefinition, ParamDefinition } from '../../types';

// Mock ParamField to isolate NodeConfigPanel and avoid ParamField's REST/file
// backends. The mock records the props it receives and exposes a button to
// trigger onChange so we can verify handleChange wiring.
vi.mock('../shared/ParamField', () => ({
  ParamField: ({ param, value, onChange }: any) => (
    <button
      type="button"
      data-testid={`paramfield-${param.name}`}
      onClick={() => onChange(param.name, 'NEW')}
    >
      field:{param.name}:{String(value)}
    </button>
  ),
}));

// Mock MathText to a plain text container (avoids KaTeX rendering).
vi.mock('../shared/MathText', () => ({
  MathText: ({ text, className }: any) => <div className={className}>{text}</div>,
}));

function makeParam(overrides: Partial<ParamDefinition> = {}): ParamDefinition {
  return {
    name: 'p1',
    param_type: 'int',
    default: 0,
    description: '',
    options: [],
    min_value: null,
    max_value: null,
    ...overrides,
  };
}

function makeDef(overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'Dense',
    category: 'CNN',
    description: '',
    inputs: [],
    outputs: [],
    params: [],
    ...overrides,
  };
}

function makeNode(overrides: Omit<Partial<Node<NodeData>>, 'data'> & { data?: Partial<NodeData> } = {}): Node<NodeData> {
  const { data, ...rest } = overrides;
  return {
    id: 'n1',
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label: 'My Node',
      type: 'Dense',
      params: {},
      definition: makeDef(),
      executionStatus: 'idle',
      ...data,
    },
    ...rest,
  } as Node<NodeData>;
}

/** Seed the active tab with the given nodes + selected id. */
function seedTab(nodes: Node<NodeData>[], selectedNodeId: string | null) {
  useTabStore.setState((state) => ({
    tabs: state.tabs.map((t) =>
      t.id === state.activeTabId ? { ...t, nodes, selectedNodeId } : t,
    ),
  }));
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ isCanvasPanning: false });
  // fresh single tab
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('NodeConfigPanel — null render paths', () => {
  it('renders null when no node is selected', () => {
    seedTab([], null);
    const { container } = render(<NodeConfigPanel />);
    expect(container.firstChild).toBeNull();
  });

  it('renders null when the selected node is a note node', () => {
    const note = makeNode({ id: 'note1', type: 'noteNode', data: { type: 'note', definition: undefined } });
    seedTab([note], 'note1');
    const { container } = render(<NodeConfigPanel />);
    expect(container.firstChild).toBeNull();
  });
});

describe('NodeConfigPanel — header & accent color', () => {
  it('shows title, node label, and category for a non-preset node', () => {
    seedTab([makeNode()], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('Node Config')).toBeInTheDocument();
    expect(screen.getByText('My Node')).toBeInTheDocument();
    expect(screen.getByText('CNN')).toBeInTheDocument();
  });

  it('renders the description via MathText when present', () => {
    seedTab([makeNode({ data: { definition: makeDef({ description: 'A dense layer' }) } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('A dense layer')).toBeInTheDocument();
  });

  it('falls back to the Utility accent color for an unknown category', () => {
    seedTab([makeNode({ data: { definition: makeDef({ category: 'TotallyUnknown' }) } })], 'n1');
    render(<NodeConfigPanel />);
    // category text still rendered
    expect(screen.getByText('TotallyUnknown')).toBeInTheDocument();
  });

  it('defaults category to Utility when definition has no category-ish value (undefined def)', () => {
    // node with no definition at all -> def undefined -> category 'Utility'
    seedTab([makeNode({ data: { definition: undefined } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('Utility')).toBeInTheDocument();
    // no params + not preset -> noParams message
    expect(screen.getByText('No configurable parameters')).toBeInTheDocument();
  });

  it('dims the panel when the canvas is panning', () => {
    useUIStore.setState({ isCanvasPanning: true });
    seedTab([makeNode()], 'n1');
    const { container } = render(<NodeConfigPanel />);
    const panel = container.firstChild as HTMLElement;
    expect(panel.style.opacity).toBe('0.4');
  });
});

describe('NodeConfigPanel — params section', () => {
  it('renders param fields, description hints, and range hints', () => {
    const def = makeDef({
      params: [
        makeParam({ name: 'lr', param_type: 'float', description: 'learning rate', min_value: 0, max_value: 1 }),
        makeParam({ name: 'units', description: '', min_value: 1, max_value: null }),
        makeParam({ name: 'bias', description: '', min_value: null, max_value: 10 }),
        makeParam({ name: 'plain', description: '', min_value: null, max_value: null }),
      ],
    });
    seedTab([makeNode({ data: { definition: def, params: { lr: 0.1, units: 8, bias: 0, plain: 3 } } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('Parameters')).toBeInTheDocument();
    // all 4 param fields rendered with their values
    expect(screen.getByTestId('paramfield-lr')).toHaveTextContent('field:lr:0.1');
    expect(screen.getByTestId('paramfield-units')).toBeInTheDocument();
    // description hint only for lr
    expect(screen.getByText('learning rate')).toBeInTheDocument();
    // range hints: lr has both min & max (-∞/+∞ not used here)
    expect(screen.getByText('Range: 0 — 1')).toBeInTheDocument();
    // units: min only -> max shows +∞
    expect(screen.getByText('Range: 1 — +∞')).toBeInTheDocument();
    // bias: max only -> min shows -∞
    expect(screen.getByText('Range: -∞ — 10')).toBeInTheDocument();
    // exactly 3 range hints (lr, units, bias) — 'plain' has neither min nor max
    expect(screen.getAllByText(/Range:/)).toHaveLength(3);
  });

  it('wires handleChange to updateNodeParams on a field change', () => {
    const def = makeDef({ params: [makeParam({ name: 'lr', param_type: 'float' })] });
    seedTab([makeNode({ data: { definition: def, params: { lr: 0.1 } } })], 'n1');
    render(<NodeConfigPanel />);
    fireEvent.click(screen.getByTestId('paramfield-lr'));
    const tab = useTabStore.getState().getActiveTab();
    expect(tab.nodes[0].data.params.lr).toBe('NEW');
  });

  it('handleChange no-ops when selectedNodeId is falsy (empty-string id edge)', () => {
    // A node whose id is '' is found by find(n => n.id === '') so the panel
    // renders, but the `!selectedNodeId` guard inside handleChange short-circuits
    // because '' is falsy — updateNodeParams is never called.
    const def = makeDef({ params: [makeParam({ name: 'lr', param_type: 'float' })] });
    const node = makeNode({ id: '', data: { definition: def, params: { lr: 0.1 } } });
    seedTab([node], '');
    const updateSpy = vi.spyOn(useTabStore.getState(), 'updateNodeParams');
    render(<NodeConfigPanel />);
    fireEvent.click(screen.getByTestId('paramfield-lr'));
    expect(updateSpy).not.toHaveBeenCalled();
  });

  it('shows the noParams message for a non-preset node with an empty params array', () => {
    seedTab([makeNode({ data: { definition: makeDef({ params: [] }) } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('No configurable parameters')).toBeInTheDocument();
  });
});

describe('NodeConfigPanel — preset node', () => {
  it('shows the preset badge, node-count hint, and a Configure button', () => {
    const presetNode = makeNode({
      id: 'p1',
      type: 'presetNode',
      data: {
        label: 'My Preset',
        type: 'preset:My Preset',
        isPreset: true,
        definition: makeDef({ category: 'CNN', params: [] }),
        presetDefinition: { nodes: [{}, {}, {}] } as any,
        params: {},
      },
    });
    seedTab([presetNode], 'p1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('PRESET')).toBeInTheDocument();
    expect(screen.getByText('3 nodes inside')).toBeInTheDocument();
    const btn = screen.getByText('Configure Preset');
    fireEvent.click(btn);
    expect(useTabStore.getState().getActiveTab().presetModalNodeId).toBe('p1');
    // preset path renders neither the params section nor noParams
    expect(screen.queryByText('No configurable parameters')).not.toBeInTheDocument();
    expect(screen.queryByText('Parameters')).not.toBeInTheDocument();
  });

  it('falls back to 0 nodes when presetDefinition is missing', () => {
    const presetNode = makeNode({
      id: 'p2',
      type: 'presetNode',
      data: {
        label: 'Empty Preset',
        type: 'preset:Empty',
        isPreset: true,
        definition: undefined,
        presetDefinition: undefined,
        params: {},
      },
    });
    seedTab([presetNode], 'p2');
    render(<NodeConfigPanel />);
    expect(screen.getByText('0 nodes inside')).toBeInTheDocument();
  });
});

describe('NodeConfigPanel — I/O ports section', () => {
  it('renders inputs (with optional badge) and outputs', () => {
    const def = makeDef({
      inputs: [
        { name: 'x', data_type: 'TENSOR', description: '', optional: false },
        { name: 'mask', data_type: 'TENSOR', description: '', optional: true },
      ],
      outputs: [{ name: 'y', data_type: 'TENSOR', description: '', optional: false }],
    });
    seedTab([makeNode({ data: { definition: def } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('Ports')).toBeInTheDocument();
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
    expect(screen.getByText('x')).toBeInTheDocument();
    expect(screen.getByText('mask')).toBeInTheDocument();
    expect(screen.getByText('optional')).toBeInTheDocument();
    expect(screen.getByText('y')).toBeInTheDocument();
  });

  it('renders only the inputs sub-section when there are no outputs', () => {
    const def = makeDef({
      inputs: [{ name: 'x', data_type: 'TENSOR', description: '', optional: false }],
      outputs: [],
    });
    seedTab([makeNode({ data: { definition: def } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.queryByText('Outputs')).not.toBeInTheDocument();
  });

  it('renders only the outputs sub-section when there are no inputs', () => {
    const def = makeDef({
      inputs: [],
      outputs: [{ name: 'y', data_type: 'TENSOR', description: '', optional: false }],
    });
    seedTab([makeNode({ data: { definition: def } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.queryByText('Inputs')).not.toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
  });

  it('omits the ports section entirely when there are no inputs or outputs', () => {
    seedTab([makeNode({ data: { definition: makeDef({ inputs: [], outputs: [] }) } })], 'n1');
    render(<NodeConfigPanel />);
    expect(screen.queryByText('Ports')).not.toBeInTheDocument();
  });
});

describe('NodeConfigPanel — execution status', () => {
  function renderWithStatus(status: NodeData['executionStatus'], error?: string) {
    seedTab(
      [makeNode({ data: { definition: makeDef(), executionStatus: status, error } })],
      'n1',
    );
    return render(<NodeConfigPanel />);
  }

  it('hides the execution section when status is idle', () => {
    renderWithStatus('idle');
    expect(screen.queryByText('Execution')).not.toBeInTheDocument();
  });

  it('shows an error status with the error message', () => {
    renderWithStatus('error', 'boom');
    expect(screen.getByText('Execution')).toBeInTheDocument();
    expect(screen.getByText('Error: boom')).toBeInTheDocument();
  });

  it('shows an error status label when no error string is set', () => {
    renderWithStatus('error', undefined);
    // falls back to the status label, not node.error
    expect(screen.getByText('Error')).toBeInTheDocument();
  });

  it('shows a completed status', () => {
    renderWithStatus('completed');
    expect(screen.getByText('Completed')).toBeInTheDocument();
  });

  it('shows a cached status', () => {
    renderWithStatus('cached');
    expect(screen.getByText('Cached')).toBeInTheDocument();
  });

  it('shows a running status (default color branch)', () => {
    renderWithStatus('running');
    expect(screen.getByText('Running')).toBeInTheDocument();
  });
});
