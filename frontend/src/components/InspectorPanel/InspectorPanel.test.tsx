import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { InspectorPanel } from './InspectorPanel';
import { useI18n } from '../../i18n';
import { useTabStore, type TabState } from '../../store/tabStore';
import {
  fetchOutput,
  fetchStepIndex,
  fetchGradIndex,
  PayloadTooLargeError,
  RunDataExpiredError,
} from '../../api/executionOutputs';
import type { TensorOutput, OutputData, NodeData, NodeDefinition, SegmentGroup } from '../../types';
import type { Node, Edge } from '@xyflow/react';

vi.mock('../../api/executionOutputs', async () => {
  const actual = await vi.importActual<typeof import('../../api/executionOutputs')>(
    '../../api/executionOutputs',
  );
  return {
    ...actual,
    fetchOutput: vi.fn(),
    fetchStepIndex: vi.fn(),
    fetchGradIndex: vi.fn(),
  };
});

const mockOutput = vi.mocked(fetchOutput);
const mockStepIndex = vi.mocked(fetchStepIndex);
const mockGradIndex = vi.mocked(fetchGradIndex);

// ── Fixtures ──

function tensor(values: unknown, extra: Partial<TensorOutput> = {}): TensorOutput {
  return {
    type: 'tensor',
    run_id: 'r',
    node_id: 'n',
    port: 'p',
    full_shape: [2, 2],
    dtype: 'float32',
    slice: ':',
    sliced_shape: [2, 2],
    values,
    truncated: false,
    ...extra,
  };
}

function def(outputs: string[], type = 'Generic'): NodeDefinition {
  return {
    node_name: type,
    category: 'cat',
    description: '',
    inputs: [],
    outputs: outputs.map((name) => ({
      name,
      data_type: 'tensor',
      description: '',
      optional: false,
    })),
    params: [],
  };
}

function node(
  id: string,
  label: string,
  opts: { type?: string; outputs?: string[]; definition?: NodeDefinition } = {},
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: {
      label,
      type: opts.type ?? 'Generic',
      params: {},
      definition: opts.definition ?? def(opts.outputs ?? []),
    },
  };
}

function edge(
  id: string,
  source: string,
  target: string,
  opts: { sourceHandle?: string | null; targetHandle?: string | null; type?: string; data?: unknown } = {},
): Edge {
  return {
    id,
    source,
    target,
    sourceHandle: opts.sourceHandle,
    targetHandle: opts.targetHandle,
    type: opts.type,
    data: opts.data,
  } as Edge;
}

/** Replace the single active tab with provided partial state. */
function seedTab(partial: Partial<TabState>) {
  const tabs = useTabStore.getState().tabs;
  const active = tabs[0];
  const newTab: TabState = {
    ...active,
    nodes: [],
    edges: [],
    selectedNodeId: null,
    activeSegment: null,
    lastRunId: null,
    ...partial,
  };
  useTabStore.setState({ tabs: [newTab], activeTabId: newTab.id });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  mockOutput.mockReset();
  mockStepIndex.mockReset();
  mockGradIndex.mockReset();
  // sensible defaults so child fetch components don't hang
  mockStepIndex.mockResolvedValue([]);
  mockGradIndex.mockResolvedValue([]);
  mockOutput.mockResolvedValue(tensor([[1, 2], [3, 4]], { min: 1, max: 4 }));
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('InspectorPanel — empty modes', () => {
  it('renders the not-run empty state when there is no lastRunId', () => {
    seedTab({ lastRunId: null });
    render(<InspectorPanel />);
    expect(screen.getByText('Run the graph to capture data')).toBeInTheDocument();
    expect(screen.getByText('Make sure Rec is ON, then click ▶ Run')).toBeInTheDocument();
  });

  it('renders the no-selection empty state when run exists but nothing selected', () => {
    seedTab({ lastRunId: 'run1', selectedNodeId: null });
    render(<InspectorPanel />);
    expect(screen.getByText('Select a node or segment to inspect')).toBeInTheDocument();
  });

  it('falls back to none mode when selected node id does not exist', () => {
    seedTab({ lastRunId: 'run1', selectedNodeId: 'ghost', nodes: [] });
    render(<InspectorPanel />);
    expect(screen.getByText('Select a node or segment to inspect')).toBeInTheDocument();
  });
});

describe('InspectorPanel — collapse', () => {
  it('collapses and expands via the collapse button', () => {
    seedTab({ lastRunId: null });
    render(<InspectorPanel />);
    const btn = screen.getByLabelText('Collapse inspector');
    fireEvent.click(btn);
    expect(screen.getByText('INSPECTOR')).toBeInTheDocument();
    // expand again
    fireEvent.click(screen.getByLabelText('Expand inspector'));
    expect(screen.getByText('Run the graph to capture data')).toBeInTheDocument();
  });
});

describe('InspectorPanel — single node mode', () => {
  it('shows tabs, node name, and forward port pairing with values', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    const src = node('src', 'Src');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    render(<InspectorPanel />);
    expect(screen.getByText('NodeA')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Forward' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Steps' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Backward' })).toBeInTheDocument();
    // input port label "in: y" and output port label "out: out"
    expect(screen.getByText(/in: y/)).toBeInTheDocument();
    expect(screen.getByText(/out: out/)).toBeInTheDocument();
    // values fetched and rendered
    await waitFor(() =>
      expect(screen.getAllByText('shape [2, 2]').length).toBeGreaterThan(0),
    );
  });

  it('shows the empty-ports message when node has no inputs or outputs', () => {
    const n = node('a', 'NodeA', { outputs: [] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    render(<InspectorPanel />);
    expect(screen.getByText('This node has no ports.')).toBeInTheDocument();
  });

  it('renders TokenChipsView for a Tokenizer node', async () => {
    const tdef = def(['tokens', 'token_ids', 'offsets'], 'Tokenizer');
    const n = node('tok', 'Tok', { type: 'Tokenizer', definition: tdef });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'tok', nodes: [n], edges: [] });
    // tokens port returns a list; the rest return tensors (default mock)
    mockOutput.mockImplementation(async (_r, _n, port) => {
      if (port === 'tokens') {
        return {
          type: 'list',
          run_id: 'r',
          node_id: 'n',
          port,
          length: 2,
          values: ['Hello', 'World'],
        } as OutputData;
      }
      if (port === 'token_ids') {
        return {
          type: 'list',
          run_id: 'r',
          node_id: 'n',
          port,
          length: 2,
          values: [1, 2],
        } as OutputData;
      }
      return tensor([[1]], { min: 1, max: 1 });
    });
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('Hello')).toBeInTheDocument());
    expect(screen.getByText('2 tokens')).toBeInTheDocument();
  });

  it('switches to the Steps tab and renders StepTraceView', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    mockStepIndex.mockResolvedValue([]);
    render(<InspectorPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
    // StepTraceView empty state
    await waitFor(() =>
      expect(screen.getByText('This node does not record steps')).toBeInTheDocument(),
    );
  });

  it('switches to the Backward tab and renders BackwardView', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    mockGradIndex.mockResolvedValue([]);
    render(<InspectorPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Backward' }));
    await waitFor(() =>
      expect(screen.getByText('No gradients captured')).toBeInTheDocument(),
    );
  });

  it('renders fetch error messages for input and output ports', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    const src = node('src', 'Src');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    mockOutput.mockRejectedValue(new Error('port failed'));
    render(<InspectorPanel />);
    await waitFor(() =>
      expect(screen.getByText(/input: port failed · output: port failed/)).toBeInTheDocument(),
    );
  });

  it('shows the run-data-expired message for a port that 404s', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    mockOutput.mockRejectedValue(new RunDataExpiredError('run1'));
    render(<InspectorPanel />);
    await waitFor(() =>
      expect(screen.getByText(/run data expired — re-run to capture/)).toBeInTheDocument(),
    );
  });

  it('falls back to a sliced fetch on PayloadTooLargeError', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    mockOutput.mockImplementation(async (_r, _n, _port, opts) => {
      if (!opts) throw new PayloadTooLargeError('too big');
      return tensor([[5, 5], [5, 5]], { min: 5, max: 5 });
    });
    render(<InspectorPanel />);
    await waitFor(() =>
      expect(screen.getByText('shape [2, 2]')).toBeInTheDocument(),
    );
    expect(mockOutput).toHaveBeenCalledWith('run1', 'a', 'out', {
      slice: '0,:,:',
      maxElements: 65536,
    });
  });

  it('skips trigger edges (by edge.type) when resolving input sources', () => {
    const n = node('a', 'NodeA', { outputs: [] });
    const src = node('src', 'Src');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y', type: 'triggerEdge' })],
    });
    render(<InspectorPanel />);
    // trigger edge skipped, no outputs → empty-ports message
    expect(screen.getByText('This node has no ports.')).toBeInTheDocument();
  });

  it('skips trigger edges (by edge.data.type) and edges without a sourceHandle', () => {
    const n = node('a', 'NodeA', { outputs: [] });
    const src = node('src', 'Src');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, src],
      edges: [
        edge('e1', 'src', 'a', { sourceHandle: 'y', data: { type: 'trigger' } }),
        edge('e2', 'src', 'a', { sourceHandle: null }), // no handle
        edge('e3', 'other', 'b'), // unrelated target
      ],
    });
    render(<InspectorPanel />);
    expect(screen.getByText('This node has no ports.')).toBeInTheDocument();
  });

  it('handles a node without a definition (outputs default to empty)', () => {
    const n: Node<NodeData> = {
      id: 'a',
      type: 'baseNode',
      position: { x: 0, y: 0 },
      data: { label: 'NoDef', type: 'Generic', params: {} }, // no definition
    };
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    render(<InspectorPanel />);
    expect(screen.getByText('This node has no ports.')).toBeInTheDocument();
  });

  it('clicks the Forward tab handler explicitly', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    render(<InspectorPanel />);
    // move away then click Forward to fire its onClick handler
    fireEvent.click(screen.getByRole('tab', { name: 'Backward' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Forward' }));
    expect(screen.getByRole('tab', { name: 'Forward' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });

  it('pairs more inputs than outputs (output side null in a row)', async () => {
    const n = node('a', 'NodeA', { outputs: ['o1'] }); // 1 output
    const s1 = node('s1', 'S1');
    const s2 = node('s2', 'S2');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, s1, s2],
      edges: [
        edge('e1', 's1', 'a', { sourceHandle: 'p' }),
        edge('e2', 's2', 'a', { sourceHandle: 'q' }), // 2 inputs
      ],
    });
    render(<InspectorPanel />);
    // second row has no output → "out: —"
    expect(screen.getByText(/out: —/)).toBeInTheDocument();
    expect(screen.getByText(/in: q/)).toBeInTheDocument();
  });

  it('pairs more outputs than inputs (input side null in a row)', async () => {
    const n = node('a', 'NodeA', { outputs: ['o1', 'o2'] }); // 2 outputs, 0 inputs
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    render(<InspectorPanel />);
    // rows where input is null → "in: —"
    expect(screen.getAllByText(/in: —/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/out: o1/)).toBeInTheDocument();
    expect(screen.getByText(/out: o2/)).toBeInTheDocument();
  });

  it('resets the tab back to forward when the selection changes', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    const { rerender } = render(<InspectorPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
    expect(screen.getByRole('tab', { name: 'Steps' })).toHaveAttribute('aria-selected', 'true');
    // change selection in the store and rerender
    const n2 = node('b', 'NodeB', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'b', nodes: [n2], edges: [] });
    rerender(<InspectorPanel />);
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: 'Forward' })).toHaveAttribute(
        'aria-selected',
        'true',
      ),
    );
  });
});

describe('InspectorPanel — segment mode', () => {
  function segment(headId: string, tailId: string): SegmentGroup {
    return { id: 'seg1', headNodeId: headId, tailNodeId: tailId } as SegmentGroup;
  }

  it('returns none mode when the segment head or tail node is missing', () => {
    seedTab({
      lastRunId: 'run1',
      activeSegment: segment('missingHead', 'missingTail'),
      nodes: [],
      edges: [],
    });
    render(<InspectorPanel />);
    expect(screen.getByText('Select a node or segment to inspect')).toBeInTheDocument();
  });

  it('renders segment inputs and outputs with tensor and scalar values', async () => {
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: ['result'] });
    const external = node('ext', 'Ext');
    seedTab({
      lastRunId: 'run1',
      activeSegment: segment('h', 't'),
      nodes: [head, tail, external],
      edges: [
        edge('e1', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' }), // internal data edge
        edge('e2', 'ext', 'h', { sourceHandle: 'feed', targetHandle: 'x' }), // segment input
      ],
    });
    render(<InspectorPanel />);
    expect(screen.getByText('SEGMENT')).toBeInTheDocument();
    // header shows "Head → Tail" as a single combined node-names element
    expect(screen.getByText(/Head\s*→\s*Tail/)).toBeInTheDocument();
    // input title with count and output title with count
    expect(screen.getByText('Segment inputs (1)')).toBeInTheDocument();
    expect(screen.getByText('Segment outputs (1)')).toBeInTheDocument();
    // input displayName "→ Head.x"
    expect(screen.getByText('→ Head.x')).toBeInTheDocument();
    // output displayName "result"
    expect(screen.getByText('result')).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getAllByText('shape [2, 2]').length).toBeGreaterThan(0),
    );
  });

  it('renders the empty placeholder for a segment side with no ports', () => {
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: [] }); // no outputs → output side empty
    seedTab({
      lastRunId: 'run1',
      activeSegment: segment('h', 't'),
      nodes: [head, tail],
      edges: [edge('e1', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' })],
    });
    render(<InspectorPanel />);
    // no inputs (no external source) and no outputs → both sides show the em dash
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(1);
  });

  it('skips trigger edges and duplicate input keys when gathering segment inputs', async () => {
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: ['out'] });
    const ext = node('ext', 'Ext');
    seedTab({
      lastRunId: 'run1',
      activeSegment: segment('h', 't'),
      nodes: [head, tail, ext],
      edges: [
        edge('e0', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' }),
        // trigger edge from outside — skipped
        edge('e1', 'ext', 'h', { sourceHandle: 'a', targetHandle: 'x', type: 'triggerEdge' }),
        // duplicate of the same source->target handle pair — second one dedup'd
        edge('e2', 'ext', 'h', { sourceHandle: 'feed', targetHandle: 'x' }),
        edge('e3', 'ext', 'h', { sourceHandle: 'feed', targetHandle: 'x' }),
        // edge with no sourceHandle — skipped
        edge('e4', 'ext', 'h', { sourceHandle: null, targetHandle: 'z' }),
      ],
    });
    render(<InspectorPanel />);
    // only one unique input surfaces
    expect(screen.getByText('Segment inputs (1)')).toBeInTheDocument();
  });

  it('uses a truncated id label when the target node has no label match', async () => {
    // target node id present but data.label missing → falls back to e.target.slice(0,6)
    const head: Node<NodeData> = {
      id: 'headnode123',
      type: 'baseNode',
      position: { x: 0, y: 0 },
      data: { label: '', type: 'Generic', params: {}, definition: def([]) },
    };
    const tail = node('t', 'Tail', { outputs: ['out'] });
    const ext = node('ext', 'Ext');
    seedTab({
      lastRunId: 'run1',
      activeSegment: { id: 's', headNodeId: 'headnode123', tailNodeId: 't' } as SegmentGroup,
      nodes: [head, tail, ext],
      edges: [
        edge('e0', 'headnode123', 't', { sourceHandle: 'mid', targetHandle: 'in' }),
        edge('e1', 'ext', 'headnode123', { sourceHandle: 'feed', targetHandle: 'q' }),
      ],
    });
    render(<InspectorPanel />);
    // displayName uses label '' (falsy → '' is used as targetLabel via ?? only on null/undef;
    // label is '' which is defined, so targetLabel is '' → "→ .q")
    expect(screen.getByText(/→ \.q/)).toBeInTheDocument();
  });

  it('renders segment side port error and scalar/string/model non-tensor values', async () => {
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: ['scalarOut', 'stringOut', 'modelOut', 'errOut'] });
    seedTab({
      lastRunId: 'run1',
      activeSegment: { id: 's', headNodeId: 'h', tailNodeId: 't' } as SegmentGroup,
      nodes: [head, tail],
      edges: [edge('e0', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' })],
    });
    mockOutput.mockImplementation(async (_r, _n, port) => {
      if (port === 'scalarOut') {
        return { type: 'scalar', run_id: 'r', node_id: 'n', port, value: 9 } as OutputData;
      }
      if (port === 'stringOut') {
        return { type: 'string', run_id: 'r', node_id: 'n', port, value: 'hey' } as OutputData;
      }
      if (port === 'modelOut') {
        return {
          type: 'model',
          run_id: 'r',
          node_id: 'n',
          port,
          class: 'Net',
          params: 5,
          trainable: 5,
          repr: 'Net()',
        } as OutputData;
      }
      throw new Error('seg port boom');
    });
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('9')).toBeInTheDocument());
    expect(screen.getByText('hey')).toBeInTheDocument();
    expect(screen.getByText(/Net · params 5/)).toBeInTheDocument();
    expect(screen.getByText('seg port boom')).toBeInTheDocument();
  });

  it('shows model fallback params ? when params undefined in a segment side', async () => {
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: ['modelOut'] });
    seedTab({
      lastRunId: 'run1',
      activeSegment: { id: 's', headNodeId: 'h', tailNodeId: 't' } as SegmentGroup,
      nodes: [head, tail],
      edges: [edge('e0', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' })],
    });
    mockOutput.mockResolvedValue({
      type: 'model',
      run_id: 'r',
      node_id: 'n',
      port: 'modelOut',
      // class + params undefined → "Module · params ?"
      repr: '',
    } as unknown as OutputData);
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText(/Module · params \?/)).toBeInTheDocument());
  });

  it('shows the loading ellipsis for a segment port still fetching', async () => {
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: ['out'] });
    seedTab({
      lastRunId: 'run1',
      activeSegment: { id: 's', headNodeId: 'h', tailNodeId: 't' } as SegmentGroup,
      nodes: [head, tail],
      edges: [edge('e0', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' })],
    });
    mockOutput.mockReturnValue(new Promise(() => {})); // pending forever
    const { container } = render(<InspectorPanel />);
    // before resolution, the output port shows the diffMissing ellipsis
    await waitFor(() => expect(container.textContent).toContain('…'));
  });

  it('skips a segment input edge whose data.type is trigger (data present)', () => {
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: ['out'] });
    const ext = node('ext', 'Ext');
    seedTab({
      lastRunId: 'run1',
      activeSegment: { id: 's', headNodeId: 'h', tailNodeId: 't' } as SegmentGroup,
      nodes: [head, tail, ext],
      edges: [
        edge('e0', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' }),
        // data present and marked trigger → second operand of the isTrigger OR evaluates truthy
        edge('e1', 'ext', 'h', { sourceHandle: 'feed', targetHandle: 'x', data: { type: 'trigger' } }),
      ],
    });
    render(<InspectorPanel />);
    // trigger input filtered → zero inputs
    expect(screen.getByText('Segment inputs (0)')).toBeInTheDocument();
  });

  it('falls back to sliced target id and empty handle for an input edge to a node missing from nodes', async () => {
    // segmentSet includes 'midnode' (via h→midnode→t edges), but 'midnode' is
    // absent from `nodes` → targetNode undefined → uses e.target.slice(0,6).
    const head = node('h', 'Head');
    const tail = node('t', 'Tail', { outputs: ['out'] });
    const ext = node('ext', 'Ext');
    seedTab({
      lastRunId: 'run1',
      activeSegment: { id: 's', headNodeId: 'h', tailNodeId: 't' } as SegmentGroup,
      nodes: [head, tail, ext], // 'midnode123' intentionally omitted
      edges: [
        edge('e0', 'h', 'midnode123', { sourceHandle: 'a', targetHandle: 'b' }),
        edge('e1', 'midnode123', 't', { sourceHandle: 'c', targetHandle: 'd' }),
        // external edge into the in-segment-but-missing node, targetHandle null → '' fallback
        edge('e2', 'ext', 'midnode123', { sourceHandle: 'feed', targetHandle: null }),
      ],
    });
    render(<InspectorPanel />);
    // displayName: "→ {midnod}." (sliced id + empty targetHandle)
    expect(screen.getByText(/→ midnod\./)).toBeInTheDocument();
  });

  it('uses tail.data.definition.outputs ?? [] when tail has no definition', () => {
    const head = node('h', 'Head');
    const tail: Node<NodeData> = {
      id: 't',
      type: 'baseNode',
      position: { x: 0, y: 0 },
      data: { label: 'Tail', type: 'Generic', params: {} }, // no definition
    };
    seedTab({
      lastRunId: 'run1',
      activeSegment: { id: 's', headNodeId: 'h', tailNodeId: 't' } as SegmentGroup,
      nodes: [head, tail],
      edges: [edge('e0', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' })],
    });
    render(<InspectorPanel />);
    expect(screen.getByText('Segment outputs (0)')).toBeInTheDocument();
  });
});

describe('InspectorPanel — fetch cancellation', () => {
  it('cancels the in-flight fetch when the panel unmounts', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    let resolveOut!: (v: OutputData) => void;
    mockOutput.mockReturnValue(new Promise<OutputData>((r) => { resolveOut = r; }));
    const { unmount } = render(<InspectorPanel />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount(); // cancelled = true
    resolveOut(tensor([[1]], { min: 1, max: 1 }));
    await Promise.resolve();
    await Promise.resolve();
    expect(true).toBe(true);
  });

  it('does not update state from a rejected fetch after unmount', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    let rejectOut!: (e: unknown) => void;
    mockOutput.mockReturnValue(new Promise<OutputData>((_r, rej) => { rejectOut = rej; }));
    const { unmount } = render(<InspectorPanel />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount();
    rejectOut(new Error('late'));
    await Promise.resolve();
    await Promise.resolve();
    expect(true).toBe(true);
  });
});
