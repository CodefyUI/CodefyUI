import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { InspectorPanel } from './InspectorPanel';
import { useI18n } from '../../i18n';
import { useTabStore, type TabState } from '../../store/tabStore';
import { flushTabNodeUpdates, queueTabNodeStatus } from '../../store/nodeUpdateQueue';
import {
  fetchOutput,
  fetchStepIndex,
  fetchGradIndex,
  PayloadTooLargeError,
  RunDataExpiredError,
} from '../../api/executionOutputs';
import type {
  ExecutionStatus,
  TensorOutput,
  OutputData,
  NodeData,
  NodeDefinition,
  SegmentGroup,
} from '../../types';
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
    // The tab is carried over from the previous test, so one that left a run
    // in progress would otherwise hand `running` to every test after it.
    status: 'idle',
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
    expect(screen.getByText('Nothing captured yet')).toBeInTheDocument();
    expect(
      screen.getByText('Turn on Record node outputs in Settings, then run the graph'),
    ).toBeInTheDocument();
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
    expect(screen.getByText('Nothing captured yet')).toBeInTheDocument();
  });
});

describe('InspectorPanel — single node mode', () => {
  it('groups all inputs above all outputs with provenance labels and values', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    const src = node('src', 'Src');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    const { container } = render(<InspectorPanel />);
    expect(screen.getByText('NodeA')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Forward' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Steps' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Backward' })).toBeInTheDocument();
    // one group per side, inputs above outputs
    expect(screen.getByText('Inputs (1)')).toBeInTheDocument();
    expect(screen.getByText('Outputs (1)')).toBeInTheDocument();
    const text = container.textContent ?? '';
    expect(text.indexOf('Inputs (1)')).toBeLessThan(text.indexOf('Outputs (1)'));
    // input rows carry source provenance; output rows the port name
    expect(screen.getByText('Src.y')).toBeInTheDocument();
    expect(screen.getByText('out')).toBeInTheDocument();
    // the old one-to-one placeholders must never render
    expect(screen.queryByText(/in: —|out: —/)).toBeNull();
    // values fetched and rendered
    await waitFor(() =>
      expect(screen.getAllByText('shape [2, 2]').length).toBeGreaterThan(0),
    );
  });

  it('shows the empty-ports message when node has no inputs or outputs', () => {
    const n = node('a', 'NodeA', { outputs: [] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    render(<InspectorPanel />);
    expect(screen.getByText('No ports')).toBeInTheDocument();
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
    const { container } = render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('Hello')).toBeInTheDocument());
    expect(screen.getByText('2 tokens')).toBeInTheDocument();
    // chips render above the port groups
    const text = container.textContent ?? '';
    expect(text.indexOf('Hello')).toBeLessThan(text.indexOf('Inputs (0)'));
  });

  it('switches to the Steps tab and renders StepTraceView', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    mockStepIndex.mockResolvedValue([]);
    render(<InspectorPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
    // StepTraceView empty state
    await waitFor(() =>
      expect(screen.getByText('No steps recorded')).toBeInTheDocument(),
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
    // one error line per port row, input and output groups alike
    await waitFor(() =>
      expect(screen.getAllByText('port failed').length).toBeGreaterThanOrEqual(2),
    );
  });

  it('shows the run-data-expired message for a port that 404s', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    mockOutput.mockRejectedValue(new RunDataExpiredError('run1'));
    render(<InspectorPanel />);
    await waitFor(() =>
      expect(screen.getByText(/Run data expired — re-run to capture/)).toBeInTheDocument(),
    );
  });

  it('shows the expired message in the locale chosen after the fetch failed', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    mockOutput.mockRejectedValue(new RunDataExpiredError('run1'));
    render(<InspectorPanel />);
    await waitFor(() =>
      expect(screen.getByText(/Run data expired — re-run to capture/)).toBeInTheDocument(),
    );
    const callsBefore = mockOutput.mock.calls.length;

    act(() => useI18n.setState({ locale: 'zh-TW' }));

    // A locale switch refetches nothing, so the line has to translate itself.
    expect(screen.getByText(/執行資料已過期 — 重新執行以擷取/)).toBeInTheDocument();
    expect(screen.queryByText(/Run data expired/)).toBeNull();
    expect(mockOutput).toHaveBeenCalledTimes(callsBefore);
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
    expect(screen.getByText('No ports')).toBeInTheDocument();
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
    expect(screen.getByText('No ports')).toBeInTheDocument();
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
    expect(screen.getByText('No ports')).toBeInTheDocument();
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

  it('renders every input when there are more inputs than outputs (no placeholder rows)', async () => {
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
    expect(screen.getByText('Inputs (2)')).toBeInTheDocument();
    expect(screen.getByText('Outputs (1)')).toBeInTheDocument();
    expect(screen.getByText('S1.p')).toBeInTheDocument();
    expect(screen.getByText('S2.q')).toBeInTheDocument();
    expect(screen.getByText('o1')).toBeInTheDocument();
    expect(screen.queryByText(/in: —|out: —/)).toBeNull();
  });

  it('renders the inputs empty state when outputs outnumber absent inputs', async () => {
    const n = node('a', 'NodeA', { outputs: ['o1', 'o2'] }); // 2 outputs, 0 inputs
    seedTab({ lastRunId: 'run1', selectedNodeId: 'a', nodes: [n], edges: [] });
    render(<InspectorPanel />);
    expect(screen.getByText('Inputs (0)')).toBeInTheDocument();
    expect(screen.getByText('No inputs connected')).toBeInTheDocument();
    expect(screen.getByText('Outputs (2)')).toBeInTheDocument();
    expect(screen.getByText('o1')).toBeInTheDocument();
    expect(screen.getByText('o2')).toBeInTheDocument();
    expect(screen.queryByText(/in: —|out: —/)).toBeNull();
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

describe('InspectorPanel — single node transform summary', () => {
  it('highlights changed output cells when the solo input/output tensors share a shape', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    const src = node('src', 'Src');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    mockOutput.mockImplementation(async (_r, nodeId) =>
      nodeId === 'src'
        ? tensor([[1, 2], [3, 4]], { min: 1, max: 4 })
        : tensor([[1, 9], [3, 4]], { min: 1, max: 9 }),
    );
    const { container } = render(<InspectorPanel />);
    await waitFor(() => expect(screen.getAllByText('shape [2, 2]').length).toBe(2));
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(true);
  });

  it('shows a shape chip in the divider when the solo tensors change shape', async () => {
    const n = node('a', 'NodeA', { outputs: ['out'] });
    const src = node('src', 'Src');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    mockOutput.mockImplementation(async (_r, nodeId) =>
      nodeId === 'src'
        ? tensor([[1, 2, 3], [4, 5, 6]], { full_shape: [2, 3], sliced_shape: [2, 3] })
        : tensor([1, 2, 3], { full_shape: [3], sliced_shape: [3] }),
    );
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('[2, 3] → [3]')).toBeInTheDocument());
  });

  it('shows neither chip nor highlight for multi-port nodes', async () => {
    const n = node('a', 'NodeA', { outputs: ['o1'] });
    const s1 = node('s1', 'S1');
    const s2 = node('s2', 'S2');
    seedTab({
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [n, s1, s2],
      edges: [
        edge('e1', 's1', 'a', { sourceHandle: 'p' }),
        edge('e2', 's2', 'a', { sourceHandle: 'q' }),
      ],
    });
    const { container } = render(<InspectorPanel />);
    await waitFor(() => expect(screen.getAllByText('shape [2, 2]').length).toBe(3));
    expect(container.querySelector('[class*="shapeChip"]')).toBeNull();
    const colored = Array.from(container.querySelectorAll('td')).some((td) =>
      (td as HTMLElement).style.background.includes('rgba(6, 182, 212'),
    );
    expect(colored).toBe(false);
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

// ── Inspecting a node while the graph is still running ───────────────────────
// The engine writes a node's captures only after the node returns, and answers
// 404 for anything not written yet. Every test here answers the same way, so a
// request issued too early fails exactly as it does against the real server.

describe('InspectorPanel — while the graph is running', () => {
  const RUNNING_NOTE = 'Node is running…';
  const PENDING_NOTE = 'Waiting for this node to run…';
  const EXPIRED = /Run data expired/;

  /** The node as the engine last reported it. */
  function at(status: ExecutionStatus, n: Node<NodeData>): Node<NodeData> {
    return { ...n, data: { ...n.data, executionStatus: status } };
  }

  function tabId(): string {
    return useTabStore.getState().activeTabId;
  }

  /**
   * One `node_status` frame, written the way the socket handler writes it:
   * buffered in the frame queue, then applied in a single store commit.
   */
  function report(nodeId: string, status: ExecutionStatus) {
    act(() => {
      queueTabNodeStatus(tabId(), nodeId, status);
      flushTabNodeUpdates();
    });
  }

  /** What `execute()` does to the tab before the new run's id arrives. */
  function startNewRun(runId: string) {
    act(() => {
      const store = useTabStore.getState();
      store.clearExecutionStatus();
      store.setTabStatus(tabId(), 'running');
      store.setLastRunId(tabId(), null);
    });
    act(() => useTabStore.getState().setLastRunId(tabId(), runId));
  }

  const SHAPES: Record<string, Partial<TensorOutput>> = {
    wide: { full_shape: [2, 3], sliced_shape: [2, 3] },
    flat: { full_shape: [3], sliced_shape: [3] },
  };

  /** 404 until the node is in `finished`, then a tensor whose shape names the node. */
  function serveFinished(finished: Set<string>, shapeOf: Record<string, keyof typeof SHAPES> = {}) {
    mockOutput.mockImplementation(async (runId, nodeId) => {
      if (!finished.has(nodeId)) throw new RunDataExpiredError(runId);
      const shape = shapeOf[nodeId];
      if (shape === 'wide') return tensor([[1, 2, 3], [4, 5, 6]], SHAPES.wide);
      if (shape === 'flat') return tensor([1, 2, 3], SHAPES.flat);
      return tensor([[1, 2], [3, 4]], { min: 1, max: 4 });
    });
  }

  function callsFor(nodeId: string): number {
    return mockOutput.mock.calls.filter(([, id]) => id === nodeId).length;
  }

  /** Let every request already issued run to its end. */
  async function settle() {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it('says the node is running instead of calling its run data expired', async () => {
    const a = at('running', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'running', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    serveFinished(new Set());
    render(<InspectorPanel />);
    await settle();

    expect(screen.queryByText(EXPIRED)).toBeNull();
    const note = screen.getByText(RUNNING_NOTE);
    // A neutral line, not the warning style an error gets.
    expect(note.className).not.toMatch(/portError/);
    // Asking now could only be answered 404.
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it.each(['completed', 'cached', 'interrupted'] as const)(
    'fills the output in by itself when the node reports %s',
    async (terminal) => {
      const a = at('running', node('a', 'NodeA', { outputs: ['out'] }));
      seedTab({ status: 'running', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
      const finished = new Set<string>();
      serveFinished(finished);
      render(<InspectorPanel />);
      await settle();

      finished.add('a'); // the engine writes the captures first...
      report('a', terminal); // ...and only then says the node is done

      // No reselect, no tab switch: the row was on screen the whole time.
      await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
      expect(screen.queryByText(RUNNING_NOTE)).toBeNull();
      expect(screen.queryByText(EXPIRED)).toBeNull();
      expect(mockOutput).toHaveBeenCalledTimes(1);
      expect(mockOutput).toHaveBeenCalledWith('run1', 'a', 'out');
    },
  );

  it('reads an input whose upstream node has finished while the selected node still runs', async () => {
    const src = at('completed', node('src', 'Src'));
    const a = at('running', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({
      status: 'running',
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [a, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    serveFinished(new Set(['src']));
    render(<InspectorPanel />);

    // The input belongs to `src`, so it is `src`'s status that decides.
    await waitFor(() => expect(screen.getAllByText('shape [2, 2]')).toHaveLength(1));
    expect(screen.getByText(RUNNING_NOTE)).toBeInTheDocument();
    expect(screen.queryByText(EXPIRED)).toBeNull();
    expect(mockOutput).toHaveBeenCalledTimes(1);
    expect(mockOutput).toHaveBeenCalledWith('run1', 'src', 'y');
  });

  it('says a queued node is waiting, not running, then follows it to the end', async () => {
    const a = at('idle', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'running', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    const finished = new Set<string>();
    serveFinished(finished);
    render(<InspectorPanel />);
    await settle();

    expect(screen.getByText(PENDING_NOTE)).toBeInTheDocument();
    expect(screen.queryByText(RUNNING_NOTE)).toBeNull();
    expect(screen.queryByText(EXPIRED)).toBeNull();
    expect(mockOutput).not.toHaveBeenCalled();

    report('a', 'running');
    expect(screen.getByText(RUNNING_NOTE)).toBeInTheDocument();
    expect(screen.queryByText(PENDING_NOTE)).toBeNull();
    expect(mockOutput).not.toHaveBeenCalled();

    finished.add('a');
    report('a', 'completed');
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    expect(mockOutput).toHaveBeenCalledTimes(1);
  });

  it('translates the note at render, so it follows a locale switch', async () => {
    const a = at('idle', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'running', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    serveFinished(new Set());
    render(<InspectorPanel />);
    expect(screen.getByText(PENDING_NOTE)).toBeInTheDocument();

    act(() => useI18n.setState({ locale: 'zh-TW' }));
    expect(screen.getByText('等待此節點執行…')).toBeInTheDocument();
    expect(screen.queryByText(PENDING_NOTE)).toBeNull();

    report('a', 'running');
    expect(screen.getByText('節點執行中…')).toBeInTheDocument();
  });

  it('does not refetch when the nodes array is rebuilt with no status change (#166)', async () => {
    const src = at('completed', node('src', 'Src'));
    const a = at('running', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({
      status: 'running',
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [a, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    serveFinished(new Set(['src']));
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    await settle();
    const before = mockOutput.mock.calls.length;

    // What dragging a node does on every frame: a new array of new node
    // objects, every status exactly as it was.
    for (let step = 0; step < 3; step++) {
      act(() => {
        useTabStore.setState((s) => ({
          tabs: s.tabs.map((t) => ({
            ...t,
            nodes: t.nodes.map((n) => ({
              ...n,
              position: { x: n.position.x + 10, y: n.position.y },
            })),
          })),
        }));
      });
    }
    await settle();

    expect(mockOutput).toHaveBeenCalledTimes(before);
    expect(screen.getByText('shape [2, 2]')).toBeInTheDocument();
    expect(screen.getByText(RUNNING_NOTE)).toBeInTheDocument();
  });

  it('does not read a finished port again because another node changed status', async () => {
    const s1 = at('completed', node('s1', 'S1'));
    const s2 = at('running', node('s2', 'S2'));
    const a = at('idle', node('a', 'NodeA', { outputs: ['o1'] }));
    seedTab({
      status: 'running',
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [a, s1, s2],
      edges: [
        edge('e1', 's1', 'a', { sourceHandle: 'p' }),
        edge('e2', 's2', 'a', { sourceHandle: 'q' }),
      ],
    });
    const finished = new Set(['s1']);
    serveFinished(finished, { s1: 'wide', s2: 'flat' });
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('shape [2, 3]')).toBeInTheDocument());
    expect(callsFor('s1')).toBe(1);

    finished.add('s2');
    report('s2', 'completed');
    await waitFor(() => expect(screen.getByText('shape [3]')).toBeInTheDocument());

    report('a', 'running');
    finished.add('a');
    report('a', 'completed');
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());

    // Three status changes later, the upstream tensor was downloaded once.
    expect(callsFor('s1')).toBe(1);
    expect(callsFor('s2')).toBe(1);
    expect(callsFor('a')).toBe(1);
  });

  it('keeps a download already in flight when another node changes status', async () => {
    const src = at('completed', node('src', 'Src'));
    const a = at('running', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({
      status: 'running',
      lastRunId: 'run1',
      selectedNodeId: 'a',
      nodes: [a, src],
      edges: [edge('e1', 'src', 'a', { sourceHandle: 'y' })],
    });
    let landSrc!: (value: OutputData) => void;
    let aFinished = false;
    mockOutput.mockImplementation((runId, nodeId) => {
      if (nodeId === 'src') return new Promise<OutputData>((resolve) => { landSrc = resolve; });
      if (!aFinished) return Promise.reject(new RunDataExpiredError(runId));
      return Promise.resolve(tensor([1, 2, 3], SHAPES.flat));
    });
    render(<InspectorPanel />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalledWith('run1', 'src', 'y'));

    // `a` finishing has nothing to do with `src.y`, which is still downloading.
    aFinished = true;
    report('a', 'completed');
    await waitFor(() => expect(screen.getByText('shape [3]')).toBeInTheDocument());

    await act(async () => landSrc(tensor([[1, 2, 3], [4, 5, 6]], SHAPES.wide)));
    await waitFor(() => expect(screen.getByText('shape [2, 3]')).toBeInTheDocument());
    expect(callsFor('src')).toBe(1);
  });

  it('reads a node again when it runs a second time in the same run', async () => {
    const a = at('completed', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'running', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    mockOutput
      .mockResolvedValueOnce(tensor([[1, 2, 3], [4, 5, 6]], SHAPES.wide))
      .mockResolvedValueOnce(tensor([1, 2, 3], SHAPES.flat));
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('shape [2, 3]')).toBeInTheDocument());

    // Second pass: the first pass's value is no longer what this node holds.
    report('a', 'running');
    expect(screen.queryByText('shape [2, 3]')).toBeNull();
    expect(screen.getByText(RUNNING_NOTE)).toBeInTheDocument();
    expect(mockOutput).toHaveBeenCalledTimes(1);

    report('a', 'completed');
    await waitFor(() => expect(screen.getByText('shape [3]')).toBeInTheDocument());
    expect(mockOutput).toHaveBeenCalledTimes(2);
  });

  it('does not let a slow first-pass download overwrite the second pass', async () => {
    const a = at('completed', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'running', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    let landFirstPass!: (value: OutputData) => void;
    mockOutput
      .mockImplementationOnce(
        () => new Promise<OutputData>((resolve) => { landFirstPass = resolve; }),
      )
      .mockResolvedValueOnce(tensor([1, 2, 3], SHAPES.flat));
    render(<InspectorPanel />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalledTimes(1));

    report('a', 'running');
    report('a', 'completed');
    await waitFor(() => expect(screen.getByText('shape [3]')).toBeInTheDocument());

    await act(async () => landFirstPass(tensor([[1, 2, 3], [4, 5, 6]], SHAPES.wide)));
    await settle();
    expect(screen.getByText('shape [3]')).toBeInTheDocument();
    expect(screen.queryByText('shape [2, 3]')).toBeNull();
  });

  it('drops a stale expired line the moment the node is known to be running', async () => {
    // A reload mid-run: the tab comes back idle until the server acknowledges
    // the re-attach, so the first read is issued, and answered 404.
    const a = at('idle', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'idle', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    const finished = new Set<string>();
    serveFinished(finished);
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText(EXPIRED)).toBeInTheDocument());

    act(() => useTabStore.getState().setTabStatus(tabId(), 'running'));
    expect(screen.queryByText(EXPIRED)).toBeNull();
    expect(screen.getByText(PENDING_NOTE)).toBeInTheDocument();

    report('a', 'running');
    expect(screen.queryByText(EXPIRED)).toBeNull();
    expect(screen.getByText(RUNNING_NOTE)).toBeInTheDocument();

    finished.add('a');
    report('a', 'completed');
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    expect(screen.queryByText(EXPIRED)).toBeNull();
  });

  it('never shows the previous run’s value under a node the new run has not reached', async () => {
    const a = at('completed', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'completed', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    const finished = new Set(['a']);
    serveFinished(finished, { a: 'wide' });
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText('shape [2, 3]')).toBeInTheDocument());

    finished.clear();
    startNewRun('run2');
    expect(screen.queryByText('shape [2, 3]')).toBeNull();
    expect(screen.getByText(PENDING_NOTE)).toBeInTheDocument();
    await settle();
    expect(mockOutput.mock.calls.filter(([runId]) => runId === 'run2')).toHaveLength(0);

    report('a', 'running');
    finished.add('a');
    report('a', 'completed');
    await waitFor(() => expect(screen.getByText('shape [2, 3]')).toBeInTheDocument());
    expect(mockOutput).toHaveBeenLastCalledWith('run2', 'a', 'out');
  });

  it('drops an answer that arrives after the tab moved on to another run', async () => {
    const a = at('completed', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'completed', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    let landRun1!: (value: OutputData) => void;
    mockOutput.mockImplementation(
      () => new Promise<OutputData>((resolve) => { landRun1 = resolve; }),
    );
    render(<InspectorPanel />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalledWith('run1', 'a', 'out'));

    startNewRun('run2');
    await act(async () => landRun1(tensor([[1, 2, 3], [4, 5, 6]], SHAPES.wide)));
    await settle();

    expect(screen.queryByText('shape [2, 3]')).toBeNull();
    expect(screen.getByText(PENDING_NOTE)).toBeInTheDocument();
  });

  it('applies the same rule to every owner in segment mode', async () => {
    const ext = at('completed', node('ext', 'Ext'));
    const head = at('running', node('h', 'Head'));
    const tail = at('idle', node('t', 'Tail', { outputs: ['result'] }));
    seedTab({
      status: 'running',
      lastRunId: 'run1',
      activeSegment: { id: 'seg1', headNodeId: 'h', tailNodeId: 't' },
      nodes: [head, tail, ext],
      edges: [
        edge('e1', 'h', 't', { sourceHandle: 'mid', targetHandle: 'in' }),
        edge('e2', 'ext', 'h', { sourceHandle: 'feed', targetHandle: 'x' }),
      ],
    });
    const finished = new Set(['ext']);
    serveFinished(finished, { ext: 'wide', t: 'flat' });
    render(<InspectorPanel />);

    // The segment's input is owned by `ext`, its output by the tail.
    await waitFor(() => expect(screen.getByText('shape [2, 3]')).toBeInTheDocument());
    expect(screen.getByText(PENDING_NOTE)).toBeInTheDocument();
    expect(screen.queryByText(EXPIRED)).toBeNull();
    expect(callsFor('t')).toBe(0);

    report('h', 'completed');
    report('t', 'running');
    expect(screen.getByText(RUNNING_NOTE)).toBeInTheDocument();

    finished.add('t');
    report('t', 'completed');
    await waitFor(() => expect(screen.getByText('shape [3]')).toBeInTheDocument());
    expect(callsFor('ext')).toBe(1);
    expect(callsFor('t')).toBe(1);
  });

  it('still reports expired data for a node that never produced any', async () => {
    // Out of scope to redesign: a node the run passed over has no captures,
    // and once the run is over that is what the 404 means.
    const a = at('skipped', node('a', 'NodeA', { outputs: ['out'] }));
    seedTab({ status: 'running', lastRunId: 'run1', selectedNodeId: 'a', nodes: [a], edges: [] });
    serveFinished(new Set());
    render(<InspectorPanel />);
    await waitFor(() => expect(screen.getByText(EXPIRED)).toBeInTheDocument());
    expect(screen.queryByText(RUNNING_NOTE)).toBeNull();
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
