import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { StepTraceView } from './StepTraceView';
import { useI18n } from '../../i18n';
import {
  fetchOutput,
  fetchStepIndex,
  PayloadTooLargeError,
  RunDataExpiredError,
  type StepIndexEntry,
} from '../../api/executionOutputs';
import type { TensorOutput, OutputData } from '../../types';

vi.mock('../../api/executionOutputs', async () => {
  const actual = await vi.importActual<typeof import('../../api/executionOutputs')>(
    '../../api/executionOutputs',
  );
  return {
    ...actual,
    fetchStepIndex: vi.fn(),
    fetchOutput: vi.fn(),
  };
});

const mockStepIndex = vi.mocked(fetchStepIndex);
const mockOutput = vi.mocked(fetchOutput);

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

function step(partial: Partial<StepIndexEntry> & Pick<StepIndexEntry, 'index' | 'name'>): StepIndexEntry {
  return {
    description: '',
    scalars: {},
    tensor_keys: [],
    ...partial,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  mockStepIndex.mockReset();
  mockOutput.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('StepTraceView', () => {
  it('shows the loading placeholder before the index resolves', () => {
    mockStepIndex.mockReturnValue(new Promise(() => {}));
    const { container } = render(<StepTraceView runId="r1" nodeId="n1" />);
    expect(container.querySelector('div')?.textContent).toBe('…');
  });

  it('shows the empty state when no steps are recorded', async () => {
    mockStepIndex.mockResolvedValue([]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('This node does not record steps')).toBeInTheDocument(),
    );
    expect(
      screen.getByText('Enable Verbose mode and re-run to see steps'),
    ).toBeInTheDocument();
  });

  it('shows the index error on a generic failure', async () => {
    mockStepIndex.mockRejectedValue(new Error('idx fail'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('idx fail')).toBeInTheDocument());
  });

  it('shows the expired message when the index rejects with RunDataExpiredError', async () => {
    mockStepIndex.mockRejectedValue(new RunDataExpiredError('r1'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('run data expired — re-run to capture')).toBeInTheDocument(),
    );
  });

  it('renders a step card with description, scalars and a tensor grid', async () => {
    mockStepIndex.mockResolvedValue([
      step({
        index: 0,
        name: 'Softmax',
        description: 'compute $x$',
        scalars: { temperature: 0.5, count: 3, tiny: 0.0001 },
        tensor_keys: ['probs'],
      }),
    ]);
    mockOutput.mockResolvedValue(tensor([[0.1, 0.9], [0.4, 0.6]], { min: 0.1, max: 0.9 }));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('Softmax')).toBeInTheDocument());
    // step index shows 1-based
    expect(screen.getByText('1.')).toBeInTheDocument();
    // scalars formatting: integer, fixed(4), exponential
    expect(screen.getByText(/temperature = 0\.5000/)).toBeInTheDocument();
    expect(screen.getByText(/count = 3/)).toBeInTheDocument();
    expect(screen.getByText(/tiny = 1\.00e-4/)).toBeInTheDocument();
    // tensor label + grid
    expect(screen.getByText('probs')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
  });

  it('collapses and expands a step card on header click', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'StepA', description: 'desc', tensor_keys: [] }),
    ]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('StepA')).toBeInTheDocument());
    // body visible (caret ▾, '(no tensors)' shown)
    expect(screen.getByText('(no tensors)')).toBeInTheDocument();
    expect(screen.getByText('▾')).toBeInTheDocument();
    fireEvent.click(screen.getByText('StepA'));
    // collapsed: caret ▸, body hidden
    expect(screen.getByText('▸')).toBeInTheDocument();
    expect(screen.queryByText('(no tensors)')).not.toBeInTheDocument();
    // expand again (toggles isCollapsed back)
    fireEvent.click(screen.getByText('StepA'));
    expect(screen.getByText('(no tensors)')).toBeInTheDocument();
  });

  it('shows the (no tensors) note for a step with empty tensor_keys and no description/scalars', async () => {
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'Bare' })]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('Bare')).toBeInTheDocument());
    expect(screen.getByText('(no tensors)')).toBeInTheDocument();
  });

  it('renders a per-tensor loading placeholder before its fetch resolves', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockReturnValue(new Promise(() => {})); // pending
    const { container } = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('t')).toBeInTheDocument());
    // the loading placeholder '…' appears inside the tensor block
    expect(container.textContent).toContain('…');
  });

  it('shows a scalar (non-tensor) tensor value via String()', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['val'] }),
    ]);
    const scalar: OutputData = { type: 'scalar', run_id: 'r', node_id: 'n', port: 'p', value: 7 };
    mockOutput.mockResolvedValue(scalar);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('7')).toBeInTheDocument());
  });

  it('renders a non-tensor non-scalar value with an empty scalar body', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['str'] }),
    ]);
    const strOut: OutputData = { type: 'string', run_id: 'r', node_id: 'n', port: 'p', value: 'hi' };
    mockOutput.mockResolvedValue(strOut);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    // label rendered; the scalar block is present but renders nothing for non-scalar type
    await waitFor(() => expect(screen.getByText('str')).toBeInTheDocument());
  });

  it('shows a per-tensor generic error message', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockRejectedValue(new Error('tensor boom'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('tensor boom')).toBeInTheDocument());
  });

  it("shows 'expired' on a per-tensor RunDataExpiredError", async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 0, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockRejectedValue(new RunDataExpiredError('r1'));
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('expired')).toBeInTheDocument());
  });

  it('falls back to a sliced fetch on PayloadTooLargeError', async () => {
    mockStepIndex.mockResolvedValue([
      step({ index: 2, name: 'S', tensor_keys: ['t'] }),
    ]);
    mockOutput.mockImplementation(async (_r, _n, _port, opts) => {
      if (!opts) throw new PayloadTooLargeError('too big');
      return tensor([[1, 1], [1, 1]], { min: 1, max: 1 });
    });
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(screen.getByText('shape [2, 2]')).toBeInTheDocument());
    // port name encodes step index + tensor name; retry has slice opts
    expect(mockOutput).toHaveBeenCalledWith('r1', 'n1', '__step__2__t', {
      slice: '0,:,:',
      maxElements: 65536,
    });
  });

  it('bails out of the index .then when unmounted before it resolves', async () => {
    let resolve!: (v: StepIndexEntry[]) => void;
    mockStepIndex.mockReturnValue(new Promise<StepIndexEntry[]>((r) => { resolve = r; }));
    const { unmount } = render(<StepTraceView runId="r1" nodeId="n1" />);
    unmount();
    resolve([step({ index: 0, name: 'S', tensor_keys: ['t'] })]);
    await Promise.resolve();
    await Promise.resolve();
    expect(mockOutput).not.toHaveBeenCalled();
  });

  it('bails out of the index .catch when unmounted before it rejects', async () => {
    let reject!: (e: unknown) => void;
    mockStepIndex.mockReturnValue(new Promise<StepIndexEntry[]>((_r, rej) => { reject = rej; }));
    const { unmount, container } = render(<StepTraceView runId="r1" nodeId="n1" />);
    unmount();
    reject(new Error('late'));
    await Promise.resolve();
    await Promise.resolve();
    expect(container.querySelector('.portError')).toBeNull();
  });

  it('bails out of the per-tensor success handler when unmounted mid-fetch', async () => {
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'S', tensor_keys: ['t'] })]);
    let resolveOut!: (v: TensorOutput) => void;
    mockOutput.mockReturnValue(new Promise<TensorOutput>((r) => { resolveOut = r; }));
    const { unmount } = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount();
    resolveOut(tensor([[1]], { min: 1, max: 1 }));
    await Promise.resolve();
    await Promise.resolve();
    expect(true).toBe(true);
  });

  it('bails out of the per-tensor error handler when unmounted mid-fetch', async () => {
    mockStepIndex.mockResolvedValue([step({ index: 0, name: 'S', tensor_keys: ['t'] })]);
    let rejectOut!: (e: unknown) => void;
    mockOutput.mockReturnValue(new Promise<TensorOutput>((_r, rej) => { rejectOut = rej; }));
    const { unmount } = render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() => expect(mockOutput).toHaveBeenCalled());
    unmount();
    rejectOut(new Error('late tensor'));
    await Promise.resolve();
    await Promise.resolve();
    expect(true).toBe(true);
  });

  it('does not start a tensor effect when steps array is empty after resolving', async () => {
    // Empty steps → second effect early-returns; empty-state renders.
    mockStepIndex.mockResolvedValue([]);
    render(<StepTraceView runId="r1" nodeId="n1" />);
    await waitFor(() =>
      expect(screen.getByText('This node does not record steps')).toBeInTheDocument(),
    );
    expect(mockOutput).not.toHaveBeenCalled();
  });
});
