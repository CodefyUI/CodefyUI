import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import * as rest from '../../api/rest';
import type { SweepDetail as SweepDetailData, SweepState } from '../../api/rest';
import { useI18n } from '../../i18n';
import { _resetSweepStoreForTesting, useSweepStore } from '../../store/sweepStore';
import { SweepDetail } from './SweepDetail';

vi.mock('./LossChart', () => ({
  LossChart: ({ series }: { series?: { name: string }[] }) => (
    <div data-testid="sweep-chart" data-series={(series ?? []).map((entry) => entry.name).join(',')} />
  ),
}));

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return { ...actual, downloadSweepCsv: vi.fn() };
});

const api = vi.mocked(rest);

function sweep(state: SweepState = 'running'): SweepDetailData {
  return {
    sweep_id: 's1', name: 'Learning-rate search', state, method: 'grid',
    seed: null, seed_variants: false,
    objective: { metric: 'val_loss', direction: 'minimize' },
    created_at: '2026-10-01T00:00:00Z', finished_at: null,
    error: 'one child failed', objective_warning: 'metric absent from one run',
    counts: { queued: 2, running: 1, succeeded: 1, failed: 1, cancelled: 0, interrupted: 0, missing: 1 },
    params: [{ node_id: 'train', param: 'rate', domain: [0.1, 0.2, 0.3] }],
    variants: [
      { index: 0, domain_index: 0, run_id: 'r0', status: 'succeeded', params: [{ node_id: 'train', param: 'rate', value: 0.1 }], seed: null, objective: 0.4, rank: 1, run_exists: true, final_metrics: { val_loss: 0.4, accuracy: 0.8 } },
      { index: 1, domain_index: 1, run_id: null, status: 'missing', params: [{ node_id: 'train', param: 'rate', value: 0.2 }], seed: null, objective: null, rank: null, run_exists: false, final_metrics: {} },
      { index: 2, domain_index: 2, run_id: 'r2', status: 'running', params: [{ node_id: 'train', param: 'rate', value: 0.3 }], seed: null, objective: 0.8, rank: 2, run_exists: true, final_metrics: { val_loss: 0.8 } },
    ],
    best: { index: 0, run_id: 'r0', objective: 0.4 },
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetSweepStoreForTesting();
  useSweepStore.setState({ loadCurves: async () => {} });
  api.downloadSweepCsv.mockResolvedValue(undefined);
});

afterEach(() => {
  act(() => {
    useSweepStore.setState({ loadCurves: async () => {} });
    _resetSweepStoreForTesting();
  });
  vi.clearAllMocks();
});

describe('SweepDetail', () => {
  it('shows state, objective, counts, persistent errors and truthful running work', () => {
    useSweepStore.setState({ detail: sweep(), selectedSweepId: 's1' });
    render(<SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />);
    const view = screen.getByTestId('sweep-detail');

    expect(within(view).getByText('Learning-rate search')).toBeInTheDocument();
    expect(within(view).getByText(/val_loss.*minimize/i)).toBeInTheDocument();
    expect(within(view).getByText(/3 active or queued/i)).toBeInTheDocument();
    expect(within(view).getByText('one child failed')).toBeInTheDocument();
    expect(within(view).getByText('metric absent from one run')).toBeInTheDocument();
  });

  it('states what cancellation requested and keeps running plus queued visible', () => {
    useSweepStore.setState({ detail: sweep('cancelling'), selectedSweepId: 's1', cancelledRequested: 2 });
    render(<SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />);
    expect(screen.getByText(/Stopping.*asked 2 runs to stop/i)).toBeInTheDocument();
    expect(screen.getByText(/3 active or queued/i)).toBeInTheDocument();
    expect(screen.getByText(/Stopping is cooperative/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Stop sweep$/i })).toBeNull();
  });

  it('does not invent a zero acknowledgement for a pre-existing cancelling sweep', () => {
    useSweepStore.setState({
      detail: sweep('cancelling'), selectedSweepId: 's1', cancelledRequested: null,
    });
    const english = render(
      <SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />,
    );

    expect(screen.getByText(/stop request was made before this view opened/i)).toBeInTheDocument();
    expect(screen.queryByText(/asked 0 runs to stop/i)).toBeNull();
    english.unmount();

    useI18n.setState({ locale: 'zh-TW' });
    const traditionalChinese = render(
      <SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />,
    );
    expect(screen.getByText(/此檢視開啟前已提出停止要求/)).toBeInTheDocument();
    traditionalChinese.unmount();
  });

  it('keeps a failed sweep cancellable while children remain active', () => {
    const cancelSweep = vi.fn().mockResolvedValue(undefined);
    useSweepStore.setState({
      detail: sweep('failed'), selectedSweepId: 's1', cancelSweep,
    });
    render(<SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />);

    expect(screen.getByText('Failed')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /^Stop sweep$/i }));
    expect(cancelSweep).toHaveBeenCalledTimes(1);
  });

  it('calls Back, cancel and the server CSV action', async () => {
    const onBack = vi.fn();
    const cancelSweep = vi.fn().mockResolvedValue(undefined);
    useSweepStore.setState({ detail: sweep(), selectedSweepId: 's1', cancelSweep });
    render(<SweepDetail onBack={onBack} onOpenRun={vi.fn()} chartHeight={160} />);

    fireEvent.click(screen.getByRole('button', { name: /^Back$/i }));
    expect(onBack).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: /^Stop sweep$/i }));
    expect(cancelSweep).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: /Download CSV/i }));
    await waitFor(() => expect(api.downloadSweepCsv).toHaveBeenCalledWith('s1'));
  });

  it('renders best, params and metrics; dead runs are disabled and live runs open', () => {
    const onOpenRun = vi.fn();
    useSweepStore.setState({ detail: sweep('finished'), selectedSweepId: 's1' });
    render(<SweepDetail onBack={vi.fn()} onOpenRun={onOpenRun} chartHeight={160} />);

    const best = screen.getByTestId('sweep-variant-0');
    expect(best).toHaveTextContent('Best');
    expect(best).toHaveTextContent('0.1');
    expect(best).toHaveTextContent('0.4');
    const missing = screen.getByTestId('sweep-variant-1');
    expect(within(missing).getByRole('button', { name: /Unavailable/i })).toBeDisabled();
    fireEvent.click(within(best).getByRole('button', { name: /Open run/i }));
    expect(onOpenRun).toHaveBeenCalledWith('r0');
  });

  it('sorts from keyboard controls stably and preserves server order on ties', () => {
    const detail = sweep('finished');
    detail.variants[2].objective = 0.4;
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    render(<SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />);
    const header = screen.getByRole('button', { name: /Objective/i });
    fireEvent.keyDown(header, { key: 'Enter' });
    const rows = screen.getAllByTestId(/sweep-variant-/);
    expect(rows[0]).toHaveAttribute('data-testid', 'sweep-variant-0');
    expect(rows[1]).toHaveAttribute('data-testid', 'sweep-variant-2');
  });

  it('loads and overlays only real objective curves', async () => {
    const loadCurves = vi.fn(async () => {
      useSweepStore.setState({ curves: [{ runId: 'r0', variantIndex: 0, name: 'Variant 1', points: [{ x: 1, y: 0.4 }] }] });
    });
    useSweepStore.setState({ detail: sweep(), selectedSweepId: 's1', loadCurves });
    render(<SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />);
    await waitFor(() => expect(loadCurves).toHaveBeenCalledTimes(1));
    expect(await screen.findByTestId('sweep-chart')).toHaveAttribute('data-series', 'Variant 1');
  });
});
