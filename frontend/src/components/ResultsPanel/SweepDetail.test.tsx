import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import * as rest from '../../api/rest';
import type { SweepDetail as SweepDetailData, SweepState, SweepVariant } from '../../api/rest';
import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { _resetSweepStoreForTesting, useSweepStore } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import type { NodeDefinition } from '../../types';
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
// Tests install vi.fn() actions with setState, which the store reset does not
// undo; the real ones go back before every test.
const realActions = (({ cancelSweep }) => ({ cancelSweep }))(useSweepStore.getState());

const trainDefinition: NodeDefinition = {
  node_name: 'Train', category: 'Test', description: '', inputs: [], outputs: [],
  params: [
    { name: 'rate', param_type: 'float', default: 0.1, description: '', options: [], min_value: 0, max_value: 1 },
  ],
};

function variant(index: number, status: SweepVariant['status'], objective: number | null, rank: number | null): SweepVariant {
  return {
    index, domain_index: index, run_id: status === 'missing' ? null : `r${index}`, status,
    params: [{ node_id: 'train', param: 'rate', value: (index + 1) / 10 }],
    seed: null, objective, rank, run_exists: status !== 'missing',
    final_metrics: status === 'missing' ? {} : { val_loss: objective ?? 0, accuracy: 0.8 },
  };
}

function sweep(state: SweepState = 'running'): SweepDetailData {
  return {
    sweep_id: 's1', name: 'Learning-rate search', state, method: 'grid',
    seed: null, seed_variants: false,
    objective: { metric: 'val_loss', direction: 'minimize' },
    created_at: '2026-10-01T00:00:00Z', finished_at: null,
    error: 'one child failed', objective_warning: 'metric absent from one run',
    counts: { queued: 2, running: 1, succeeded: 1, failed: 1, cancelled: 0, interrupted: 0, missing: 1 },
    params: [{ node_id: 'train', param: 'rate', domain: [0.1, 0.2, 0.3] }],
    variants: [variant(0, 'succeeded', 0.4, 1), variant(1, 'missing', null, null), variant(2, 'running', 0.8, 2)],
    best: { index: 0, run_id: 'r0', objective: 0.4 },
  };
}

function view() {
  return render(<SweepDetail onBack={vi.fn()} onOpenRun={vi.fn()} chartHeight={160} />);
}

/** The variant indexes in the order the table shows them. */
function rowOrder(): string[] {
  return screen.getAllByTestId(/sweep-variant-/).map((row) => row.getAttribute('data-testid')!.replace('sweep-variant-', ''));
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetSweepStoreForTesting();
  useSweepStore.setState(realActions);
  useNodeDefStore.setState({ definitions: [trainDefinition] });
  useTabStore.setState({ tabs: [], activeTabId: '' });
  useTabStore.getState().addTab('test');
  useTabStore.getState().setNodes([{
    id: 'train', type: 'customNode', position: { x: 0, y: 0 },
    data: { type: 'Train', label: 'Trainer', params: { rate: 0.1 } },
  }] as never);
  api.downloadSweepCsv.mockResolvedValue(undefined);
});

afterEach(() => {
  act(() => {
    _resetSweepStoreForTesting();
  });
  vi.clearAllMocks();
});

describe('SweepDetail', () => {
  it('shows state, objective, every count there is, and persistent errors', () => {
    const detail = sweep();
    detail.counts = { ...detail.counts, cancelled: 2, interrupted: 1 };
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    const page = screen.getByTestId('sweep-detail');

    expect(within(page).getByText('Learning-rate search')).toBeInTheDocument();
    expect(within(page).getByText(/val_loss.*minimize/i)).toBeInTheDocument();
    for (const count of ['3 active or queued', '1 succeeded', '1 failed', '2 cancelled', '1 interrupted', '1 missing']) {
      expect(within(page).getByText(count)).toBeInTheDocument();
    }
    expect(within(page).getByText('one child failed')).toBeInTheDocument();
  });

  it('says nothing about the objective while every variant is still running or queued', () => {
    // The server warns whenever nothing is ranked yet, which for real
    // training is every sweep until its first variant ends, hours later.
    const detail = sweep();
    detail.counts = { queued: 2, running: 1, succeeded: 0, failed: 0, cancelled: 0, interrupted: 0, missing: 0 };
    detail.variants = [variant(0, 'running', null, null), variant(1, 'queued', null, null), variant(2, 'queued', null, null)];
    detail.best = null;
    detail.objective_warning = "no variant recorded a metric named 'val_loss'";
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    expect(screen.queryByText(/recorded a metric named/i)).toBeNull();
  });

  it('warns in the active language once a variant finished without the objective', () => {
    const detail = sweep();
    detail.counts = { queued: 0, running: 1, succeeded: 1, failed: 0, cancelled: 0, interrupted: 0, missing: 0 };
    detail.variants = [
      { ...variant(0, 'succeeded', null, null), final_metrics: { train_loss: 0.5, lr: 0.1 } },
      { ...variant(1, 'running', null, null), final_metrics: { train_loss: 0.7 } },
    ];
    detail.best = null;
    detail.objective_warning = "no variant recorded a metric named 'val_loss'";
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    const english = view();
    expect(screen.getByText('No variant recorded a metric named "val_loss"; the runs recorded lr, train_loss.')).toBeInTheDocument();
    expect(screen.queryByText(detail.objective_warning)).toBeNull();
    english.unmount();

    useI18n.setState({ locale: 'zh-TW' });
    view();
    expect(screen.getByText('沒有任何變體記錄名為「val_loss」的指標；各執行記錄的是 lr, train_loss。')).toBeInTheDocument();
  });

  it('does not warn while a running variant has already recorded the objective', () => {
    // Ranks come only once a variant ends, but a live run that logs the
    // objective proves the name right.
    const detail = sweep();
    detail.counts = { queued: 0, running: 1, succeeded: 1, failed: 0, cancelled: 0, interrupted: 0, missing: 0 };
    detail.variants = [
      { ...variant(0, 'succeeded', null, null), final_metrics: { train_loss: 0.5 } },
      { ...variant(1, 'running', null, null), final_metrics: { val_loss: 0.9 } },
    ];
    detail.best = null;
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    expect(screen.queryByText(/recorded a metric named/i)).toBeNull();
  });

  it('warns when the sweep ended with nothing ranked', () => {
    const detail = sweep('finished');
    detail.counts = { queued: 0, running: 0, succeeded: 0, failed: 2, cancelled: 0, interrupted: 0, missing: 0 };
    detail.variants = [
      { ...variant(0, 'failed', null, null), final_metrics: {} },
      { ...variant(1, 'failed', null, null), final_metrics: {} },
    ];
    detail.best = null;
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    expect(screen.getByText('No variant recorded a metric named "val_loss".')).toBeInTheDocument();
  });

  it('does not warn once a variant recorded the objective', () => {
    // A ranked variant recorded it; whatever the server says, nothing is wrong.
    const detail = sweep('finished');
    detail.objective_warning = "no variant recorded a metric named 'val_loss'";
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    expect(screen.queryByText(/recorded a metric named/i)).toBeNull();
  });

  it('names the objective node and explains an ambiguous name in each language (#641)', () => {
    const detail = sweep('finished');
    detail.counts = { queued: 0, running: 0, succeeded: 2, failed: 0, cancelled: 0, interrupted: 0, missing: 0 };
    detail.variants = [
      { ...variant(0, 'succeeded', 0.4, 1), objective_node_id: 'train' },
      { ...variant(1, 'succeeded', null, null), ambiguous_producers: [null, 'train', 'block/inner'] },
    ];
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    const english = view();
    expect(screen.getByText('"val_loss" was logged by more than one node (run level, block/inner, Trainer (train)), so those variants are not ranked. Start a new sweep with an objective node chosen.')).toBeInTheDocument();
    const cell = within(screen.getByTestId('sweep-variant-1')).getByText('Ambiguous');
    expect(cell).toHaveAttribute('title', 'run level, Trainer (train), block/inner');
    // The ambiguity is the explanation; the "never recorded" warning is not.
    expect(screen.queryByText(/recorded a metric named/i)).toBeNull();
    english.unmount();

    useI18n.setState({ locale: 'zh-TW' });
    detail.objective = { metric: 'val_loss', direction: 'minimize', node_id: 'train' };
    view();
    expect(screen.getByText(/val_loss（來自 Trainer \(train\)）/)).toBeInTheDocument();
    expect(screen.getByText(/由多個節點記錄（執行層級, block\/inner, Trainer \(train\)）/)).toBeInTheDocument();
  });

  it('warns by node when the chosen node never logged the objective', () => {
    // The summary says another node logged val_loss; only a harvested value
    // would show that the chosen node did.
    const detail = sweep('finished');
    detail.objective = { metric: 'val_loss', direction: 'minimize', node_id: 'block/inner' };
    detail.counts = { queued: 0, running: 0, succeeded: 1, failed: 0, cancelled: 0, interrupted: 0, missing: 0 };
    detail.variants = [{ ...variant(0, 'succeeded', null, null), final_metrics: { val_loss: 0.3 } }];
    detail.best = null;
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    expect(screen.getByText(/val_loss from block\/inner · Minimize/)).toBeInTheDocument();
    expect(screen.getByText('No variant recorded "val_loss" from block/inner.')).toBeInTheDocument();
  });

  it('does not warn by node once the chosen node recorded the objective', () => {
    const detail = sweep('running');
    detail.objective = { metric: 'val_loss', direction: 'minimize', node_id: 'train' };
    detail.counts = { queued: 0, running: 1, succeeded: 1, failed: 0, cancelled: 0, interrupted: 0, missing: 0 };
    detail.variants = [{ ...variant(0, 'succeeded', 0.2, null), objective_node_id: 'train' }];
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    expect(screen.queryByText(/No variant recorded/)).toBeNull();
  });

  it('leaves out a status no variant has', () => {
    const detail = sweep('finished');
    detail.counts = { queued: 0, running: 0, succeeded: 2, failed: 0, cancelled: 1, interrupted: 0, missing: 0 };
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    const summary = screen.getByText('2 succeeded').parentElement!;
    expect(summary).toHaveTextContent('1 cancelled');
    expect(summary).not.toHaveTextContent(/active or queued|failed|interrupted|missing/);
  });

  it('says once that the sweep is stopping, with what the stop asked for', () => {
    useSweepStore.setState({ detail: sweep('cancelling'), selectedSweepId: 's1', cancelledRequested: 2 });
    view();
    // The state chip says "Stopping"; the banner says what was asked.
    expect(screen.getAllByText(/stopping/i)).toHaveLength(1);
    expect(screen.getByText('Asked 2 runs to stop.')).toBeInTheDocument();
    expect(screen.getByText('A running node may finish its current step first.')).toBeInTheDocument();
    expect(screen.getByText('3 active or queued')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Stop sweep$/i })).toBeNull();
  });

  it('counts a single stopped run in the singular', () => {
    useSweepStore.setState({ detail: sweep('cancelling'), selectedSweepId: 's1', cancelledRequested: 1 });
    view();
    expect(screen.getByText('Asked 1 run to stop.')).toBeInTheDocument();
  });

  it('does not invent a zero acknowledgement for a stop it did not send', () => {
    useSweepStore.setState({
      detail: sweep('cancelling'), selectedSweepId: 's1', cancelledRequested: null,
    });
    const english = view();
    expect(screen.getByText('A stop was already requested.')).toBeInTheDocument();
    expect(screen.queryByText(/asked \d+ runs? to stop/i)).toBeNull();
    english.unmount();

    // A second request that found nothing left to stop is the same case.
    act(() => {
      useSweepStore.setState({ cancelledRequested: 0 });
      useI18n.setState({ locale: 'zh-TW' });
    });
    view();
    expect(screen.getByText('先前已要求停止。')).toBeInTheDocument();
  });

  it('keeps a failed sweep cancellable while children remain active', () => {
    const cancelSweep = vi.fn().mockResolvedValue(undefined);
    useSweepStore.setState({ detail: sweep('failed'), selectedSweepId: 's1', cancelSweep });
    view();

    expect(screen.getByText('Failed')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /^Stop sweep$/i }));
    expect(cancelSweep).toHaveBeenCalledTimes(1);
  });

  it('holds Stop while a stop is in flight and keeps the focus in the view', () => {
    const cancelSweep = vi.fn(async () => {
      useSweepStore.setState({ cancelPending: true });
    });
    useSweepStore.setState({ detail: sweep(), selectedSweepId: 's1', cancelSweep });
    view();
    const stop = screen.getByRole('button', { name: /^Stop sweep$/i });
    stop.focus();
    fireEvent.click(stop);

    expect(screen.getByRole('button', { name: /^Stop sweep$/i })).toBeDisabled();
    // The button is about to go: focus moves to the sweep's own heading.
    expect(document.activeElement).toBe(screen.getByRole('heading', { name: 'Learning-rate search' }));
  });

  it('keeps a failed stop on screen', () => {
    useSweepStore.setState({ detail: sweep(), selectedSweepId: 's1', cancelError: 'session token rejected' });
    view();
    expect(screen.getByText('Could not stop the sweep: session token rejected')).toBeInTheDocument();
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

  it('heads a swept param by node label and param, with the full address as its title', () => {
    // Palette ids are UUIDs: as a header the id alone named nothing, and two
    // params of one node read the same once the column cut it short.
    const detail = sweep('finished');
    detail.params = [
      ...detail.params,
      { node_id: '3f1c9ab2-c04e-4d5f-8b1a-7e6d2c930f45', param: 'lr', domain: [0.01] },
    ];
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();

    const named = screen.getByRole('columnheader', { name: 'Trainer · rate' });
    expect(named).toHaveAttribute('title', 'train.rate');
    // A node the open graph does not have keeps its id, shortened.
    const unknown = screen.getByRole('columnheader', { name: '3f1c9ab2 · lr' });
    expect(unknown).toHaveAttribute('title', '3f1c9ab2-c04e-4d5f-8b1a-7e6d2c930f45.lr');
  });

  it('names columns from the graph the sweep was created from, not the active one', () => {
    // Two graphs can share a short node id; a label from the wrong one names
    // the wrong parameter.
    const origin = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('other');
    useTabStore.getState().setNodes([{
      id: 'train', type: 'customNode', position: { x: 0, y: 0 },
      data: { type: 'Train', label: 'Loader', params: { rate: 0.1 } },
    }] as never);
    expect(useTabStore.getState().activeTabId).not.toBe(origin);
    useSweepStore.setState({ detail: sweep('finished'), selectedSweepId: 's1', origins: { s1: origin } });
    view();
    expect(screen.getByRole('columnheader', { name: 'Trainer · rate' })).toBeInTheDocument();

    // Once that tab is closed the id stays, rather than another graph's label.
    act(() => {
      useSweepStore.setState({ origins: { s1: 'closed-tab' } });
    });
    expect(screen.getByRole('columnheader', { name: 'train · rate' })).toHaveAttribute('title', 'train.rate');
  });

  it('words each status in the active language and sorts statuses in a fixed order', () => {
    useI18n.setState({ locale: 'zh-TW' });
    useSweepStore.setState({ detail: sweep('finished'), selectedSweepId: 's1' });
    view();
    const statusOf = (index: string) => within(screen.getByTestId(`sweep-variant-${index}`)).getAllByRole('cell')[2].textContent;
    expect([statusOf('0'), statusOf('1'), statusOf('2')]).toEqual(['已成功', '已遺失', '執行中']);
    // The summary counts in the same words as the cells.
    expect(screen.getByText('1 個已成功')).toBeInTheDocument();
    expect(screen.getByText('1 個已遺失')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '狀態' }));
    // Live work first, then outcomes; not the alphabetical order of English tokens.
    expect(rowOrder()).toEqual(['2', '0', '1']);
  });

  it('keeps rows without a value last whichever way a column sorts', () => {
    useSweepStore.setState({ detail: sweep('finished'), selectedSweepId: 's1' });
    view();
    const objective = screen.getByRole('button', { name: /^Objective/ });
    fireEvent.click(objective);
    expect(rowOrder()).toEqual(['0', '2', '1']);
    fireEvent.click(objective);
    expect(rowOrder()).toEqual(['2', '0', '1']);
  });

  it('keeps the server order between tied rows', () => {
    // The server sends variants best first, so between two equal objectives
    // its order is the ranking. Here that is NOT index order, so a tie broken
    // by index (or by an unstable sort) would show 0 before 2.
    const detail = sweep('finished');
    detail.variants = [variant(2, 'succeeded', 0.4, 1), variant(0, 'succeeded', 0.4, 2), variant(1, 'missing', null, null)];
    useSweepStore.setState({ detail, selectedSweepId: 's1' });
    view();
    fireEvent.keyDown(screen.getByRole('button', { name: /^Objective/ }), { key: 'Enter' });
    expect(rowOrder()).toEqual(['2', '0', '1']);
  });

  it('labels each curve in the active language', () => {
    useSweepStore.setState({
      detail: sweep(), selectedSweepId: 's1',
      curves: [{ runId: 'r0', variantIndex: 0, points: [{ x: 1, y: 0.4 }] }],
    });
    const english = view();
    expect(screen.getByTestId('sweep-chart')).toHaveAttribute('data-series', 'Variant 1');
    english.unmount();

    useI18n.setState({ locale: 'zh-TW' });
    view();
    expect(screen.getByTestId('sweep-chart')).toHaveAttribute('data-series', '變體 1');
  });

  it('says when the server no longer has the sweep', () => {
    useSweepStore.setState({ selectedSweepId: 'gone', detail: null, notFound: true, error: "sweep 'gone' not found" });
    view();
    expect(screen.getByText('This sweep no longer exists on the server.')).toBeInTheDocument();
    expect(screen.queryByText(/Loading sweep/)).toBeNull();
  });
});
