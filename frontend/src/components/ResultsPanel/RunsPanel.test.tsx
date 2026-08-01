import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within, act } from '@testing-library/react';
import { RunsPanel, formatDuration, runDevice } from './RunsPanel';
import * as rest from '../../api/rest';
import type { RunStatus, RunSummary } from '../../api/rest';
import { _resetRunStoreForTesting, useRunStore } from '../../store/runStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useDialogStore } from '../../store/dialogStore';
import { useI18n } from '../../i18n';

// Stub the chart: its SVG sub-tree would swamp the assertions and its
// ResizeObserver bookkeeping is LossChart.test.tsx's business, not this
// file's. The props it receives are asserted through data attributes.
vi.mock('./LossChart', () => ({
  LossChart: ({ series, xLabel }: { series?: { name: string }[]; xLabel?: string }) => (
    <div
      data-testid="loss-chart"
      data-series={(series ?? []).map((s) => s.name).join(',')}
      data-xlabel={xLabel}
    />
  ),
}));

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listRuns: vi.fn(),
    getRun: vi.fn(),
    getRunEvents: vi.fn(),
    getRunMetrics: vi.fn(),
    getRunArtifacts: vi.fn(),
    cancelRun: vi.fn(),
    deleteRun: vi.fn(),
    downloadRunMetricsCsv: vi.fn(),
  };
});

const api = vi.mocked(rest);
const t = (k: any, vars?: any) => useI18n.getState().t(k, vars);

/** The most recent toast. `Array.prototype.at` is outside the project lib. */
function lastToast() {
  const { toasts } = useToastStore.getState();
  return toasts[toasts.length - 1];
}

function makeRun(partial: Partial<RunSummary> & { id: string }): RunSummary {
  return {
    name: null,
    status: 'succeeded' as RunStatus,
    error: null,
    options: {},
    queue_key: 'cpu',
    created_at: '2026-08-01T10:00:00.000000Z',
    started_at: '2026-08-01T10:00:01.000000Z',
    finished_at: '2026-08-01T10:00:09.000000Z',
    git_commit: null,
    git_dirty: null,
    plugin_pins: null,
    queue_position: null,
    final_metrics: {},
    active: false,
    ...partial,
  };
}

function listing(runs: RunSummary[], total = runs.length) {
  return { runs, total, limit: 50, offset: 0 };
}

/** Render the panel and wait for the first poll to land. */
async function renderPanel(runs: RunSummary[], total = runs.length) {
  api.listRuns.mockResolvedValue(listing(runs, total));
  const utils = render(<RunsPanel panelHeight={400} />);
  if (runs.length > 0) {
    await screen.findByText(runs[0].name || t('runs.unnamed'));
  } else {
    await waitFor(() => expect(api.listRuns).toHaveBeenCalled());
  }
  return utils;
}

function rowOf(runId: string): HTMLElement {
  return document.querySelector(`[data-run-id="${runId}"]`) as HTMLElement;
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  useDialogStore.setState({ active: null, resolve: null });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
  _resetRunStoreForTesting();

  api.listRuns.mockResolvedValue(listing([]));
  api.getRun.mockResolvedValue(null);
  api.getRunEvents.mockResolvedValue({
    run_id: 'r1', status: 'succeeded', active: false, events: [], cursor: 0,
  });
  api.getRunMetrics.mockResolvedValue({ run_id: 'r1', names: [], metrics: [] });
  api.getRunArtifacts.mockResolvedValue({ run_id: 'r1', artifacts: [] });
  api.cancelRun.mockResolvedValue({ run_id: 'r1', status: 'running', cancelled: true });
  api.deleteRun.mockResolvedValue({ run_id: 'r1', deleted: true });
  api.downloadRunMetricsCsv.mockResolvedValue(undefined);
});

afterEach(() => {
  _resetRunStoreForTesting();
  vi.clearAllMocks();
  vi.useRealTimers();
});

// ── list rendering ────────────────────────────────────────────────────────

describe('RunsPanel — the table', () => {
  it('renders a row per run with name, status, device, duration and final loss', async () => {
    await renderPanel([
      makeRun({
        id: 'a',
        name: 'mnist-cnn',
        status: 'succeeded',
        queue_key: 'cuda:0',
        started_at: '2026-08-01T10:00:00.000000Z',
        finished_at: '2026-08-01T10:01:30.000000Z',
        final_metrics: { train_loss: 0.0312, lr: 0.001 },
      }),
    ], 7);

    const row = rowOf('a');
    expect(within(row).getByText('mnist-cnn')).toBeInTheDocument();
    expect(within(row).getByText(t('runs.status.succeeded'))).toBeInTheDocument();
    expect(within(row).getByText('cuda:0')).toBeInTheDocument();
    expect(within(row).getByText('1m 30s')).toBeInTheDocument();
    // train_loss wins over lr for the loss column.
    expect(within(row).getByText('0.0312')).toBeInTheDocument();
    expect(screen.getByText(t('runs.showing', { shown: 1, total: 7 }))).toBeInTheDocument();
  });

  it('labels an unnamed run and dashes a run with no loss series', async () => {
    await renderPanel([makeRun({ id: 'a', started_at: null, finished_at: null })]);
    const row = rowOf('a');
    expect(within(row).getByText(t('runs.unnamed'))).toBeInTheDocument();
    // No metrics, no start time -> three dashes, none of them a zero.
    expect(within(row).getAllByText('-').length).toBeGreaterThanOrEqual(3);
  });

  it('shows a queue position for a queued run and a dash when the server declined to say', async () => {
    await renderPanel([
      makeRun({ id: 'q1', name: 'queued-one', status: 'queued', queue_position: 3, started_at: null, finished_at: null }),
      makeRun({ id: 'q2', name: 'queued-two', status: 'queued', queue_position: null, started_at: null, finished_at: null }),
    ]);
    expect(within(rowOf('q1')).getByText(/Queue #3/)).toBeInTheDocument();
    // A null position must NOT render as 0.
    expect(within(rowOf('q2')).queryByText(/Queue #/)).not.toBeInTheDocument();
    expect(within(rowOf('q2')).queryByText(/Queue #0/)).not.toBeInTheDocument();
  });

  it('falls back to the requested device when the run has not been scheduled yet', async () => {
    await renderPanel([
      makeRun({ id: 'a', status: 'queued', queue_key: null, options: { device: 'auto' } }),
    ]);
    expect(within(rowOf('a')).getByText('auto')).toBeInTheDocument();
  });

  it('shows the empty state, and a different one once a filter is on', async () => {
    await renderPanel([]);
    expect(screen.getByText(t('runs.empty'))).toBeInTheDocument();

    fireEvent.click(screen.getByText(t('runs.status.failed')));
    await waitFor(() =>
      expect(api.listRuns).toHaveBeenCalledWith(expect.objectContaining({ status: ['failed'] })),
    );
    expect(await screen.findByText(t('runs.emptyFiltered'))).toBeInTheDocument();
  });

  it('surfaces a list error banner', async () => {
    api.listRuns.mockRejectedValue(new Error('backend asleep'));
    render(<RunsPanel panelHeight={400} />);
    expect(await screen.findByText('backend asleep')).toBeInTheDocument();
  });

  it('refreshes on demand', async () => {
    await renderPanel([makeRun({ id: 'a', name: 'one' })]);
    api.listRuns.mockClear();
    fireEvent.click(screen.getByTitle(t('runs.refresh')));
    await waitFor(() => expect(api.listRuns).toHaveBeenCalledTimes(1));
  });
});

// ── status transitions ────────────────────────────────────────────────────

describe('RunsPanel — status transitions', () => {
  it('repaints a row when a poll moves it from running to succeeded', async () => {
    const { rerender } = await renderPanel([
      makeRun({ id: 'a', name: 'training', status: 'running', finished_at: null }),
    ]);
    expect(within(rowOf('a')).getByText(t('runs.status.running'))).toBeInTheDocument();
    // Stop is offered while it is going; Delete is not.
    expect(within(rowOf('a')).getByText(t('runs.action.cancel'))).toBeInTheDocument();
    expect(within(rowOf('a')).queryByText(t('runs.action.delete'))).not.toBeInTheDocument();

    api.listRuns.mockResolvedValue(listing([
      makeRun({ id: 'a', name: 'training', status: 'succeeded', final_metrics: { train_loss: 0.5 } }),
    ]));
    await act(async () => { await useRunStore.getState().refresh(); });
    rerender(<RunsPanel panelHeight={400} />);

    expect(within(rowOf('a')).getByText(t('runs.status.succeeded'))).toBeInTheDocument();
    expect(within(rowOf('a')).getByText('0.5000')).toBeInTheDocument();
    // Now it is history: deletable, and no longer stoppable.
    expect(within(rowOf('a')).getByText(t('runs.action.delete'))).toBeInTheDocument();
    expect(within(rowOf('a')).queryByText(t('runs.action.cancel'))).not.toBeInTheDocument();
  });

  it('ticks the duration of a live run without a new poll', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const started = new Date(Date.now() - 5000).toISOString();
    await renderPanel([
      makeRun({ id: 'a', name: 'live', status: 'running', started_at: started, finished_at: null }),
    ]);
    expect(within(rowOf('a')).getByText('5s')).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(within(rowOf('a')).getByText('8s')).toBeInTheDocument();
  });
});

// ── row actions ───────────────────────────────────────────────────────────

describe('RunsPanel — row actions', () => {
  it('cancels a running run and does not open the detail while doing it', async () => {
    await renderPanel([makeRun({ id: 'a', name: 'live', status: 'running', finished_at: null })]);
    fireEvent.click(within(rowOf('a')).getByText(t('runs.action.cancel')));
    await waitFor(() => expect(api.cancelRun).toHaveBeenCalledWith('a'));
    // The click was inside the row but must not have selected it.
    expect(useRunStore.getState().selectedRunId).toBeNull();
    expect(api.getRun).not.toHaveBeenCalled();
  });

  it('exports metrics as CSV', async () => {
    await renderPanel([makeRun({ id: 'a', name: 'done' })]);
    fireEvent.click(within(rowOf('a')).getByText(t('runs.action.csv')));
    await waitFor(() => expect(api.downloadRunMetricsCsv).toHaveBeenCalledWith('a'));
  });

  it('confirms before deleting, and does nothing when the user backs out', async () => {
    await renderPanel([makeRun({ id: 'a', name: 'old-run' })]);
    fireEvent.click(within(rowOf('a')).getByText(t('runs.action.delete')));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    expect(useDialogStore.getState().active).toMatchObject({
      kind: 'confirm',
      variant: 'danger',
    });

    await act(async () => { useDialogStore.getState().close(false); });
    expect(api.deleteRun).not.toHaveBeenCalled();
  });

  it('deletes once confirmed', async () => {
    await renderPanel([makeRun({ id: 'a', name: 'old-run' })]);
    fireEvent.click(within(rowOf('a')).getByText(t('runs.action.delete')));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    await act(async () => { useDialogStore.getState().close(true); });
    await waitFor(() => expect(api.deleteRun).toHaveBeenCalledWith('a'));
  });
});

// ── re-attach ─────────────────────────────────────────────────────────────

describe('RunsPanel — re-attach', () => {
  function stubSocket(connected = true) {
    const send = vi.fn();
    const connect = vi.fn().mockResolvedValue(undefined);
    const tabId = useTabStore.getState().activeTabId;
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((tab) =>
        tab.id === tabId
          ? { ...tab, ws: { connected, send, connect } as never }
          : tab,
      ),
    }));
    return { send, connect, tabId };
  }

  it('attaches the active tab to the run from cursor 0 and clears the old log', async () => {
    const { send, tabId } = stubSocket();
    useTabStore.getState().addTabLog(tabId, { message: 'from an older run', type: 'info' });
    await renderPanel([makeRun({ id: 'live-1', name: 'live', status: 'running', finished_at: null })]);

    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(send).toHaveBeenCalled());

    expect(send).toHaveBeenCalledWith({ action: 'attach', run_id: 'live-1', cursor: 0 });
    expect(useTabStore.getState().getActiveTab().logs).toHaveLength(0);
    expect(useTabStore.getState().getActiveTab().lastRunId).toBe('live-1');
    // The tab flips to `running` only when the server acknowledges the attach.
    expect(useTabStore.getState().getActiveTab().status).not.toBe('running');
  });

  it('connects first when the socket is down', async () => {
    const { send, connect } = stubSocket(false);
    await renderPanel([makeRun({ id: 'live-1', status: 'running', finished_at: null })]);
    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(connect).toHaveBeenCalled());
    expect(send).toHaveBeenCalled();
  });

  it('reports an unreachable execution server instead of attaching into the void', async () => {
    const tabId = useTabStore.getState().activeTabId;
    const send = vi.fn();
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((tab) => (tab.id === tabId
        ? { ...tab, ws: { connected: false, send, connect: vi.fn().mockRejectedValue(new Error('nope')) } as never }
        : tab)),
    }));
    await renderPanel([makeRun({ id: 'live-1', status: 'running', finished_at: null })]);
    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() =>
      expect(useToastStore.getState().toasts[0]?.message).toBe(t('runs.reattach.offline')),
    );
    expect(send).not.toHaveBeenCalled();
  });

  it('asks before stealing a tab that is already following another run', async () => {
    const { send, tabId } = stubSocket();
    useTabStore.getState().setTabStatus(tabId, 'running');
    useTabStore.getState().setLastRunId(tabId, 'other-run-id');
    await renderPanel([makeRun({ id: 'live-1', status: 'running', finished_at: null })]);

    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    await act(async () => { useDialogStore.getState().close(false); });
    expect(send).not.toHaveBeenCalled();

    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    await act(async () => { useDialogStore.getState().close(true); });
    await waitFor(() => expect(send).toHaveBeenCalled());
  });

  it('does not ask when the tab is already following THIS run', async () => {
    const { send, tabId } = stubSocket();
    useTabStore.getState().setTabStatus(tabId, 'running');
    useTabStore.getState().setLastRunId(tabId, 'live-1');
    await renderPanel([makeRun({ id: 'live-1', status: 'running', finished_at: null })]);
    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(send).toHaveBeenCalled());
    expect(useDialogStore.getState().active).toBeNull();
  });

  it('offers no live-watch button for a finished run', async () => {
    await renderPanel([makeRun({ id: 'a', status: 'succeeded' })]);
    expect(within(rowOf('a')).queryByText(t('runs.action.reattach'))).not.toBeInTheDocument();
  });

  it('tells the dock to reveal the log, but only once it actually attached', async () => {
    const { send } = stubSocket();
    const onWatchRun = vi.fn();
    api.listRuns.mockResolvedValue(listing([
      makeRun({ id: 'live-1', name: 'live', status: 'running', finished_at: null }),
    ]));
    render(<RunsPanel panelHeight={400} onWatchRun={onWatchRun} />);
    await screen.findByText('live');

    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(send).toHaveBeenCalled());
    expect(onWatchRun).toHaveBeenCalledTimes(1);
  });

  it('does not reveal the log when the user declines the switch', async () => {
    const { tabId } = stubSocket();
    const onWatchRun = vi.fn();
    useTabStore.getState().setTabStatus(tabId, 'running');
    useTabStore.getState().setLastRunId(tabId, 'other-run-id');
    api.listRuns.mockResolvedValue(listing([
      makeRun({ id: 'live-1', name: 'live', status: 'running', finished_at: null }),
    ]));
    render(<RunsPanel panelHeight={400} onWatchRun={onWatchRun} />);
    await screen.findByText('live');

    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    await act(async () => { useDialogStore.getState().close(false); });
    expect(onWatchRun).not.toHaveBeenCalled();
  });

  it('abandons the attach when the user switched canvas tabs mid-confirm', async () => {
    // clearLogs/clearExecutionStatus act on whichever tab is active NOW, so
    // finishing here would wipe the wrong tab's panel.
    const { send, tabId } = stubSocket();
    useTabStore.getState().setTabStatus(tabId, 'running');
    useTabStore.getState().setLastRunId(tabId, 'other-run-id');
    useTabStore.getState().addTabLog(tabId, { message: 'keep me', type: 'info' });
    await renderPanel([makeRun({ id: 'live-1', status: 'running', finished_at: null })]);

    fireEvent.click(within(rowOf('live-1')).getByText(t('runs.action.reattach')));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    // A second canvas tab becomes active while the confirm is open.
    act(() => { useTabStore.getState().addTab('second'); });
    await act(async () => { useDialogStore.getState().close(true); });

    expect(send).not.toHaveBeenCalled();
    const original = useTabStore.getState().tabs.find((tab) => tab.id === tabId)!;
    expect(original.logs).toHaveLength(1);
  });
});

// ── detail view ───────────────────────────────────────────────────────────

describe('RunsPanel — detail view', () => {
  const liveRun = makeRun({
    id: 'r1', name: 'resnet', status: 'running', finished_at: null,
    queue_key: 'cuda:0', options: { device: 'cuda', seed: 1234 },
  });

  beforeEach(() => {
    api.getRun.mockResolvedValue({ ...liveRun, last_cursor: 4 });
    api.getRunMetrics.mockResolvedValue({
      run_id: 'r1',
      names: ['train_loss', 'val_loss'],
      metrics: [
        { node_id: 'loop', name: 'train_loss', step: 1, value: 2 },
        { node_id: 'loop', name: 'val_loss', step: 1, value: 2.4 },
      ],
    });
    api.getRunArtifacts.mockResolvedValue({
      run_id: 'r1',
      artifacts: [{
        id: 1, kind: 'checkpoint', path: 'runs/r1/interrupt.pt',
        meta: { epoch: 3 }, created_at: '2026-08-01T10:00:00.000000Z',
      }],
    });
    // One live page, then a terminal empty one so the follower finishes
    // instead of long-polling past the end of the test.
    api.getRunEvents
      .mockResolvedValueOnce({
        run_id: 'r1', status: 'running', active: true, cursor: 5,
        events: [{
          cursor: 5, type: 'node_status', ts: '2026-08-01T10:00:05.000000Z',
          payload: { node_id: 'abcdef1234', status: 'completed' },
        }],
      })
      .mockResolvedValue({
        run_id: 'r1', status: 'succeeded', active: false, events: [], cursor: 5,
      });
  });

  it('opens on row click with multi-series chart, artifacts and a log tail', async () => {
    await renderPanel([liveRun]);
    fireEvent.click(rowOf('r1'));

    const chart = await screen.findByTestId('loss-chart');
    expect(chart.getAttribute('data-series')).toBe('train_loss,val_loss');
    expect(chart.getAttribute('data-xlabel')).toBe('step');

    const detail = within(screen.getByTestId('run-detail'));
    expect(detail.getByText('resnet')).toBeInTheDocument();
    expect(detail.getByText('runs/r1/interrupt.pt')).toBeInTheDocument();
    expect(detail.getByText('checkpoint')).toBeInTheDocument();
    expect(detail.getByText(`${t('runs.detail.seed')} 1234`)).toBeInTheDocument();
    // The scheduled device, not the requested one.
    expect(detail.getByText('cuda:0')).toBeInTheDocument();

    await waitFor(() =>
      expect(detail.getByText(t('runs.log.node', { node: 'abcdef12', status: 'completed' })))
        .toBeInTheDocument(),
    );
  });

  it('closes on a second click of the same row, and via the close button', async () => {
    await renderPanel([liveRun]);
    fireEvent.click(rowOf('r1'));
    await screen.findByTestId('loss-chart');

    fireEvent.click(rowOf('r1'));
    await waitFor(() => expect(screen.queryByTestId('loss-chart')).not.toBeInTheDocument());

    fireEvent.click(rowOf('r1'));
    await screen.findByTestId('loss-chart');
    fireEvent.click(screen.getByLabelText(t('runs.detail.close')));
    await waitFor(() => expect(screen.queryByTestId('loss-chart')).not.toBeInTheDocument());
  });

  it('shows the empty states when a run recorded nothing', async () => {
    api.getRunMetrics.mockResolvedValue({ run_id: 'r1', names: [], metrics: [] });
    api.getRunArtifacts.mockResolvedValue({ run_id: 'r1', artifacts: [] });
    // Reset first: the enclosing beforeEach queued a mockResolvedValueOnce
    // page, and a queued `once` outranks a later mockResolvedValue.
    api.getRunEvents.mockReset();
    api.getRunEvents.mockResolvedValue({
      run_id: 'r1', status: 'succeeded', active: false, events: [], cursor: 4,
    });
    await renderPanel([liveRun]);
    fireEvent.click(rowOf('r1'));
    expect(await screen.findByText(t('runs.detail.noMetrics'))).toBeInTheDocument();
    expect(screen.getByText(t('runs.detail.noArtifacts'))).toBeInTheDocument();
    expect(screen.getByText(t('runs.detail.noLog'))).toBeInTheDocument();
  });

  it('shows the error message of a failed run', async () => {
    const failed = makeRun({ id: 'r1', name: 'boom', status: 'failed', error: 'CUDA out of memory' });
    api.getRun.mockResolvedValue({ ...failed, last_cursor: 1 });
    await renderPanel([failed]);
    fireEvent.click(rowOf('r1'));
    expect(await screen.findByText(/CUDA out of memory/)).toBeInTheDocument();
  });

  it('copies an artifact path to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true, value: { writeText },
    });
    await renderPanel([liveRun]);
    fireEvent.click(rowOf('r1'));
    const copy = await screen.findByTitle(t('runs.detail.copyPath'));
    fireEvent.click(copy);
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('runs/r1/interrupt.pt'));
    expect(lastToast()).toMatchObject({ type: 'success' });
  });

  it('reports a clipboard the browser refused', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn().mockRejectedValue(new Error('denied')) },
    });
    await renderPanel([liveRun]);
    fireEvent.click(rowOf('r1'));
    fireEvent.click(await screen.findByTitle(t('runs.detail.copyPath')));
    await waitFor(() =>
      expect(lastToast()).toMatchObject({ type: 'error' }),
    );
  });

  it('renders a live-updating curve as metric events arrive', async () => {
    await renderPanel([liveRun]);
    fireEvent.click(rowOf('r1'));
    await screen.findByTestId('loss-chart');

    await act(async () => {
      useRunStore.setState((state) => ({
        detail: {
          ...state.detail!,
          series: { ...state.detail!.series, lr: { 1: 0.01 } },
        },
      }));
    });
    expect(screen.getByTestId('loss-chart').getAttribute('data-series'))
      .toBe('lr,train_loss,val_loss');
  });
});

// ── i18n ──────────────────────────────────────────────────────────────────

describe('RunsPanel — i18n', () => {
  it('renders the zh-TW strings when that locale is active', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    await renderPanel([makeRun({ id: 'a', name: 'run-a', status: 'running', finished_at: null })]);
    expect(within(rowOf('a')).getByText('執行中')).toBeInTheDocument();
    expect(within(rowOf('a')).getByText('停止')).toBeInTheDocument();
  });
});

// ── pure helpers ──────────────────────────────────────────────────────────

describe('RunsPanel helpers', () => {
  it('formats durations in seconds, minutes and hours', () => {
    const start = '2026-08-01T10:00:00.000000Z';
    expect(formatDuration(null, null)).toBe('-');
    expect(formatDuration(start, '2026-08-01T10:00:09.000000Z')).toBe('9s');
    expect(formatDuration(start, '2026-08-01T10:01:04.000000Z')).toBe('1m 04s');
    expect(formatDuration(start, '2026-08-01T12:05:00.000000Z')).toBe('2h 05m');
    expect(formatDuration('not-a-date', null)).toBe('-');
  });

  it('measures a still-running run against now', () => {
    const now = Date.parse('2026-08-01T10:00:30.000Z');
    expect(formatDuration('2026-08-01T10:00:00.000000Z', null, now)).toBe('30s');
  });

  it('prefers the scheduled device over the requested one', () => {
    expect(runDevice(makeRun({ id: 'a', queue_key: 'cuda:0', options: { device: 'auto' } })))
      .toBe('cuda:0');
    expect(runDevice(makeRun({ id: 'a', queue_key: null, options: { device: 'cpu' } })))
      .toBe('cpu');
    expect(runDevice(makeRun({ id: 'a', queue_key: null, options: {} }))).toBe('-');
  });
});
