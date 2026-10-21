import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import {
  StudiesApiError,
  cancelStudy,
  getStudy,
  getStudyEvents,
  listStudies,
  resumeStudy,
  type StudyAttempt,
  type StudyDetail,
  type StudySummary,
  type StudyVariant,
} from '../../api/studies';
import { useI18n } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { _resetStudyStoreForTesting, useStudyStore } from '../../store/studyStore';
import { useWorkerStore } from '../../store/workerStore';
import { settle } from '../../test/utils';
import { StudiesPanel } from './StudiesPanel';

vi.mock('../../api/studies', async (importOriginal) => {
  const original = await importOriginal<typeof import('../../api/studies')>();
  return {
    StudiesApiError: original.StudiesApiError,
    listStudies: vi.fn(),
    getStudy: vi.fn(),
    getStudyEvents: vi.fn(),
    cancelStudy: vi.fn(),
    resumeStudy: vi.fn(),
  };
});

const list = vi.mocked(listStudies);
const detailOf = vi.mocked(getStudy);
const eventsOf = vi.mocked(getStudyEvents);
const cancelOf = vi.mocked(cancelStudy);
const resumeOf = vi.mocked(resumeStudy);

function summary(over: Partial<StudySummary> = {}): StudySummary {
  return {
    study_id: 's1', name: 'lr study', status: 'running', revision: 3,
    objective: { node_id: 'metrics', metric: 'mse', direction: 'minimize' },
    repetitions: 2, seed: 7, budget: { max_runs: null, max_parallel_runs: 1 },
    resume_policy: 'explicit', idempotency_key: null, error: null, halt: null,
    owner: null, created_at: '2026-10-21T00:00:00.000000Z', updated_at: 'x',
    finished_at: null, counts: { succeeded: 1, timed_out: 1, pending: 2 },
    ...over,
  };
}

function attempt(over: Partial<StudyAttempt> = {}): StudyAttempt {
  return {
    attempt_id: 'a', variant_index: 0, repetition: 0, attempt: 1, seed: 7,
    run_id: 'run-0123456789', status: 'succeeded', error: null,
    started_at: null, finished_at: null, objective_value: 1, ...over,
  };
}

function variant(index: number, label: string, currents: Array<StudyAttempt | null>,
  over: Partial<StudyVariant> = {}): StudyVariant {
  return {
    index, label,
    repetitions: currents.map((current, repetition) => ({ repetition, seed: 7 + repetition, current })),
    attempts: currents.filter((a): a is StudyAttempt => a !== null),
    aggregate: { count: 2, mean: 1, std: 0, values: [1, 1] },
    eligible: true, ineligible_reasons: [], rank: 1, tie: false, tied_with: [],
    ...over,
  };
}

function detail(over: Partial<StudyDetail> = {}): StudyDetail {
  return {
    ...summary(),
    budget_usage: {
      runs: { used: 3, limit: null },
      parallel: { active: 1, limit: 1 },
      active_seconds: { used: 1.5, limit: null },
      run_seconds: { longest_active: 0, limit: null },
      unfinished_repetitions: 2,
      exhausted: null,
    },
    attempts_submitted: 3,
    variants: [
      variant(0, 'A', [attempt(), attempt({ repetition: 1, attempt: 2, run_id: null })]),
      variant(1, 'B', [attempt({ status: 'failed', objective_value: 2.123456789 }), null],
        { eligible: false, rank: null, aggregate: { count: 1, mean: 2.123456789, std: null, values: [2.1, null] } }),
      variant(2, 'C', [attempt({ status: 'succeeded', objective_value: null }), attempt({ objective_value: 3 })],
        { eligible: false, rank: null }),
    ],
    rankings: [
      { rank: 1, variant_index: 0, label: 'A', mean: 1, std: 0, count: 2, tie: true, tied_with: [3] },
      { rank: 3, variant_index: 1, label: 'B', mean: 2.5, std: null, count: 2, tie: false, tied_with: [] },
    ],
    last_cursor: 2,
    ...over,
  };
}

const originalSelect = useRunStore.getState().select;
const originalReveal = useWorkerStore.getState().revealRuns;

beforeEach(() => {
  _resetStudyStoreForTesting();
  vi.clearAllMocks();
  useI18n.setState({ locale: 'en' });
  list.mockResolvedValue({ studies: [], total: 0 });
  detailOf.mockResolvedValue(detail());
  eventsOf.mockResolvedValue({ events: [], cursor: 0 });
});

afterEach(() => {
  // Unmount first: a reset under a mounted panel is an update outside act().
  cleanup();
  _resetStudyStoreForTesting();
  useRunStore.setState({ select: originalSelect });
  useWorkerStore.setState({ revealRuns: originalReveal });
});

async function openPanel(studyId?: string) {
  const view = render(<StudiesPanel />);
  act(() => useStudyStore.getState().openPanel(studyId));
  await settle();
  return view;
}

function dialog() {
  return screen.getByRole('dialog', { name: 'Studies' });
}

describe('StudiesPanel: the list', () => {
  it('renders nothing while closed', () => {
    const { container } = render(<StudiesPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it('says it is loading, then that there are no studies', async () => {
    let resolve!: (value: { studies: StudySummary[]; total: number }) => void;
    list.mockReturnValueOnce(new Promise((res) => { resolve = res; }));
    render(<StudiesPanel />);
    act(() => useStudyStore.getState().openPanel());
    expect(screen.getByText('Loading studies...')).toBeInTheDocument();
    await act(async () => resolve({ studies: [], total: 0 }));
    expect(screen.getByText(/No studies yet/)).toBeInTheDocument();
  });

  it('lists studies with owner, progress and status, and opens one', async () => {
    list.mockResolvedValue({
      studies: [
        summary(),
        summary({ study_id: 's2', name: null, owner: 'graph-copilot', status: 'halted', counts: {}, created_at: 'not a date' }),
        summary({ study_id: 's3', status: 'from_the_future' }),
      ],
      total: 3,
    });
    await openPanel();
    const first = screen.getByTestId('study-row-s1');
    expect(within(first).getByText('API')).toBeInTheDocument();
    expect(within(first).getByText('2 of 4')).toBeInTheDocument();
    expect(within(first).getByText('Running')).toBeInTheDocument();
    const second = screen.getByTestId('study-row-s2');
    expect(within(second).getByRole('button', { name: 'Unnamed study' })).toHaveAttribute('title', 'Open Unnamed study');
    expect(within(second).getByText('graph-copilot')).toBeInTheDocument();
    expect(within(second).getByText('Halted')).toBeInTheDocument();
    expect(within(second).getByText('0 of 0')).toBeInTheDocument();
    expect(within(second).getByText('not a date')).toBeInTheDocument();
    expect(within(screen.getByTestId('study-row-s3')).getByText('from_the_future')).toBeInTheDocument();

    fireEvent.click(within(first).getByRole('button', { name: 'lr study' }));
    await settle();
    expect(detailOf).toHaveBeenCalledWith('s1');
    expect(screen.getByRole('heading', { name: 'lr study' })).toBeInTheDocument();
  });

  it('shows a failed read', async () => {
    list.mockRejectedValue(new Error('Failed to fetch'));
    await openPanel();
    expect(within(dialog()).getByRole('alert')).toBeInTheDocument();
  });

  it('refreshes, closes from the button, the backdrop and Escape, and restores focus', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const view = await openPanel();
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Close' }));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await settle();
    expect(list.mock.calls.length).toBeGreaterThanOrEqual(2);

    // A press inside the dialog does not close it; one on the backdrop does.
    fireEvent.mouseDown(dialog());
    expect(useStudyStore.getState().panelOpen).toBe(true);
    fireEvent.mouseDown(dialog().parentElement!);
    expect(useStudyStore.getState().panelOpen).toBe(false);
    view.rerender(<StudiesPanel />);
    expect(document.activeElement).toBe(opener);

    act(() => useStudyStore.getState().openPanel());
    await settle();
    fireEvent.keyDown(dialog(), { key: 'Tab' });
    fireEvent.keyDown(dialog(), { key: 'Escape' });
    expect(useStudyStore.getState().panelOpen).toBe(false);

    act(() => useStudyStore.getState().openPanel());
    await settle();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(useStudyStore.getState().panelOpen).toBe(false);
    opener.remove();
  });
});

describe('StudiesPanel: one study', () => {
  it('shows loading, a 404 and an error before a detail exists', async () => {
    const rejects: Array<(error: unknown) => void> = [];
    detailOf.mockImplementation(() => new Promise((_res, rej) => { rejects.push(rej); }));
    await openPanel('s1');
    expect(screen.getByText('Loading study...')).toBeInTheDocument();
    await act(async () => rejects.forEach((reject) => reject(new StudiesApiError('gone', 404))));
    expect(screen.getByText('This study no longer exists.')).toBeInTheDocument();

    detailOf.mockRejectedValueOnce(new Error('Failed to fetch'));
    await act(async () => useStudyStore.getState().refresh());
    expect(within(dialog()).getByRole('alert')).toBeInTheDocument();
    // Escape from a study goes back to the list.
    fireEvent.keyDown(dialog(), { key: 'Escape' });
    await settle();
    expect(useStudyStore.getState().selectedId).toBeNull();
    expect(useStudyStore.getState().panelOpen).toBe(true);
  });

  it('shows the objective, budget, rankings with ties, repetitions and the log', async () => {
    detailOf.mockResolvedValue(detail({
      owner: 'graph-copilot',
      budget_usage: {
        runs: { used: 3, limit: 6 },
        parallel: { active: 1, limit: 2 },
        active_seconds: { used: 12.5, limit: 10 },
        run_seconds: { longest_active: 2.25, limit: 30 },
        unfinished_repetitions: 2,
        exhausted: 'max_active_seconds',
      },
    }));
    eventsOf.mockResolvedValueOnce({
      events: [
        { cursor: 1, type: 'study_created', payload: null, ts: '2026-10-21T00:00:00Z' },
        { cursor: 2, type: 'study_started', payload: null, ts: '' },
      ],
      cursor: 2,
    });
    await openPanel('s1');
    expect(screen.getByText('Objective: mse of node metrics, lower is better')).toBeInTheDocument();
    expect(screen.getByText('3 variants x 2 repetitions, seed 7')).toBeInTheDocument();
    expect(screen.getByText('Submitted by the plugin graph-copilot')).toBeInTheDocument();

    const budget = screen.getByRole('region', { name: 'Budget' });
    expect(within(budget).getByText('3 of 6')).toBeInTheDocument();
    expect(within(budget).getByText('1 of 2')).toBeInTheDocument();
    expect(within(budget).getByText(/12.5 s of 10 s/)).toBeInTheDocument();
    expect(within(budget).getByText('used up')).toBeInTheDocument();
    expect(within(budget).getByText('30 s')).toBeInTheDocument();
    expect(within(budget).getByText(/longest running 2.25 s/)).toBeInTheDocument();

    const rankings = screen.getByRole('region', { name: 'Rankings' });
    expect(within(screen.getByTestId('study-rank-0')).getByText('Tie with 3')).toBeInTheDocument();
    const cells = within(screen.getByTestId('study-rank-1')).getAllByRole('cell');
    expect(cells[cells.length - 1]).toHaveTextContent('-');
    expect(within(rankings).getByText('2.5')).toBeInTheDocument();

    const a = screen.getByTestId('study-variant-0');
    expect(within(a).getByText('attempt 2')).toBeInTheDocument();
    expect(within(a).getByText('Yes')).toBeInTheDocument();
    const b = screen.getByTestId('study-variant-1');
    expect(within(b).getAllByText('2.12346')).toHaveLength(2);   // r1 and the mean
    expect(within(b).getByText('Not ranked: r1 Failed; r2 -')).toBeInTheDocument();
    const c = screen.getByTestId('study-variant-2');
    expect(within(c).getByText('Not ranked: r1 recorded no objective')).toBeInTheDocument();

    const log = screen.getByRole('region', { name: 'Event log' });
    const items = within(log).getAllByRole('listitem');
    expect(items[0]).toHaveTextContent('study_started');
    expect(items[0]).toHaveTextContent('-');
    expect(items[1]).toHaveTextContent('study_created');
  });

  it('opens a child run in the Runs panel', async () => {
    const select = vi.fn().mockResolvedValue(undefined);
    const revealRuns = vi.fn();
    useRunStore.setState({ select });
    useWorkerStore.setState({ revealRuns });
    await openPanel('s1');
    fireEvent.click(screen.getByRole('button', { name: 'Open the run of A r1' }));
    expect(select).toHaveBeenCalledWith('run-0123456789');
    expect(revealRuns).toHaveBeenCalled();
    expect(useStudyStore.getState().panelOpen).toBe(false);
  });

  it('cancels a running study and shows a refusal', async () => {
    await openPanel('s1');
    expect(screen.queryByRole('button', { name: 'Resume' })).toBeNull();
    cancelOf.mockRejectedValueOnce(new StudiesApiError('already succeeded', 409));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel study' }));
    await settle();
    expect(screen.getByText(/The server refused: already succeeded/)).toBeInTheDocument();
  });

  it('resumes an interrupted study and says why it waits', async () => {
    detailOf.mockResolvedValue(detail({ status: 'interrupted', error: 'a child failed to submit' }));
    resumeOf.mockResolvedValue(detail());
    await openPanel('s1');
    expect(screen.getByText(/waiting for a resume/)).toBeInTheDocument();
    expect(screen.getByText('a child failed to submit')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Resume' }));
    await settle();
    expect(resumeOf).toHaveBeenCalledWith('s1', {});
  });

  it('a cancelled study offers a resume and no cancel', async () => {
    detailOf.mockResolvedValue(detail({ status: 'cancelled', rankings: [] }));
    await openPanel('s1');
    expect(screen.queryByRole('button', { name: 'Cancel study' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Resume' })).toBeInTheDocument();
    expect(screen.getByText(/No variant is ranked yet/)).toBeInTheDocument();
  });

  it('shows the read error under a study that is already shown', async () => {
    await openPanel('s1');
    detailOf.mockRejectedValueOnce(new Error('Failed to fetch'));
    await act(async () => useStudyStore.getState().refresh());
    expect(within(dialog()).getByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'lr study' })).toBeInTheDocument();
  });
});

describe('StudiesPanel: a halted study', () => {
  const halted = (halt: StudyDetail['halt'], budget: StudyDetail['budget']) => detail({
    status: 'halted', halt, budget, name: null,
    budget_usage: { ...detail().budget_usage, runs: { used: 4, limit: 4 }, exhausted: 'max_runs' },
  });

  it('names the max_runs budget and resumes with a raised one', async () => {
    detailOf.mockResolvedValue(halted(
      { budget: 'max_runs', limit: 4, used: 4, unfinished_repetitions: 2, at: 't' },
      { max_runs: 4, max_parallel_runs: 1 },
    ));
    resumeOf.mockResolvedValue(detail());
    await openPanel('s1');
    expect(screen.getByRole('heading', { name: 'Unnamed study' })).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('budget.max_runs is used up (4 of 4 runs submitted), 2 repetition(s) unfinished');
    // A halted study resumes only through the budget form.
    expect(screen.queryByRole('button', { name: 'Resume' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Cancel study' })).toBeInTheDocument();
    const runs = screen.getByLabelText('Max runs');
    const active = screen.getByLabelText('Max active seconds');
    expect(runs).toHaveValue('4');
    expect(active).toHaveValue('');
    const submit = screen.getByRole('button', { name: 'Resume with this budget' });

    fireEvent.change(runs, { target: { value: '2.5' } });
    expect(submit).toBeDisabled();
    expect(screen.getByText(/Enter a whole number/)).toBeInTheDocument();
    fireEvent.submit(submit.closest('form')!);
    expect(resumeOf).not.toHaveBeenCalled();
    fireEvent.change(runs, { target: { value: 'abc' } });
    expect(submit).toBeDisabled();
    fireEvent.change(runs, { target: { value: '8' } });
    fireEvent.change(active, { target: { value: '0' } });
    expect(submit).toBeDisabled();
    fireEvent.change(active, { target: { value: '90.5' } });
    expect(submit).toBeEnabled();
    fireEvent.click(submit);
    await settle();
    expect(resumeOf).toHaveBeenCalledWith('s1', {
      budget: { max_runs: 8, max_parallel_runs: 1, max_active_seconds: 90.5 },
    });
  });

  it('names the max_active_seconds budget and keeps its limit in the form', async () => {
    detailOf.mockResolvedValue(halted(
      { budget: 'max_active_seconds', limit: 10, used: 10.25, unfinished_repetitions: 1, at: 't' },
      { max_runs: null, max_parallel_runs: 1, max_active_seconds: 10, max_run_seconds: null },
    ));
    resumeOf.mockResolvedValue(detail());
    await openPanel('s1');
    expect(screen.getByRole('status')).toHaveTextContent('budget.max_active_seconds is used up (10.25 s of 10 s)');
    expect(screen.getByLabelText('Max active seconds')).toHaveValue('10');
    fireEvent.change(screen.getByLabelText('Max active seconds'), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Resume with this budget' }));
    await settle();
    expect(resumeOf).toHaveBeenCalledWith('s1', {
      budget: { max_runs: null, max_parallel_runs: 1, max_active_seconds: null, max_run_seconds: null },
    });
  });

  it('words a budget it does not know generically', async () => {
    detailOf.mockResolvedValue(halted(
      { budget: 'max_gpu_hours', limit: 1, used: 1, unfinished_repetitions: 1, at: 't' },
      { max_runs: null, max_parallel_runs: 1 },
    ));
    await openPanel('s1');
    expect(screen.getByRole('status')).toHaveTextContent('Halted by the budget max_gpu_hours.');
  });

  it('reads in Traditional Chinese', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    detailOf.mockResolvedValue(halted(
      { budget: 'max_runs', limit: 4, used: 4, unfinished_repetitions: 2, at: 't' },
      { max_runs: 4, max_parallel_runs: 1 },
    ));
    await openPanel('s1');
    expect(screen.getByRole('status')).toHaveTextContent('已暫停：budget.max_runs 已用完');
    expect(screen.getByRole('button', { name: '以此預算繼續' })).toBeInTheDocument();
    expect(screen.getByText('已暫停（預算）')).toBeInTheDocument();
  });
});
