import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  StudiesApiError,
  cancelStudy,
  getStudy,
  getStudyEvents,
  listStudies,
  resumeStudy,
  type StudyDetail,
} from '../api/studies';
import {
  STUDY_EVENTS_KEPT,
  STUDY_POLL_MS,
  _resetStudyStoreForTesting,
  useStudyStore,
} from './studyStore';

vi.mock('../api/studies', async (importOriginal) => {
  const original = await importOriginal<typeof import('../api/studies')>();
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

function detail(over: Partial<StudyDetail> = {}): StudyDetail {
  return { study_id: 's1', status: 'running', revision: 3, ...over } as StudyDetail;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  _resetStudyStoreForTesting();
  vi.clearAllMocks();
  list.mockResolvedValue({ studies: [{ study_id: 's1' } as never], total: 1 });
  detailOf.mockResolvedValue(detail());
  eventsOf.mockResolvedValue({ events: [], cursor: 0 });
});

afterEach(() => {
  vi.useRealTimers();
  _resetStudyStoreForTesting();
});

describe('studyStore: the list', () => {
  it('opens the panel and reads the list on refresh', async () => {
    useStudyStore.getState().openPanel();
    expect(useStudyStore.getState().panelOpen).toBe(true);
    expect(detailOf).not.toHaveBeenCalled();
    await useStudyStore.getState().refresh();
    const state = useStudyStore.getState();
    expect(state.studies).toHaveLength(1);
    expect(state.total).toBe(1);
    expect(state.loaded).toBe(true);
    expect(state.listError).toBeNull();
  });

  it('keeps the error of a failed read', async () => {
    list.mockRejectedValueOnce(new Error('offline'));
    await useStudyStore.getState().refresh();
    expect(useStudyStore.getState().listError).toBe('offline');
    expect(useStudyStore.getState().loaded).toBe(true);
    list.mockRejectedValueOnce('plain');
    await useStudyStore.getState().refresh();
    expect(useStudyStore.getState().listError).toBe('plain');
  });

  it('drops a list reply that lands after a study was opened', async () => {
    const pending = deferred<{ studies: never[]; total: number }>();
    list.mockReturnValueOnce(pending.promise);
    const read = useStudyStore.getState().refresh();
    await useStudyStore.getState().openStudy('s1');
    pending.resolve({ studies: [], total: 0 });
    await read;
    expect(useStudyStore.getState().loaded).toBe(false);
  });

  it('drops a list refusal that lands after the panel closed', async () => {
    const pending = deferred<{ studies: never[]; total: number }>();
    list.mockReturnValueOnce(pending.promise);
    const read = useStudyStore.getState().refresh();
    useStudyStore.getState().closePanel();
    pending.reject(new Error('late'));
    await read;
    expect(useStudyStore.getState().listError).toBeNull();
  });
});

describe('studyStore: one study', () => {
  it('opens a study from openPanel and appends its events', async () => {
    eventsOf.mockResolvedValueOnce({
      events: [{ cursor: 1, type: 'study_created', payload: null, ts: 't' }],
      cursor: 1,
    });
    useStudyStore.getState().openPanel('s1');
    await vi.waitFor(() => expect(useStudyStore.getState().events).toHaveLength(1));
    expect(useStudyStore.getState().detail?.study_id).toBe('s1');
    expect(useStudyStore.getState().cursor).toBe(1);
    expect(eventsOf).toHaveBeenLastCalledWith('s1', 0);
    // The next read asks after the cursor, and an empty page changes nothing.
    await useStudyStore.getState().refresh();
    expect(eventsOf).toHaveBeenLastCalledWith('s1', 1);
    expect(useStudyStore.getState().events).toHaveLength(1);
  });

  it('keeps the newest events only', async () => {
    const many = Array.from({ length: STUDY_EVENTS_KEPT + 5 }, (_, index) => ({
      cursor: index + 1, type: 'x', payload: null, ts: 't',
    }));
    eventsOf.mockResolvedValueOnce({ events: many, cursor: many.length });
    await useStudyStore.getState().openStudy('s1');
    const { events } = useStudyStore.getState();
    expect(events).toHaveLength(STUDY_EVENTS_KEPT);
    expect(events[0].cursor).toBe(6);
  });

  it('marks a 404 as not found and keeps any other error', async () => {
    detailOf.mockRejectedValueOnce(new StudiesApiError('gone', 404));
    await useStudyStore.getState().openStudy('s1');
    expect(useStudyStore.getState().notFound).toBe(true);
    expect(useStudyStore.getState().detailError).toBe('gone');
    detailOf.mockRejectedValueOnce(new Error('boom'));
    await useStudyStore.getState().refresh();
    expect(useStudyStore.getState().notFound).toBe(false);
    expect(useStudyStore.getState().detailError).toBe('boom');
  });

  it('drops replies for a study that is no longer open', async () => {
    const first = deferred<StudyDetail>();
    detailOf.mockReturnValueOnce(first.promise);
    const opening = useStudyStore.getState().openStudy('s1');
    useStudyStore.getState().closeStudy();
    first.resolve(detail());
    await opening;
    expect(useStudyStore.getState().detail).toBeNull();

    const page = deferred<{ events: never[]; cursor: number }>();
    eventsOf.mockReturnValueOnce(page.promise);
    const second = useStudyStore.getState().openStudy('s1');
    await vi.waitFor(() => expect(eventsOf).toHaveBeenCalled());
    useStudyStore.getState().closeStudy();
    page.resolve({ events: [], cursor: 9 });
    await second;
    expect(useStudyStore.getState().cursor).toBe(0);

    const failing = deferred<StudyDetail>();
    detailOf.mockReturnValueOnce(failing.promise);
    const third = useStudyStore.getState().openStudy('s1');
    useStudyStore.getState().closeStudy();
    failing.reject(new Error('late'));
    await third;
    expect(useStudyStore.getState().detailError).toBeNull();
  });

  it('closing a study reads the list again', async () => {
    await useStudyStore.getState().openStudy('s1');
    useStudyStore.getState().closeStudy();
    await vi.waitFor(() => expect(useStudyStore.getState().loaded).toBe(true));
    expect(useStudyStore.getState().selectedId).toBeNull();
  });
});

describe('studyStore: cancel and resume', () => {
  it('does nothing without an open study', async () => {
    await useStudyStore.getState().cancel();
    expect(cancelOf).not.toHaveBeenCalled();
  });

  it('cancels, then reads the study again', async () => {
    await useStudyStore.getState().openStudy('s1');
    cancelOf.mockResolvedValueOnce(detail({ status: 'cancel_requested' }));
    detailOf.mockResolvedValueOnce(detail({ status: 'cancelled' }));
    await useStudyStore.getState().cancel();
    expect(cancelOf).toHaveBeenCalledWith('s1');
    expect(useStudyStore.getState().detail?.status).toBe('cancelled');
    expect(useStudyStore.getState().pending).toBeNull();
  });

  it('resumes with and without a budget', async () => {
    await useStudyStore.getState().openStudy('s1');
    resumeOf.mockResolvedValue(detail());
    await useStudyStore.getState().resume();
    expect(resumeOf).toHaveBeenLastCalledWith('s1', {});
    const budget = { max_runs: 4, max_parallel_runs: 1 };
    await useStudyStore.getState().resume(budget);
    expect(resumeOf).toHaveBeenLastCalledWith('s1', { budget });
  });

  it('shows a refusal and ignores a second click while one is in flight', async () => {
    await useStudyStore.getState().openStudy('s1');
    const pending = deferred<StudyDetail>();
    resumeOf.mockReturnValueOnce(pending.promise);
    const first = useStudyStore.getState().resume();
    expect(useStudyStore.getState().pending).toBe('resume');
    await useStudyStore.getState().cancel();
    expect(cancelOf).not.toHaveBeenCalled();
    pending.reject(new StudiesApiError('budget used up', 409));
    await first;
    expect(useStudyStore.getState().actionError).toBe('budget used up');
    expect(useStudyStore.getState().pending).toBeNull();
  });

  it('drops the outcome of an action for a study no longer open', async () => {
    await useStudyStore.getState().openStudy('s1');
    const ok = deferred<StudyDetail>();
    cancelOf.mockReturnValueOnce(ok.promise);
    const done = useStudyStore.getState().cancel();
    useStudyStore.getState().closeStudy();
    ok.resolve(detail({ status: 'cancelled' }));
    await done;
    expect(useStudyStore.getState().detail).toBeNull();

    await useStudyStore.getState().openStudy('s1');
    const bad = deferred<StudyDetail>();
    cancelOf.mockReturnValueOnce(bad.promise);
    const failed = useStudyStore.getState().cancel();
    useStudyStore.getState().closeStudy();
    bad.reject(new Error('late'));
    await failed;
    expect(useStudyStore.getState().actionError).toBeNull();
  });
});

describe('studyStore: polling', () => {
  it('polls while watched and stops when the last watcher leaves', async () => {
    vi.useFakeTimers();
    const release = useStudyStore.getState().watch();
    const second = useStudyStore.getState().watch();
    expect(list).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(STUDY_POLL_MS);
    expect(list).toHaveBeenCalledTimes(2);
    second();
    second();
    await vi.advanceTimersByTimeAsync(STUDY_POLL_MS);
    expect(list).toHaveBeenCalledTimes(3);
    release();
    await vi.advanceTimersByTimeAsync(STUDY_POLL_MS * 3);
    expect(list).toHaveBeenCalledTimes(3);
  });

  it('a reset clears a running poller', async () => {
    vi.useFakeTimers();
    useStudyStore.getState().watch();
    _resetStudyStoreForTesting();
    await vi.advanceTimersByTimeAsync(STUDY_POLL_MS * 2);
    expect(list).toHaveBeenCalledTimes(1);
  });
});
