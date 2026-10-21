import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  StudiesApiError,
  cancelStudy,
  getStudy,
  getStudyEvents,
  listStudies,
  resumeStudy,
} from './studies';
import { _setSessionTokenForTesting } from './_auth';

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

function mockFetch(status: number, body: unknown, jsonThrows = false) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'mock',
    json: async () => {
      if (jsonThrows) throw new SyntaxError('not json');
      return body;
    },
  } as unknown as Response;
  const fn = vi.fn().mockResolvedValue(response);
  g.fetch = fn as unknown as typeof fetch;
  return fn;
}

beforeEach(() => {
  originalFetch = g.fetch;
  _setSessionTokenForTesting('test-token');
});

afterEach(() => {
  g.fetch = originalFetch;
  _setSessionTokenForTesting(null);
});

describe('listStudies', () => {
  it('reads GET /api/studies, newest first', async () => {
    const fetchMock = mockFetch(200, { studies: [{ study_id: 's1' }], total: 1 });
    const signal = new AbortController().signal;
    expect(await listStudies(signal)).toEqual({ studies: [{ study_id: 's1' }], total: 1 });
    expect(fetchMock).toHaveBeenCalledWith('/api/studies?limit=100', { signal });
  });

  it('tolerates a body without the fields', async () => {
    const fetchMock = mockFetch(200, {});
    expect(await listStudies()).toEqual({ studies: [], total: 0 });
    expect(fetchMock).toHaveBeenCalledWith('/api/studies?limit=100', undefined);
  });

  it('throws the server detail with its status', async () => {
    mockFetch(503, { detail: 'study coordinator not initialised' });
    const error = await listStudies().catch((caught) => caught);
    expect(error).toBeInstanceOf(StudiesApiError);
    expect(error.message).toBe('study coordinator not initialised');
    expect(error.status).toBe(503);
    expect(error.name).toBe('StudiesApiError');
  });

  it('falls back to the status text for a body that is not JSON', async () => {
    mockFetch(500, null, true);
    await expect(listStudies()).rejects.toThrow('Studies failed: mock');
  });
});

describe('getStudy and getStudyEvents', () => {
  it('reads one study, its id encoded', async () => {
    const fetchMock = mockFetch(200, { study_id: 'a/b' });
    expect(await getStudy('a/b')).toEqual({ study_id: 'a/b' });
    expect(fetchMock).toHaveBeenCalledWith('/api/studies/a%2Fb', undefined);
    const signal = new AbortController().signal;
    await getStudy('s', signal);
    expect(fetchMock).toHaveBeenLastCalledWith('/api/studies/s', { signal });
  });

  it('refuses a missing study with 404', async () => {
    mockFetch(404, { detail: "study 's' not found" });
    await expect(getStudy('s')).rejects.toMatchObject({ status: 404 });
  });

  it('reads the events after a cursor', async () => {
    const fetchMock = mockFetch(200, { events: [{ cursor: 4 }], cursor: 4 });
    expect(await getStudyEvents('s', 3)).toEqual({ events: [{ cursor: 4 }], cursor: 4 });
    expect(fetchMock).toHaveBeenCalledWith('/api/studies/s/events?cursor=3', undefined);
    const signal = new AbortController().signal;
    await getStudyEvents('s', 0, signal);
    expect(fetchMock).toHaveBeenLastCalledWith('/api/studies/s/events?cursor=0', { signal });
  });

  it('keeps the cursor when the page has none', async () => {
    mockFetch(200, {});
    expect(await getStudyEvents('s', 7)).toEqual({ events: [], cursor: 7 });
  });

  it('throws on a refused events read', async () => {
    mockFetch(404, {});
    await expect(getStudyEvents('s', 0)).rejects.toThrow('Study events failed: mock');
  });
});

describe('cancelStudy and resumeStudy', () => {
  it('cancels with the session token and an optional revision', async () => {
    const fetchMock = mockFetch(200, { study_id: 's', status: 'cancel_requested' });
    await cancelStudy('s');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/studies/s/cancel');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({});
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    await cancelStudy('s', 4);
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ expected_revision: 4 });
  });

  it('resumes with an optional budget and revision', async () => {
    const fetchMock = mockFetch(200, { study_id: 's', status: 'running' });
    await resumeStudy('s');
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({});
    const budget = { max_runs: null, max_parallel_runs: 1, max_active_seconds: 60 };
    await resumeStudy('s', { budget, expectedRevision: 2 });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ budget, expected_revision: 2 });
  });

  it('throws the 409 the server sends', async () => {
    mockFetch(409, { detail: 'budget.max_runs (1) is used up' });
    await expect(resumeStudy('s')).rejects.toMatchObject({
      status: 409,
      message: 'budget.max_runs (1) is used up',
    });
    mockFetch(409, {});
    await expect(cancelStudy('s')).rejects.toThrow('Cancel failed: mock');
  });
});
