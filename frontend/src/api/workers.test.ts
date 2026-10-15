import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { drainWorker, fetchWorkers, undrainWorker } from './workers';
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

describe('fetchWorkers', () => {
  it('reads the list from GET /api/workers', async () => {
    const fetchMock = mockFetch(200, { workers: [{ id: 'w1' }] });
    expect(await fetchWorkers()).toEqual([{ id: 'w1' }]);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/workers');
  });

  it('answers an empty list for a body without one', async () => {
    mockFetch(200, {});
    expect(await fetchWorkers()).toEqual([]);
  });

  it('surfaces the backend detail on failure', async () => {
    mockFetch(503, { detail: 'worker registry is not available' });
    await expect(fetchWorkers()).rejects.toThrow('worker registry is not available');
  });

  it('falls back to the status text when the error body is not JSON', async () => {
    mockFetch(500, null, true);
    await expect(fetchWorkers()).rejects.toThrow('Workers failed: mock');
  });
});

describe('drainWorker / undrainWorker', () => {
  it('POSTs drain with the session token and returns the worker', async () => {
    const fetchMock = mockFetch(200, { id: 'gpu box', draining: true });
    expect(await drainWorker('gpu box')).toEqual({ id: 'gpu box', draining: true });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/gpu%20box/drain');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
  });

  it('POSTs undrain', async () => {
    const fetchMock = mockFetch(200, { id: 'w1', draining: false });
    await undrainWorker('w1');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/workers/w1/undrain');
  });

  it('surfaces a 404 detail', async () => {
    mockFetch(404, { detail: "no worker named 'w1'" });
    await expect(drainWorker('w1')).rejects.toThrow("no worker named 'w1'");
  });

  it('names the action when the body says nothing usable', async () => {
    mockFetch(500, { detail: { nested: true } });
    await expect(drainWorker('w1')).rejects.toThrow('Drain failed: mock');
    mockFetch(500, null, true);
    await expect(undrainWorker('w1')).rejects.toThrow('Undrain failed: mock');
  });
});
