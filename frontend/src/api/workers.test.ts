import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  agentCommand,
  createEnrollment,
  deleteEnrollment,
  drainWorker,
  fetchEnrollments,
  fetchWorkers,
  removeWorker,
  undrainWorker,
} from './workers';
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

describe('removeWorker', () => {
  it('DELETEs the worker with the session token', async () => {
    const fetchMock = mockFetch(200, { id: 'gpu box', revoked: true });
    await removeWorker('gpu box');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/gpu%20box');
    expect(init.method).toBe('DELETE');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
  });

  it('surfaces the failure', async () => {
    mockFetch(404, { detail: "no worker named 'x'" });
    await expect(removeWorker('x')).rejects.toThrow("no worker named 'x'");
    mockFetch(500, null, true);
    await expect(removeWorker('x')).rejects.toThrow('Remove failed: mock');
  });
});

describe('enrollment tokens', () => {
  it('lists them with the session token, which a GET needs here', async () => {
    const list = { enrollments: [], control_url: 'http://h:1', other_machine_urls: [], listening_on_loopback: false };
    const fetchMock = mockFetch(200, list);
    expect(await fetchEnrollments()).toEqual(list);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/enrollments');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    mockFetch(403, { detail: 'Missing or invalid X-CodefyUI-Token header' });
    await expect(fetchEnrollments()).rejects.toThrow('Missing or invalid');
  });

  it('creates one with a lifetime and a number of uses', async () => {
    const fetchMock = mockFetch(200, { id: 'abcdef', token: 'abcdef.0123456789abcdef' });
    expect(await createEnrollment({ ttl_s: 3600, uses: 2 })).toMatchObject({ id: 'abcdef' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/enrollments');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ ttl_s: 3600, uses: 2 });
    mockFetch(422, null, true);
    await expect(createEnrollment({ ttl_s: 1, uses: 1 })).rejects.toThrow('Generate token failed: mock');
  });

  it('deletes one', async () => {
    const fetchMock = mockFetch(200, { id: 'abc def', deleted: true });
    await deleteEnrollment('abc def');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/workers/enrollments/abc%20def');
    expect(fetchMock.mock.calls[0][1].method).toBe('DELETE');
    mockFetch(404, { detail: "no enrollment token 'x'" });
    await expect(deleteEnrollment('x')).rejects.toThrow("no enrollment token 'x'");
  });

  it('creates one for another address', async () => {
    const fetchMock = mockFetch(200, { id: 'abcdef' });
    await createEnrollment({ ttl_s: 3600, uses: 1, control_url: 'https://cp.lab' });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ ttl_s: 3600, uses: 1, control_url: 'https://cp.lab' });
  });

  it('asks the server for the agent command at an address, the token in the body', async () => {
    const fetchMock = mockFetch(200, { control_url: 'https://cp.lab', command: 'cdui agent start ...' });
    const body = { control_url: 'https://cp.lab', token: 'abcdef.0123456789abcdef', worker: null };
    expect(await agentCommand(body)).toBe('cdui agent start ...');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/agent-command');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual(body);
    mockFetch(422, { detail: 'control_url must be http(s)://host[:port][/path]' });
    await expect(agentCommand(body)).rejects.toThrow('control_url must be');
    mockFetch(500, null, true);
    await expect(agentCommand(body)).rejects.toThrow('Command failed: mock');
  });
});
