import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  agentCommand,
  addSshWorker,
  createEnrollment,
  deleteEnrollment,
  deleteSshWorker,
  drainWorker,
  fetchEnrollments,
  fetchWorkerList,
  fetchSshWorkers,
  fetchWorkers,
  removeWorker,
  setDevicePolicy,
  setSshWorkerEnabled,
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

describe('fetchWorkerList', () => {
  it('reads the workers and the refused handshakes', async () => {
    const rejection = { name: 'old-box', reason: 'version mismatch', at: 1 };
    mockFetch(200, { workers: [{ id: 'w1' }], rejections: [rejection] });
    expect(await fetchWorkerList()).toEqual({ workers: [{ id: 'w1' }], rejections: [rejection] });
  });

  it('answers empty lists for a body without them', async () => {
    mockFetch(200, {});
    expect(await fetchWorkerList()).toEqual({ workers: [], rejections: [] });
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

describe('setDevicePolicy (#788)', () => {
  it('PUTs the policy with the session token and returns the worker', async () => {
    const fetchMock = mockFetch(200, { id: 'gpu box' });
    expect(await setDevicePolicy('gpu box', 'cuda:1', 'shared')).toEqual({ id: 'gpu box' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/gpu%20box/devices/cuda%3A1/policy');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body)).toEqual({ policy: 'shared' });
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
  });

  it('surfaces the failure', async () => {
    mockFetch(422, { detail: 'cpu cannot be shared by memory' });
    await expect(setDevicePolicy('w', 'cpu', 'shared')).rejects.toThrow('cpu cannot be shared');
    mockFetch(500, null, true);
    await expect(setDevicePolicy('w', 'cuda:0', 'exclusive')).rejects.toThrow('Policy change failed: mock');
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

describe('SSH workers (#784)', () => {
  it('lists them with the session token', async () => {
    const list = { ssh_workers: [], ssh_available: true, port_range: [18000, 18100], default_remote_cdui: '~/CodefyUI/cdui' };
    const fetchMock = mockFetch(200, list);
    expect(await fetchSshWorkers()).toEqual(list);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/ssh');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    mockFetch(503, { detail: 'SSH workers are not available' });
    await expect(fetchSshWorkers()).rejects.toMatchObject({ message: 'SSH workers are not available', status: 503 });
  });

  it('adds one', async () => {
    const fetchMock = mockFetch(200, { id: 'gpu-box', state: 'connecting' });
    expect(await addSshWorker({ target: 'alice@gpu-box', port: 2222 })).toMatchObject({ id: 'gpu-box' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/workers/ssh');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ target: 'alice@gpu-box', port: 2222 });
    mockFetch(409, { detail: 'an SSH worker named gpu-box exists already' });
    await expect(addSshWorker({ target: 'a@gpu-box' })).rejects.toThrow('exists already');
  });

  it('enables, disables and deletes one', async () => {
    let fetchMock = mockFetch(200, { id: 'gpu box', enabled: false });
    await setSshWorkerEnabled('gpu box', false);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/workers/ssh/gpu%20box/disable');
    expect(fetchMock.mock.calls[0][1].method).toBe('POST');
    fetchMock = mockFetch(200, { id: 'gpu-box', enabled: true });
    await setSshWorkerEnabled('gpu-box', true);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/workers/ssh/gpu-box/enable');
    mockFetch(500, null, true);
    await expect(setSshWorkerEnabled('x', true)).rejects.toThrow('Enable failed: mock');
    await expect(setSshWorkerEnabled('x', false)).rejects.toThrow('Disable failed: mock');

    fetchMock = mockFetch(200, { id: 'gpu-box', deleted: true });
    await deleteSshWorker('gpu-box');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/workers/ssh/gpu-box');
    expect(fetchMock.mock.calls[0][1].method).toBe('DELETE');
    mockFetch(404, { detail: "no SSH worker named 'x'" });
    await expect(deleteSshWorker('x')).rejects.toThrow("no SSH worker named 'x'");
  });
});
