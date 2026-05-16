import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  fetchOutput,
  listRunOutputs,
  deleteRun,
  RunDataExpiredError,
  InvalidSliceError,
  PayloadTooLargeError,
} from './executionOutputs';
import { _setSessionTokenForTesting } from './_auth';

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

function mockFetch(status: number, body: unknown) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'mock',
    json: async () => body,
  } as unknown as Response;
  g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
  return g.fetch as unknown as ReturnType<typeof vi.fn>;
}

beforeEach(() => {
  originalFetch = g.fetch;
  // Pre-seed the cached session token so apiFetch doesn't bootstrap.
  _setSessionTokenForTesting('test-token');
});

afterEach(() => {
  g.fetch = originalFetch;
  _setSessionTokenForTesting(null);
});

describe('fetchOutput', () => {
  it('constructs URL without query when no options given', async () => {
    const fetchMock = mockFetch(200, { type: 'scalar', value: 1 });
    await fetchOutput('run-1', 'node-1', 'port-1');
    expect(fetchMock).toHaveBeenCalledWith('/api/execution/outputs/run-1/node-1/port-1');
  });

  it('appends slice and max_elements query params', async () => {
    const fetchMock = mockFetch(200, { type: 'tensor' });
    await fetchOutput('r', 'n', 'p', { slice: '0,:,:', maxElements: 1024 });
    const call = fetchMock.mock.calls[0][0] as string;
    expect(call).toContain('slice=0%2C%3A%2C%3A');
    expect(call).toContain('max_elements=1024');
  });

  it('url-encodes path segments', async () => {
    const fetchMock = mockFetch(200, {});
    await fetchOutput('r', 'node with space', 'p/q');
    const call = fetchMock.mock.calls[0][0] as string;
    expect(call).toContain('node%20with%20space');
    expect(call).toContain('p%2Fq');
  });

  it('throws RunDataExpiredError on 404', async () => {
    mockFetch(404, { detail: 'missing' });
    await expect(fetchOutput('r', 'n', 'p')).rejects.toBeInstanceOf(RunDataExpiredError);
  });

  it('throws InvalidSliceError on 400', async () => {
    mockFetch(400, { detail: 'bad slice' });
    await expect(fetchOutput('r', 'n', 'p')).rejects.toBeInstanceOf(InvalidSliceError);
  });

  it('throws PayloadTooLargeError on 413', async () => {
    mockFetch(413, { detail: 'too big' });
    await expect(fetchOutput('r', 'n', 'p')).rejects.toBeInstanceOf(PayloadTooLargeError);
  });

  it('throws generic Error on 500', async () => {
    mockFetch(500, { detail: 'server crash' });
    await expect(fetchOutput('r', 'n', 'p')).rejects.toThrow(/server crash/);
  });
});

describe('listRunOutputs', () => {
  it('returns the array on 200', async () => {
    mockFetch(200, [
      { node_id: 'n1', port: 'out', type: 'tensor', full_shape: [2, 3] },
    ]);
    const out = await listRunOutputs('r');
    expect(out).toHaveLength(1);
    expect(out[0].node_id).toBe('n1');
  });

  it('throws RunDataExpiredError on 404', async () => {
    mockFetch(404, { detail: 'missing' });
    await expect(listRunOutputs('r')).rejects.toBeInstanceOf(RunDataExpiredError);
  });
});

describe('deleteRun', () => {
  it('resolves on 200', async () => {
    const fetchMock = mockFetch(200, { deleted: true });
    await deleteRun('r');
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'DELETE' });
  });

  it('throws on 404', async () => {
    mockFetch(404, { detail: 'missing' });
    await expect(deleteRun('r')).rejects.toBeInstanceOf(RunDataExpiredError);
  });
});
