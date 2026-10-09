import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  fetchOutput,
  fetchPortStats,
  listRunOutputs,
  deleteRun,
  fetchStepIndex,
  fetchGradIndex,
  NoValueError,
  RunDataExpiredError,
  InvalidSliceError,
  PayloadTooLargeError,
  StatsNotCapturedError,
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
    text: async () => '',
  } as unknown as Response;
  g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
  return g.fetch as unknown as ReturnType<typeof vi.fn>;
}

// Like mockFetch but lets res.json() reject — exercises readDetail's catch path.
function mockFetchJsonThrows(status: number) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'mock-status-text',
    json: async () => {
      throw new SyntaxError('Unexpected end of JSON input');
    },
    text: async () => '',
  } as unknown as Response;
  g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
  return g.fetch as unknown as ReturnType<typeof vi.fn>;
}

// A 204 has no body, so a real `res.json()` on it throws. Returns the json
// spy so a test can prove nobody tried.
function mockFetchNoContent() {
  const json = vi.fn(async () => {
    throw new SyntaxError('Unexpected end of JSON input');
  });
  const response = {
    ok: true,
    status: 204,
    statusText: 'No Content',
    json,
    text: async () => '',
  } as unknown as Response;
  g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
  return json;
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

  it('asks for a bounded preview with preview=true (#640)', async () => {
    const fetchMock = mockFetch(200, { type: 'tensor' });
    await fetchOutput('r', 'n', 'p', { preview: true, maxElements: 65536 });
    const call = fetchMock.mock.calls[0][0] as string;
    expect(call).toContain('preview=true');
    expect(call).toContain('max_elements=65536');
    expect(call).not.toContain('slice=');
  });

  it('url-encodes path segments', async () => {
    const fetchMock = mockFetch(200, {});
    await fetchOutput('r', 'node with space', 'p q');
    const call = fetchMock.mock.calls[0][0] as string;
    expect(call).toContain('node%20with%20space');
    expect(call).toContain('p%20q');
  });

  it('throws RunDataExpiredError on 404', async () => {
    mockFetch(404, { detail: 'missing' });
    await expect(fetchOutput('r', 'n', 'p')).rejects.toBeInstanceOf(RunDataExpiredError);
  });

  it('throws NoValueError on 204 without reading the empty body', async () => {
    // The node ran and the port produced None; the server says so with 204.
    const json = mockFetchNoContent();
    const err = await fetchOutput('r', 'n', 'p').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NoValueError);
    expect(err).not.toBeInstanceOf(RunDataExpiredError);
    expect((err as Error).message).toContain('n.p');
    expect(json).not.toHaveBeenCalled();
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

  it('returns the parsed body on success', async () => {
    mockFetch(200, { type: 'scalar', value: 42 });
    await expect(fetchOutput('r', 'n', 'p')).resolves.toEqual({
      type: 'scalar',
      value: 42,
    });
  });

  it('falls back to statusText when error body has no detail field (readDetail)', async () => {
    // body is a plain object without a string `detail` → readDetail returns statusText.
    mockFetch(500, { somethingElse: true });
    await expect(fetchOutput('r', 'n', 'p')).rejects.toThrow(/fetchOutput failed: mock/);
  });

  it('falls back to statusText when error body is null (readDetail)', async () => {
    // body is null → the `body && ...` guard short-circuits to statusText.
    mockFetch(500, null);
    await expect(fetchOutput('r', 'n', 'p')).rejects.toThrow(/fetchOutput failed: mock/);
  });

  it('falls back to statusText when the error body is not JSON (readDetail catch)', async () => {
    mockFetchJsonThrows(500);
    await expect(fetchOutput('r', 'n', 'p')).rejects.toThrow(
      /fetchOutput failed: mock-status-text/,
    );
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

  it('url-encodes the run id', async () => {
    const fetchMock = mockFetch(200, []);
    await listRunOutputs('run/with space');
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%2Fwith%20space',
    );
  });

  it('throws a generic Error with detail on a non-404 failure', async () => {
    mockFetch(500, { detail: 'boom' });
    await expect(listRunOutputs('r')).rejects.toThrow(/listRunOutputs failed: boom/);
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

  it('throws a generic Error with detail on a non-404 failure', async () => {
    mockFetch(500, { detail: 'nope' });
    await expect(deleteRun('r')).rejects.toThrow(/deleteRun failed: nope/);
  });
});

describe('fetchStepIndex', () => {
  it('builds the __steps_index URL and returns the parsed list on 200', async () => {
    const fetchMock = mockFetch(200, [
      { index: 0, name: 'forward', description: '', scalars: {}, tensor_keys: [] },
    ]);
    const out = await fetchStepIndex('run 1', 'node a');
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%201/node%20a/__steps_index',
    );
    expect(out).toHaveLength(1);
    expect(out[0].name).toBe('forward');
  });

  it('returns an empty list on 404 (no steps recorded)', async () => {
    mockFetch(404, { detail: 'missing' });
    await expect(fetchStepIndex('r', 'n')).resolves.toEqual([]);
  });

  it('throws a generic Error on other failures', async () => {
    mockFetch(500, { detail: 'index boom' });
    await expect(fetchStepIndex('r', 'n')).rejects.toThrow(
      /fetchStepIndex failed: index boom/,
    );
  });
});

describe('fetchGradIndex', () => {
  it('builds the __grad_index URL and returns the parsed list on 200', async () => {
    const fetchMock = mockFetch(200, [
      { port: 'out', kind: 'port', has_grad: true, health: null },
    ]);
    const out = await fetchGradIndex('run 1', 'node a');
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%201/node%20a/__grad_index',
    );
    expect(out).toHaveLength(1);
    expect(out[0].port).toBe('out');
  });

  it('throws RunDataExpiredError on 404: the server holds nothing for the run', async () => {
    // A node with no gradients in a run the server holds is a 200 with [];
    // a 404 means the run itself is gone, which the Backward tab says.
    mockFetch(404, { detail: 'missing' });
    await expect(fetchGradIndex('r', 'n')).rejects.toBeInstanceOf(RunDataExpiredError);
  });

  it('throws a generic Error on other failures', async () => {
    mockFetch(500, { detail: 'grad boom' });
    await expect(fetchGradIndex('r', 'n')).rejects.toThrow(
      /fetchGradIndex failed: grad boom/,
    );
  });
});

describe('fetchPortStats', () => {
  it('builds the /stats URL and returns the parsed body on 200', async () => {
    const fetchMock = mockFetch(200, { kind: 'tensor' });
    await expect(fetchPortStats('run 1', 'node a', 'out')).resolves.toEqual({
      kind: 'tensor',
    });
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%201/node%20a/out/stats',
    );
  });

  it('throws StatsNotCapturedError with the server detail on 404', async () => {
    mockFetch(404, { detail: 'nothing captured' });
    const err = await fetchPortStats('r', 'n', 'p').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(StatsNotCapturedError);
    expect((err as Error).message).toBe('nothing captured');
  });

  it('throws NoValueError on 204 without reading the empty body', async () => {
    const json = mockFetchNoContent();
    const err = await fetchPortStats('r', 'n', 'p').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NoValueError);
    expect(err).not.toBeInstanceOf(StatsNotCapturedError);
    expect(json).not.toHaveBeenCalled();
  });
});

// ── An id with a slash: a node inside a block (#621) ─────────────────────────
// The run names it `<instance>/<inner>`, and the server decodes `%2F` before
// routing, so in a path segment it reaches no route at all. Such an id goes
// in the query; every other id keeps the path form above, byte for byte.

describe('an id the path cannot carry', () => {
  it('fetchOutput asks the query form', async () => {
    const fetchMock = mockFetch(200, { type: 'scalar', value: 1 });
    await fetchOutput('r', 'blk/mul', 'tensor');
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/execution/outputs/r/value?node_id=blk%2Fmul&port=tensor',
    );
  });

  it('fetchOutput keeps slice and max_elements in the query form', async () => {
    const fetchMock = mockFetch(200, { type: 'tensor' });
    await fetchOutput('run 1', 'blk/nest/mul', 'tensor', { slice: '0,:,:', maxElements: 1024 });
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%201/value?node_id=blk%2Fnest%2Fmul&port=tensor' +
        '&slice=0%2C%3A%2C%3A&max_elements=1024',
    );
  });

  it('fetchOutput sends a port with a slash in the query too', async () => {
    const fetchMock = mockFetch(200, {});
    await fetchOutput('r', 'n', 'p/q');
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/r/value?node_id=n&port=p%2Fq',
    );
  });

  it('fetchOutput maps the answers exactly as the path form does', async () => {
    mockFetch(404, { detail: 'missing' });
    await expect(fetchOutput('r', 'blk/mul', 'p')).rejects.toBeInstanceOf(RunDataExpiredError);
    mockFetchNoContent();
    const err = await fetchOutput('r', 'blk/mul', 'p').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(NoValueError);
    expect((err as Error).message).toContain('blk/mul.p');
    mockFetch(400, { detail: 'bad slice' });
    await expect(fetchOutput('r', 'blk/mul', 'p')).rejects.toBeInstanceOf(InvalidSliceError);
    mockFetch(413, { detail: 'too big' });
    await expect(fetchOutput('r', 'blk/mul', 'p')).rejects.toBeInstanceOf(PayloadTooLargeError);
  });

  it('fetchPortStats asks the query form', async () => {
    const fetchMock = mockFetch(200, { kind: 'tensor' });
    await fetchPortStats('run 1', 'blk/mul', 'out');
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%201/stats?node_id=blk%2Fmul&port=out',
    );
    mockFetch(404, { detail: 'nothing captured' });
    await expect(fetchPortStats('r', 'blk/mul', 'p')).rejects.toBeInstanceOf(
      StatsNotCapturedError,
    );
  });

  it('fetchStepIndex asks the query form', async () => {
    const fetchMock = mockFetch(200, []);
    await fetchStepIndex('run 1', 'blk/mul');
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%201/steps?node_id=blk%2Fmul',
    );
  });

  it('fetchGradIndex asks the query form', async () => {
    const fetchMock = mockFetch(200, []);
    await fetchGradIndex('run 1', 'blk/mul');
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/execution/outputs/run%201/grads?node_id=blk%2Fmul',
    );
  });
});
