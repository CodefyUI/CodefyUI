import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { exportGraph } from './rest';

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
});

afterEach(() => {
  g.fetch = originalFetch;
});

describe('exportGraph', () => {
  it('sends name when provided so the script header uses it', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    await exportGraph([], [], 'Train CNN on MNIST');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/graph/export');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ nodes: [], edges: [], name: 'Train CNN on MNIST' });
  });

  it('omits name when not provided so backend falls back to its default', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    await exportGraph([], []);
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ nodes: [], edges: [] });
  });

  it('throws when the endpoint fails', async () => {
    mockFetch(500, {});
    await expect(exportGraph([], [], 'x')).rejects.toThrow(/Export failed/);
  });
});
