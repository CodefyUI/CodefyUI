import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  exportGraph,
  fetchNodeDefinitions,
  fetchPresetDefinitions,
  fetchDevices,
  validateGraph,
  saveGraph,
  loadGraph,
  listGraphs,
  resetWeights,
  createPreset,
  listExamples,
  loadExample,
  reloadNodes,
  listCustomNodes,
  toggleCustomNode,
  uploadCustomNode,
  deleteCustomNode,
  listModelFiles,
  uploadModelFile,
  deleteModelFile,
  downloadModelFile,
  listImageFiles,
  uploadImageFile,
  deleteImageFile,
  downloadImageFile,
} from './rest';
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

// Error response whose .json() rejects — exercises the `.catch(() => ({}))`
// fallbacks in createPreset / upload* / download* error handlers.
function mockFetchJsonThrows(status: number) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'mock',
    json: async () => {
      throw new SyntaxError('not json');
    },
    text: async () => '',
  } as unknown as Response;
  g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
  return g.fetch as unknown as ReturnType<typeof vi.fn>;
}

beforeEach(() => {
  originalFetch = g.fetch;
  // Pre-seed the cached session token so apiFetch doesn't try to bootstrap
  // (mocking that round-trip in every test would be noisy).
  _setSessionTokenForTesting('test-token');
});

afterEach(() => {
  g.fetch = originalFetch;
  _setSessionTokenForTesting(null);
  vi.restoreAllMocks();
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

  it('returns the parsed script payload on success', async () => {
    mockFetch(200, { script: 'print(1)' });
    await expect(exportGraph([], [])).resolves.toEqual({ script: 'print(1)' });
  });
});

// ── Simple GET endpoints (fetch, success + error) ──

describe('GET endpoints', () => {
  const cases: Array<{
    name: string;
    fn: () => Promise<unknown>;
    url: string;
    errorRe: RegExp;
  }> = [
    {
      name: 'fetchNodeDefinitions',
      fn: () => fetchNodeDefinitions(),
      url: '/api/nodes',
      errorRe: /Failed to fetch node definitions/,
    },
    {
      name: 'fetchPresetDefinitions',
      fn: () => fetchPresetDefinitions(),
      url: '/api/presets',
      errorRe: /Failed to fetch presets/,
    },
    {
      name: 'fetchDevices',
      fn: () => fetchDevices(),
      url: '/api/system/devices',
      errorRe: /Failed to fetch devices/,
    },
    {
      name: 'listGraphs',
      fn: () => listGraphs(),
      url: '/api/graph/list',
      errorRe: /List failed/,
    },
    {
      name: 'listExamples',
      fn: () => listExamples(),
      url: '/api/examples/list',
      errorRe: /Failed to list examples/,
    },
    {
      name: 'listCustomNodes',
      fn: () => listCustomNodes(),
      url: '/api/custom-nodes',
      errorRe: /Failed to list custom nodes/,
    },
    {
      name: 'listModelFiles',
      fn: () => listModelFiles(),
      url: '/api/models',
      errorRe: /Failed to list model files/,
    },
    {
      name: 'listImageFiles',
      fn: () => listImageFiles(),
      url: '/api/images',
      errorRe: /Failed to list image files/,
    },
  ];

  for (const c of cases) {
    it(`${c.name} fetches ${c.url} and returns the body on success`, async () => {
      const fetchMock = mockFetch(200, [{ id: 1 }]);
      const out = await c.fn();
      expect(fetchMock).toHaveBeenCalledWith(c.url);
      expect(out).toEqual([{ id: 1 }]);
    });

    it(`${c.name} throws on a non-ok response`, async () => {
      mockFetch(500, {});
      await expect(c.fn()).rejects.toThrow(c.errorRe);
    });
  }
});

describe('loadGraph', () => {
  it('url-encodes the name and returns the body', async () => {
    const fetchMock = mockFetch(200, { nodes: [] });
    await loadGraph('My Graph/v2');
    expect(fetchMock).toHaveBeenCalledWith('/api/graph/load/My%20Graph%2Fv2');
  });

  it('throws on failure', async () => {
    mockFetch(404, {});
    await expect(loadGraph('x')).rejects.toThrow(/Load failed/);
  });
});

describe('loadExample', () => {
  it('url-encodes the path query param', async () => {
    const fetchMock = mockFetch(200, { nodes: [] });
    await loadExample('examples/foo bar.json');
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/examples/load?path=examples%2Ffoo%20bar.json',
    );
  });

  it('throws on failure', async () => {
    mockFetch(500, {});
    await expect(loadExample('p')).rejects.toThrow(/Failed to load example/);
  });
});

// ── Mutating JSON endpoints (apiFetch → token header) ──

describe('validateGraph', () => {
  it('POSTs nodes/edges with the token header and returns the body', async () => {
    const fetchMock = mockFetch(200, { valid: true });
    const out = await validateGraph([{ id: 'a' }], [{ id: 'e' }]);
    expect(out).toEqual({ valid: true });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/graph/validate');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ nodes: [{ id: 'a' }], edges: [{ id: 'e' }] });
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
  });

  it('throws on failure', async () => {
    mockFetch(422, {});
    await expect(validateGraph([], [])).rejects.toThrow(/Validation failed/);
  });
});

describe('saveGraph', () => {
  it('POSTs the data and returns the body', async () => {
    const fetchMock = mockFetch(200, { saved: true });
    await saveGraph({ name: 'g', nodes: [], edges: [] } as never);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/graph/save');
    expect(JSON.parse(init.body)).toEqual({ name: 'g', nodes: [], edges: [] });
  });

  it('throws on failure', async () => {
    mockFetch(500, {});
    await expect(saveGraph({} as never)).rejects.toThrow(/Save failed/);
  });
});

describe('resetWeights', () => {
  it('omits node_ids when no ids are provided', async () => {
    const fetchMock = mockFetch(200, { graph_id: 'g', scope: 'graph', evicted: 3 });
    const out = await resetWeights('g');
    expect(out).toEqual({ graph_id: 'g', scope: 'graph', evicted: 3 });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ graph_id: 'g' });
  });

  it('omits node_ids when an empty array is provided', async () => {
    const fetchMock = mockFetch(200, {});
    await resetWeights('g', []);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ graph_id: 'g' });
  });

  it('includes node_ids when a non-empty array is provided', async () => {
    const fetchMock = mockFetch(200, {});
    await resetWeights('g', ['n1', 'n2']);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      graph_id: 'g',
      node_ids: ['n1', 'n2'],
    });
  });

  it('throws on failure', async () => {
    mockFetch(500, {});
    await expect(resetWeights('g')).rejects.toThrow(/Reset weights failed/);
  });
});

describe('createPreset', () => {
  it('POSTs the payload and returns the created preset', async () => {
    const fetchMock = mockFetch(200, { name: 'p', nodes: [], edges: [] });
    const out = await createPreset({ name: 'p', nodes: [], edges: [] });
    expect((out as unknown as { name: string }).name).toBe('p');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/presets/create');
  });

  it('surfaces the backend detail message on failure', async () => {
    mockFetch(400, { detail: 'name already exists' });
    await expect(createPreset({ name: 'p', nodes: [], edges: [] })).rejects.toThrow(
      /name already exists/,
    );
  });

  it('falls back to a generic message when the error body has no detail', async () => {
    mockFetch(500, {});
    await expect(createPreset({ name: 'p', nodes: [], edges: [] })).rejects.toThrow(
      /Export failed/,
    );
  });

  it('falls back to a generic message when the error body is not JSON', async () => {
    mockFetchJsonThrows(500);
    await expect(createPreset({ name: 'p', nodes: [], edges: [] })).rejects.toThrow(
      /Export failed/,
    );
  });
});

describe('reloadNodes', () => {
  it('POSTs to /api/nodes/reload and returns the body', async () => {
    const fetchMock = mockFetch(200, { reloaded: true });
    const out = await reloadNodes();
    expect(out).toEqual({ reloaded: true });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/nodes/reload');
    expect(init.method).toBe('POST');
  });

  it('throws on failure', async () => {
    mockFetch(500, {});
    await expect(reloadNodes()).rejects.toThrow(/Reload failed/);
  });
});

describe('toggleCustomNode', () => {
  it('POSTs the filename and returns the body', async () => {
    const fetchMock = mockFetch(200, { enabled: false });
    await toggleCustomNode('my_node.py');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/custom-nodes/toggle');
    expect(JSON.parse(init.body)).toEqual({ filename: 'my_node.py' });
  });

  it('throws on failure', async () => {
    mockFetch(500, {});
    await expect(toggleCustomNode('x')).rejects.toThrow(/Toggle failed/);
  });
});

// ── FormData upload endpoints ──

describe('uploadCustomNode', () => {
  it('POSTs a FormData with the file and returns the body', async () => {
    const fetchMock = mockFetch(200, { filename: 'x.py' });
    const file = new File(['code'], 'x.py');
    await uploadCustomNode(file);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/custom-nodes/upload');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('file')).toBe(file);
  });

  it('throws on failure', async () => {
    mockFetch(500, {});
    await expect(uploadCustomNode(new File([''], 'x'))).rejects.toThrow(/Upload failed/);
  });
});

describe('uploadModelFile', () => {
  it('POSTs a FormData and returns the body', async () => {
    const fetchMock = mockFetch(200, { filename: 'm.pt', size: 10 });
    const file = new File(['weights'], 'm.pt');
    const out = await uploadModelFile(file);
    expect(out).toEqual({ filename: 'm.pt', size: 10 });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/models/upload');
    expect((fetchMock.mock.calls[0][1].body as FormData).get('file')).toBe(file);
  });

  it('surfaces the backend detail on failure', async () => {
    mockFetch(413, { detail: 'file too large' });
    await expect(uploadModelFile(new File([''], 'm.pt'))).rejects.toThrow(/file too large/);
  });

  it('falls back to a generic message when the error body is not JSON', async () => {
    mockFetchJsonThrows(500);
    await expect(uploadModelFile(new File([''], 'm.pt'))).rejects.toThrow(/Upload failed/);
  });
});

describe('uploadImageFile', () => {
  it('POSTs a FormData and returns the body', async () => {
    const fetchMock = mockFetch(200, { filename: 'i.png', size: 4 });
    const file = new File(['img'], 'i.png');
    const out = await uploadImageFile(file);
    expect(out).toEqual({ filename: 'i.png', size: 4 });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/images/upload');
  });

  it('surfaces the backend detail on failure', async () => {
    mockFetch(413, { detail: 'image too large' });
    await expect(uploadImageFile(new File([''], 'i.png'))).rejects.toThrow(/image too large/);
  });

  it('falls back to a generic message when the error body is not JSON', async () => {
    mockFetchJsonThrows(500);
    await expect(uploadImageFile(new File([''], 'i.png'))).rejects.toThrow(/Upload failed/);
  });
});

// ── DELETE endpoints ──

describe('delete endpoints', () => {
  const cases: Array<{
    name: string;
    fn: (f: string) => Promise<unknown>;
    url: string;
    errorRe: RegExp;
  }> = [
    {
      name: 'deleteCustomNode',
      fn: (f) => deleteCustomNode(f),
      url: '/api/custom-nodes/a%20b.py',
      errorRe: /Delete failed/,
    },
    {
      name: 'deleteModelFile',
      fn: (f) => deleteModelFile(f),
      url: '/api/models/a%20b.py',
      errorRe: /Delete failed/,
    },
    {
      name: 'deleteImageFile',
      fn: (f) => deleteImageFile(f),
      url: '/api/images/a%20b.py',
      errorRe: /Delete failed/,
    },
  ];

  for (const c of cases) {
    it(`${c.name} DELETEs the url-encoded filename`, async () => {
      const fetchMock = mockFetch(200, { deleted: true });
      await c.fn('a b.py');
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe(c.url);
      expect(init.method).toBe('DELETE');
      expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    });

    it(`${c.name} throws on failure`, async () => {
      mockFetch(500, {});
      await expect(c.fn('x')).rejects.toThrow(c.errorRe);
    });
  }
});

// ── Download endpoints (blob → anchor click) ──

describe('download endpoints', () => {
  let createObjectURL: ReturnType<typeof vi.fn>;
  let revokeObjectURL: ReturnType<typeof vi.fn>;
  let clickSpy: ReturnType<typeof vi.spyOn>;
  let appendSpy: ReturnType<typeof vi.spyOn>;

  function mockFetchBlob(status: number, body: unknown) {
    const response = {
      ok: status >= 200 && status < 300,
      status,
      statusText: 'mock',
      json: async () => body,
      text: async () => '',
      blob: async () => new Blob(['data']),
    } as unknown as Response;
    g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
    return g.fetch as unknown as ReturnType<typeof vi.fn>;
  }

  beforeEach(() => {
    createObjectURL = vi.fn().mockReturnValue('blob:mock-url');
    revokeObjectURL = vi.fn();
    // jsdom doesn't implement these; install spies we can assert against.
    (URL as unknown as { createObjectURL: unknown }).createObjectURL = createObjectURL;
    (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = revokeObjectURL;
    // Don't actually navigate when the synthetic anchor is clicked.
    clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {});
    appendSpy = vi.spyOn(document.body, 'appendChild');
  });

  afterEach(() => {
    clickSpy.mockRestore();
    appendSpy.mockRestore();
    delete (URL as unknown as { createObjectURL?: unknown }).createObjectURL;
    delete (URL as unknown as { revokeObjectURL?: unknown }).revokeObjectURL;
  });

  describe('downloadModelFile', () => {
    it('fetches a nested path, encoding each segment, and triggers a download', async () => {
      const fetchMock = mockFetchBlob(200, null);
      await downloadModelFile('runs/exp 1/model.pt');
      // Slashes preserved, each segment encoded.
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/models/download/runs/exp%201/model.pt',
      );
      expect(createObjectURL).toHaveBeenCalledTimes(1);
      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
      // The anchor's download attribute is just the basename.
      const anchor = appendSpy.mock.calls[0][0] as HTMLAnchorElement;
      expect(anchor.download).toBe('model.pt');
      expect(anchor.href).toContain('blob:mock-url');
      // The transient anchor is removed from the DOM after clicking.
      expect(document.body.contains(anchor)).toBe(false);
    });

    it('uses the whole filename as the download name when there is no slash', async () => {
      mockFetchBlob(200, null);
      await downloadModelFile('model.pt');
      const anchor = appendSpy.mock.calls[0][0] as HTMLAnchorElement;
      expect(anchor.download).toBe('model.pt');
    });

    it('surfaces the backend detail on failure', async () => {
      mockFetchBlob(404, { detail: 'gone' });
      await expect(downloadModelFile('m.pt')).rejects.toThrow(/gone/);
      expect(clickSpy).not.toHaveBeenCalled();
    });

    it('falls back to a generic message when the error body is not JSON', async () => {
      const response = {
        ok: false,
        status: 500,
        statusText: 'mock',
        json: async () => {
          throw new SyntaxError('not json');
        },
        blob: async () => new Blob(['x']),
      } as unknown as Response;
      g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
      await expect(downloadModelFile('m.pt')).rejects.toThrow(/Download failed/);
    });
  });

  describe('downloadImageFile', () => {
    it('fetches a nested path and triggers a download', async () => {
      const fetchMock = mockFetchBlob(200, null);
      await downloadImageFile('runs/img 1.png');
      expect(fetchMock).toHaveBeenCalledWith('/api/images/download/runs/img%201.png');
      expect(clickSpy).toHaveBeenCalledTimes(1);
      const anchor = appendSpy.mock.calls[0][0] as HTMLAnchorElement;
      expect(anchor.download).toBe('img 1.png');
    });

    it('surfaces the backend detail on failure', async () => {
      mockFetchBlob(404, { detail: 'no image' });
      await expect(downloadImageFile('i.png')).rejects.toThrow(/no image/);
    });

    it('falls back to a generic message when the error body is not JSON', async () => {
      const response = {
        ok: false,
        status: 500,
        statusText: 'mock',
        json: async () => {
          throw new SyntaxError('not json');
        },
        blob: async () => new Blob(['x']),
      } as unknown as Response;
      g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
      await expect(downloadImageFile('i.png')).rejects.toThrow(/Download failed/);
    });
  });
});
