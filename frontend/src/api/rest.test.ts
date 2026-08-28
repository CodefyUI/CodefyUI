import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  exportGraph,
  fetchNodeDefinitions,
  fetchPresetDefinitions,
  fetchDevices,
  fetchHealth,
  validateGraph,
  saveGraph,
  loadGraph,
  listGraphs,
  resetWeights,
  createPreset,
  listExamples,
  loadExample,
  listPlugins,
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
  fetchCodexStatus,
  startCodexLogin,
  logoutCodex,
  getRun,
  listRuns,
  cancelRun,
  deleteRun,
  getRunEvents,
  getRunMetrics,
  getRunArtifacts,
  downloadRunMetricsCsv,
  ACTIVE_RUN_STATUSES,
  TERMINAL_RUN_STATUSES,
  listPacks,
  installPack,
  cancelPackJob,
  getPackJobEvents,
  removePackItem,
  PackApiError,
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

  // core#136 review, M-6. The exported script had no seed at all, so an
  // exported augmenting graph drew fresh entropy on every invocation while
  // the docs promised the same crops every time. The canvas seed now
  // travels with the export and becomes the script's `--seed` default.
  it('sends the canvas seed so the exported script reproduces the run', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    await exportGraph([], [], undefined, undefined, { seed: 4321, deterministic: true });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      nodes: [],
      edges: [],
      seed: 4321,
      deterministic: true,
    });
  });

  it('sends seed 0 rather than dropping it as falsy', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    await exportGraph([], [], undefined, undefined, { seed: 0, deterministic: false });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ nodes: [], edges: [], seed: 0 });
  });

  it('omits seed entirely when the canvas has none, so the export is unseeded', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    await exportGraph([], [], undefined, undefined, { seed: null, deterministic: false });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ nodes: [], edges: [] });
  });

  it('sends embedded preset definitions needed for portable expansion', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    const preset = { preset_name: 'Portable', nodes: [], edges: [] } as any;
    await exportGraph([], [], 'portable', [preset]);
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      nodes: [],
      edges: [],
      name: 'portable',
      presets: [preset],
    });
  });

  // core#137 review, MAJOR 2. `exportGraph` had no `subgraphs` parameter at
  // all, so Export -> Python posted a graph whose `subgraph:<id>` instance
  // nodes named definitions the request did not carry. The backend has no
  // registry to look them up in and answered 400 `Unknown subgraph: <id>`,
  // making every graph with a collapsed block un-exportable from the UI.
  it('sends subgraph definitions so a collapsed block can be expanded server-side', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    const definition = {
      id: 'blk',
      name: 'Block',
      description: '',
      nodes: [{ id: 'inner', type: 'Add', position: { x: 0, y: 0 }, data: { params: {} } }],
      edges: [],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    };
    const instance = {
      id: 'one', type: 'subgraph:blk', position: { x: 0, y: 0 }, data: { params: {} },
    };
    await exportGraph([instance], [], 'blocks', undefined, undefined, [definition]);
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      nodes: [instance],
      edges: [],
      name: 'blocks',
      subgraphs: [definition],
    });
  });

  it('omits subgraphs when the graph has no blocks, so the body is unchanged', async () => {
    const fetchMock = mockFetch(200, { script: '...' });
    await exportGraph([], [], 'plain', undefined, undefined, []);
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ nodes: [], edges: [], name: 'plain' });
  });

  it('throws when the endpoint fails', async () => {
    mockFetch(500, {});
    await expect(exportGraph([], [], 'x')).rejects.toThrow(/Export failed/);
  });

  it('surfaces backend validation details when export fails', async () => {
    mockFetch(400, { detail: ['Unknown node type: Missing', 'No start node defined'] });
    await expect(exportGraph([], [], 'x')).rejects.toThrow(
      /Unknown node type: Missing; No start node defined/,
    );
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
      name: 'listPlugins',
      fn: () => listPlugins(),
      url: '/api/plugins',
      errorRe: /Failed to list plugins/,
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

// fetchHealth is NOT a plain passthrough GET (unlike the `cases` table
// above): it normalizes the backend's project-mode-only `project` key so
// callers never see `undefined`. Covered separately so that distinction is
// pinned explicitly (Task 12 controller item 1).
describe('fetchHealth', () => {
  it('passes through the project path string when present (project mode)', async () => {
    const fetchMock = mockFetch(200, {
      status: 'ok', version: '2.2.0', nodes_loaded: 3, presets_loaded: 1,
      caches: { execution_cache: { instances: 1, entries: 2, bytes: 2048, max_bytes_each: 4096 } },
      project: '/home/me/my-service',
    });
    const out = await fetchHealth();
    expect(fetchMock).toHaveBeenCalledWith('/api/health');
    expect(out).toEqual({
      status: 'ok', version: '2.2.0', nodes_loaded: 3, presets_loaded: 1,
      caches: { execution_cache: { instances: 1, entries: 2, bytes: 2048, max_bytes_each: 4096 } },
      project: '/home/me/my-service',
    });
  });

  it('normalizes an ABSENT project key (non-project mode) to null, never undefined', async () => {
    mockFetch(200, { status: 'ok', nodes_loaded: 3, presets_loaded: 1 });
    const out = await fetchHealth();
    expect(out.project).toBeNull();
    expect('project' in out).toBe(true);
  });

  // #193 item 2: the settings panel maps over `caches` on every render and
  // prints `version` into a value column, so neither may arrive as undefined.
  it('normalizes an absent caches block to an empty map and an absent version to null', async () => {
    mockFetch(200, { status: 'ok', nodes_loaded: 3, presets_loaded: 1 });
    const out = await fetchHealth();
    expect(out.caches).toEqual({});
    expect(out.version).toBeNull();
  });

  it('passes the per-store cache usage through untouched', async () => {
    // Each store reports a DIFFERENT set of keys (see CacheUsage); the client
    // must not narrow them to some common subset.
    mockFetch(200, {
      status: 'ok', version: '2.2.0', nodes_loaded: 3, presets_loaded: 1,
      caches: {
        execution_cache: { instances: 2, entries: 5, bytes: 1024, max_bytes_each: 2048 },
        run_output_store: { runs: 1, max_runs: 20, bytes: 4096, max_bytes: 8192 },
        node_state_store: { modules: 3, max_modules: 64, bytes: 512, max_bytes: 1024 },
      },
    });
    const out = await fetchHealth();
    expect(out.caches.run_output_store).toEqual({
      runs: 1, max_runs: 20, bytes: 4096, max_bytes: 8192,
    });
    expect(Object.keys(out.caches)).toEqual([
      'execution_cache', 'run_output_store', 'node_state_store',
    ]);
  });

  // The Package Center's restart flow polls /api/health to tell "the server
  // answered again" from "a NEW server came up": only a changed boot_id
  // proves the second one, so the field has to survive the normalization.
  it('exposes boot_id when present, and undefined when the server predates it', async () => {
    mockFetch(200, {
      status: 'ok', nodes_loaded: 3, presets_loaded: 1, boot_id: 'boot-abc',
    });
    expect((await fetchHealth()).boot_id).toBe('boot-abc');

    mockFetch(200, { status: 'ok', nodes_loaded: 3, presets_loaded: 1 });
    expect((await fetchHealth()).boot_id).toBeUndefined();
  });

  it('throws on a non-ok response', async () => {
    mockFetch(500, {});
    await expect(fetchHealth()).rejects.toThrow(/Health failed/);
  });
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
    expect(JSON.parse(init.body)).toEqual({
      nodes: [{ id: 'a' }],
      edges: [{ id: 'e' }],
      presets: [],
      // core#137: definitions are graph-local, so validation only sees inside
      // a subgraph if the request carries them.
      subgraphs: [],
    });
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
  });

  it('POSTs graph-embedded presets so portable graphs validate (#84)', async () => {
    const fetchMock = mockFetch(200, { valid: true });
    const presets = [{ preset_name: 'EmbeddedPr' }];
    await validateGraph([{ id: 'a' }], [], presets);
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body).presets).toEqual(presets);
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


describe('Codex auth endpoints', () => {
  it('fetchCodexStatus GETs the status endpoint', async () => {
    const fetchMock = mockFetch(200, { status: 'logged_in', email: 'me@example.com' });
    await expect(fetchCodexStatus()).resolves.toEqual({
      status: 'logged_in',
      email: 'me@example.com',
    });
    expect(fetchMock).toHaveBeenCalledWith('/api/llm/codex/status');
  });

  it('fetchCodexStatus throws on failure', async () => {
    mockFetch(500, {});
    await expect(fetchCodexStatus()).rejects.toThrow(/Codex status failed/);
  });

  it('startCodexLogin POSTs with the auth token and returns auth_url', async () => {
    const fetchMock = mockFetch(200, { auth_url: 'https://auth.example' });
    await expect(startCodexLogin()).resolves.toEqual({ auth_url: 'https://auth.example' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/llm/codex/login');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
  });

  it('logoutCodex POSTs with the auth token', async () => {
    const fetchMock = mockFetch(200, { status: 'logged_out' });
    await expect(logoutCodex()).resolves.toEqual({ status: 'logged_out' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/llm/codex/logout');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
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

// ── Runs (#120/#123/#124) ────────────────────────────────────────────────

describe('run endpoints', () => {
  it('exposes status vocabularies that mirror the backend', () => {
    expect([...ACTIVE_RUN_STATUSES]).toEqual(['queued', 'running']);
    expect([...TERMINAL_RUN_STATUSES]).toEqual([
      'succeeded', 'failed', 'cancelled', 'interrupted',
    ]);
  });

  describe('getRun', () => {
    it('returns null for a run the server has never heard of', async () => {
      mockFetch(404, {});
      expect(await getRun('gone')).toBeNull();
    });

    it('throws on any other failure', async () => {
      mockFetch(500, {});
      await expect(getRun('x')).rejects.toThrow(/Failed to fetch run/);
    });
  });

  describe('listRuns', () => {
    // The run readers go through `apiFetch` (#132), which normalises the
    // init argument, so these assert the URL rather than the whole call.
    it('sends no query at all when nothing is narrowed', async () => {
      const fetchMock = mockFetch(200, { runs: [], total: 0, limit: 50, offset: 0 });
      await listRuns();
      expect(fetchMock.mock.calls[0][0]).toBe('/api/runs');
    });

    it('repeats ?status= per status rather than joining with commas', async () => {
      const fetchMock = mockFetch(200, { runs: [], total: 0, limit: 50, offset: 0 });
      await listRuns({ status: ['queued', 'running'], limit: 10, offset: 20 });
      expect(fetchMock.mock.calls[0][0]).toBe(
        '/api/runs?status=queued&status=running&limit=10&offset=20',
      );
    });

    it('throws when the list fails', async () => {
      mockFetch(503, {});
      await expect(listRuns()).rejects.toThrow(/Failed to list runs/);
    });
  });

  describe('cancelRun', () => {
    it('POSTs with the session token and returns the outcome', async () => {
      const fetchMock = mockFetch(200, { run_id: 'r1', status: 'running', cancelled: true });
      expect(await cancelRun('r1')).toEqual({
        run_id: 'r1', status: 'running', cancelled: true,
      });
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe('/api/runs/r1/cancel');
      expect(init.method).toBe('POST');
      expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    });

    it('surfaces the backend detail on failure', async () => {
      mockFetch(404, { detail: 'run not found' });
      await expect(cancelRun('r1')).rejects.toThrow(/run not found/);
    });

    it('falls back to the status text when the error body is not JSON', async () => {
      mockFetchJsonThrows(500);
      await expect(cancelRun('r1')).rejects.toThrow(/Cancel failed/);
    });
  });

  describe('deleteRun', () => {
    it('DELETEs the run row with the session token', async () => {
      const fetchMock = mockFetch(200, { run_id: 'r1', deleted: true });
      expect(await deleteRun('r1')).toEqual({ run_id: 'r1', deleted: true });
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe('/api/runs/r1');
      expect(init.method).toBe('DELETE');
      expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    });

    it('surfaces the 409 the server sends for a live run', async () => {
      mockFetch(409, { detail: 'queued or running; cancel it first' });
      await expect(deleteRun('r1')).rejects.toThrow(/cancel it first/);
    });

    it('falls back to the status text when the error body is not JSON', async () => {
      mockFetchJsonThrows(500);
      await expect(deleteRun('r1')).rejects.toThrow(/Delete failed/);
    });
  });

  describe('getRunEvents', () => {
    it('defaults to no query and passes the abort signal through', async () => {
      const fetchMock = mockFetch(200, { events: [], cursor: 0 });
      const controller = new AbortController();
      await getRunEvents('r1', { signal: controller.signal });
      expect(fetchMock).toHaveBeenCalledWith('/api/runs/r1/events', {
        signal: controller.signal,
      });
    });

    it('carries the cursor, the long-poll wait and the page limit', async () => {
      const fetchMock = mockFetch(200, { events: [], cursor: 7 });
      await getRunEvents('r1', { cursor: 7, wait: 25, limit: 100 });
      expect(fetchMock.mock.calls[0][0]).toBe(
        '/api/runs/r1/events?cursor=7&wait=25&limit=100',
      );
    });

    it('throws when the run vanishes mid-follow', async () => {
      mockFetch(404, {});
      await expect(getRunEvents('r1')).rejects.toThrow(/Failed to fetch run events/);
    });
  });

  describe('getRunMetrics', () => {
    it('fetches every series, or one by name', async () => {
      const fetchMock = mockFetch(200, { run_id: 'r1', names: [], metrics: [] });
      await getRunMetrics('r1');
      expect(fetchMock.mock.calls[0][0]).toBe('/api/runs/r1/metrics');
      await getRunMetrics('r1', 'train loss');
      expect(fetchMock.mock.calls[1][0]).toBe('/api/runs/r1/metrics?name=train%20loss');
    });

    it('throws when metrics are unavailable', async () => {
      mockFetch(500, {});
      await expect(getRunMetrics('r1')).rejects.toThrow(/Failed to fetch run metrics/);
    });
  });

  describe('getRunArtifacts', () => {
    it('fetches all artifacts, or one kind', async () => {
      const fetchMock = mockFetch(200, { run_id: 'r1', artifacts: [] });
      await getRunArtifacts('r1');
      expect(fetchMock.mock.calls[0][0]).toBe('/api/runs/r1/artifacts');
      await getRunArtifacts('r1', 'checkpoint');
      expect(fetchMock.mock.calls[1][0]).toBe('/api/runs/r1/artifacts?kind=checkpoint');
    });

    it('throws when artifacts are unavailable', async () => {
      mockFetch(404, {});
      await expect(getRunArtifacts('r1')).rejects.toThrow(/Failed to fetch run artifacts/);
    });
  });

  describe('downloadRunMetricsCsv', () => {
    let createObjectURL: ReturnType<typeof vi.fn>;
    let revokeObjectURL: ReturnType<typeof vi.fn>;
    let clickSpy: ReturnType<typeof vi.spyOn>;

    function mockFetchBlob(status: number, body: unknown) {
      const response = {
        ok: status >= 200 && status < 300,
        status,
        statusText: 'mock',
        json: async () => body,
        blob: async () => new Blob(['run_id,name\n']),
      } as unknown as Response;
      g.fetch = vi.fn().mockResolvedValue(response) as unknown as typeof fetch;
      return g.fetch as unknown as ReturnType<typeof vi.fn>;
    }

    beforeEach(() => {
      createObjectURL = vi.fn().mockReturnValue('blob:mock-url');
      revokeObjectURL = vi.fn();
      (URL as unknown as { createObjectURL: unknown }).createObjectURL = createObjectURL;
      (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = revokeObjectURL;
      clickSpy = vi
        .spyOn(HTMLAnchorElement.prototype, 'click')
        .mockImplementation(() => {});
    });

    afterEach(() => {
      clickSpy.mockRestore();
      delete (URL as unknown as { createObjectURL?: unknown }).createObjectURL;
      delete (URL as unknown as { revokeObjectURL?: unknown }).revokeObjectURL;
    });

    it('asks for CSV and saves it under a run-scoped filename', async () => {
      const appendSpy = vi.spyOn(document.body, 'appendChild');
      const fetchMock = mockFetchBlob(200, null);
      await downloadRunMetricsCsv('abc123');
      expect(fetchMock).toHaveBeenCalledWith('/api/runs/abc123/metrics?format=csv');
      const anchor = appendSpy.mock.calls[0][0] as HTMLAnchorElement;
      expect(anchor.download).toBe('run-abc123-metrics.csv');
      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
      expect(document.body.contains(anchor)).toBe(false);
      appendSpy.mockRestore();
    });

    it('surfaces the backend detail rather than downloading an error page', async () => {
      mockFetchBlob(404, { detail: 'run not found' });
      await expect(downloadRunMetricsCsv('nope')).rejects.toThrow(/run not found/);
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
      await expect(downloadRunMetricsCsv('r1')).rejects.toThrow(/Download failed/);
    });
  });
});

// ── Optional packs / Package Center ──────────────────────────────────────

describe('pack endpoints', () => {
  /** Await a call that must fail, and hand back the PackApiError it threw. */
  async function packError(call: Promise<unknown>): Promise<PackApiError> {
    const err = await call.then(() => null, (e: unknown) => e);
    expect(err).toBeInstanceOf(PackApiError);
    return err as PackApiError;
  }

  describe('listPacks', () => {
    // The panel polls this route and maps over `packs`, `items`, `pip` and
    // `blocked_by` on every render, so an absent key has to arrive as an
    // empty list rather than as a crash halfway through a repaint.
    it('normalizes a sparse payload', async () => {
      const fetchMock = mockFetch(200, {
        packs: [{
          id: 'word-vectors', title: 'Word vectors',
          description: 'GloVe embeddings', install_mode: 'live',
        }],
        launch_mode: 'systemd',
      });
      const out = await listPacks();
      expect(fetchMock).toHaveBeenCalledWith('/api/packs');
      expect(out).toEqual({
        packs: [{
          id: 'word-vectors',
          title: 'Word vectors',
          description: 'GloVe embeddings',
          install_mode: 'live',
          status: 'not_installed',
          pip_ready: false,
          usable: false,
          depends_on: [],
          blocked_by: [],
          pip: [],
          items: [],
          size_bytes_total: 0,
          install_command: null,
        }],
        active_job: null,
        last_restart_job: null,
        // Absent is read as "allowed": the server is what actually refuses a
        // remote install, with a 403 the panel then reports.
        remote_install_allowed: true,
        // A launch mode outside {start, dev} is one nothing here can act on.
        launch_mode: 'unknown',
        gpu: null,
      });
    });

    // The other half of normalizing: a mapped payload is also a payload a
    // forgotten key can silently vanish from.
    it('carries a full payload through unchanged', async () => {
      mockFetch(200, {
        packs: [{
          id: 'gpu-torch', title: 'GPU PyTorch', description: 'CUDA wheels',
          install_mode: 'restart', status: 'partial', pip_ready: true,
          usable: false, depends_on: ['word-vectors'], blocked_by: ['word-vectors'],
          pip: [{ spec: 'torch==2.6.0' }],
          items: [{
            id: 'glove-6b-100d', kind: 'hf', repo_id: 'stanfordnlp/glove',
            url: null, size_bytes: 347116733, license: 'PDDL', status: 'downloading',
          }],
          size_bytes_total: 347116733,
          install_command: 'cdui install --gpu cu128',
        }],
        active_job: { job_id: 'j1', pack_id: 'gpu-torch' },
        last_restart_job: { job_id: 'j0', status: 'done' },
        remote_install_allowed: false,
        launch_mode: 'dev',
        gpu: {
          detected_label: 'NVIDIA GeForce RTX 4080', recommended_variant: 'cu128',
          installed_variant: 'cpu', variants: ['cpu', 'cu128'],
          install_command: 'cdui install --gpu cu128',
        },
      });
      const out = await listPacks();
      expect(out.packs[0].items[0].status).toBe('downloading');
      expect(out.packs[0].pip).toEqual([{ spec: 'torch==2.6.0' }]);
      expect(out.packs[0].blocked_by).toEqual(['word-vectors']);
      expect(out.active_job).toEqual({ job_id: 'j1', pack_id: 'gpu-torch' });
      expect(out.last_restart_job).toEqual({ job_id: 'j0', status: 'done' });
      expect(out.remote_install_allowed).toBe(false);
      expect(out.launch_mode).toBe('dev');
      expect(out.gpu?.recommended_variant).toBe('cu128');
    });

    it('throws PackApiError with the status on 404', async () => {
      mockFetch(404, { detail: 'Package Center is not available' });
      const err = await packError(listPacks());
      expect(err.status).toBe(404);
      expect(err.message).toBe('Package Center is not available');
    });
  });

  describe('installPack', () => {
    it('posts the token header and omits undefined body keys', async () => {
      const fetchMock = mockFetch(202, { job_id: 'j1' });
      // `items: undefined` is what a caller spreading a partly filled options
      // object sends; it must not reach the wire as `items: null`.
      expect(
        await installPack('gpu-torch', {
          items: undefined, mode: 'restart', variant: 'cu128',
        }),
      ).toEqual({ job_id: 'j1' });
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe('/api/packs/gpu-torch/install');
      expect(init.method).toBe('POST');
      expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
      // No `items` key at all: omitting it is what means "the whole pack",
      // and the backend's InstallRequest forbids anything it did not declare.
      expect(JSON.parse(init.body)).toEqual({ mode: 'restart', variant: 'cu128' });
    });

    it('sends an empty object when the caller narrows nothing', async () => {
      const fetchMock = mockFetch(202, { job_id: 'j1' });
      await installPack('word-vectors');
      expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({});
    });

    it('surfaces 409 as PackApiError carrying the body', async () => {
      const body = { detail: 'an install is already running', job_id: 'j-running' };
      mockFetch(409, body);
      const err = await packError(installPack('word-vectors'));
      expect(err.status).toBe(409);
      expect(err.message).toBe('an install is already running');
      // The panel offers "follow that job instead", which needs its id.
      expect(err.body).toEqual(body);
    });

    it('falls back to the status text when the error body is not JSON', async () => {
      mockFetchJsonThrows(500);
      const err = await packError(installPack('word-vectors'));
      expect(err.status).toBe(500);
      expect(err.message).toBe('mock');
      expect(err.body).toBeNull();
    });
  });

  describe('cancelPackJob', () => {
    it('POSTs to the job with the session token', async () => {
      const fetchMock = mockFetch(200, { job_id: 'j1', cancelled: true });
      expect(await cancelPackJob('j1')).toEqual({ job_id: 'j1', cancelled: true });
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe('/api/packs/jobs/j1/cancel');
      expect(init.method).toBe('POST');
      expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    });

    it('throws PackApiError when the job is already forgotten', async () => {
      mockFetch(404, { detail: "job 'j1' not found" });
      expect((await packError(cancelPackJob('j1'))).status).toBe(404);
    });
  });

  describe('getPackJobEvents', () => {
    it('passes cursor, wait, limit and the signal', async () => {
      const fetchMock = mockFetch(200, {
        job_id: 'j1', status: 'running', events: [], cursor: 7,
      });
      const controller = new AbortController();
      await getPackJobEvents('j1', {
        cursor: 7, wait: 25, limit: 100, signal: controller.signal,
      });
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/packs/jobs/j1/events?cursor=7&wait=25&limit=100',
        { signal: controller.signal },
      );
    });

    it('sends no query when following from the beginning', async () => {
      const fetchMock = mockFetch(200, {
        job_id: 'j1', status: 'done', events: [], cursor: 0,
      });
      const page = await getPackJobEvents('j1');
      expect(fetchMock).toHaveBeenCalledWith('/api/packs/jobs/j1/events', {
        signal: undefined,
      });
      expect(page.status).toBe('done');
    });

    it('throws PackApiError when the job is gone mid-follow', async () => {
      mockFetch(404, { detail: "job 'j1' not found" });
      expect((await packError(getPackJobEvents('j1'))).status).toBe(404);
    });
  });

  describe('removePackItem', () => {
    it('issues DELETE', async () => {
      const fetchMock = mockFetch(200, {
        pack_id: 'word-vectors', item_id: 'glove-6b-100d', removed: true,
      });
      expect(await removePackItem('word-vectors', 'glove-6b-100d')).toEqual({
        pack_id: 'word-vectors', item_id: 'glove-6b-100d', removed: true,
      });
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe('/api/packs/word-vectors/items/glove-6b-100d');
      expect(init.method).toBe('DELETE');
      expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    });

    it('surfaces the 409 sent while a job targets the pack', async () => {
      mockFetch(409, { detail: 'an install is running', job_id: 'j1' });
      const err = await packError(removePackItem('word-vectors', 'glove-6b-100d'));
      expect(err.status).toBe(409);
      expect(err.body).toEqual({ detail: 'an install is running', job_id: 'j1' });
    });
  });
});
