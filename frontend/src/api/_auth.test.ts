import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getSessionToken,
  _setSessionTokenForTesting,
  apiFetch,
  wsUrlWithToken,
} from './_auth';

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

function okResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
    text: async () => '',
  } as unknown as Response;
}

function errorResponse(status: number, statusText: string): Response {
  return {
    ok: false,
    status,
    statusText,
    json: async () => ({}),
    text: async () => '',
  } as unknown as Response;
}

beforeEach(() => {
  originalFetch = g.fetch;
  // Ensure each test starts from a clean (unbootstrapped) state.
  _setSessionTokenForTesting(null);
});

afterEach(() => {
  g.fetch = originalFetch;
  _setSessionTokenForTesting(null);
  vi.restoreAllMocks();
});

describe('getSessionToken', () => {
  it('fetches the token from the bootstrap endpoint and caches it', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ token: 'abc123' }));
    g.fetch = fetchMock as unknown as typeof fetch;

    const token = await getSessionToken();
    expect(token).toBe('abc123');
    expect(fetchMock).toHaveBeenCalledWith('/api/auth/bootstrap');

    // Second call must hit the cache, not the network.
    const again = await getSessionToken();
    expect(again).toBe('abc123');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('returns the pre-seeded cached token without calling fetch', async () => {
    _setSessionTokenForTesting('seeded');
    const fetchMock = vi.fn();
    g.fetch = fetchMock as unknown as typeof fetch;

    await expect(getSessionToken()).resolves.toBe('seeded');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('coalesces concurrent calls into a single in-flight request', async () => {
    let resolveFetch!: (r: Response) => void;
    const fetchMock = vi.fn().mockReturnValue(
      new Promise<Response>((resolve) => {
        resolveFetch = resolve;
      }),
    );
    g.fetch = fetchMock as unknown as typeof fetch;

    const p1 = getSessionToken();
    const p2 = getSessionToken();
    // Both callers share the same inflight promise → one network request.
    expect(fetchMock).toHaveBeenCalledTimes(1);

    resolveFetch(okResponse({ token: 'shared' }));
    await expect(p1).resolves.toBe('shared');
    await expect(p2).resolves.toBe('shared');
  });

  it('throws and clears inflight when the bootstrap endpoint is not ok', async () => {
    const fetchMock = vi.fn().mockResolvedValue(errorResponse(403, 'Forbidden'));
    g.fetch = fetchMock as unknown as typeof fetch;

    await expect(getSessionToken()).rejects.toThrow(
      /Failed to bootstrap auth token: 403 Forbidden/,
    );

    // inflight was reset → a retry issues a fresh request (now succeeding).
    g.fetch = vi
      .fn()
      .mockResolvedValue(okResponse({ token: 'recovered' })) as unknown as typeof fetch;
    await expect(getSessionToken()).resolves.toBe('recovered');
  });

  it('throws and clears inflight when the response is missing a token', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ notToken: 1 }));
    g.fetch = fetchMock as unknown as typeof fetch;

    await expect(getSessionToken()).rejects.toThrow(/Bootstrap response missing token/);

    // A non-object body also fails the `typeof body?.token` guard.
    g.fetch = vi.fn().mockResolvedValue(okResponse(null)) as unknown as typeof fetch;
    await expect(getSessionToken()).rejects.toThrow(/Bootstrap response missing token/);
  });
});

describe('_setSessionTokenForTesting', () => {
  it('clears the cached token when passed null so the next call re-bootstraps', async () => {
    _setSessionTokenForTesting('first');
    await expect(getSessionToken()).resolves.toBe('first');

    _setSessionTokenForTesting(null);
    g.fetch = vi
      .fn()
      .mockResolvedValue(okResponse({ token: 'second' })) as unknown as typeof fetch;
    await expect(getSessionToken()).resolves.toBe('second');
  });
});

describe('apiFetch', () => {
  it('passes GET requests through unchanged without attaching the token header', async () => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch('/api/thing');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/thing');
    // GET passthrough: the (defaulted) init is forwarded as-is, with no
    // X-CodefyUI-Token header injected.
    const headers = new Headers(init?.headers);
    expect(headers.has('X-CodefyUI-Token')).toBe(false);
  });

  it('treats an explicit lowercase get method as non-mutating', async () => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch('/api/thing', { method: 'get' });
    const init = fetchMock.mock.calls[0][1];
    // No X-CodefyUI-Token header was attached for the read-only request.
    const headers = new Headers(init?.headers);
    expect(headers.has('X-CodefyUI-Token')).toBe(false);
  });

  it('attaches the session token header on mutating (POST) requests', async () => {
    _setSessionTokenForTesting('mut-token');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch('/api/thing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/thing');
    const headers = new Headers(init.headers);
    expect(headers.get('X-CodefyUI-Token')).toBe('mut-token');
    // Existing headers are preserved alongside the injected token.
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(init.body).toBe('{}');
  });

  it.each(['PUT', 'PATCH', 'DELETE'])(
    'attaches the token header on %s requests too',
    async (method) => {
      _setSessionTokenForTesting('mtok');
      const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
      g.fetch = fetchMock as unknown as typeof fetch;

      await apiFetch('/api/thing', { method });
      const headers = new Headers(fetchMock.mock.calls[0][1].headers);
      expect(headers.get('X-CodefyUI-Token')).toBe('mtok');
    },
  );

  it('bootstraps the token first when none is cached before a mutating request', async () => {
    // No token seeded → apiFetch must call getSessionToken → fetch bootstrap.
    const fetchMock = vi
      .fn()
      // First call: bootstrap. Second call: the actual POST.
      .mockResolvedValueOnce(okResponse({ token: 'bootstrapped' }))
      .mockResolvedValueOnce(okResponse({ done: true }));
    g.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch('/api/thing', { method: 'POST' });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/auth/bootstrap');
    const headers = new Headers(fetchMock.mock.calls[1][1].headers);
    expect(headers.get('X-CodefyUI-Token')).toBe('bootstrapped');
  });
});

describe('wsUrlWithToken', () => {
  let originalLocation: Location;

  beforeEach(() => {
    originalLocation = window.location;
  });

  afterEach(() => {
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      configurable: true,
    });
  });

  function setLocation(protocol: string, host: string) {
    Object.defineProperty(window, 'location', {
      value: { protocol, host },
      configurable: true,
    });
  }

  it('builds a ws:// URL with the token query param on http pages', async () => {
    _setSessionTokenForTesting('wstok');
    setLocation('http:', 'localhost:8000');

    const url = await wsUrlWithToken('/ws/execution');
    const parsed = new URL(url);
    expect(parsed.protocol).toBe('ws:');
    expect(parsed.host).toBe('localhost:8000');
    expect(parsed.pathname).toBe('/ws/execution');
    expect(parsed.searchParams.get('token')).toBe('wstok');
  });

  it('builds a wss:// URL on https pages', async () => {
    _setSessionTokenForTesting('securetok');
    setLocation('https:', 'example.com');

    const url = await wsUrlWithToken('/ws/execution');
    const parsed = new URL(url);
    expect(parsed.protocol).toBe('wss:');
    expect(parsed.searchParams.get('token')).toBe('securetok');
  });
});
