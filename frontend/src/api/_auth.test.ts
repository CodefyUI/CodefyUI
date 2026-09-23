import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getSessionToken,
  invalidateSessionToken,
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

/**
 * The server mints a new token every time its process starts, and a browser
 * tab outlives a restart — the Package Center restarts the server itself to
 * finish a pack that was already imported. Without a retry, every POST from
 * that tab (install a plugin, install a pack, run a graph) answers 403 until
 * the user reloads the page.
 */
describe('apiFetch after the server rotates its token', () => {
  it('re-bootstraps and replays the request once on a 403', async () => {
    _setSessionTokenForTesting('stale');
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(errorResponse(403, 'Forbidden'))
      .mockResolvedValueOnce(okResponse({ token: 'fresh' }))
      .mockResolvedValueOnce(okResponse({ done: true }));
    g.fetch = fetchMock as unknown as typeof fetch;

    const res = await apiFetch('/api/plugins/install', {
      method: 'POST',
      body: '{"inspection_id":"i1"}',
    });

    expect(res.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get('X-CodefyUI-Token'))
      .toBe('stale');
    expect(fetchMock.mock.calls[1][0]).toBe('/api/auth/bootstrap');
    const replay = fetchMock.mock.calls[2];
    expect(replay[0]).toBe('/api/plugins/install');
    expect(new Headers(replay[1].headers).get('X-CodefyUI-Token')).toBe('fresh');
    expect(replay[1].body).toBe('{"inspection_id":"i1"}');
  });

  it('keeps the 403 when the token has not changed', async () => {
    // A refusal that is about the request, not the token: the server refuses
    // a remote plugin or pack install with 403 as well. One request, not two.
    _setSessionTokenForTesting('same');
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(errorResponse(403, 'Forbidden'))
      .mockResolvedValueOnce(okResponse({ token: 'same' }));
    g.fetch = fetchMock as unknown as typeof fetch;

    const res = await apiFetch('/api/plugins/install', { method: 'POST' });

    expect(res.status).toBe(403);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('keeps the 403 when the bootstrap endpoint is unreachable', async () => {
    _setSessionTokenForTesting('stale');
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(errorResponse(403, 'Forbidden'))
      .mockRejectedValueOnce(new Error('network down'));
    g.fetch = fetchMock as unknown as typeof fetch;

    const res = await apiFetch('/api/plugins/install', { method: 'POST' });
    expect(res.status).toBe(403);
  });

  it('leaves other error statuses alone', async () => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(errorResponse(500, 'Server Error'));
    g.fetch = fetchMock as unknown as typeof fetch;

    const res = await apiFetch('/api/plugins/install', { method: 'POST' });
    expect(res.status).toBe(500);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does not replay a request whose body is a stream', async () => {
    // The first request consumed it; a second would throw instead of sending.
    _setSessionTokenForTesting('stale');
    const fetchMock = vi.fn().mockResolvedValue(errorResponse(403, 'Forbidden'));
    g.fetch = fetchMock as unknown as typeof fetch;

    const body = new ReadableStream();
    const res = await apiFetch('/api/plugins/install', { method: 'POST', body });
    expect(res.status).toBe(403);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

/**
 * Plugins reach `apiFetch` through `api.http.fetch` with any URL they like, so
 * the token stops at the page's own origin here, where it is attached (#482).
 * This keeps a well-meaning plugin from sending the token to another server
 * by accident; plugin code still runs in the editor page and can reach the
 * token another way. The jsdom page is http://localhost:3000.
 */
describe('apiFetch sends the token only to its own origin', () => {
  it('refuses a POST to another origin before bootstrapping, and sends nothing', async () => {
    // No token cached, so the refusal has to come before the bootstrap GET too.
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ token: 'tok' }));
    g.fetch = fetchMock as unknown as typeof fetch;

    const sent = apiFetch('https://api.example.com/v1/chat', { method: 'POST', body: '{}' });

    await expect(sent).rejects.toThrow(/api\.example\.com/);
    // What `fetch` itself rejects with, so a caller's existing catch still fits.
    await expect(sent).rejects.toBeInstanceOf(TypeError);
    // And it says what to use instead -- a path for this server, which is the
    // fix when the plugin only spelled this server another way (`localhost`
    // for `127.0.0.1`), and `window.fetch` for a server that really is another.
    await expect(sent).rejects.toThrow(
      /Use a path such as \/api\/\.\.\. for this server and window\.fetch for other servers\./,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each(['PUT', 'PATCH', 'DELETE'])('refuses a %s to another origin', async (method) => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await expect(apiFetch('https://api.example.com/v1/items/1', { method }))
      .rejects.toThrow(/api\.example\.com/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refuses a protocol-relative POST, which leaves the page origin', async () => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await expect(apiFetch('//api.example.com/x', { method: 'POST' }))
      .rejects.toThrow(/api\.example\.com/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.skipIf(typeof Request === 'undefined')(
    'refuses a Request aimed at another origin, which untyped plugin code can pass',
    async () => {
      _setSessionTokenForTesting('tok');
      const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
      g.fetch = fetchMock as unknown as typeof fetch;

      const request = new Request('https://api.example.com/v1/chat');
      await expect(apiFetch(request as never, { method: 'POST' }))
        .rejects.toThrow(/api\.example\.com/);
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it('sends the URL it checked, even if a URL object is changed while the token loads', async () => {
    // The check runs before the token bootstrap and the send after it. A URL
    // object read twice would let a host changed in between receive the token.
    let resolveBootstrap!: (r: Response) => void;
    const fetchMock = vi.fn()
      .mockReturnValueOnce(new Promise<Response>((resolve) => { resolveBootstrap = resolve; }))
      .mockResolvedValueOnce(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;
    const url = new URL('/api/thing', window.location.origin);

    const sent = apiFetch(url as never, { method: 'POST' });
    // Checked, and the bootstrap GET is out: the only call so far.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    url.host = 'api.example.com';
    resolveBootstrap(okResponse({ token: 'tok' }));
    await sent;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [target, init] = fetchMock.mock.calls[1];
    expect(String(target)).toBe(`${window.location.origin}/api/thing`);
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('tok');
  });

  // A `Request` carries its own method, headers and body, and `fetch` uses
  // them when `init` names none. Reading the method from `init` alone took a
  // Request's POST for a GET: sent without the token to this server (a 403),
  // and sent unrefused to any other.
  it.skipIf(typeof Request === 'undefined')(
    'sends the token with a same-origin Request that carries its own POST, keeping its headers',
    async () => {
      _setSessionTokenForTesting('tok');
      const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
      g.fetch = fetchMock as unknown as typeof fetch;
      const request = new Request(`${window.location.origin}/api/thing`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
      });

      await apiFetch(request as never);

      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [target, init] = fetchMock.mock.calls[0];
      expect(target).toBe(request);
      const headers = new Headers(init.headers);
      expect(headers.get('X-CodefyUI-Token')).toBe('tok');
      // `init.headers` replaces a Request's own, so they have to be carried over.
      expect(headers.get('Content-Type')).toBe('application/json');
    },
  );

  it.skipIf(typeof Request === 'undefined')(
    'refuses a Request that carries its own POST to another origin',
    async () => {
      _setSessionTokenForTesting('tok');
      const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
      g.fetch = fetchMock as unknown as typeof fetch;
      const request = new Request('https://api.example.com/v1/chat', { method: 'POST' });

      await expect(apiFetch(request as never)).rejects.toThrow(/POST to https:\/\/api\.example\.com/);
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it.skipIf(typeof Request === 'undefined')(
    'keeps the 403 when the first send used up a Request body, and drops the token',
    async () => {
      // Sending a Request uses up its body. A replay after the 403 would reject
      // with a TypeError, which is what this fetch stand-in does, like `fetch`.
      // The 403 may still mean the server restarted, so the token is dropped
      // and the next call bootstraps again.
      _setSessionTokenForTesting('stale');
      const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
        if (!(input instanceof Request)) return okResponse({ token: 'fresh' });
        if (input.bodyUsed) throw new TypeError('body used already');
        await input.text();
        return errorResponse(403, 'Forbidden');
      });
      g.fetch = fetchMock as unknown as typeof fetch;
      const request = new Request(`${window.location.origin}/api/thing`, {
        method: 'POST', body: '{}',
      });

      const res = await apiFetch(request as never);

      expect(res.status).toBe(403);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      // It went out as the POST it is, with the token, not as a bare GET.
      expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get('X-CodefyUI-Token'))
        .toBe('stale');

      await apiFetch('/api/thing', { method: 'POST' });

      expect(fetchMock).toHaveBeenCalledTimes(3);
      expect(fetchMock.mock.calls[1][0]).toBe('/api/auth/bootstrap');
      expect(new Headers(fetchMock.mock.calls[2][1]?.headers).get('X-CodefyUI-Token'))
        .toBe('fresh');
    },
  );

  it('still sends the token with a POST to its own origin written out in full', async () => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch(`${window.location.origin}/api/thing`, { method: 'POST' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get('X-CodefyUI-Token'))
      .toBe('tok');
  });

  it('still sends the token with a POST to a relative path with no leading slash', async () => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch('api/thing', { method: 'POST' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get('X-CodefyUI-Token'))
      .toBe('tok');
  });

  it('lets a GET go to another origin, without the token', async () => {
    _setSessionTokenForTesting('tok');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({}));
    g.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch('https://api.example.com/x');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('https://api.example.com/x');
    expect(new Headers(init?.headers).has('X-CodefyUI-Token')).toBe(false);
  });
});

describe('invalidateSessionToken', () => {
  it('makes the next call re-read the bootstrap endpoint', async () => {
    _setSessionTokenForTesting('old');
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ token: 'new' }));
    g.fetch = fetchMock as unknown as typeof fetch;

    invalidateSessionToken();
    await expect(getSessionToken()).resolves.toBe('new');
    expect(fetchMock).toHaveBeenCalledWith('/api/auth/bootstrap');
  });
});

/**
 * A bootstrap that failed, or that was dropped, is never handed out again.
 * The bootstrap GET fails outright while the server restarts; a failure kept
 * in the in-flight slot answered every later mutating request with the same
 * error until the page was reloaded.
 */
describe('a failed or dropped bootstrap is not kept', () => {
  it('a POST after a bootstrap that failed outright bootstraps again, with the new token', async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(okResponse({ token: 'fresh' }))
      .mockResolvedValueOnce(okResponse({ done: true }));
    g.fetch = fetchMock as unknown as typeof fetch;

    await expect(apiFetch('/api/thing', { method: 'POST' })).rejects.toThrow(/Failed to fetch/);
    const res = await apiFetch('/api/thing', { method: 'POST' });

    expect(res.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[1][0]).toBe('/api/auth/bootstrap');
    expect(new Headers(fetchMock.mock.calls[2][1].headers).get('X-CodefyUI-Token'))
      .toBe('fresh');
  });

  it('a bootstrap answer that cannot be read is not kept either', async () => {
    const unreadable = {
      ok: true, status: 200, statusText: 'OK',
      json: async () => { throw new SyntaxError('Unexpected token < in JSON'); },
    } as unknown as Response;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(unreadable)
      .mockResolvedValueOnce(okResponse({ token: 'fresh' }));
    g.fetch = fetchMock as unknown as typeof fetch;

    await expect(getSessionToken()).rejects.toThrow(/Unexpected token/);
    await expect(getSessionToken()).resolves.toBe('fresh');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('a bootstrap dropped by invalidateSessionToken does not fill the cache when it lands', async () => {
    // It was sent before the restart was noticed, so it reads the old token.
    // Its own caller still gets that answer; the cache must not.
    let resolveStale!: (r: Response) => void;
    const fetchMock = vi.fn()
      .mockReturnValueOnce(new Promise<Response>((resolve) => { resolveStale = resolve; }))
      .mockResolvedValueOnce(okResponse({ token: 'new' }));
    g.fetch = fetchMock as unknown as typeof fetch;

    const stale = getSessionToken();
    invalidateSessionToken();
    await expect(getSessionToken()).resolves.toBe('new');
    resolveStale(okResponse({ token: 'old' }));
    await expect(stale).resolves.toBe('old');

    await expect(getSessionToken()).resolves.toBe('new');
    expect(fetchMock).toHaveBeenCalledTimes(2);
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
