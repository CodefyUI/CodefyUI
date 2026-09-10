/**
 * Session-token bootstrap + fetch wrapper.
 *
 * The backend (see backend/app/core/auth.py) generates a fresh URL-safe token
 * per process. Mutating requests under /api/* must echo that token back in
 * the X-CodefyUI-Token header; otherwise the auth_guard middleware returns
 * 403. The frontend grabs the token on init via /api/auth/bootstrap (a GET
 * route that's only reachable when the Host header is whitelisted by the
 * host_guard middleware — so DNS-rebinding attackers can't read it).
 *
 * Why a wrapper instead of monkey-patching window.fetch:
 *  - Keeps test-mocking ergonomic (vitest can replace this module wholesale).
 *  - Lets us skip the header on GETs and read-only routes so we don't pollute
 *    the request payload unnecessarily.
 */

const TOKEN_HEADER = 'X-CodefyUI-Token';
const BOOTSTRAP_URL = '/api/auth/bootstrap';
const MUTATING_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

let cachedToken: string | null = null;
let inflight: Promise<string> | null = null;

/**
 * Fetch the session token from the backend (cached after first call).
 *
 * Throws if the bootstrap endpoint is unreachable — the rest of the app
 * cannot make mutating requests until this resolves. Callers should `await`
 * this once at app startup; subsequent calls return the cached value.
 */
export async function getSessionToken(): Promise<string> {
  if (cachedToken !== null) return cachedToken;
  if (inflight !== null) return inflight;

  inflight = (async () => {
    const res = await fetch(BOOTSTRAP_URL);
    if (!res.ok) {
      inflight = null;
      throw new Error(
        `Failed to bootstrap auth token: ${res.status} ${res.statusText}`,
      );
    }
    const body = await res.json();
    if (typeof body?.token !== 'string') {
      inflight = null;
      throw new Error('Bootstrap response missing token');
    }
    cachedToken = body.token;
    return body.token as string;
  })();
  return inflight;
}

/**
 * Drop the cached token so the next call re-reads /api/auth/bootstrap.
 *
 * The backend mints a new token every time its process starts (see
 * `auth.py`), and a browser tab outlives a restart: the Package Center
 * restarts the server itself to finish a pack that was already imported, and
 * `cdui start` after a stop is the same event. From that moment the tab holds
 * a token the server has never heard of, so every POST — install a plugin,
 * install a pack, run a graph — comes back 403, and the WebSocket handshake
 * is refused on every reconnect attempt. Re-reading the bootstrap endpoint is
 * a local GET and gets the tab back in step.
 */
export function invalidateSessionToken(): void {
  cachedToken = null;
  inflight = null;
}

/**
 * Test-only escape hatch. Vitest setup pre-populates the token so we don't
 * have to mock the bootstrap endpoint in every test file.
 */
export function _setSessionTokenForTesting(token: string | null): void {
  cachedToken = token;
  inflight = null;
}

/**
 * Drop-in replacement for ``fetch(url, init)`` that auto-attaches the session
 * token header on mutating requests. GET / HEAD / OPTIONS are passed through
 * unchanged.
 *
 * A 403 gets one retry with a freshly bootstrapped token, because the token
 * this tab cached is refused verbatim after the server restarts — see
 * {@link invalidateSessionToken}. The retry only goes out when the new token
 * differs from the one just refused: 403 is also how the server refuses a
 * remote plugin or pack install (`routes_plugins.py`, `routes_packs.py`), and
 * that refusal must cost one request, not two.
 */
export async function apiFetch(
  url: string,
  init: RequestInit = {},
): Promise<Response> {
  const method = (init.method ?? 'GET').toUpperCase();
  if (!MUTATING_METHODS.has(method)) {
    return fetch(url, init);
  }
  const token = await getSessionToken();
  const res = await fetch(url, { ...init, headers: withToken(init, token) });
  if (res.status !== 403 || !isReplayable(init.body)) return res;

  invalidateSessionToken();
  let fresh: string;
  try {
    fresh = await getSessionToken();
  } catch {
    // The server is unreachable now; the 403 we already have is the more
    // useful answer to give the caller.
    return res;
  }
  if (fresh === token) return res;
  return fetch(url, { ...init, headers: withToken(init, fresh) });
}

/** *init*'s headers plus the session token. */
function withToken(init: RequestInit, token: string): Headers {
  const headers = new Headers(init.headers);
  headers.set(TOKEN_HEADER, token);
  return headers;
}

/**
 * Whether this body can be sent a second time.
 *
 * Strings, `FormData` and blobs can. A `ReadableStream` cannot — the first
 * request consumed it — so a request built from one keeps its 403 rather than
 * being retried into a `TypeError`.
 */
function isReplayable(body: BodyInit | null | undefined): boolean {
  return !(typeof ReadableStream !== 'undefined' && body instanceof ReadableStream);
}

/**
 * Build a WebSocket URL with the session token appended as a query parameter.
 * Browsers can't set custom headers on WebSocket handshakes, so this is the
 * cleanest way to authenticate the upgrade.
 */
export async function wsUrlWithToken(path: string): Promise<string> {
  const token = await getSessionToken();
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const base = `${protocol}//${window.location.host}${path}`;
  const u = new URL(base);
  u.searchParams.set('token', token);
  return u.toString();
}
