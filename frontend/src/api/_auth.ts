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
  const headers = new Headers(init.headers);
  headers.set(TOKEN_HEADER, token);
  return fetch(url, { ...init, headers });
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
