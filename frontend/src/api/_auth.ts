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
 *
 * A failed attempt is never kept: the GET fails outright while the server
 * restarts, and a rejection left in `inflight` would answer every later call,
 * so every mutating request, until the page was reloaded. The next call tries
 * again instead.
 */
export async function getSessionToken(): Promise<string> {
  if (cachedToken !== null) return cachedToken;
  if (inflight !== null) return inflight;

  const attempt: Promise<string> = (async () => {
    const res = await fetch(BOOTSTRAP_URL);
    if (!res.ok) {
      throw new Error(
        `Failed to bootstrap auth token: ${res.status} ${res.statusText}`,
      );
    }
    const body = await res.json();
    if (typeof body?.token !== 'string') {
      throw new Error('Bootstrap response missing token');
    }
    return body.token as string;
  })().then(
    (token) => {
      // Only the attempt still in the slot fills the cache. One dropped by
      // `invalidateSessionToken` while it ran read the token from before a
      // restart: it answers its own caller, and nobody after.
      if (inflight === attempt) cachedToken = token;
      return token;
    },
    (error: unknown) => {
      if (inflight === attempt) inflight = null;
      throw error;
    },
  );
  inflight = attempt;
  return attempt;
}

/**
 * Drop the cached token, and any bootstrap still in flight, so the next call
 * re-reads /api/auth/bootstrap.
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
 * unchanged, to any origin.
 *
 * The token is only ever sent to this page's own origin. A POST, PUT, PATCH or
 * DELETE aimed anywhere else rejects with a `TypeError` before the token is
 * read, and nothing is sent: `api.http.fetch` hands plugins this function with
 * any URL they like (#482), while every host call site passes a relative path.
 * That keeps a well-meaning plugin from sending the token to another server by
 * accident. Plugin code still runs in this page, with the token in reach.
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
  // Typed as a string, but plugin JavaScript is untyped and can pass a
  // `Request`, whose own method, headers and body `fetch` uses when `init`
  // names none -- so its POST is a POST here too.
  const request = requestOf(url);
  const method = (init.method ?? request?.method ?? 'GET').toUpperCase();
  if (!MUTATING_METHODS.has(method)) {
    return fetch(url, init);
  }
  // Every send below, the retry included, uses what was checked.
  const target = sameOriginTarget(url, request, method);
  // Headers in `init` replace a Request's own, so the token joins whichever
  // set `fetch` would have sent.
  const headers = init.headers ?? request?.headers;
  const token = await getSessionToken();
  const res = await fetch(target, { ...init, headers: withToken(headers, token) });
  if (res.status !== 403) return res;

  // Dropped on every 403, even one that cannot be replayed below: the token
  // may be one the server no longer knows, and keeping it would fail the next
  // POST as well. When the 403 was a real refusal, this costs one bootstrap GET.
  invalidateSessionToken();
  // A Request's body is used up by the first send, and a replay would reject
  // with a TypeError instead of handing back the 403.
  if (!isReplayable(init.body) || request?.bodyUsed) return res;
  let fresh: string;
  try {
    fresh = await getSessionToken();
  } catch {
    // The server is unreachable now; the 403 we already have is the more
    // useful answer to give the caller.
    return res;
  }
  if (fresh === token) return res;
  return fetch(target, { ...init, headers: withToken(headers, fresh) });
}

/** *url* as a `Request`, when untyped plugin code passed one. */
function requestOf(url: unknown): Request | null {
  return typeof Request !== 'undefined' && url instanceof Request ? url : null;
}

/**
 * What to send for *url*, once it is known to resolve to this page's own
 * origin; a `TypeError` for any other origin.
 *
 * The answer is what was checked -- the URL as a string, or a `Request`, whose
 * URL is fixed when it is built -- because the send comes after the token
 * bootstrap: a `URL` object read a second time by `fetch` could by then point
 * at a host the caller changed in the meantime.
 *
 * Resolved the way `fetch` resolves it, against the document's base URL, so
 * `/api/x` and `api/x` are this origin and a protocol-relative `//host/x` is
 * not. Refused rather than sent without the token: a plugin that spelled this
 * server another way (`localhost` for `127.0.0.1`) would otherwise get a 403
 * that says nothing about why, which is also why the message points at a path
 * first. A `TypeError` because that is what `fetch` itself rejects with, so a
 * caller's existing `catch` still fits.
 */
function sameOriginTarget(url: unknown, request: Request | null, method: string): RequestInfo {
  // A `Request` is sent to its own URL; anything else, a `URL` object
  // included, is checked and sent as its string form.
  const target = request ? request.url : String(url);
  let origin: string | null = null;
  try {
    origin = new URL(target, document.baseURI).origin;
  } catch {
    // Not a URL `fetch` could send either; refused below like any other.
  }
  const own = window.location.origin;
  if (origin !== own) {
    throw new TypeError(
      `${method} to ${origin ?? target} would carry the CodefyUI session token, `
      + `which is only sent to ${own}. Use a path such as /api/... for this server `
      + 'and window.fetch for other servers.',
    );
  }
  return request ?? target;
}

/** *headers* plus the session token. */
function withToken(headers: HeadersInit | undefined, token: string): Headers {
  const merged = new Headers(headers);
  merged.set(TOKEN_HEADER, token);
  return merged;
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
