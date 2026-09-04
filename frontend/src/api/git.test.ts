import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  GIT_TIMEOUTS_S,
  GitApiError,
  getGitConfig,
  getGitStatus,
  gitApiError,
  gitCommit,
  gitDiscard,
  gitInit,
  gitStage,
  gitUnstage,
  setGitConfig,
} from './git';
import { ApiError, errorDetail } from './rest';
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

/** A refusal whose body is not JSON at all — a proxy page, or an empty 502. */
function mockFetchJsonThrows(status: number) {
  const response = {
    ok: false,
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

/** The status a `ready` repository with nothing in it answers with. */
function cleanStatus() {
  return {
    branch: 'main',
    detached: false,
    head: 'abc123',
    unborn: false,
    upstream: null,
    ahead: null,
    behind: null,
    upstream_gone: false,
    staged: [],
    unstaged: [],
    untracked: [],
    conflicted: [],
    stash_count: 0,
    merge_in_progress: false,
    rebase_in_progress: false,
  };
}

/** Await a call that must fail, and hand back the GitApiError it threw. */
async function gitError(call: Promise<unknown>): Promise<GitApiError> {
  const err = await call.catch((e: unknown) => e);
  expect(err).toBeInstanceOf(GitApiError);
  return err as GitApiError;
}

beforeEach(() => {
  originalFetch = g.fetch;
  // Pre-seed the cached session token so apiFetch does not try to bootstrap.
  _setSessionTokenForTesting('test-token');
});

afterEach(() => {
  g.fetch = originalFetch;
  _setSessionTokenForTesting(null);
  vi.restoreAllMocks();
});

describe('GIT_TIMEOUTS_S', () => {
  // The server sends a 504 with a code and no number, so these three are the
  // only place the sentence's "{seconds}" can come from. They mirror
  // backend/app/core/git/runner.py's T_STATUS / T_LOCAL / T_READ.
  it('mirrors the three server-side deadlines', () => {
    expect(GIT_TIMEOUTS_S.status).toBe(10);
    expect(GIT_TIMEOUTS_S.mutation).toBe(30);
    expect(GIT_TIMEOUTS_S.read).toBe(20);
  });
});

describe('getGitStatus', () => {
  it('GETs the open route with no session token', async () => {
    const fetchMock = mockFetch(200, {
      repo: { state: 'ready', project_dir: '/p', git_version: '2.53.0' },
      status: cleanStatus(),
    });
    const out = await getGitStatus();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/git/status');
    // A read is a bare `fetch(url)`: no init at all, so no token header and
    // no method to make the auth guard ask for one.
    expect(fetchMock.mock.calls[0][1]).toBeUndefined();
    expect(out.repo.state).toBe('ready');
    expect(out.status?.branch).toBe('main');
  });

  it('keeps a null status, which every state but ready answers with', async () => {
    mockFetch(200, { repo: { state: 'no_project' }, status: null });
    const out = await getGitStatus();
    expect(out.status).toBeNull();
    expect(out.repo).toEqual({
      state: 'no_project',
      project_dir: null,
      git_version: null,
      nested_toplevel: null,
    });
  });

  it('normalizes the four groups to lists so the tab can map over them', async () => {
    mockFetch(200, { repo: { state: 'ready' }, status: { branch: 'main' } });
    const status = (await getGitStatus()).status;
    expect(status?.staged).toEqual([]);
    expect(status?.unstaged).toEqual([]);
    expect(status?.untracked).toEqual([]);
    expect(status?.conflicted).toEqual([]);
    expect(status?.stash_count).toBe(0);
    expect(status?.merge_in_progress).toBe(false);
    expect(status?.ahead).toBeNull();
  });

  it('normalizes each file, filling the two optional fields', async () => {
    mockFetch(200, {
      repo: { state: 'ready' },
      status: {
        ...cleanStatus(),
        staged: [{ path: 'graphs/a.graph.json', kind: 'added', xy: 'A.' }],
        unstaged: [
          { path: 'b.py', orig_path: 'a.py', kind: 'renamed', xy: 'R.', score: 95 },
        ],
      },
    });
    const status = (await getGitStatus()).status;
    expect(status?.staged[0]).toEqual({
      path: 'graphs/a.graph.json',
      orig_path: null,
      kind: 'added',
      xy: 'A.',
      score: null,
    });
    expect(status?.unstaged[0].orig_path).toBe('a.py');
    expect(status?.unstaged[0].score).toBe(95);
  });

  it('degrades a kind and a state this build does not know', async () => {
    mockFetch(200, {
      repo: { state: 'quantum' },
      status: { ...cleanStatus(), unstaged: [{ path: 'x', kind: 'wobbled', xy: '.M' }] },
    });
    const out = await getGitStatus();
    expect(out.repo.state).toBe('not_repo');
    expect(out.status?.unstaged[0].kind).toBe('modified');
  });

  it('reports a server with no git service as a coded error', async () => {
    mockFetch(503, {
      detail: {
        code: 'git_service_unavailable',
        message: 'source control is not available on this server',
        hint: 'the server started without it; restart the server',
        stderr: null,
      },
    });
    const err = await gitError(getGitStatus());
    expect(err.status).toBe(503);
    expect(err.code).toBe('git_service_unavailable');
    expect(err.hint).toBe('the server started without it; restart the server');
  });
});

describe('getGitConfig', () => {
  it('GETs the open config route and keeps both scopes', async () => {
    const fetchMock = mockFetch(200, {
      name: 'Ada',
      email: 'ada@example.com',
      name_scope: 'local',
      email_scope: 'global',
    });
    const out = await getGitConfig();
    expect(fetchMock.mock.calls[0][0]).toBe('/api/git/config');
    expect(fetchMock.mock.calls[0][1]).toBeUndefined();
    expect(out).toEqual({
      name: 'Ada',
      email: 'ada@example.com',
      name_scope: 'local',
      email_scope: 'global',
    });
  });

  it('nulls an unset identity and a scope outside the union', async () => {
    mockFetch(200, { name: null, name_scope: null, email_scope: 'machine' });
    expect(await getGitConfig()).toEqual({
      name: null,
      email: null,
      name_scope: null,
      email_scope: null,
    });
  });
});

describe('mutations', () => {
  function mutationBody() {
    return {
      status: cleanStatus(),
      changed_paths: ['graphs/a.graph.json'],
      head: 'abc123',
      detail: { paths: ['graphs/a.graph.json'] },
    };
  }

  it('gitInit POSTs with the token and no body', async () => {
    const fetchMock = mockFetch(200, mutationBody());
    const out = await gitInit();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/git/init');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    // There is nothing to choose, and the route ignores a body if one is sent.
    expect(init.body).toBeUndefined();
    expect(out.status.branch).toBe('main');
    expect(out.changed_paths).toEqual(['graphs/a.graph.json']);
    expect(out.detail).toEqual({ paths: ['graphs/a.graph.json'] });
  });

  it('gitStage sends the named paths', async () => {
    const fetchMock = mockFetch(200, mutationBody());
    await gitStage(['graphs/a.graph.json', 'layout/a.layout.json']);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/git/stage');
    expect(init.method).toBe('POST');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    // `paths` alone: `PathsRequest` refuses a body carrying both forms.
    expect(JSON.parse(init.body)).toEqual({
      paths: ['graphs/a.graph.json', 'layout/a.layout.json'],
    });
  });

  it("gitStage('all') sends all:true and no paths key", async () => {
    const fetchMock = mockFetch(200, mutationBody());
    await gitStage('all');
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ all: true });
  });

  it('gitUnstage sends both forms to the unstage route', async () => {
    const fetchMock = mockFetch(200, mutationBody());
    await gitUnstage(['a.py']);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/git/unstage');
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ paths: ['a.py'] });

    const allMock = mockFetch(200, mutationBody());
    await gitUnstage('all');
    expect(JSON.parse(allMock.mock.calls[0][1].body)).toEqual({ all: true });
  });

  it('gitDiscard sends both forms to the discard route', async () => {
    const fetchMock = mockFetch(200, mutationBody());
    await gitDiscard(['a.py']);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/git/discard');
    expect(new Headers(fetchMock.mock.calls[0][1].headers).get('X-CodefyUI-Token')).toBe(
      'test-token',
    );
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ paths: ['a.py'] });

    const allMock = mockFetch(200, mutationBody());
    await gitDiscard('all');
    expect(JSON.parse(allMock.mock.calls[0][1].body)).toEqual({ all: true });
  });

  it('gitCommit spells out both flags even when the caller omits them', async () => {
    const fetchMock = mockFetch(200, {
      ...mutationBody(),
      detail: { sha: 'abc1234def', short: 'abc1234' },
    });
    const out = await gitCommit({ message: 'save the graph' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/git/commit');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    expect(JSON.parse(init.body)).toEqual({
      message: 'save the graph',
      all: false,
      amend: false,
    });
    expect(out.detail).toEqual({ sha: 'abc1234def', short: 'abc1234' });
  });

  it('gitCommit passes all and amend through', async () => {
    const fetchMock = mockFetch(200, mutationBody());
    await gitCommit({ message: 'redo it', all: true, amend: true });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      message: 'redo it',
      all: true,
      amend: true,
    });
  });

  it('normalizes a result whose optional halves are absent', async () => {
    mockFetch(200, { status: { branch: 'main' } });
    const out = await gitStage('all');
    expect(out.changed_paths).toEqual([]);
    expect(out.head).toBeNull();
    expect(out.detail).toEqual({});
    expect(out.status.staged).toEqual([]);
  });

  it('refuses an empty selection without asking the server', async () => {
    // A selection that turned out to be empty is the one bad body a caller
    // can build by accident, and `{paths: []}` is a 422 the tab can do
    // nothing with. Refused here, in the shape every caller already handles.
    const fetchMock = mockFetch(200, mutationBody());
    const err = await gitError(gitStage([]));
    expect(err.status).toBe(400);
    expect(err.code).toBe('invalid');
    expect(fetchMock).not.toHaveBeenCalled();

    expect((await gitError(gitUnstage([]))).code).toBe('invalid');
    expect((await gitError(gitDiscard([]))).code).toBe('invalid');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refuses a write the server could not read back', async () => {
    // `MutationResult.status` is required and never null. A body without it
    // must not normalize to an empty status: that reads as a clean
    // repository, so the tab would draw the user's changes away and claim
    // the stage they just asked for did nothing.
    mockFetch(200, { changed_paths: ['a.py'] });
    const err = await gitError(gitStage(['a.py']));
    expect(err.status).toBe(502);
    expect(err.code).toBe('unknown');
    expect(err.message).toBe('the write was not read back');
  });

  it('surfaces a refused mutation as a GitApiError with its code', async () => {
    mockFetch(409, {
      detail: {
        code: 'nothing_to_commit',
        message: 'there is nothing to commit',
        hint: 'stage a change first',
        stderr: null,
      },
    });
    const err = await gitError(gitCommit({ message: 'empty' }));
    expect(err.status).toBe(409);
    expect(err.code).toBe('nothing_to_commit');
    expect(err.message).toBe('there is nothing to commit');
  });
});

describe('setGitConfig', () => {
  it('PUTs the identity with the session token', async () => {
    const fetchMock = mockFetch(200, {
      name: 'Ada',
      email: 'ada@example.com',
      name_scope: 'local',
      email_scope: 'local',
    });
    const out = await setGitConfig({ name: 'Ada', email: 'ada@example.com' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/git/config');
    // The app's first PUT. `apiFetch` counts it as mutating, so the header is
    // attached exactly as it is on a POST.
    expect(init.method).toBe('PUT');
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    expect(JSON.parse(init.body)).toEqual({ name: 'Ada', email: 'ada@example.com' });
    expect(out.name_scope).toBe('local');
  });

  it('omits the half the caller did not fill', async () => {
    const fetchMock = mockFetch(200, { name: 'Ada' });
    await setGitConfig({ name: 'Ada' });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ name: 'Ada' });
  });
});

describe('gitApiError', () => {
  /** Build a Response the way a refused call would arrive, and read it. */
  function refusal(status: number, body: unknown): Promise<GitApiError> {
    const res = {
      ok: false,
      status,
      statusText: 'mock',
      json: async () => body,
    } as unknown as Response;
    return gitApiError(res);
  }

  it('reads the four keys out of a coded detail', async () => {
    const err = await refusal(400, {
      detail: {
        code: 'invalid_path',
        message: 'notes/keys.txt is not allowed',
        hint: 'it points outside the project',
        stderr: 'fatal: pathspec did not match',
      },
    });
    expect(err.status).toBe(400);
    expect(err.code).toBe('invalid_path');
    expect(err.message).toBe('notes/keys.txt is not allowed');
    expect(err.hint).toBe('it points outside the project');
    expect(err.stderr).toBe('fatal: pathspec did not match');
    expect(err.op).toBeNull();
    expect(err.name).toBe('GitApiError');
  });

  it('keeps the fifth key busy carries', async () => {
    const err = await refusal(409, {
      detail: {
        code: 'busy',
        message: 'another git operation is already running (commit)',
        hint: null,
        stderr: null,
        op: 'commit',
      },
    });
    expect(err.code).toBe('busy');
    expect(err.op).toBe('commit');
    expect(err.hint).toBeNull();
  });

  it('folds a 422 list detail into one invalid error', async () => {
    // FastAPI's own refusal: `PathsRequest` failed its validator, or the body
    // carried a key the model forbids. Not the git envelope at all.
    const err = await refusal(422, {
      detail: [
        {
          type: 'value_error',
          loc: ['body'],
          msg: 'Value error, send either paths or all=true, not both',
        },
        { type: 'extra_forbidden', loc: ['body', 'force'], msg: 'Extra inputs are not permitted' },
      ],
    });
    expect(err.status).toBe(422);
    expect(err.code).toBe('invalid');
    expect(err.message).toBe(
      'Value error, send either paths or all=true, not both; Extra inputs are not permitted',
    );
    expect(err.hint).toBeNull();
    expect(err.stderr).toBeNull();
  });

  it('falls back to the status text when a list detail names no messages', async () => {
    const err = await refusal(422, { detail: [{ loc: ['body'] }] });
    expect(err.code).toBe('invalid');
    expect(err.message).toBe('mock');
  });

  it('degrades the auth guard string detail to unknown', async () => {
    const err = await refusal(403, {
      detail: 'Missing or invalid X-CodefyUI-Token header',
    });
    expect(err.status).toBe(403);
    expect(err.code).toBe('unknown');
    expect(err.message).toBe('Missing or invalid X-CodefyUI-Token header');
  });

  it('degrades a code from a newer server, keeping the raw one on the body', async () => {
    const err = await refusal(409, {
      detail: { code: 'shallow_update_not_allowed', message: 'nope', hint: null, stderr: null },
    });
    expect(err.code).toBe('unknown');
    expect(err.message).toBe('nope');
    expect(errorDetail(err)?.code).toBe('shallow_update_not_allowed');
  });

  it('answers with the status text when the body is not JSON', async () => {
    mockFetchJsonThrows(502);
    const err = await gitError(getGitStatus());
    expect(err.status).toBe(502);
    expect(err.code).toBe('unknown');
    expect(err.message).toBe('mock');
    expect(err.body).toBeNull();
  });

  it('is an ApiError, so the shared helpers keep working on it', async () => {
    const err = await refusal(409, {
      detail: { code: 'not_repo', message: 'not a repository', hint: null, stderr: null },
    });
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toBeInstanceOf(Error);
    expect(errorDetail(err)?.message).toBe('not a repository');
  });
});
