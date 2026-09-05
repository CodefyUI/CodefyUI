import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  GIT_ERROR_CODES,
  GIT_TIMEOUTS_S,
  GitApiError,
  getGitBranches,
  getGitConfig,
  getGitRemotes,
  getGitStashes,
  getGitStatus,
  gitAbortMerge,
  gitAddRemote,
  gitApiError,
  gitCheckout,
  gitCommit,
  gitCreateBranch,
  gitDeleteBranch,
  gitDiscard,
  gitFetch,
  gitInit,
  gitPull,
  gitPush,
  gitRemoveRemote,
  gitRenameBranch,
  gitResolve,
  gitSetRemoteUrl,
  gitStage,
  gitStashApply,
  gitStashDrop,
  gitStashPop,
  gitStashPush,
  gitSync,
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
  // The server sends a 504 with a code and no number, so these four are the
  // only place the sentence's "{seconds}" can come from. The first three are
  // backend/app/core/git/runner.py's T_STATUS / T_LOCAL / T_READ exactly; the
  // fourth is T_NETWORK (120) plus a ten-second browser-facing grace, because
  // a network operation is several git processes and one request rather than
  // the single process that deadline is applied to. 130 is not a typo.
  it('mirrors the four server-side deadlines', () => {
    expect(GIT_TIMEOUTS_S.status).toBe(10);
    expect(GIT_TIMEOUTS_S.mutation).toBe(30);
    expect(GIT_TIMEOUTS_S.read).toBe(20);
    expect(GIT_TIMEOUTS_S.network).toBe(130);
  });
});

describe('GIT_ERROR_CODES', () => {
  it('recognises the two G3 remote refusal codes', () => {
    expect(GIT_ERROR_CODES).toContain('remote_exists');
    expect(GIT_ERROR_CODES).toContain('remote_rejected');
  });

  // A code this list does not know degrades to `unknown`, and `unknown` shows
  // git's own words -- which for this refusal is a sentence about
  // `push.default` that only somebody who already knew the setting could act
  // on. Knowing the code is what lets the tab say it in the reader's language.
  it('recognises a push refused by the host git configuration', () => {
    expect(GIT_ERROR_CODES).toContain('push_config');
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

describe('reference reads', () => {
  it('GETs and normalizes every branch field', async () => {
    const fetchMock = mockFetch(200, {
      current: 'main',
      detached: true,
      local: [
        {
          name: 'main',
          sha: 'abc1234',
          current: true,
          upstream: 'origin/main',
          ahead: 2,
          behind: 3,
          gone: true,
          subject: 'Ship it',
          committed_at: 123,
        },
        { name: 'empty' },
      ],
      remote: [
        {
          name: 'feat/x',
          remote: 'origin',
          sha: 'def5678',
          subject: 'Remote work',
          committed_at: 456,
        },
        {},
      ],
    });

    const out = await getGitBranches();

    expect(fetchMock.mock.calls[0]).toEqual(['/api/git/branches']);
    expect(out).toEqual({
      current: 'main',
      detached: true,
      local: [
        {
          name: 'main',
          sha: 'abc1234',
          current: true,
          upstream: 'origin/main',
          ahead: 2,
          behind: 3,
          gone: true,
          subject: 'Ship it',
          committed_at: 123,
        },
        {
          name: 'empty',
          sha: '',
          current: false,
          upstream: null,
          ahead: null,
          behind: null,
          gone: false,
          subject: '',
          committed_at: 0,
        },
      ],
      remote: [
        {
          name: 'feat/x',
          remote: 'origin',
          sha: 'def5678',
          subject: 'Remote work',
          committed_at: 456,
        },
        { name: '', remote: '', sha: '', subject: '', committed_at: 0 },
      ],
    });
  });

  it('defaults a partial branches response to the empty answer', async () => {
    mockFetch(200, {});
    expect(await getGitBranches()).toEqual({
      current: null,
      detached: false,
      local: [],
      remote: [],
    });
  });

  it('GETs and normalizes every remote field', async () => {
    const fetchMock = mockFetch(200, [
      { name: 'origin', fetch_url: 'https://example/fetch', push_url: 'ssh://example/push' },
      {},
    ]);
    expect(await getGitRemotes()).toEqual([
      { name: 'origin', fetch_url: 'https://example/fetch', push_url: 'ssh://example/push' },
      { name: '', fetch_url: '', push_url: '' },
    ]);
    expect(fetchMock.mock.calls[0]).toEqual(['/api/git/remotes']);
  });

  it('GETs and normalizes every stash field', async () => {
    const fetchMock = mockFetch(200, [
      { index: 4, message: 'WIP on main: work', branch: 'main', created_at: 789 },
      {},
    ]);
    expect(await getGitStashes()).toEqual([
      { index: 4, message: 'WIP on main: work', branch: 'main', created_at: 789 },
      { index: 0, message: '', branch: null, created_at: 0 },
    ]);
    expect(fetchMock.mock.calls[0]).toEqual(['/api/git/stashes']);
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

  it.each([
    {
      name: 'gitFetch',
      call: () => gitFetch(),
      url: '/api/git/fetch',
      method: 'POST',
      body: { remote: null },
    },
    {
      name: 'gitPull',
      call: () => gitPull({ strategy: 'merge' }),
      url: '/api/git/pull',
      method: 'POST',
      body: { strategy: 'merge' },
    },
    {
      name: 'gitPush',
      call: () => gitPush({ remote: 'upstream', setUpstream: true }),
      url: '/api/git/push',
      method: 'POST',
      body: { remote: 'upstream', set_upstream: true },
    },
    {
      name: 'gitSync',
      call: () => gitSync(),
      url: '/api/git/sync',
      method: 'POST',
      body: undefined,
    },
    {
      name: 'gitCreateBranch',
      call: () => gitCreateBranch('feat/a b', false, 'main'),
      url: '/api/git/branches',
      method: 'POST',
      body: { name: 'feat/a b', checkout: false, start_point: 'main' },
    },
    {
      name: 'gitCheckout',
      call: () => gitCheckout('origin/feat/a', 'remote'),
      url: '/api/git/checkout',
      method: 'POST',
      body: { target: 'origin/feat/a', kind: 'remote' },
    },
    {
      name: 'gitRenameBranch',
      call: () => gitRenameBranch('feat/a b', 'feat/c'),
      url: '/api/git/branches/feat%2Fa%20b',
      method: 'PUT',
      body: { new_name: 'feat/c' },
    },
    {
      name: 'gitDeleteBranch',
      call: () => gitDeleteBranch('feat/a b', false),
      url: '/api/git/branches/feat%2Fa%20b',
      method: 'DELETE',
      body: undefined,
    },
    {
      name: 'gitDeleteBranch force',
      call: () => gitDeleteBranch('feat/a b', true),
      url: '/api/git/branches/feat%2Fa%20b?force=1',
      method: 'DELETE',
      body: undefined,
    },
    {
      name: 'gitAddRemote',
      call: () => gitAddRemote('origin', 'https://example/repo.git'),
      url: '/api/git/remotes',
      method: 'POST',
      body: { name: 'origin', url: 'https://example/repo.git' },
    },
    {
      name: 'gitSetRemoteUrl',
      call: () => gitSetRemoteUrl('origin', 'ssh://example/repo.git'),
      url: '/api/git/remotes/origin',
      method: 'PUT',
      body: { url: 'ssh://example/repo.git' },
    },
    {
      name: 'gitRemoveRemote',
      call: () => gitRemoveRemote('origin'),
      url: '/api/git/remotes/origin',
      method: 'DELETE',
      body: undefined,
    },
    {
      name: 'gitStashPush',
      call: () => gitStashPush(null, true),
      url: '/api/git/stashes',
      method: 'POST',
      body: { message: null, include_untracked: true },
    },
    {
      name: 'gitStashPop',
      call: () => gitStashPop(4),
      url: '/api/git/stashes/4/pop',
      method: 'POST',
      body: undefined,
    },
    {
      name: 'gitStashApply',
      call: () => gitStashApply(3),
      url: '/api/git/stashes/3/apply',
      method: 'POST',
      body: undefined,
    },
    {
      name: 'gitStashDrop',
      call: () => gitStashDrop(2),
      url: '/api/git/stashes/2',
      method: 'DELETE',
      body: undefined,
    },
    {
      name: 'gitAbortMerge',
      call: () => gitAbortMerge(),
      url: '/api/git/merge/abort',
      method: 'POST',
      body: undefined,
    },
    {
      name: 'gitResolve',
      call: () => gitResolve('graphs/demo.graph.json', 'theirs'),
      url: '/api/git/resolve',
      method: 'POST',
      body: { path: 'graphs/demo.graph.json', side: 'theirs' },
    },
  ])('$name uses its route, method and exact wire body', async ({ call, url, method, body }) => {
    const fetchMock = mockFetch(200, mutationBody());
    const out = await call();
    const [actualUrl, init] = fetchMock.mock.calls[0];

    expect(actualUrl).toBe(url);
    expect(init.method).toBe(method);
    expect(new Headers(init.headers).get('X-CodefyUI-Token')).toBe('test-token');
    if (body === undefined) {
      expect(init.body).toBeUndefined();
    } else {
      expect(JSON.parse(init.body)).toEqual(body);
    }
    expect(out.status.branch).toBe('main');
  });

  it('gitPush defaults to a plain push with no remote', async () => {
    const fetchMock = mockFetch(200, mutationBody());
    await gitPush({ setUpstream: false });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      remote: null,
      set_upstream: false,
    });
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

    // The other half of the guard: a key that is there and null is the same
    // hole as a key that is missing, and JSON has both spellings.
    mockFetch(200, { status: null, changed_paths: ['a.py'] });
    const explicit = await gitError(gitStage(['a.py']));
    expect(explicit.status).toBe(502);
    expect(explicit.message).toBe('the write was not read back');
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
