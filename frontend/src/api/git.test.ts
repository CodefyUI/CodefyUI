import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  GIT_ERROR_CODES,
  GIT_TIMEOUTS_S,
  GitApiError,
  getGitBranches,
  getGitCommitFiles,
  getGitConfig,
  getGitDiff,
  getGitFile,
  getGitLog,
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

/* ── The history and diff reads ──────────────────────────────────────── */

/** The URL a read asked for, parsed so a test names keys instead of encoding. */
function calledUrl(fetchMock: ReturnType<typeof mockFetch>): URL {
  return new URL(String(fetchMock.mock.calls[0][0]), 'http://localhost');
}

/** One commit as `%H%x1f%h%x1f%P...` reaches the route, snake_case and all. */
function wireCommit(over: Record<string, unknown> = {}) {
  return {
    sha: 'a'.repeat(40),
    short: 'aaaaaaa',
    parents: ['b'.repeat(40)],
    author_name: 'Ada',
    author_email: 'ada@example.com',
    authored_at: 1_700_000_000,
    refs: ['HEAD -> main', 'origin/main'],
    subject: 'Ship it',
    body: 'A longer note.\n',
    ...over,
  };
}

describe('getGitLog', () => {
  it('GETs one page, with the window it was asked for on the query', async () => {
    const fetchMock = mockFetch(200, { commits: [], has_more: false, unborn: false });
    await getGitLog(30, 30);
    const url = calledUrl(fetchMock);
    expect(url.pathname).toBe('/api/git/log');
    expect(url.searchParams.get('skip')).toBe('30');
    expect(url.searchParams.get('limit')).toBe('30');
    // Nothing else: the route takes exactly these two, and a third key would
    // be a 422 from the signature rather than something the server ignores.
    expect([...url.searchParams.keys()]).toEqual(['skip', 'limit']);
  });

  it('normalizes every commit field, camelCased off the wire', async () => {
    mockFetch(200, {
      commits: [wireCommit(), {}],
      has_more: true,
      unborn: false,
    });
    const page = await getGitLog(0, 30);
    expect(page.hasMore).toBe(true);
    expect(page.unborn).toBe(false);
    expect(page.commits[0]).toEqual({
      sha: 'a'.repeat(40),
      short: 'aaaaaaa',
      parents: ['b'.repeat(40)],
      authorName: 'Ada',
      authorEmail: 'ada@example.com',
      // Epoch SECONDS, which is what git's `%at` answers with -- not a string
      // and not milliseconds. `relativeTime` reads it as it stands.
      authoredAt: 1_700_000_000,
      refs: ['HEAD -> main', 'origin/main'],
      subject: 'Ship it',
      body: 'A longer note.\n',
    });
    // A row the server could not fill still arrives as a whole commit: the
    // history list maps over these on every repaint.
    expect(page.commits[1]).toEqual({
      sha: '',
      short: '',
      parents: [],
      authorName: '',
      authorEmail: '',
      authoredAt: 0,
      refs: [],
      subject: '',
      body: '',
    });
  });

  it('defaults a partial page to the empty answer', async () => {
    mockFetch(200, {});
    expect(await getGitLog(0, 30)).toEqual({ commits: [], hasMore: false, unborn: true });
  });

  it('reads an unborn branch as a page rather than a refusal', async () => {
    // `rev-parse --verify HEAD` failing is a 200 with `unborn: true`, never an
    // error: a repository with no commits is a state the section draws.
    mockFetch(200, { commits: [], has_more: false, unborn: true });
    const page = await getGitLog(0, 30);
    expect(page.unborn).toBe(true);
    expect(page.commits).toEqual([]);
  });

  it('throws on a refusal rather than reading it as an empty history', async () => {
    // The one answer that must NOT be normalised: an error body has no
    // `unborn` key, so a page built from it would say "no commits yet" about
    // a repository the server simply could not read.
    mockFetch(500, {
      detail: { code: 'git_failed', message: 'fatal: bad revision', hint: null, stderr: null },
    });
    const err = await gitError(getGitLog(0, 30));
    expect(err.status).toBe(500);
    expect(err.code).toBe('git_failed');
    expect(err.message).toBe('fatal: bad revision');
  });
});

describe('getGitCommitFiles', () => {
  it('GETs the bare list under the commit, with the sha in the path', async () => {
    const fetchMock = mockFetch(200, []);
    await getGitCommitFiles('abc123');
    expect(fetchMock.mock.calls[0]).toEqual(['/api/git/commits/abc123/files']);
  });

  it('normalizes each file and degrades a kind this build does not know', async () => {
    mockFetch(200, [
      { path: 'b.py', orig_path: 'a.py', kind: 'renamed', xy: 'R.', score: 95 },
      { path: 'graphs/a.graph.json', kind: 'added', xy: 'A.' },
      { path: 'x', kind: 'wobbled', xy: '?.' },
    ]);
    expect(await getGitCommitFiles('abc123')).toEqual([
      { path: 'b.py', origPath: 'a.py', kind: 'renamed' },
      { path: 'graphs/a.graph.json', origPath: null, kind: 'added' },
      { path: 'x', origPath: null, kind: 'modified' },
    ]);
  });

  it('keeps the not_found code a commit nobody has answers with', async () => {
    mockFetch(404, {
      detail: { code: 'not_found', message: 'deadbee', hint: null, stderr: null },
    });
    const err = await gitError(getGitCommitFiles('deadbee'));
    expect(err.status).toBe(404);
    expect(err.code).toBe('not_found');
    expect(err.message).toBe('deadbee');
  });
});

describe('getGitDiff', () => {
  function wireDiff(over: Record<string, unknown> = {}) {
    return {
      patch: '@@ -1 +1 @@\n-old\n+new\n',
      binary: false,
      truncated: false,
      old_ref: 'HEAD',
      new_ref: 'index',
      old_text: null,
      new_text: null,
      old_missing: false,
      new_missing: false,
      ...over,
    };
  }

  it('GETs the path, the scope and the sha, and never asks for blobs', async () => {
    const fetchMock = mockFetch(200, wireDiff({ old_ref: 'abc^', new_ref: 'abc' }));
    await getGitDiff({ path: 'graphs/a.graph.json', scope: 'commit', sha: 'abc123' });
    const url = calledUrl(fetchMock);
    expect(url.pathname).toBe('/api/git/diff');
    expect(url.searchParams.get('path')).toBe('graphs/a.graph.json');
    expect(url.searchParams.get('scope')).toBe('commit');
    expect(url.searchParams.get('sha')).toBe('abc123');
    // `blobs=1` costs two extra git reads and this build's side-by-side view
    // is derived from the patch, so the key is never sent at all.
    expect(url.searchParams.has('blobs')).toBe(false);
  });

  it('omits the sha for the two scopes the route refuses one from', async () => {
    // `scope=worktree` (or `index`) carrying a sha is a 400 by the route's own
    // first check, so an absent sha has to arrive as an absent KEY.
    for (const scope of ['worktree', 'index'] as const) {
      const fetchMock = mockFetch(200, wireDiff());
      await getGitDiff({ path: 'notes.md', scope });
      expect([...calledUrl(fetchMock).searchParams.keys()]).toEqual(['path', 'scope']);
    }
  });

  it('drops an empty sha rather than sending the key with nothing in it', async () => {
    const fetchMock = mockFetch(200, wireDiff());
    await getGitDiff({ path: 'notes.md', scope: 'worktree', sha: '' });
    expect(calledUrl(fetchMock).searchParams.has('sha')).toBe(false);
  });

  it('normalizes every field, camelCased, and keeps the blob keys out', async () => {
    mockFetch(200, wireDiff({ truncated: true, old_text: 'ignored', new_text: 'ignored' }));
    expect(await getGitDiff({ path: 'notes.md', scope: 'index' })).toEqual({
      patch: '@@ -1 +1 @@\n-old\n+new\n',
      binary: false,
      truncated: true,
      oldRef: 'HEAD',
      newRef: 'index',
      oldMissing: false,
      newMissing: false,
    });
  });

  it('defaults a partial diff to an empty patch of a root commit', async () => {
    // A root commit has no `old_ref` at all, and an untracked file has no old
    // side -- both arrive as nulls the view has to draw rather than crash on.
    mockFetch(200, { old_missing: true });
    expect(await getGitDiff({ path: 'new.md', scope: 'worktree' })).toEqual({
      patch: '',
      binary: false,
      truncated: false,
      oldRef: null,
      newRef: null,
      oldMissing: true,
      newMissing: false,
    });
  });

  it('keeps the ignored code a refused path answers with', async () => {
    // 403, raised before any git call, for an ignored worktree file and for
    // anything `.env`-shaped at any ref.
    mockFetch(403, {
      detail: { code: 'ignored', message: '.env is not readable', hint: null, stderr: null },
    });
    const err = await gitError(getGitDiff({ path: '.env', scope: 'worktree' }));
    expect(err.status).toBe(403);
    expect(err.code).toBe('ignored');
  });
});

describe('getGitFile', () => {
  it('GETs the path and the ref it was asked for', async () => {
    const fetchMock = mockFetch(200, { text: 'x', binary: false, size: 1, truncated: false });
    await getGitFile({ path: 'graphs/a.graph.json', ref: 'HEAD' });
    const url = calledUrl(fetchMock);
    expect(url.pathname).toBe('/api/git/file');
    expect(url.searchParams.get('path')).toBe('graphs/a.graph.json');
    expect(url.searchParams.get('ref')).toBe('HEAD');
    expect([...url.searchParams.keys()]).toEqual(['path', 'ref']);
  });

  it('normalizes every field, keeping the pre-truncation size', async () => {
    // A blob over the 2 MiB cap is never read: the text is empty, `size` is
    // the real one, and `truncated` is how a reader tells that apart from a
    // file that really is empty.
    mockFetch(200, { text: '', binary: false, size: 4_194_304, truncated: true });
    expect(await getGitFile({ path: 'big.bin', ref: 'worktree' })).toEqual({
      text: '',
      binary: false,
      size: 4_194_304,
      truncated: true,
    });
  });

  it('defaults a partial answer to an empty, readable file', async () => {
    mockFetch(200, {});
    expect(await getGitFile({ path: 'a.md', ref: 'index' })).toEqual({
      text: '',
      binary: false,
      size: 0,
      truncated: false,
    });
  });

  it('keeps the ignored code at every ref', async () => {
    mockFetch(403, {
      detail: { code: 'ignored', message: 'ignored by git', hint: null, stderr: null },
    });
    const err = await gitError(getGitFile({ path: 'dist/app.js', ref: 'worktree' }));
    expect(err.code).toBe('ignored');
  });
});

describe('the four history and diff reads', () => {
  // All four are OPEN GETs on the server's `read` deadline -- the bucket whose
  // number `git.error.timeout {seconds}` fills in, because a 504 carries the
  // code and nothing else. A bare `fetch(url)` with no init at all is what
  // makes them open: no method for the auth guard to catch, so no token.
  it('are bare GETs with no init, like every other read', async () => {
    expect(GIT_TIMEOUTS_S.read).toBe(20);

    const reads: Array<[unknown, () => Promise<unknown>]> = [
      [{}, () => getGitLog(0, 30)],
      [[], () => getGitCommitFiles('abc')],
      [{}, () => getGitDiff({ path: 'a.md', scope: 'worktree' })],
      [{}, () => getGitFile({ path: 'a.md', ref: 'HEAD' })],
    ];
    for (const [body, call] of reads) {
      const fetchMock = mockFetch(200, body);
      await call();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0][1]).toBeUndefined();
    }
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
