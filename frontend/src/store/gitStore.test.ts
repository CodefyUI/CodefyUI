import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as gitApi from '../api/git';
import type {
  FileKind,
  GitFile,
  GitStatus,
  Identity,
  MutationResult,
  RepoInfo,
  RepoState,
  StatusResponse,
} from '../api/git';

/**
 * The Source Control store, without a server and without a clock.
 *
 * Partial mock, the way `pluginStore.test.ts` mocks `../api/rest`: the store
 * narrows refusals with `instanceof GitApiError` and builds them here with
 * the REAL `gitApiError`, so the three body shapes the backend can answer
 * with (a coded dict, FastAPI's 422 list, the auth guard's 403 string) are
 * exercised through the same reader the app uses rather than through a
 * hand-built stand-in.
 */
vi.mock('../api/git', async (importOriginal) => {
  const actual = await importOriginal<typeof gitApi>();
  return {
    ...actual,
    getGitStatus: vi.fn(),
    getGitConfig: vi.fn(),
    gitInit: vi.fn(),
    gitStage: vi.fn(),
    gitUnstage: vi.fn(),
    gitDiscard: vi.fn(),
    gitCommit: vi.fn(),
    setGitConfig: vi.fn(),
  };
});

// The reload confirmation is an in-app modal driven by a promise; mocking the
// helper keeps these cases about the STORE's decisions.
vi.mock('../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

// `GraphMissingError` is a real class the store narrows on, so only the
// reload itself is stubbed -- the real one fetches a graph and installs it
// into a tab, neither of which this file is about.
vi.mock('../utils/openSavedGraph', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../utils/openSavedGraph')>();
  return { ...actual, reloadTabFromDisk: vi.fn(async () => false) };
});

import {
  GIT_POLL_MS,
  GIT_REVISIT_DEBOUNCE_MS,
  GIT_WRITE_DEBOUNCE_MS,
  _resetGitStoreForTesting,
  gitOpKey,
  useGitStore,
} from './gitStore';
import { GraphMissingError, reloadTabFromDisk } from '../utils/openSavedGraph';
import { announceWorktreeWrite } from '../utils/worktreeWrite';
import { confirm } from '../utils/dialog';
import { useProjectStore } from './projectStore';
import { useTabStore } from './tabStore';
import { useToastStore } from './toastStore';
import { useI18n, type TranslationKey } from '../i18n';

const api = vi.mocked(gitApi);
const confirmMock = vi.mocked(confirm);
const reloadMock = vi.mocked(reloadTabFromDisk);

/**
 * What a key renders as while the `git.*` strings are still Task 3's.
 *
 * The fake `t` below answers with this, so every assertion names a KEY and
 * its interpolated values rather than a sentence -- which is what keeps this
 * file green when the real copy lands, and what makes a renamed key fail
 * here instead of on screen.
 */
function say(key: string, vars?: Record<string, string | number>): string {
  return vars === undefined ? key : `${key}:${JSON.stringify(vars)}`;
}

// Recorded before anything replaces it: `_resetGitStoreForTesting` restores
// this store's data, and nothing restores another store's actions.
const realT = useI18n.getState().t;

/* ── wire fixtures ───────────────────────────────────────────────────── */

function gitFile(path: string, kind: FileKind = 'modified'): GitFile {
  return { path, orig_path: null, kind, xy: '.M', score: null };
}

function status(over: Partial<GitStatus> = {}): GitStatus {
  return {
    branch: 'main',
    detached: false,
    head: 'abc1234def',
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
    ...over,
  };
}

function repo(state: RepoState = 'ready', over: Partial<RepoInfo> = {}): RepoInfo {
  return {
    state,
    project_dir: '/proj',
    git_version: '2.45.0',
    nested_toplevel: null,
    ...over,
  };
}

function statusResponse(
  state: RepoState = 'ready',
  over: Partial<GitStatus> = {},
): StatusResponse {
  return { repo: repo(state), status: state === 'ready' ? status(over) : null };
}

function mutation(over: Partial<MutationResult> = {}): MutationResult {
  return { status: status(), changed_paths: [], head: 'abc1234def', detail: {}, ...over };
}

function identity(over: Partial<Identity> = {}): Identity {
  return {
    name: 'Ada',
    email: 'ada@example.com',
    name_scope: 'global',
    email_scope: 'global',
    ...over,
  };
}

/** A refusal built by the real `gitApiError`, from a body off the wire. */
function refusal(httpStatus: number, body: unknown): Promise<gitApi.GitApiError> {
  const response = {
    ok: false,
    status: httpStatus,
    statusText: 'mock',
    json: async () => body,
  } as unknown as Response;
  return gitApi.gitApiError(response);
}

/** A refusal in the git envelope, which is what almost every one of them is. */
function coded(
  httpStatus: number,
  code: string,
  over: { message?: string; hint?: string | null; stderr?: string | null } = {},
): Promise<gitApi.GitApiError> {
  return refusal(httpStatus, {
    detail: {
      code,
      message: over.message ?? `git said ${code}`,
      hint: over.hint ?? null,
      stderr: over.stderr ?? null,
    },
  });
}

/* ── harness ─────────────────────────────────────────────────────────── */

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
} {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason: unknown) => void = () => undefined;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Flush the microtasks a resolved mock leaves behind. */
const settle = () => vi.advanceTimersByTimeAsync(0);

function setVisibility(value: 'visible' | 'hidden'): void {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => value,
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

const git = () => useGitStore.getState();
const toasts = () => useToastStore.getState().toasts;

/** Open a tab bound to *file*, stamped with *origin*, and answer its id. */
function openTab(name: string, file: string | null, origin: string | null): string {
  const tabs = useTabStore.getState();
  tabs.addTab(name);
  const id = useTabStore.getState().activeTabId;
  useTabStore.getState().setCurrentGraphFile(file);
  useTabStore.getState().stampActiveTabProject(origin);
  return id;
}

beforeEach(() => {
  vi.useFakeTimers();
  useToastStore.setState({ toasts: [] });
  useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
  useTabStore.setState({ tabs: [], activeTabId: '' });
  localStorage.clear();
  _resetGitStoreForTesting();
  // A fresh `vi.fn()` through `setState`, never `vi.spyOn` on an action read
  // off `getState()`: the spy would keep a stale object and carry its call
  // history into the next case.
  useI18n.setState({
    t: vi.fn((key: TranslationKey, vars?: Record<string, string | number>) =>
      say(key, vars),
    ),
  });

  api.getGitStatus.mockResolvedValue(statusResponse('ready'));
  api.getGitConfig.mockResolvedValue(identity());
  api.gitInit.mockResolvedValue(mutation());
  api.gitStage.mockResolvedValue(mutation());
  api.gitUnstage.mockResolvedValue(mutation());
  api.gitDiscard.mockResolvedValue(mutation());
  api.gitCommit.mockResolvedValue(mutation());
  api.setGitConfig.mockResolvedValue(identity());
  confirmMock.mockResolvedValue(true);
  reloadMock.mockResolvedValue(false);
});

afterEach(() => {
  _resetGitStoreForTesting();
  useI18n.setState({ t: realT });
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  Reflect.deleteProperty(document, 'visibilityState');
  vi.useRealTimers();
  vi.clearAllMocks();
});

/* ── the poll ────────────────────────────────────────────────────────── */

describe('attach / detach', () => {
  it('reads once on the first attach and then every fifteen seconds', async () => {
    git().attach();
    await settle();
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(GIT_POLL_MS);
    expect(api.getGitStatus).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(GIT_POLL_MS);
    expect(api.getGitStatus).toHaveBeenCalledTimes(3);
  });

  it('stops the poll on detach', async () => {
    git().attach();
    await settle();
    git().detach();

    await vi.advanceTimersByTimeAsync(GIT_POLL_MS * 3);
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
  });

  it('two attaches share one poll, and one detach does not stop it', async () => {
    git().attach();
    git().attach();
    await settle();
    // The second attach joins the poll rather than starting a second read.
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(GIT_POLL_MS);
    expect(api.getGitStatus).toHaveBeenCalledTimes(2);

    git().detach();
    await vi.advanceTimersByTimeAsync(GIT_POLL_MS);
    expect(api.getGitStatus).toHaveBeenCalledTimes(3);

    git().detach();
    await vi.advanceTimersByTimeAsync(GIT_POLL_MS * 3);
    expect(api.getGitStatus).toHaveBeenCalledTimes(3);
  });

  it('pauses while the page is hidden and reads again when it comes back', async () => {
    git().attach();
    await settle();
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);

    setVisibility('hidden');
    await vi.advanceTimersByTimeAsync(GIT_POLL_MS * 3);
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);

    setVisibility('visible');
    await vi.advanceTimersByTimeAsync(GIT_REVISIT_DEBOUNCE_MS);
    // Coming back is exactly when the status is most likely to be wrong.
    expect(api.getGitStatus).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(GIT_POLL_MS);
    expect(api.getGitStatus).toHaveBeenCalledTimes(3);
  });

  it('attaching on a hidden page reads once and starts no interval', async () => {
    setVisibility('hidden');
    git().attach();
    await settle();
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(GIT_POLL_MS * 3);
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
  });

  it('refreshes when the window is focused', async () => {
    git().attach();
    await settle();

    window.dispatchEvent(new Event('focus'));
    await vi.advanceTimersByTimeAsync(GIT_REVISIT_DEBOUNCE_MS);
    expect(api.getGitStatus).toHaveBeenCalledTimes(2);
  });

  it('answers a focus and a visibilitychange in one tick with a single read', async () => {
    git().attach();
    await settle();
    api.getGitStatus.mockClear();

    // Returning to the tab fires both, in the same tick: two identical
    // `GET /status` at the same millisecond, on every single return.
    window.dispatchEvent(new Event('focus'));
    setVisibility('visible');
    await vi.advanceTimersByTimeAsync(GIT_REVISIT_DEBOUNCE_MS - 1);
    expect(api.getGitStatus).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
  });

  it('drops a pending revisit read when the tab closes first', async () => {
    git().attach();
    await settle();
    api.getGitStatus.mockClear();

    window.dispatchEvent(new Event('focus'));
    git().detach();
    await vi.advanceTimersByTimeAsync(GIT_REVISIT_DEBOUNCE_MS * 4);
    expect(api.getGitStatus).not.toHaveBeenCalled();
  });

  it('ignores focus and visibility once nothing is attached', async () => {
    git().attach();
    await settle();
    git().detach();

    window.dispatchEvent(new Event('focus'));
    setVisibility('hidden');
    setVisibility('visible');
    await settle();
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
  });
});

/* ── refresh ─────────────────────────────────────────────────────────── */

describe('refresh', () => {
  it('shows loading only until the first answer lands', async () => {
    const first = deferred<StatusResponse>();
    api.getGitStatus.mockReturnValueOnce(first.promise);

    git().attach();
    expect(git().loading).toBe(true);
    expect(git().repoState).toBe('unknown');

    first.resolve(statusResponse('ready', { staged: [gitFile('graphs/a.graph.json')] }));
    await settle();
    expect(git().loading).toBe(false);
    expect(git().repoState).toBe('ready');
    expect(git().status?.staged).toHaveLength(1);
    expect(git().repo?.project_dir).toBe('/proj');
  });

  it('treats every repository state as an answer, not a failure', async () => {
    api.getGitStatus.mockResolvedValue(statusResponse('not_repo'));
    await git().refresh();

    expect(git().repoState).toBe('not_repo');
    expect(git().status).toBeNull();
    expect(git().loadError).toBeNull();
  });

  it('reports a read that failed and leaves the repository state alone', async () => {
    api.getGitStatus.mockRejectedValue(new Error('Failed to fetch'));
    await git().refresh();

    expect(git().loadError).toBe('Failed to fetch');
    expect(git().repoState).toBe('unknown');
    expect(git().loading).toBe(false);
  });

  it('says how long the status read waited before it gave up', async () => {
    // A status read is two real git processes under the server's deadline, not
    // a lookup, so it can be stopped at ten seconds -- and the 504 carries the
    // code and no number, which is why the sentence is written on this side.
    api.getGitStatus.mockRejectedValue(await coded(504, 'timeout'));
    await git().refresh();

    expect(git().loadError).toBe(say('git.error.timeout', { seconds: 10 }));
    expect(git().repoState).toBe('unknown');
  });

  it('drops a slow read that a newer one has already overtaken', async () => {
    const slow = deferred<StatusResponse>();
    api.getGitStatus.mockReturnValueOnce(slow.promise);
    const stale = git().refresh();

    api.getGitStatus.mockResolvedValueOnce(
      statusResponse('ready', { untracked: [gitFile('graphs/new.graph.json', 'untracked')] }),
    );
    await git().refresh();
    expect(git().status?.untracked).toHaveLength(1);

    slow.resolve(statusResponse('not_repo'));
    await stale;
    await settle();
    expect(git().repoState).toBe('ready');
    expect(git().status?.untracked).toHaveLength(1);
  });

  it('drops a slow read that a write has already overtaken', async () => {
    const slow = deferred<StatusResponse>();
    api.getGitStatus.mockReturnValueOnce(slow.promise);
    const stale = git().refresh();

    api.gitStage.mockResolvedValue(
      mutation({ status: status({ staged: [gitFile('graphs/a.graph.json', 'added')] }) }),
    );
    await git().stage(['graphs/a.graph.json']);
    expect(git().status?.staged).toHaveLength(1);

    // The read started before the stage: its answer describes a repository
    // that no longer exists, and applying it would put the file back.
    slow.resolve(statusResponse('ready'));
    await stale;
    await settle();
    expect(git().status?.staged).toHaveLength(1);
  });
});

/* ── writes ──────────────────────────────────────────────────────────── */

describe('writes', () => {
  it('applies the status each write answered with', async () => {
    api.gitStage.mockResolvedValue(
      mutation({ status: status({ staged: [gitFile('a.py', 'added')] }) }),
    );
    expect(await git().stage(['a.py'])).toBe(true);
    expect(git().status?.staged).toHaveLength(1);

    api.gitUnstage.mockResolvedValue(
      mutation({ status: status({ unstaged: [gitFile('a.py')] }) }),
    );
    await git().unstage('all');
    expect(git().status?.staged).toHaveLength(0);
    expect(git().status?.unstaged).toHaveLength(1);

    api.gitDiscard.mockResolvedValue(mutation({ status: status() }));
    await git().discard('all');
    expect(git().status?.unstaged).toHaveLength(0);

    git().setCommitMessage('a message');
    api.gitCommit.mockResolvedValue(
      mutation({ status: status({ head: 'ffff111' }), head: 'ffff111' }),
    );
    await git().commit();
    expect(git().status?.head).toBe('ffff111');
  });

  it('passes the selection through as given', async () => {
    await git().stage(['a.py', 'b.py']);
    expect(api.gitStage).toHaveBeenCalledWith(['a.py', 'b.py']);

    await git().unstage('all');
    expect(api.gitUnstage).toHaveBeenCalledWith('all');
  });

  it('refuses a second write while one is running', async () => {
    const slow = deferred<MutationResult>();
    api.gitStage.mockReturnValueOnce(slow.promise);
    const running = git().stage(['a.py']);
    expect(git().busyOp).toBe('stage');

    expect(await git().unstage(['a.py'])).toBe(false);
    expect(api.gitUnstage).not.toHaveBeenCalled();
    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe(say('git.error.busy'));
    expect(toasts()[0].type).toBe('warning');

    slow.resolve(mutation());
    await running;
    expect(git().busyOp).toBeNull();
  });

  it('clears busyOp when a write fails', async () => {
    api.gitStage.mockRejectedValue(await coded(500, 'git_failed'));
    expect(await git().stage(['a.py'])).toBe(false);
    expect(git().busyOp).toBeNull();
    expect(git().lastError?.code).toBe('git_failed');
  });

  it('makes no request at all for an empty selection', async () => {
    expect(await git().stage([])).toBe(false);
    expect(await git().unstage([])).toBe(false);
    expect(await git().discard([])).toBe(false);

    expect(api.gitStage).not.toHaveBeenCalled();
    expect(api.gitUnstage).not.toHaveBeenCalled();
    expect(api.gitDiscard).not.toHaveBeenCalled();
    expect(toasts()).toHaveLength(0);
    expect(git().lastError).toBeNull();
  });

  it('names the new commit in a toast and in the live region', async () => {
    git().setCommitMessage('  a message  ');
    api.gitCommit.mockResolvedValue(
      mutation({ detail: { sha: 'abc1234def5678', short: 'abc1234' } }),
    );
    await git().commit();

    expect(api.gitCommit).toHaveBeenCalledWith({
      message: 'a message',
      all: false,
      amend: false,
    });
    expect(git().liveMessage).toBe(say('git.toast.committed', { sha: 'abc1234' }));
    expect(toasts()[0].message).toBe(say('git.toast.committed', { sha: 'abc1234' }));
    expect(toasts()[0].type).toBe('success');
  });

  it('falls back to the long id, then to the new HEAD, for the commit name', async () => {
    git().setCommitMessage('m');
    api.gitCommit.mockResolvedValue(mutation({ detail: { sha: 'abc1234def5678' } }));
    await git().commit();
    expect(git().liveMessage).toBe(say('git.toast.committed', { sha: 'abc1234def5678' }));

    git().setCommitMessage('m');
    api.gitCommit.mockResolvedValue(mutation({ detail: {}, head: 'ffff111' }));
    await git().commit();
    expect(git().liveMessage).toBe(say('git.toast.committed', { sha: 'ffff111' }));
  });

  it('clears the message and the amend flag after a commit', async () => {
    git().setCommitMessage('a message');
    git().setAmend(true);
    await git().commit({ all: true });

    expect(api.gitCommit).toHaveBeenCalledWith({
      message: 'a message',
      all: true,
      amend: true,
    });
    expect(git().commitMessage).toBe('');
    expect(git().amend).toBe(false);
  });

  it('keeps the message when the commit is refused', async () => {
    git().setCommitMessage('a message');
    api.gitCommit.mockRejectedValue(await coded(409, 'nothing_to_commit'));
    expect(await git().commit()).toBe(false);
    expect(git().commitMessage).toBe('a message');
  });

  it('never commits an empty message', async () => {
    git().setCommitMessage('   ');
    expect(await git().commit()).toBe(false);
    expect(api.gitCommit).not.toHaveBeenCalled();
  });

  it('re-reads the repository after init, so the screen moves on', async () => {
    api.getGitStatus.mockResolvedValueOnce(statusResponse('not_repo'));
    await git().refresh();
    expect(git().repoState).toBe('not_repo');

    api.getGitStatus.mockResolvedValueOnce(statusResponse('ready'));
    expect(await git().init()).toBe(true);

    expect(git().repoState).toBe('ready');
    expect(toasts()[0].message).toBe(say('git.toast.initialized'));
    expect(git().liveMessage).toBe(say('git.toast.initialized'));
  });

  it('says how many files a whole-tree write skipped', async () => {
    api.gitStage.mockResolvedValue(
      mutation({ detail: { skipped: ['notes/a.txt', 'notes/b.txt'] } }),
    );
    await git().stage('all');

    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe(say('git.group.skipped', { count: 2 }));
    expect(toasts()[0].type).toBe('info');
  });

  it('says nothing when nothing was skipped', async () => {
    api.gitStage.mockResolvedValue(mutation({ detail: { skipped: [] } }));
    await git().stage('all');
    api.gitUnstage.mockResolvedValue(mutation({ detail: { skipped: 'all of them' } }));
    await git().unstage('all');

    expect(toasts()).toHaveLength(0);
  });

  it('announces the group counts after a stage', async () => {
    api.gitStage.mockResolvedValue(
      mutation({
        status: status({
          staged: [gitFile('a.py', 'added'), gitFile('b.py', 'added')],
          unstaged: [gitFile('c.py')],
          untracked: [gitFile('d.py', 'untracked')],
        }),
      }),
    );
    await git().stage('all');

    expect(git().liveMessage).toBe('git.group.staged 2, git.group.changes 2');
  });

  it('announces the counts the panel is showing, not the ones it hides', async () => {
    // "Hide layout files" takes the layout half of every save out of the
    // Changes group, and the sentence has to say what the heading beside it
    // says -- 1, not the 3 that are in the status.
    git().setHideLayout(true);
    api.gitStage.mockResolvedValue(
      mutation({
        status: status({
          staged: [gitFile('graphs/a.graph.json', 'added')],
          unstaged: [gitFile('graphs/b.graph.json'), gitFile('layout/b.layout.json')],
          untracked: [gitFile('layout/c.layout.json', 'untracked')],
        }),
      }),
    );
    await git().stage(['graphs/a.graph.json']);
    expect(git().liveMessage).toBe('git.group.staged 1, git.group.changes 1');

    // The counterfactual: the same status with the filter off counts all
    // three, which is what the panel draws then too.
    git().setHideLayout(false);
    await git().stage(['graphs/a.graph.json']);
    expect(git().liveMessage).toBe('git.group.staged 1, git.group.changes 3');
  });

  it('lets a component say something the writes never would', () => {
    // The commit chord's refusal never reaches the server, so nothing in
    // `runOp` can announce it.
    git().announce('Enter a message');
    expect(git().liveMessage).toBe('Enter a message');
  });
});

/* ── refusals ────────────────────────────────────────────────────────── */

describe('refusals', () => {
  it('opens the identity form and reads the config when git has no identity', async () => {
    git().setCommitMessage('a message');
    api.gitCommit.mockRejectedValue(await coded(409, 'identity_missing'));
    api.getGitConfig.mockResolvedValue(identity({ name: null, name_scope: null }));

    await git().commit();
    await settle();

    expect(git().identityFormOpen).toBe(true);
    expect(git().lastError?.code).toBe('identity_missing');
    expect(api.getGitConfig).toHaveBeenCalledTimes(1);
    expect(git().identity?.name).toBeNull();
  });

  it('answers a busy refusal with a toast and no error line', async () => {
    api.gitStage.mockRejectedValue(
      await refusal(409, { detail: { code: 'busy', message: 'a commit is running', op: 'commit' } }),
    );
    expect(await git().stage(['a.py'])).toBe(false);

    expect(git().lastError).toBeNull();
    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe(say('git.error.busy'));
    expect(toasts()[0].type).toBe('warning');
  });

  it('says how long the server waited before it gave up', async () => {
    api.gitCommit.mockRejectedValue(await coded(504, 'timeout'));
    git().setCommitMessage('a message');
    await git().commit();
    expect(git().lastError?.message).toBe(say('git.error.timeout', { seconds: 30 }));

    git().dismissError();
    api.getGitConfig.mockRejectedValue(await coded(504, 'timeout'));
    git().openIdentityForm();
    await settle();
    // The config is a READ, and reads get twenty seconds, not thirty.
    expect(git().lastError?.message).toBe(say('git.error.timeout', { seconds: 20 }));
  });

  it('keeps every part of a coded refusal', async () => {
    api.gitDiscard.mockRejectedValue(
      await coded(400, 'invalid_path', {
        message: 'notes is a submodule',
        hint: 'submodules are not managed here',
        stderr: 'fatal: pathspec',
      }),
    );
    await git().discard(['notes']);

    expect(git().lastError).toEqual({
      code: 'invalid_path',
      message: 'notes is a submodule',
      hint: 'submodules are not managed here',
      stderr: 'fatal: pathspec',
    });
  });

  it('folds a 422 list detail into one invalid error', async () => {
    api.gitStage.mockRejectedValue(
      await refusal(422, {
        detail: [
          { msg: 'exactly one of paths or all' },
          { msg: 'paths must not be empty' },
        ],
      }),
    );
    await git().stage(['a.py']);

    expect(git().lastError?.code).toBe('invalid');
    expect(git().lastError?.message).toBe(
      'exactly one of paths or all; paths must not be empty',
    );
  });

  it('reads the auth guard string detail as an unknown refusal', async () => {
    api.gitStage.mockRejectedValue(
      await refusal(403, { detail: 'Missing or invalid X-CodefyUI-Token header' }),
    );
    await git().stage(['a.py']);

    expect(git().lastError?.code).toBe('unknown');
    expect(git().lastError?.message).toBe('Missing or invalid X-CodefyUI-Token header');
  });

  it('reads a rejection that is not a git refusal at all as unknown', async () => {
    api.gitStage.mockRejectedValue(new Error('socket hung up'));
    await git().stage(['a.py']);

    expect(git().lastError).toEqual({
      code: 'unknown',
      message: 'socket hung up',
      hint: null,
      stderr: null,
    });
  });

  it('clears the previous error when the next write starts', async () => {
    api.gitStage.mockRejectedValue(await coded(500, 'git_failed'));
    await git().stage(['a.py']);
    expect(git().lastError).not.toBeNull();

    api.gitStage.mockResolvedValue(mutation());
    await git().stage(['a.py']);
    expect(git().lastError).toBeNull();
  });

  it('dismissError takes the line down', async () => {
    api.gitStage.mockRejectedValue(await coded(500, 'git_failed'));
    await git().stage(['a.py']);
    git().dismissError();
    expect(git().lastError).toBeNull();
  });
});

/* ── the save hook ───────────────────────────────────────────────────── */

describe('noteWorktreeWrite', () => {
  it('coalesces a burst of saves into one read', async () => {
    git().attach();
    await settle();
    api.getGitStatus.mockClear();

    git().noteWorktreeWrite();
    git().noteWorktreeWrite();
    git().noteWorktreeWrite();
    await vi.advanceTimersByTimeAsync(GIT_WRITE_DEBOUNCE_MS - 1);
    expect(api.getGitStatus).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
  });

  it('does nothing at all when the tab has never been opened', async () => {
    git().noteWorktreeWrite();
    await vi.advanceTimersByTimeAsync(GIT_WRITE_DEBOUNCE_MS * 4);
    expect(api.getGitStatus).not.toHaveBeenCalled();
  });

  it('drops a pending read when the tab closes', async () => {
    git().attach();
    await settle();
    api.getGitStatus.mockClear();

    git().noteWorktreeWrite();
    git().detach();
    await vi.advanceTimersByTimeAsync(GIT_WRITE_DEBOUNCE_MS * 4);
    expect(api.getGitStatus).not.toHaveBeenCalled();
  });

  it('hears a save announced by the save path while the tab is attached', async () => {
    git().attach();
    await settle();
    api.getGitStatus.mockClear();

    announceWorktreeWrite();
    await vi.advanceTimersByTimeAsync(GIT_WRITE_DEBOUNCE_MS);
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
  });

  it('an announced save reaches nobody once the tab is gone', async () => {
    git().attach();
    await settle();
    git().detach();
    api.getGitStatus.mockClear();

    announceWorktreeWrite();
    await vi.advanceTimersByTimeAsync(GIT_WRITE_DEBOUNCE_MS * 4);
    expect(api.getGitStatus).not.toHaveBeenCalled();
  });
});

/* ── worktree changes under open tabs ────────────────────────────────── */

describe('applyWorktreeChange', () => {
  const pair = ['graphs/demo.graph.json', 'layout/demo.layout.json'];

  it('offers to reload the open graph a discard changed', async () => {
    openTab('demo', 'demo', '/proj');
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: pair }));
    await git().discard('all');

    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe(say('git.toast.changedOnDisk', { count: 1 }));
    expect(toasts()[0].type).toBe('warning');
    expect(toasts()[0].action?.label).toBe(say('git.toast.reload'));

    // Sticky: a warning that vanishes after four seconds is a warning nobody
    // who looked away will ever see, and this one is the only thing standing
    // between an open tab and a silently older graph.
    await vi.advanceTimersByTimeAsync(10_000);
    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].action?.label).toBe(say('git.toast.reload'));
  });

  it('never offers after a commit, whose changed_paths are already on screen', async () => {
    openTab('demo', 'demo', '/proj');
    git().setCommitMessage('a message');
    api.gitCommit.mockResolvedValue(
      mutation({ changed_paths: pair, detail: { short: 'abc1234' } }),
    );
    await git().commit();

    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe(say('git.toast.committed', { sha: 'abc1234' }));
  });

  it('reads the layout half and the legacy spelling as the same graph', async () => {
    openTab('demo', 'demo', '/proj');
    api.gitDiscard.mockResolvedValue(
      mutation({ changed_paths: ['layout/demo.layout.json'] }),
    );
    await git().discard('all');
    expect(toasts()).toHaveLength(1);

    useToastStore.setState({ toasts: [] });
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: ['graphs/demo.json'] }));
    await git().discard('all');
    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe(say('git.toast.changedOnDisk', { count: 1 }));
  });

  it('ignores paths that are not saved graphs', async () => {
    openTab('demo', 'demo', '/proj');
    api.gitDiscard.mockResolvedValue(
      mutation({
        changed_paths: [
          '.gitignore',
          'graphs/demo.graph.json.bak',
          'nested/graphs/demo.graph.json',
          'README.md',
        ],
      }),
    );
    await git().discard('all');
    expect(toasts()).toHaveLength(0);
  });

  it('leaves a tab stamped with another project alone', async () => {
    openTab('demo', 'demo', '/somewhere-else');
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: pair }));
    await git().discard('all');
    expect(toasts()).toHaveLength(0);
  });

  it('includes a tab that has never been saved into a project', async () => {
    openTab('demo', 'demo', null);
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: pair }));
    await git().discard('all');
    expect(toasts()).toHaveLength(1);
  });

  it('replaces the standing offer instead of stacking a second one', async () => {
    openTab('demo', 'demo', '/proj');
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: pair }));
    await git().discard('all');
    const first = toasts()[0].id;

    openTab('other', 'other', '/proj');
    api.gitDiscard.mockResolvedValue(
      mutation({ changed_paths: [...pair, 'graphs/other.graph.json'] }),
    );
    await git().discard('all');

    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].id).not.toBe(first);
    expect(toasts()[0].message).toBe(say('git.toast.changedOnDisk', { count: 2 }));
  });

  it('asks once and then reloads every affected tab', async () => {
    const one = openTab('demo', 'demo', '/proj');
    const two = openTab('other', 'other', '/proj');
    api.gitDiscard.mockResolvedValue(
      mutation({ changed_paths: [...pair, 'graphs/other.graph.json'] }),
    );
    await git().discard('all');

    toasts()[0].action?.onClick();
    await settle();

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(confirmMock.mock.calls[0][0]).toEqual({
      title: say('git.toast.reloadConfirm', { count: 2 }),
      variant: 'danger',
    });
    expect(reloadMock).toHaveBeenCalledTimes(2);
    expect(reloadMock).toHaveBeenCalledWith(one, 'demo');
    expect(reloadMock).toHaveBeenCalledWith(two, 'other');
  });

  it('reloads nothing when the confirmation is declined', async () => {
    openTab('demo', 'demo', '/proj');
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: pair }));
    await git().discard('all');

    confirmMock.mockResolvedValue(false);
    toasts()[0].action?.onClick();
    await settle();
    expect(reloadMock).not.toHaveBeenCalled();
  });

  it('keeps the tab when the graph is gone on this branch', async () => {
    const one = openTab('demo', 'demo', '/proj');
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: pair }));
    await git().discard('all');

    reloadMock.mockRejectedValue(new GraphMissingError('demo'));
    toasts()[0].action?.onClick();
    await settle();

    // Whatever the tab is showing is now the only copy of it anywhere.
    expect(useTabStore.getState().getTab(one)).toBeDefined();
    const info = toasts().filter((toast) => toast.type === 'info');
    expect(info).toHaveLength(1);
    expect(info[0].message).toBe(say('git.toast.missingOnDisk', { name: 'demo' }));
  });

  it('skips a tab that was closed between the offer and the click', async () => {
    const one = openTab('demo', 'demo', '/proj');
    openTab('keep', null, '/proj');
    api.gitDiscard.mockResolvedValue(mutation({ changed_paths: pair }));
    await git().discard('all');

    useTabStore.getState().removeTab(one);
    toasts()[0].action?.onClick();
    await settle();
    expect(reloadMock).not.toHaveBeenCalled();
  });
});

/* ── preferences and vocabulary ──────────────────────────────────────── */

describe('hideLayout', () => {
  it('remembers the choice across a reload', () => {
    expect(git().hideLayout).toBe(false);

    git().setHideLayout(true);
    expect(localStorage.getItem('codefyui-git-hide-layout')).toBe('true');
    _resetGitStoreForTesting();
    expect(git().hideLayout).toBe(true);

    git().setHideLayout(false);
    _resetGitStoreForTesting();
    expect(git().hideLayout).toBe(false);
  });
});

describe('the identity form', () => {
  it('reads the config when it opens and closes on a successful save', async () => {
    git().openIdentityForm();
    await settle();
    expect(git().identityFormOpen).toBe(true);
    expect(git().identity?.email).toBe('ada@example.com');

    api.setGitConfig.mockResolvedValue(
      identity({ name: 'Grace', email: 'grace@example.com', name_scope: 'local' }),
    );
    expect(await git().saveIdentity({ name: ' Grace ', email: ' grace@example.com ' })).toBe(
      true,
    );

    expect(api.setGitConfig).toHaveBeenCalledWith({
      name: 'Grace',
      email: 'grace@example.com',
    });
    expect(git().identity?.name_scope).toBe('local');
    expect(git().identityFormOpen).toBe(false);
  });

  it('stays open when the save is refused', async () => {
    git().openIdentityForm();
    api.setGitConfig.mockRejectedValue(await coded(400, 'invalid_value'));
    expect(await git().saveIdentity({ name: 'Grace', email: 'x' })).toBe(false);

    expect(git().identityFormOpen).toBe(true);
    expect(git().lastError?.code).toBe('invalid_value');
  });

  it('sends only the half that was filled in', async () => {
    // An absent key means "leave that one alone"; an empty string is a value
    // the route refuses with a 400. Setting only the email -- because the
    // name is already global -- must not drag `name: ""` along with it.
    await git().saveIdentity({ name: '', email: 'ada@example.com' });
    expect(api.setGitConfig).toHaveBeenLastCalledWith({ email: 'ada@example.com' });

    await git().saveIdentity({ name: 'Ada', email: '   ' });
    expect(api.setGitConfig).toHaveBeenLastCalledWith({ name: 'Ada' });
  });

  it('sends nothing when both halves are empty', async () => {
    expect(await git().saveIdentity({ name: '  ', email: '' })).toBe(false);
    expect(api.setGitConfig).not.toHaveBeenCalled();
  });

  it('a slow config read cannot put the old identity back after a save', async () => {
    const slow = deferred<Identity>();
    api.getGitConfig.mockReturnValueOnce(slow.promise);
    git().openIdentityForm();

    api.setGitConfig.mockResolvedValue(
      identity({ name: 'Grace', email: 'grace@example.com', name_scope: 'local' }),
    );
    await git().saveIdentity({ name: 'Grace', email: 'grace@example.com' });
    expect(git().identity?.name).toBe('Grace');

    // The read was started by the form opening, before the write existed.
    slow.resolve(identity({ name: 'Ada', email: 'ada@example.com' }));
    await settle();
    expect(git().identity?.name).toBe('Grace');
    expect(git().identity?.name_scope).toBe('local');
  });

  it('closeIdentityForm closes it', () => {
    git().openIdentityForm();
    git().closeIdentityForm();
    expect(git().identityFormOpen).toBe(false);
  });
});

describe('gitOpKey', () => {
  it('spells the identity op the way the key does, not the way the wire does', () => {
    expect(gitOpKey('identity')).toBe('git.op.identity');
    expect(gitOpKey('status')).toBe('git.op.status');
    expect(gitOpKey('commit')).toBe('git.op.commit');
  });
});
