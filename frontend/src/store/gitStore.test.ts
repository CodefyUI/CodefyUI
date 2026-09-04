import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as gitApi from '../api/git';
import type {
  BranchesResponse,
  FileKind,
  GitFile,
  GitStatus,
  Identity,
  MutationResult,
  RemoteInfo,
  RepoInfo,
  RepoState,
  StashInfo,
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
    getGitBranches: vi.fn(),
    getGitRemotes: vi.fn(),
    getGitStashes: vi.fn(),
    gitInit: vi.fn(),
    gitStage: vi.fn(),
    gitUnstage: vi.fn(),
    gitDiscard: vi.fn(),
    gitCommit: vi.fn(),
    setGitConfig: vi.fn(),
    gitFetch: vi.fn(),
    gitPull: vi.fn(),
    gitPush: vi.fn(),
    gitSync: vi.fn(),
    gitCreateBranch: vi.fn(),
    gitCheckout: vi.fn(),
    gitRenameBranch: vi.fn(),
    gitDeleteBranch: vi.fn(),
    gitAddRemote: vi.fn(),
    gitSetRemoteUrl: vi.fn(),
    gitRemoveRemote: vi.fn(),
    gitStashPush: vi.fn(),
    gitStashPop: vi.fn(),
    gitStashApply: vi.fn(),
    gitStashDrop: vi.fn(),
    gitAbortMerge: vi.fn(),
    gitResolve: vi.fn(),
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

// Recorded before anything replaces them: `_resetGitStoreForTesting` restores
// this store's data, and nothing restores another store's actions.
const realT = useI18n.getState().t;
const realApplyWorktreeChange = useGitStore.getState().applyWorktreeChange;

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

function branches(name = 'main'): BranchesResponse {
  return {
    current: name,
    detached: false,
    local: [
      {
        name,
        sha: 'abc1234',
        current: true,
        upstream: `origin/${name}`,
        ahead: 0,
        behind: 0,
        gone: false,
        subject: 'Current work',
        committed_at: 123,
      },
    ],
    remote: [],
  };
}

function remotes(name = 'origin'): RemoteInfo[] {
  return [{ name, fetch_url: `https://example/${name}`, push_url: `https://example/${name}` }];
}

function stashes(index = 0): StashInfo[] {
  return [{ index, message: `stash ${index}`, branch: 'main', created_at: 123 }];
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
  useGitStore.setState({ applyWorktreeChange: realApplyWorktreeChange });
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
  api.getGitBranches.mockResolvedValue(branches());
  api.getGitRemotes.mockResolvedValue(remotes());
  api.getGitStashes.mockResolvedValue(stashes());
  api.gitInit.mockResolvedValue(mutation());
  api.gitStage.mockResolvedValue(mutation());
  api.gitUnstage.mockResolvedValue(mutation());
  api.gitDiscard.mockResolvedValue(mutation());
  api.gitCommit.mockResolvedValue(mutation());
  api.setGitConfig.mockResolvedValue(identity());
  api.gitFetch.mockResolvedValue(mutation());
  api.gitPull.mockResolvedValue(mutation());
  api.gitPush.mockResolvedValue(mutation());
  api.gitSync.mockResolvedValue(mutation());
  api.gitCreateBranch.mockResolvedValue(mutation());
  api.gitCheckout.mockResolvedValue(mutation());
  api.gitRenameBranch.mockResolvedValue(mutation());
  api.gitDeleteBranch.mockResolvedValue(mutation());
  api.gitAddRemote.mockResolvedValue(mutation());
  api.gitSetRemoteUrl.mockResolvedValue(mutation());
  api.gitRemoveRemote.mockResolvedValue(mutation());
  api.gitStashPush.mockResolvedValue(mutation());
  api.gitStashPop.mockResolvedValue(mutation());
  api.gitStashApply.mockResolvedValue(mutation());
  api.gitStashDrop.mockResolvedValue(mutation());
  api.gitAbortMerge.mockResolvedValue(mutation());
  api.gitResolve.mockResolvedValue(mutation());
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

  it('drops a pending revisit read when the page hides first', async () => {
    git().attach();
    await settle();
    api.getGitStatus.mockClear();

    // A focus and then the page going away inside the window. The read the
    // focus queued would land on a page nobody is looking at, which is what
    // the pause is for.
    window.dispatchEvent(new Event('focus'));
    setVisibility('hidden');
    await vi.advanceTimersByTimeAsync(GIT_REVISIT_DEBOUNCE_MS * 4);
    expect(api.getGitStatus).not.toHaveBeenCalled();
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

  // A commit is the one local write that certainly moves the current branch,
  // and an open Branches section shows that branch's sha, subject and ahead
  // count. Without this the header would say "1 to push" (which comes from the
  // status) beside a row that still said 0 until the fifteen-second poll.
  it('refreshes the branch list after a commit and after an init', async () => {
    git().setCommitMessage('a message');
    api.getGitBranches.mockResolvedValue(branches('after-commit'));

    expect(await git().commit()).toBe(true);
    expect(git().branches?.current).toBe('after-commit');

    api.getGitBranches.mockClear();
    api.getGitBranches.mockResolvedValue(branches('after-init'));
    expect(await git().init()).toBe(true);
    expect(api.getGitBranches).toHaveBeenCalledTimes(1);
    expect(git().branches?.current).toBe('after-init');
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

/* ── refs, network lane and G3 writes ─────────────────────────────────── */

describe('reference sections', () => {
  it('starts collapsed and loads only a section when it opens', async () => {
    expect(git().sections).toEqual({ branches: false, remotes: false, stashes: false });
    expect(git().branches).toBeNull();
    expect(git().remotes).toBeNull();
    expect(git().stashes).toBeNull();

    git().setSectionOpen('branches', true);
    await settle();
    expect(api.getGitBranches).toHaveBeenCalledTimes(1);
    expect(api.getGitRemotes).not.toHaveBeenCalled();
    expect(api.getGitStashes).not.toHaveBeenCalled();
    expect(git().branches?.current).toBe('main');
    expect(localStorage.getItem('codefyui-git-sections')).toBe(
      JSON.stringify({ branches: true, remotes: false, stashes: false }),
    );

    git().setSectionOpen('remotes', true);
    git().setSectionOpen('stashes', true);
    await settle();
    expect(api.getGitRemotes).toHaveBeenCalledTimes(1);
    expect(api.getGitStashes).toHaveBeenCalledTimes(1);
    expect(git().remotes?.[0].name).toBe('origin');
    expect(git().stashes?.[0].index).toBe(0);

    api.getGitRemotes.mockClear();
    git().setSectionOpen('remotes', false);
    await settle();
    expect(api.getGitRemotes).not.toHaveBeenCalled();
  });

  it('restores all three open states and rejects malformed stored values', async () => {
    git().setSectionOpen('branches', true);
    git().setSectionOpen('stashes', true);
    await settle();
    _resetGitStoreForTesting();
    expect(git().sections).toEqual({ branches: true, remotes: false, stashes: true });

    localStorage.setItem(
      'codefyui-git-sections',
      JSON.stringify({ branches: 'yes', remotes: true, stashes: 1 }),
    );
    _resetGitStoreForTesting();
    expect(git().sections).toEqual({ branches: false, remotes: true, stashes: false });

    localStorage.setItem('codefyui-git-sections', '{not json');
    _resetGitStoreForTesting();
    expect(git().sections).toEqual({ branches: false, remotes: false, stashes: false });
  });

  it('refreshes only expanded refs on the initial read and each poll', async () => {
    git().setSectionOpen('branches', true);
    await settle();
    api.getGitBranches.mockClear();
    api.getGitRemotes.mockClear();
    api.getGitStashes.mockClear();

    git().attach();
    await settle();
    expect(api.getGitBranches).toHaveBeenCalledTimes(1);
    expect(api.getGitRemotes).not.toHaveBeenCalled();
    expect(api.getGitStashes).not.toHaveBeenCalled();

    api.getGitBranches.mockClear();
    await vi.advanceTimersByTimeAsync(GIT_POLL_MS);
    expect(api.getGitBranches).toHaveBeenCalledTimes(1);
    expect(api.getGitRemotes).not.toHaveBeenCalled();
    expect(api.getGitStashes).not.toHaveBeenCalled();
  });

  it('drops an older ref read after a newer one has answered', async () => {
    const slow = deferred<BranchesResponse>();
    api.getGitBranches.mockReturnValueOnce(slow.promise);
    const stale = git().refreshRefs('branches');

    api.getGitBranches.mockResolvedValueOnce(branches('newer'));
    await git().refreshRefs('branches');
    expect(git().branches?.current).toBe('newer');

    slow.resolve(branches('older'));
    await stale;
    expect(git().branches?.current).toBe('newer');
  });
});

describe('network operations', () => {
  it('fetch applies status, refreshes branches, and announces success', async () => {
    api.gitFetch.mockResolvedValue(
      mutation({ status: status({ head: 'after-fetch' }), detail: { remote: 'origin' } }),
    );
    api.getGitBranches.mockResolvedValue(branches('after-fetch'));

    expect(await git().fetch()).toBe(true);

    expect(api.gitFetch).toHaveBeenCalledTimes(1);
    expect(git().status?.head).toBe('after-fetch');
    expect(git().branches?.current).toBe('after-fetch');
    expect(git().liveMessage).toBe(say('git.toast.fetched'));
    expect(toasts().map((item) => [item.message, item.type])).toEqual([
      [say('git.toast.fetched'), 'success'],
    ]);
  });

  it.each([
    { headMoved: true, key: 'git.toast.pulled' },
    { headMoved: false, key: 'git.toast.upToDate' },
  ])('pull uses $key when head_moved is $headMoved', async ({ headMoved, key }) => {
    api.gitPull.mockResolvedValue(
      mutation({
        status: status({ head: headMoved ? 'after-pull' : 'abc1234def' }),
        detail: { head_moved: headMoved, remote: 'origin' },
      }),
    );
    api.getGitBranches.mockResolvedValue(branches('after-pull'));

    expect(await git().pull('ff-only')).toBe(true);

    expect(api.gitPull).toHaveBeenCalledWith({ strategy: 'ff-only' });
    expect(git().branches?.current).toBe('after-pull');
    expect(git().liveMessage).toBe(say(key));
    expect(toasts()[0].message).toBe(say(key));
    expect(toasts()[0].type).toBe('success');
  });

  it('push and sync apply their final status, refresh refs, and use their own toasts', async () => {
    api.gitPush.mockResolvedValue(
      mutation({ status: status({ head: 'after-push' }), detail: { published: false } }),
    );
    expect(await git().push()).toBe(true);
    expect(api.gitPush).toHaveBeenCalledWith({ setUpstream: false });
    expect(git().status?.head).toBe('after-push');
    expect(git().liveMessage).toBe(say('git.toast.pushed'));
    expect(api.getGitBranches).toHaveBeenCalledTimes(1);

    useToastStore.setState({ toasts: [] });
    api.getGitBranches.mockClear();
    api.gitSync.mockResolvedValue(
      mutation({
        status: status({ head: 'after-sync' }),
        detail: {
          steps: ['fetch', 'merge', 'push'],
          head_moved: true,
          published: false,
          remote: 'origin',
          branch: 'main',
        },
      }),
    );
    expect(await git().sync()).toBe(true);
    expect(api.gitSync).toHaveBeenCalledTimes(1);
    expect(git().status?.head).toBe('after-sync');
    expect(git().liveMessage).toBe(say('git.toast.synced'));
    expect(toasts()[0].message).toBe(say('git.toast.synced'));
    expect(api.getGitBranches).toHaveBeenCalledTimes(1);
  });

  it('says a sync that published a branch published it, not that it synced', async () => {
    api.gitSync.mockResolvedValue(
      mutation({
        status: status({ branch: 'topic', upstream: 'origin/topic' }),
        detail: {
          steps: ['publish'],
          head_moved: false,
          published: true,
          remote: 'origin',
          branch: 'topic',
        },
      }),
    );

    expect(await git().sync()).toBe(true);

    expect(git().liveMessage).toBe(
      say('git.toast.published', { branch: 'topic', remote: 'origin' }),
    );
    expect(toasts()[0].message).toBe(
      say('git.toast.published', { branch: 'topic', remote: 'origin' }),
    );
  });

  it('publish chooses the only loaded remote and trusts detail.published for its toast', async () => {
    useGitStore.setState({ remotes: remotes('upstream') });
    api.gitPush.mockResolvedValue(
      mutation({
        status: status({ branch: 'topic', upstream: 'upstream/topic' }),
        detail: { branch: 'topic', remote: 'upstream', published: true },
      }),
    );

    expect(await git().publish()).toBe(true);

    expect(api.gitPush).toHaveBeenCalledWith({ remote: 'upstream', setUpstream: true });
    expect(git().liveMessage).toBe(
      say('git.toast.published', { branch: 'topic', remote: 'upstream' }),
    );
    expect(toasts()[0].message).toBe(
      say('git.toast.published', { branch: 'topic', remote: 'upstream' }),
    );
    expect(api.getGitBranches).toHaveBeenCalledTimes(1);

    useToastStore.setState({ toasts: [] });
    api.gitPush.mockResolvedValue(
      mutation({ detail: { branch: 'topic', remote: 'upstream', published: false } }),
    );
    await git().publish('upstream');
    expect(toasts()[0].message).toBe(say('git.toast.pushed'));
  });

  // The publish that sends NO remote is the one request in the tab that can
  // come back 400 `invalid_value` for "several remotes and no upstream", and
  // both halves of it are reachable from the header: the list is `null` until
  // something reads it, and a repository with two remotes leaves nothing to
  // pick from without asking. `scm.ts` reads that pair as "not published yet"
  // and offers the button whose picker answers the question.
  it.each([
    { name: 'the remote list has never been read', loaded: null },
    {
      name: 'several remotes leave nothing to resolve',
      loaded: [...remotes('origin'), ...remotes('upstream')],
    },
  ])('publish lets the server resolve the remote when $name', async ({ loaded }) => {
    useGitStore.setState({ remotes: loaded });
    api.gitPush.mockResolvedValue(
      mutation({ detail: { branch: 'topic', remote: 'origin', published: true } }),
    );

    expect(await git().publish()).toBe(true);

    expect(api.gitPush).toHaveBeenCalledWith({ setUpstream: true });
  });

  it('records publish as the op behind an ambiguous remote refusal', async () => {
    useGitStore.setState({ remotes: [...remotes('origin'), ...remotes('upstream')] });
    api.gitPush.mockRejectedValueOnce(await coded(400, 'invalid_value'));

    expect(await git().publish()).toBe(false);

    expect(api.gitPush).toHaveBeenCalledWith({ setUpstream: true });
    expect(git().lastError?.code).toBe('invalid_value');
    expect(git().lastError?.op).toBe('publish');
  });

  it('allows one local and one network operation, but refuses a second network operation', async () => {
    const slowLocal = deferred<MutationResult>();
    api.gitStage.mockReturnValueOnce(slowLocal.promise);
    const local = git().stage(['a.py']);
    expect(git().busyOp).toBe('stage');

    const slowNetwork = deferred<MutationResult>();
    api.gitFetch.mockReturnValueOnce(slowNetwork.promise);
    const network = git().fetch();
    expect(git().busyOp).toBe('stage');
    expect(git().netOp).toBe('fetch');
    expect(api.gitFetch).toHaveBeenCalledTimes(1);

    expect(await git().pull('ff-only')).toBe(false);
    expect(api.gitPull).not.toHaveBeenCalled();
    expect(toasts()[0].message).toBe(say('git.error.busy'));

    slowLocal.resolve(mutation());
    slowNetwork.resolve(mutation());
    await Promise.all([local, network]);
    expect(git().busyOp).toBeNull();
    expect(git().netOp).toBeNull();
  });

  it('uses the network timeout and keeps the wire code beside the op that got it', async () => {
    api.gitFetch.mockRejectedValueOnce(await coded(504, 'timeout'));
    await git().fetch();
    expect(git().lastError?.message).toBe(say('git.error.timeout', { seconds: 130 }));
    expect(git().lastError?.op).toBe('fetch');

    // The ambiguous-remote refusal is presented as "not published yet", but
    // that is `scm.ts` reading the pair: the store keeps the server's own code
    // so a bug report quotes what was actually sent.
    api.gitPull.mockRejectedValueOnce(await coded(400, 'invalid_value'));
    await git().pull('ff-only');
    expect(git().lastError?.code).toBe('invalid_value');
    expect(git().lastError?.op).toBe('pull');
  });

  it('records publish as the op behind its own input refusal', async () => {
    api.gitPush.mockRejectedValueOnce(await coded(400, 'invalid_value'));

    await git().publish('origin');

    expect(git().lastError?.code).toBe('invalid_value');
    expect(git().lastError?.op).toBe('publish');
  });

  it.each([
    {
      name: 'pull conflict',
      arrange: async () => api.gitPull.mockRejectedValue(await coded(409, 'conflict')),
      run: () => git().pull('merge'),
    },
    {
      // The fetch half still ran, so the ahead/behind counts on screen have
      // moved even though the merge refused to touch the tree.
      name: 'pull refused after its fetch',
      arrange: async () => api.gitPull.mockRejectedValue(await coded(409, 'dirty_tree')),
      run: () => git().pull('ff-only'),
    },
    {
      name: 'failed sync',
      arrange: async () => api.gitSync.mockRejectedValue(await coded(409, 'remote_rejected')),
      run: () => git().sync(),
    },
  ])(
    'refreshes status after $name because an earlier step may have changed the tree',
    async ({ arrange, run }) => {
      await arrange();
      api.getGitStatus.mockResolvedValue(
        statusResponse('ready', { head: 'read-after-failure', merge_in_progress: true }),
      );

      expect(await run()).toBe(false);

      expect(api.getGitStatus).toHaveBeenCalledTimes(1);
      expect(git().status?.head).toBe('read-after-failure');
    },
  );

  it('reads nothing back when the lane was busy, because nothing ran', async () => {
    api.gitSync.mockRejectedValue(await coded(409, 'busy'));

    expect(await git().sync()).toBe(false);

    expect(api.getGitStatus).not.toHaveBeenCalled();
    expect(toasts()[0].message).toBe(say('git.error.busy'));
  });
});

describe('branch and remote operations', () => {
  it('creates, switches, renames, and deletes branches with fresh branch lists', async () => {
    const applyWorktreeChange = vi.fn();
    useGitStore.setState({ applyWorktreeChange });

    api.gitCreateBranch.mockResolvedValue(
      mutation({
        status: status({ branch: 'feat/new', head: 'created' }),
        changed_paths: ['graphs/a.graph.json'],
      }),
    );
    api.getGitBranches.mockResolvedValueOnce(branches('feat/new'));
    expect(await git().createBranch('feat/new', true, 'main')).toBe(true);
    expect(api.gitCreateBranch).toHaveBeenCalledWith('feat/new', true, 'main');
    expect(git().status?.head).toBe('created');
    expect(git().branches?.current).toBe('feat/new');
    expect(git().liveMessage).toBe('git.group.staged 0, git.group.changes 0');
    expect(toasts()).toHaveLength(0);
    expect(applyWorktreeChange).toHaveBeenLastCalledWith(['graphs/a.graph.json']);

    api.gitCheckout.mockResolvedValue(
      mutation({
        status: status({ branch: 'main', head: 'switched' }),
        changed_paths: ['graphs/b.graph.json'],
        detail: { branch: 'main' },
      }),
    );
    api.getGitBranches.mockResolvedValueOnce(branches('main'));
    expect(await git().checkout('main', 'local')).toBe(true);
    expect(api.gitCheckout).toHaveBeenCalledWith('main', 'local');
    expect(git().status?.head).toBe('switched');
    expect(git().liveMessage).toBe(say('git.toast.switched', { name: 'main' }));
    expect(toasts()[0].message).toBe(say('git.toast.switched', { name: 'main' }));
    expect(applyWorktreeChange).toHaveBeenLastCalledWith(['graphs/b.graph.json']);

    useToastStore.setState({ toasts: [] });
    api.gitRenameBranch.mockResolvedValue(
      mutation({ status: status({ branch: 'renamed', head: 'renamed' }) }),
    );
    api.getGitBranches.mockResolvedValueOnce(branches('renamed'));
    expect(await git().renameBranch('main', 'renamed')).toBe(true);
    expect(api.gitRenameBranch).toHaveBeenCalledWith('main', 'renamed');
    expect(git().status?.head).toBe('renamed');
    expect(git().branches?.current).toBe('renamed');
    expect(toasts()).toHaveLength(0);

    api.gitDeleteBranch.mockResolvedValue(
      mutation({ status: status({ head: 'deleted' }) }),
    );
    api.getGitBranches.mockResolvedValueOnce(branches('main'));
    expect(await git().deleteBranch('old', true)).toBe(true);
    expect(api.gitDeleteBranch).toHaveBeenCalledWith('old', true);
    expect(git().status?.head).toBe('deleted');
    expect(git().branches?.current).toBe('main');
    expect(api.getGitBranches).toHaveBeenCalledTimes(4);
  });

  it('does not offer a reload when a created branch was not checked out', async () => {
    const applyWorktreeChange = vi.fn();
    useGitStore.setState({ applyWorktreeChange });
    api.gitCreateBranch.mockResolvedValue(
      mutation({ changed_paths: ['graphs/a.graph.json'] }),
    );

    await git().createBranch('feat/no-switch', false, null);

    expect(applyWorktreeChange).not.toHaveBeenCalled();
  });

  it('adds, edits, and removes remotes with a fresh remote list and no toast', async () => {
    api.getGitRemotes
      .mockResolvedValueOnce(remotes('origin'))
      .mockResolvedValueOnce(remotes('upstream'))
      .mockResolvedValueOnce([]);

    api.gitAddRemote.mockResolvedValue(mutation({ status: status({ head: 'added-remote' }) }));
    expect(await git().addRemote('origin', 'https://example/repo')).toBe(true);
    expect(api.gitAddRemote).toHaveBeenCalledWith('origin', 'https://example/repo');
    expect(git().remotes?.[0].name).toBe('origin');
    expect(git().status?.head).toBe('added-remote');

    api.gitSetRemoteUrl.mockResolvedValue(
      mutation({ status: status({ head: 'changed-remote' }) }),
    );
    expect(await git().setRemoteUrl('origin', 'ssh://example/repo')).toBe(true);
    expect(api.gitSetRemoteUrl).toHaveBeenCalledWith('origin', 'ssh://example/repo');
    expect(git().remotes?.[0].name).toBe('upstream');
    expect(git().status?.head).toBe('changed-remote');

    api.gitRemoveRemote.mockResolvedValue(
      mutation({ status: status({ head: 'removed-remote' }) }),
    );
    expect(await git().removeRemote('upstream')).toBe(true);
    expect(api.gitRemoveRemote).toHaveBeenCalledWith('upstream');
    expect(git().remotes).toEqual([]);
    expect(git().status?.head).toBe('removed-remote');
    expect(api.getGitRemotes).toHaveBeenCalledTimes(3);
    expect(git().liveMessage).toBe('git.group.staged 0, git.group.changes 0');
    expect(toasts()).toHaveLength(0);
  });
});

describe('stash and merge operations', () => {
  it('stashes, pops, applies, and drops by git index while refreshing the stash list', async () => {
    const applyWorktreeChange = vi.fn();
    useGitStore.setState({ applyWorktreeChange });
    api.getGitStashes
      .mockResolvedValueOnce(stashes(0))
      .mockResolvedValueOnce(stashes(3))
      .mockResolvedValueOnce(stashes(4))
      .mockResolvedValueOnce([]);

    api.gitStashPush.mockResolvedValue(
      mutation({
        status: status({ head: 'stashed' }),
        changed_paths: ['graphs/a.graph.json'],
      }),
    );
    // A box the user left blank is `null`, never `""` -- the second is a 400.
    expect(await git().stashPush('   ', true)).toBe(true);
    expect(api.gitStashPush).toHaveBeenCalledWith(null, true);
    expect(git().status?.head).toBe('stashed');
    expect(git().stashes?.[0].index).toBe(0);
    expect(git().liveMessage).toBe(say('git.toast.stashed'));
    expect(toasts()[0].message).toBe(say('git.toast.stashed'));
    expect(applyWorktreeChange).toHaveBeenLastCalledWith(['graphs/a.graph.json']);

    useToastStore.setState({ toasts: [] });
    api.gitStashPop.mockResolvedValue(
      mutation({ status: status({ head: 'popped' }), changed_paths: ['graphs/b.graph.json'] }),
    );
    expect(await git().stashPop(3)).toBe(true);
    expect(api.gitStashPop).toHaveBeenCalledWith(3);
    expect(git().stashes?.[0].index).toBe(3);
    expect(git().status?.head).toBe('popped');
    expect(toasts()).toHaveLength(0);

    api.gitStashApply.mockResolvedValue(
      mutation({ status: status({ head: 'applied' }), changed_paths: ['graphs/c.graph.json'] }),
    );
    expect(await git().stashApply(4)).toBe(true);
    expect(api.gitStashApply).toHaveBeenCalledWith(4);
    expect(git().stashes?.[0].index).toBe(4);
    expect(git().status?.head).toBe('applied');

    api.gitStashDrop.mockResolvedValue(mutation({ status: status({ head: 'dropped' }) }));
    expect(await git().stashDrop(4)).toBe(true);
    expect(api.gitStashDrop).toHaveBeenCalledWith(4);
    expect(git().stashes).toEqual([]);
    expect(git().status?.head).toBe('dropped');
    expect(api.getGitStashes).toHaveBeenCalledTimes(4);
    expect(applyWorktreeChange).toHaveBeenCalledTimes(3);
  });

  it('sends a typed stash message without the whitespace around it', async () => {
    await git().stashPush('  half-done sweep  ', false);

    expect(api.gitStashPush).toHaveBeenCalledWith('half-done sweep', false);
  });

  it.each([
    {
      name: 'pop',
      arrange: async () => api.gitStashPop.mockRejectedValue(await coded(409, 'conflict')),
      run: () => git().stashPop(7),
    },
    {
      name: 'apply',
      arrange: async () => api.gitStashApply.mockRejectedValue(await coded(409, 'conflict')),
      run: () => git().stashApply(8),
    },
  ])('refreshes status after a stash $name conflict', async ({ arrange, run }) => {
    await arrange();
    api.getGitStatus.mockResolvedValue(
      statusResponse('ready', { head: 'after-stash-conflict', merge_in_progress: true }),
    );

    expect(await run()).toBe(false);

    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
    expect(git().status?.head).toBe('after-stash-conflict');
    expect(git().lastError?.code).toBe('conflict');
  });

  it('aborts and resolves a merge with status feedback and reload offers only', async () => {
    const applyWorktreeChange = vi.fn();
    useGitStore.setState({ applyWorktreeChange });

    api.gitAbortMerge.mockResolvedValue(
      mutation({ status: status({ head: 'aborted' }), changed_paths: ['graphs/a.graph.json'] }),
    );
    expect(await git().abortMerge()).toBe(true);
    expect(api.gitAbortMerge).toHaveBeenCalledTimes(1);
    expect(git().status?.head).toBe('aborted');
    expect(git().liveMessage).toBe('git.group.staged 0, git.group.changes 0');
    expect(toasts()).toHaveLength(0);

    api.gitResolve.mockResolvedValue(
      mutation({ status: status({ head: 'resolved' }), changed_paths: ['graphs/b.graph.json'] }),
    );
    expect(await git().resolve('graphs/b.graph.json', 'ours')).toBe(true);
    expect(api.gitResolve).toHaveBeenCalledWith('graphs/b.graph.json', 'ours');
    expect(git().status?.head).toBe('resolved');
    expect(toasts()).toHaveLength(0);
    expect(applyWorktreeChange.mock.calls).toEqual([
      [['graphs/a.graph.json']],
      [['graphs/b.graph.json']],
    ]);
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
      op: 'discard',
    });
  });

  it('supplies the local credential recovery hint for an auth refusal', async () => {
    api.gitFetch.mockRejectedValue(await coded(409, 'auth_required'));

    await git().fetch();

    expect(git().lastError?.hint).toBe(say('git.error.authRequiredHint'));
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
      op: 'stage',
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
