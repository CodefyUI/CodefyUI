import { create } from 'zustand';
import {
  GIT_TIMEOUTS_S,
  GitApiError,
  getGitBranches,
  getGitConfig,
  getGitRemotes,
  getGitStashes,
  getGitStatus,
  gitAbortMerge,
  gitAddRemote,
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
  type BranchesResponse,
  type GitCheckoutKind,
  type GitErrorCode,
  type GitPathSelection,
  type GitPullStrategy,
  type GitResolveSide,
  type GitStatus,
  type Identity,
  type MutationResult,
  type RemoteInfo,
  type RepoInfo,
  type RepoState,
  type StashInfo,
} from '../api/git';
import { GraphMissingError, reloadTabFromDisk } from '../utils/openSavedGraph';
import { setWorktreeWriteListener } from '../utils/worktreeWrite';
import { confirm } from '../utils/dialog';
import { useProjectStore } from './projectStore';
import { useTabStore } from './tabStore';
import { useToastStore, type ToastAction, type ToastType } from './toastStore';
import { useI18n, type TranslationKey } from '../i18n';
import { errorHint } from '../components/SourceControl/scm';

/**
 * Everything the Source Control tab knows and everything it does.
 *
 * One store rather than state inside the panel, for the same reason
 * `pluginStore` is one: what it holds outlives the component. A stage is a
 * request, a commit is a request, and the poll behind them keeps running
 * while the user is on the canvas -- so the tab is a pure view of this, and
 * mounting it twice (or closing and reopening it) changes nothing but the
 * reference count.
 *
 * Three decisions worth reading before the code:
 *
 *  - **`repoState` drives the screen, `loadError` reports a failure.**
 *    `GET /api/git/status` answers 200 for every repository state -- "there
 *    is no project", "git is not installed", "this directory is not a
 *    repository" are all normal answers with a null `status` beside them --
 *    so those are states the tab draws, not errors it shows. `loadError` is
 *    for what really did fail: a server with no git service (503), a server
 *    that cannot be reached, and git itself failing or being stopped at the
 *    status deadline (500 `git_failed`, 504 `timeout`) -- the poll runs two
 *    real git processes, and either of them can go wrong.
 *
 *  - **`busyOp` and `netOp` are locks, not spinners.** The backend serialises
 *    writes behind TWO locks -- one for the worktree, one for the network --
 *    and refuses a second holder with `409 busy`; this store refuses it a step
 *    earlier, so two clicks on Stage cannot race each other into a 409 the
 *    user has to read. Two lanes rather than one because a fetch can take two
 *    minutes and a commit during it is perfectly safe: the lanes only refuse
 *    each other. Reads take neither, which is why the poll runs straight
 *    through a commit.
 *
 *  - **Nothing is reloaded from disk without asking.** A discard can put an
 *    older version of a graph a tab is showing back on disk, and the tab has
 *    no unsaved flag to check -- so the store raises one sticky toast with a
 *    Reload button and lets the person holding the unsaved edits decide.
 */

/* ── Vocabulary ─────────────────────────────────────────────────────── */

/**
 * The operations this store runs on the LOCAL lane, in ITS spelling.
 *
 * `status` is the poll; the rest are the writes that take the worktree lock.
 * Two spellings differ from the wire deliberately:
 *
 *  - the identity write is `set_identity` on the server (it is the word that
 *    comes back in a `busy` refusal's `op` field), and `identity` here,
 *    because the i18n key is `git.op.identity`;
 *  - `status` is an operation here even though `refresh()` never goes through
 *    `runOp`: it is one of the four timeout buckets, and a union that omits
 *    it would make the map below partial for no gain.
 */
export type GitOp =
  | 'status'
  | 'init'
  | 'stage'
  | 'unstage'
  | 'discard'
  | 'commit'
  | 'identity'
  | 'create_branch'
  | 'checkout'
  | 'rename_branch'
  | 'delete_branch'
  | 'add_remote'
  | 'set_remote_url'
  | 'remove_remote'
  | 'stash_push'
  | 'stash_pop'
  | 'stash_apply'
  | 'stash_drop'
  | 'abort_merge'
  | 'resolve';

/**
 * The operations that talk to a remote, and hold the OTHER lock.
 *
 * `publish` is not a route: it is `POST /push` with `set_upstream: true`, and
 * a separate name here because it is a separate button, a separate label in
 * the busy line and a separate toast. A `busy` refusal from the server can
 * therefore say `push` while this store says `publish`; the two are the same
 * lane either way.
 */
export type GitNetOp = 'fetch' | 'pull' | 'push' | 'sync' | 'publish';
export type GitAnyOp = GitOp | GitNetOp;

/** Which of the server's four deadlines an operation runs under. */
type TimeoutBucket = keyof typeof GIT_TIMEOUTS_S;

/**
 * The operation -> deadline map, complete by construction.
 *
 * A 504 arrives as `{code: "timeout"}` and nothing else -- the number of
 * seconds is not in the body -- so `git.error.timeout {seconds}` can only be
 * filled in from this side. `identity` is the WRITE (30 s); reading the
 * config back is the `read` bucket and passes it explicitly.
 */
const OP_TIMEOUT_BUCKET: Record<GitAnyOp, TimeoutBucket> = {
  status: 'status',
  init: 'mutation',
  stage: 'mutation',
  unstage: 'mutation',
  discard: 'mutation',
  commit: 'mutation',
  identity: 'mutation',
  create_branch: 'mutation',
  checkout: 'mutation',
  rename_branch: 'mutation',
  delete_branch: 'mutation',
  add_remote: 'mutation',
  set_remote_url: 'mutation',
  remove_remote: 'mutation',
  stash_push: 'mutation',
  stash_pop: 'mutation',
  stash_apply: 'mutation',
  stash_drop: 'mutation',
  abort_merge: 'mutation',
  resolve: 'mutation',
  fetch: 'network',
  pull: 'network',
  push: 'network',
  sync: 'network',
  publish: 'network',
};

/**
 * A refusal, reduced to the four things the tab shows.
 *
 * `code` is what the error line switches on to pick its sentence, so the
 * store keeps the server's own `message` rather than translating it -- with
 * ONE exception, `timeout`, whose sentence needs a number only this file
 * knows. `hint` is the actionable half of a refusal and `stderr` is git's own
 * tail, shown behind Details.
 */
export interface GitStoreError {
  code: GitErrorCode;
  message: string;
  hint: string | null;
  stderr: string | null;
  /**
   * What was being run when this came back, or null for a read.
   *
   * One code does not say the same thing after every operation: a 400
   * `invalid_value` from a fetch or a push is the server telling us it could
   * not decide WHICH remote (`network.resolve_remote`), which is a different
   * sentence and a different button from an `invalid_value` about a value the
   * user typed. `scm.ts` is where that decision lives; this is the fact it
   * needs. Optional so a component or a test can build one of these from a
   * code alone -- the store always fills it in.
   */
  op?: GitAnyOp | null;
}

/* ── i18n ───────────────────────────────────────────────────────────── */

/**
 * Translate, off the store rather than through the hook.
 *
 * `TranslationKey` is `keyof typeof en`, so every key this file names is
 * checked against the locale file itself and a typo is a compile error. It
 * once took a narrower union of its own, standing in for strings that had not
 * landed yet; they have, and the real type is strictly better.
 */
function t(key: TranslationKey, vars?: Record<string, string | number>): string {
  return useI18n.getState().t(key, vars);
}

/** `pluginStore`'s helper, for the same reason: no store action via a hook. */
function toast(
  message: string,
  type: ToastType,
  opts?: { action?: ToastAction; sticky?: boolean },
): string {
  return useToastStore.getState().addToast(message, type, opts);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/* ── The layout-file filter ─────────────────────────────────────────── */

/**
 * A layout file: the half of a saved graph that holds positions and notes.
 *
 * One Save writes a PAIR (`graphs/<name>.graph.json` and
 * `layout/<name>.layout.json`), and only the first of the two is a change
 * anybody reviews -- which is what the "Hide layout files" filter is for.
 *
 * It lives here rather than beside the panel that draws the list because both
 * sides of the filter have to agree on what it hides: the Changes group drops
 * these rows, and the sentence the live region announces after a write counts
 * what that group is showing. One definition, both callers.
 */
export function isLayoutFile(path: string): boolean {
  return /^layout\/.+\.layout\.json$/.test(path);
}

/* ── Persistence ────────────────────────────────────────────────────── */

/** "Hide layout files" survives a reload; nothing else in this store does. */
const HIDE_LAYOUT_KEY = 'codefyui-git-hide-layout';

function loadHideLayout(): boolean {
  try {
    return localStorage.getItem(HIDE_LAYOUT_KEY) === 'true';
  } catch {
    // Private mode, a disabled-storage policy, a quota already blown. The
    // preference is a convenience; losing it is not worth a broken tab.
    return false;
  }
}

function saveHideLayout(hide: boolean): void {
  try {
    localStorage.setItem(HIDE_LAYOUT_KEY, String(hide));
  } catch {
    /* see `loadHideLayout` */
  }
}

export type GitRefKind = 'branches' | 'remotes' | 'stashes';
export type GitSections = Record<GitRefKind, boolean>;

const SECTIONS_KEY = 'codefyui-git-sections';
const CLOSED_SECTIONS: GitSections = {
  branches: false,
  remotes: false,
  stashes: false,
};

function loadSections(): GitSections {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(SECTIONS_KEY) ?? 'null');
    if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
      return { ...CLOSED_SECTIONS };
    }
    const stored = raw as Record<string, unknown>;
    return {
      branches: stored.branches === true,
      remotes: stored.remotes === true,
      stashes: stored.stashes === true,
    };
  } catch {
    return { ...CLOSED_SECTIONS };
  }
}

function saveSections(sections: GitSections): void {
  try {
    localStorage.setItem(SECTIONS_KEY, JSON.stringify(sections));
  } catch {
    /* A display preference must never make Source Control unusable. */
  }
}

/* ── The poll ───────────────────────────────────────────────────────── */

/**
 * Seconds between status polls while the tab is open and the page visible.
 *
 * One poll is two short git processes (`rev-parse`, `status --porcelain=v2`),
 * both run with `GIT_OPTIONAL_LOCKS=0` so they never fight a real operation
 * for the index lock. Fifteen seconds is the interval the backend's own
 * runner docstring argues for; the save hook below is what makes it feel
 * immediate anyway, because the write the user just did announces itself.
 */
export const GIT_POLL_MS = 15_000;

/**
 * How long a burst of saves is allowed to keep coalescing.
 *
 * One Save writes TWO files (`graphs/<name>.graph.json` and
 * `layout/<name>.layout.json`), and a user saving three tabs in a row is one
 * intention, not six. 600 ms is short enough that the panel still updates
 * while the hand is off the keyboard.
 */
export const GIT_WRITE_DEBOUNCE_MS = 600;

/**
 * How long the events that mean "the page is back" are allowed to coalesce.
 *
 * Returning to the tab fires `visibilitychange` AND `focus` in the same tick,
 * and each of them wanted a read of its own: every return sent two identical
 * `GET /api/git/status`, at the same millisecond, for one status. A short
 * trailing debounce makes a burst of them one read. Short on purpose, and not
 * the save hook's 600 ms: nothing was written here, the page has simply come
 * back, and the panel should be right by the time the eye reaches it.
 */
export const GIT_REVISIT_DEBOUNCE_MS = 100;

/** How many mounted views want the poll running. */
let attachCount = 0;
let pollTimer: ReturnType<typeof setInterval> | null = null;
let writeTimer: ReturnType<typeof setTimeout> | null = null;
let revisitTimer: ReturnType<typeof setTimeout> | null = null;
let listening = false;

/**
 * The generation of the newest status read.
 *
 * Requests carry no `AbortSignal` -- the client deliberately leaves the
 * deadline to the server, which already enforces one -- so a slow poll can
 * still be in flight when a newer read, or a mutation carrying its own fresh
 * status, has already answered. Every write that lands a status bumps this,
 * and a read only writes when its own stamp is still the newest. Without it a
 * poll that started before a stage lands afterwards and puts the file back
 * where it was.
 */
let readSeq = 0;

/** Independent generations keep each lazy reference list race-safe. */
let refReadSeq: Record<GitRefKind, number> = {
  branches: 0,
  remotes: 0,
  stashes: 0,
};

/**
 * The generation of the newest identity read or write, for the same reason
 * `readSeq` exists: a config read that was slow enough to outlive the write
 * that replaced it would otherwise put the OLD name and email back into a
 * form the user has just saved. Bumped by both, applied only by the newest.
 */
let identitySeq = 0;

/** The sticky "changed on disk" toast, so the next one replaces it. */
let changedToastId: string | null = null;

function pageVisible(): boolean {
  return document.visibilityState === 'visible';
}

const REF_KINDS: readonly GitRefKind[] = ['branches', 'remotes', 'stashes'];

function refreshExpandedRefs(): void {
  const state = useGitStore.getState();
  for (const kind of REF_KINDS) {
    if (state.sections[kind]) void state.refreshRefs(kind);
  }
}

function startPoll(): void {
  if (pollTimer !== null || !pageVisible()) return;
  pollTimer = setInterval(() => {
    void useGitStore.getState().refresh();
    refreshExpandedRefs();
  }, GIT_POLL_MS);
}

function stopPoll(): void {
  if (pollTimer === null) return;
  clearInterval(pollTimer);
  pollTimer = null;
}

function cancelRevisitDebounce(): void {
  if (revisitTimer === null) return;
  clearTimeout(revisitTimer);
  revisitTimer = null;
}

/** One read for however many "the page is back" events arrive together. */
function refreshOnRevisit(): void {
  cancelRevisitDebounce();
  revisitTimer = setTimeout(() => {
    revisitTimer = null;
    void useGitStore.getState().refresh();
    refreshExpandedRefs();
  }, GIT_REVISIT_DEBOUNCE_MS);
}

/**
 * A hidden page polls nothing and a returning one is refreshed at once.
 *
 * Half of what this tab shows is written by something other than this tab --
 * the command line, an editor, another browser tab -- so the moment the page
 * comes back is exactly the moment its status is most likely to be wrong.
 *
 * The read goes through the debounce because this event does not arrive
 * alone; the poll it starts and stops does not, because that is a schedule
 * rather than a request.
 */
function onVisibilityChange(): void {
  if (attachCount === 0) return;
  if (pageVisible()) {
    startPoll();
    refreshOnRevisit();
  } else {
    stopPoll();
    // The read a `focus` scheduled a moment ago goes with it. Leaving it
    // queued would send a status read against a page nobody is looking at,
    // which is the one thing pausing here exists to avoid -- and the window
    // is real: the two events arrive together on the way OUT of a tab too.
    cancelRevisitDebounce();
  }
}

/** The other half of one return to the tab -- see `refreshOnRevisit`. */
function onFocus(): void {
  if (attachCount === 0) return;
  refreshOnRevisit();
}

function listen(): void {
  if (listening) return;
  listening = true;
  document.addEventListener('visibilitychange', onVisibilityChange);
  window.addEventListener('focus', onFocus);
}

function unlisten(): void {
  if (!listening) return;
  listening = false;
  document.removeEventListener('visibilitychange', onVisibilityChange);
  window.removeEventListener('focus', onFocus);
}

function cancelWriteDebounce(): void {
  if (writeTimer === null) return;
  clearTimeout(writeTimer);
  writeTimer = null;
}

/* ── Reading the identity ───────────────────────────────────────────── */

/**
 * Read `user.name` / `user.email` and which config file each came from.
 *
 * Not part of the poll: the identity changes about once per machine, and a
 * third git process every fifteen seconds to re-read it would be the most
 * expensive part of the poll for the least reason. It is read when the form
 * that shows it opens -- including the time it opens by itself, because a
 * commit was refused for the lack of one.
 */
async function loadIdentity(): Promise<void> {
  const seq = (identitySeq += 1);
  try {
    const answer = await getGitConfig();
    // A read that a write has already overtaken describes an identity that
    // no longer exists -- see `identitySeq`.
    if (seq !== identitySeq) return;
    useGitStore.setState({ identity: answer });
  } catch (err) {
    if (seq !== identitySeq) return;
    // Reported rather than swallowed: the form is on screen because the user
    // asked for it, and a form that stays blank with no reason given is worse
    // than an error line above it.
    useGitStore.setState({ lastError: toStoreError(err, 'read') });
  }
}

function revealIdentityForm(): void {
  useGitStore.setState({ identityFormOpen: true });
  void loadIdentity();
}

/* ── Running one operation ──────────────────────────────────────────── */

/**
 * A refusal in the shape the error line reads.
 *
 * `timeout` is the one code whose message is written here instead of by the
 * server: the 504 body carries the code and nothing else, so the sentence
 * ("git did not finish within {seconds}s") has to be filled in from the
 * bucket the operation ran under. Every other code keeps git's own words and
 * the component picks the sentence from `code` -- and, for the one code whose
 * meaning depends on what was asked, from `op` beside it.
 */
function toStoreError(
  err: unknown,
  bucket: TimeoutBucket,
  op: GitAnyOp | null = null,
): GitStoreError {
  if (!(err instanceof GitApiError)) {
    return { code: 'unknown', message: errorMessage(err), hint: null, stderr: null, op };
  }
  return {
    code: err.code,
    message:
      err.code === 'timeout'
        ? t('git.error.timeout', { seconds: GIT_TIMEOUTS_S[bucket] })
        : err.message,
    hint: errorHint(err.code, err.hint, t),
    stderr: err.stderr,
    op,
  };
}

/** The commit this made, shortest readable form first. */
function commitSha(result: MutationResult): string {
  const short = result.detail.short;
  if (typeof short === 'string') return short;
  const sha = result.detail.sha;
  if (typeof sha === 'string') return sha;
  return result.head ?? '';
}

/**
 * What the two group headings now count.
 *
 * The live region's job after a stage or a discard is to say the thing the
 * screen says by moving rows between groups, which no screen reader
 * announces. Built from the headings the panel already shows rather than
 * from a sentence of its own, so there is nothing here to translate twice.
 *
 * Which means the counts on SCREEN, not the ones in the status: with "Hide
 * layout files" on, the Changes group is missing the layout half of every
 * save, and a sentence built from the raw status announced 11 beside a
 * heading that said 10. The staged group hides nothing, so it is not
 * filtered here either -- which is `SourceControlTab` exactly.
 */
function groupCounts(status: GitStatus): string {
  const changed = [...status.unstaged, ...status.untracked];
  const changes = useGitStore.getState().hideLayout
    ? changed.filter((file) => !isLayoutFile(file.path)).length
    : changed.length;
  return `${t('git.group.staged')} ${status.staged.length}, `
    + `${t('git.group.changes')} ${changes}`;
}

/** What one operation needs from `runOp` beyond the call itself. */
interface RunOpOptions {
  /** This write can put DIFFERENT bytes under an open tab -- see `runOp`. */
  worktree?: boolean;
  /** The reference lists this write can have changed. */
  refs?: GitRefKind[];
  /** Whether a refusal still needs a fresh status -- see `alwaysRefresh`. */
  refreshStatusOnError?: (err: unknown) => boolean;
  /**
   * Fallbacks for the success sentence.
   *
   * `MutationResult.detail` is deliberately open, and a build of the server
   * that does not fill in `detail.branch` or `detail.remote` should still
   * produce a sentence with a name in it rather than one with a hole. These
   * are what the caller already knows, used only when the answer said nothing.
   */
  name?: string;
  remote?: string;
}

/** The lane an operation runs on: these five take the network lock. */
const NETWORK_OPS: ReadonlySet<GitAnyOp> = new Set([
  'fetch',
  'pull',
  'push',
  'sync',
  'publish',
]);

/**
 * The operations whose news is NOT visible in the panel, and so get a toast.
 *
 * Everything else says what it did in the live region only: a stage moves a
 * row between two groups on screen, a rename rewrites a line in the branch
 * list, and a toast on top of that is the same fact twice. What a toast is for
 * is the fact the panel cannot show -- a commit id, a repository that now
 * exists, a branch that reached a remote.
 */
const SUCCESS_TOAST_OPS: ReadonlySet<GitAnyOp> = new Set([
  'commit',
  'init',
  'fetch',
  'pull',
  'push',
  'sync',
  'publish',
  'stash_push',
  'checkout',
]);

function isNetworkOp(op: GitAnyOp): op is GitNetOp {
  return NETWORK_OPS.has(op);
}

function detailString(result: MutationResult, key: string): string | null {
  const value = result.detail[key];
  return typeof value === 'string' && value !== '' ? value : null;
}

function upstreamRemote(upstream: string | null): string | null {
  if (upstream === null) return null;
  const slash = upstream.indexOf('/');
  return slash > 0 ? upstream.slice(0, slash) : upstream;
}

/**
 * The sentence an operation ends with, or null when it has nothing to say.
 *
 * Most operations fall through to the group counts, which is the live region
 * saying what the screen already shows by moving rows between groups -- the
 * one change no screen reader announces. The named ones above it are the
 * operations whose result is a FACT rather than a rearrangement: which commit,
 * which branch, whether anything came down, where a branch went.
 *
 * The identity write is the one that answers null: it returns an identity
 * rather than a status, nothing in the panel moved, and the form redrawing
 * with the saved values is the feedback.
 */
function announcement(
  op: GitAnyOp,
  result: MutationResult | Identity,
  opts: RunOpOptions,
): string | null {
  if (!('status' in result)) return null;
  if (op === 'commit') return t('git.toast.committed', { sha: commitSha(result) });
  if (op === 'init') return t('git.toast.initialized');
  if (op === 'fetch') return t('git.toast.fetched');
  if (op === 'pull') {
    return t(result.detail.head_moved === true ? 'git.toast.pulled' : 'git.toast.upToDate');
  }
  // `detail.published` is the server's own answer to "did this create the
  // upstream", and it is the only thing that separates the three sentences:
  // a sync whose last step was a publish has published a branch, and saying
  // "Synced" would hide the one part of it the user did not already know.
  //
  // `detail.remote` is read ONLY on this branch. A plain push's is git's push
  // destination, which can be a URL rather than a remote NAME and arrives
  // credential-masked (backend `network._pushed_remote`); a publish's is the
  // name that was resolved, which is the word this sentence needs.
  if (op === 'push' || op === 'publish' || op === 'sync') {
    if (result.detail.published === true) {
      return t('git.toast.published', {
        branch: detailString(result, 'branch') ?? result.status.branch ?? '',
        remote:
          detailString(result, 'remote')
          ?? opts.remote
          ?? upstreamRemote(result.status.upstream)
          ?? '',
      });
    }
    return t(op === 'sync' ? 'git.toast.synced' : 'git.toast.pushed');
  }
  if (op === 'stash_push') return t('git.toast.stashed');
  if (op === 'checkout') {
    return t('git.toast.switched', {
      name: detailString(result, 'branch') ?? result.status.branch ?? opts.name ?? '',
    });
  }
  return groupCounts(result.status);
}

/**
 * Run one write on its lane, with the error mapping and the fresh status.
 *
 * Two lanes, because the server has two locks: a network operation takes
 * `netOp` and a local write takes `busyOp`, so a commit during a fetch is
 * allowed on both sides and only a second operation on the SAME lane is
 * refused. The 409 the server would answer with is the same toast, so a race
 * this store lets through still reads correctly.
 *
 * `worktree` is the bit that decides whether the "changed on disk" offer is
 * raised, and it is deliberately NOT `changed_paths.length > 0`: a commit
 * reports every file it swallowed, and every one of those is byte-identical
 * to what the tab already holds. Only an operation that can put DIFFERENT
 * bytes under an open tab sets it -- `discard`, the merge half of a pull and
 * of a sync, a checkout, a branch created WITH a checkout, every stash write
 * that touches the tree, and the two merge-resolution writes.
 *
 * Answers whether the operation succeeded, so a caller can clear its form.
 */
async function runOp(
  op: GitAnyOp,
  call: () => Promise<MutationResult | Identity>,
  opts: RunOpOptions = {},
): Promise<boolean> {
  const network = isNetworkOp(op);
  const state = useGitStore.getState();
  if (network ? state.netOp !== null : state.busyOp !== null) {
    // The same sentence the server's own 409 gets, because it is the same
    // fact: one operation at a time on this lane.
    toast(t('git.error.busy'), 'warning');
    return false;
  }
  // `network` is a const initialised by a type guard, so each branch below
  // knows which of the two unions `op` is in without a cast.
  if (network) useGitStore.setState({ netOp: op, lastError: null });
  else useGitStore.setState({ busyOp: op, lastError: null });

  try {
    const result = await call();
    if ('status' in result) {
      // Newer than any status read still in flight -- see `readSeq`. Also
      // clears `loadError`: a write that answered is proof the repository
      // can be read, so a failure from an older poll is no longer true.
      readSeq += 1;
      useGitStore.setState({ status: result.status, loadError: null });
      const skipped = result.detail.skipped;
      if (Array.isArray(skipped) && skipped.length > 0) {
        // A whole-tree write leaves link-parents alone. Without this the tab
        // would silently do less than "Stage All" says it does.
        toast(t('git.group.skipped', { count: skipped.length }), 'info');
      }
    } else {
      // Newer than any config read still in flight -- see `identitySeq`.
      identitySeq += 1;
      useGitStore.setState({ identity: result });
    }

    const said = announcement(op, result, opts);
    if (said !== null) {
      useGitStore.setState({ liveMessage: said });
      if (SUCCESS_TOAST_OPS.has(op)) toast(said, 'success');
    }
    if (opts.worktree === true && 'status' in result) {
      useGitStore.getState().applyWorktreeChange(result.changed_paths);
    }
    if (opts.refs !== undefined) {
      await Promise.all(opts.refs.map((kind) => useGitStore.getState().refreshRefs(kind)));
    }
    return true;
  } catch (err) {
    if (err instanceof GitApiError && err.code === 'busy') {
      // Somebody else holds the lock -- another browser tab, or the command
      // line. Nothing to show in the error line: it is over in a second and
      // the button is still there to press again.
      toast(t('git.error.busy'), 'warning');
      return false;
    }
    // The code is the server's, unchanged: rewriting one here would put a
    // word in a bug report that no server ever sent. The refusal carries the
    // operation instead, and `scm.ts` reads the pair -- see `GitStoreError.op`.
    useGitStore.setState({ lastError: toStoreError(err, OP_TIMEOUT_BUCKET[op], op) });
    // The one refusal with a form behind it: git will not commit without a
    // name and an email, and asking for them here is the whole fix.
    if (err instanceof GitApiError && err.code === 'identity_missing') {
      revealIdentityForm();
    }
    if (opts.refreshStatusOnError?.(err) === true) {
      await useGitStore.getState().refresh();
    }
    return false;
  } finally {
    if (network) useGitStore.setState({ netOp: null });
    else useGitStore.setState({ busyOp: null });
  }
}

/** An explicitly empty selection: a request that could only be refused. */
function emptySelection(paths: GitPathSelection): boolean {
  return paths !== 'all' && paths.length === 0;
}

/**
 * A fresh status after ANY refusal, which is what a sequence needs.
 *
 * `pull` and `sync` are sequences: a pull fetches and then merges, and a sync
 * fetches, merges and then pushes. A refusal from the second or third step
 * does not undo the first -- a conflicted merge leaves markers on disk and the
 * merge in progress, and a fetch that succeeded has already moved the tracking
 * refs and with them the ahead/behind counts on screen. A pre-flight refusal
 * (no upstream, a detached HEAD) buys one wasted read; the alternative is a
 * panel describing a repository that stopped existing a second ago.
 *
 * Unconditional rather than "every refusal but `busy`", which is what this
 * said when it was a predicate: the `busy` refusal -- the one that means the
 * request never started -- returns from `runOp` before this is consulted,
 * whether it came from the server's 409 or from this store's own same-lane
 * guard. Excluding it here was a branch no caller could reach.
 */
const alwaysRefresh = (): boolean => true;

/* ── Worktree changes under open tabs ───────────────────────────────── */

/**
 * The graph a repository path belongs to, or null.
 *
 * One save writes a PAIR -- `graphs/<name>.graph.json` and
 * `layout/<name>.layout.json` -- and `graphs/<name>.json` is the older
 * single-file spelling still on some disks. All three reduce to the stem,
 * which is what `TabState.currentGraphFile` holds. Order matters: the
 * `.graph.json` pattern is tried first, or `foo.graph.json` would reduce to
 * `foo.graph` under the legacy one.
 */
function graphBaseName(path: string): string | null {
  const patterns = [
    /^graphs\/([^/]+)\.graph\.json$/,
    /^layout\/([^/]+)\.layout\.json$/,
    /^graphs\/([^/]+)\.json$/,
  ];
  for (const pattern of patterns) {
    const match = pattern.exec(path);
    if (match !== null) return match[1];
  }
  return null;
}

/** One open tab a worktree change landed under. */
interface AffectedTab {
  id: string;
  file: string;
}

/**
 * Re-read each tab from disk, after asking once for all of them.
 *
 * One confirmation, not one per tab: the user is answering a single question
 * ("take what is on disk"), and a modal per tab would be the same answer
 * three times. A tab closed between the toast and the click is skipped
 * before the fetch rather than after it.
 */
async function reloadAffected(tabs: AffectedTab[]): Promise<void> {
  const ok = await confirm({
    title: t('git.toast.reloadConfirm', { count: tabs.length }),
    variant: 'danger',
  });
  if (!ok) return;
  for (const tab of tabs) {
    if (useTabStore.getState().getTab(tab.id) === undefined) continue;
    try {
      await reloadTabFromDisk(tab.id, tab.file);
    } catch (err) {
      if (err instanceof GraphMissingError) {
        // The file is gone on this branch. The tab keeps what it is showing,
        // which is now the only copy of it anywhere.
        toast(t('git.toast.missingOnDisk', { name: tab.file }), 'info');
      } else {
        toast(errorMessage(err), 'error');
      }
    }
  }
}

/* ── The store ──────────────────────────────────────────────────────── */

interface GitState {
  /** `unknown` until the first status answers -- not a repository state. */
  repoState: 'unknown' | RepoState;
  repo: RepoInfo | null;
  /** Null for every repository state except `ready`. */
  status: GitStatus | null;
  identity: Identity | null;
  /**
   * The three reference lists, `null` until each one is first read.
   *
   * `null` and `[]` are different answers and the panel draws them
   * differently: "not read yet" is a section that has never been opened, and
   * the empty array is a repository that really has no remotes, no stashes.
   * Anything deciding whether to offer Publish needs the second one, so it
   * has to ask for the read (`refreshRefs('remotes')`) rather than read a
   * length off a list nobody has fetched.
   */
  branches: BranchesResponse | null;
  remotes: RemoteInfo[] | null;
  stashes: StashInfo[] | null;
  /** Which of the three sections are open; persisted, see `SECTIONS_KEY`. */
  sections: GitSections;
  /** No status has come back yet. Not "a request is in flight". */
  loading: boolean;
  /** The status could not be READ -- a 503 or an unreachable server. */
  loadError: string | null;
  /** The local lane's lock: a write that touches the worktree or the index. */
  busyOp: GitOp | null;
  /** The network lane's lock, held independently of `busyOp`. */
  netOp: GitNetOp | null;
  lastError: GitStoreError | null;
  /** The visually hidden `role="status"` region's text. */
  liveMessage: string;
  commitMessage: string;
  amend: boolean;
  hideLayout: boolean;
  identityFormOpen: boolean;

  attach: () => void;
  detach: () => void;
  refresh: () => Promise<void>;
  refreshRefs: (kind: GitRefKind) => Promise<void>;
  setSectionOpen: (kind: GitRefKind, open: boolean) => void;
  noteWorktreeWrite: () => void;
  announce: (message: string) => void;
  setCommitMessage: (message: string) => void;
  setAmend: (amend: boolean) => void;
  stage: (paths: GitPathSelection) => Promise<boolean>;
  unstage: (paths: GitPathSelection) => Promise<boolean>;
  discard: (paths: GitPathSelection) => Promise<boolean>;
  commit: (opts?: { all?: boolean }) => Promise<boolean>;
  init: () => Promise<boolean>;
  saveIdentity: (identity: { name: string; email: string }) => Promise<boolean>;
  fetch: () => Promise<boolean>;
  pull: (strategy: GitPullStrategy) => Promise<boolean>;
  push: () => Promise<boolean>;
  sync: () => Promise<boolean>;
  publish: (remote?: string) => Promise<boolean>;
  createBranch: (name: string, checkout?: boolean, startPoint?: string | null) => Promise<boolean>;
  checkout: (target: string, kind: GitCheckoutKind) => Promise<boolean>;
  renameBranch: (name: string, newName: string) => Promise<boolean>;
  deleteBranch: (name: string, force: boolean) => Promise<boolean>;
  addRemote: (name: string, url: string) => Promise<boolean>;
  setRemoteUrl: (name: string, url: string) => Promise<boolean>;
  removeRemote: (name: string) => Promise<boolean>;
  stashPush: (message: string | null, includeUntracked: boolean) => Promise<boolean>;
  stashPop: (index: number) => Promise<boolean>;
  stashApply: (index: number) => Promise<boolean>;
  stashDrop: (index: number) => Promise<boolean>;
  abortMerge: () => Promise<boolean>;
  resolve: (path: string, side: GitResolveSide) => Promise<boolean>;
  setHideLayout: (hide: boolean) => void;
  openIdentityForm: () => void;
  closeIdentityForm: () => void;
  dismissError: () => void;
  applyWorktreeChange: (changedPaths: string[]) => void;
}

export const useGitStore = create<GitState>((set, get) => ({
  repoState: 'unknown',
  repo: null,
  status: null,
  identity: null,
  branches: null,
  remotes: null,
  stashes: null,
  sections: loadSections(),
  loading: false,
  loadError: null,
  busyOp: null,
  netOp: null,
  lastError: null,
  liveMessage: '',
  commitMessage: '',
  amend: false,
  hideLayout: loadHideLayout(),
  identityFormOpen: false,

  /**
   * One more view wants the status kept fresh.
   *
   * Reference-counted rather than a boolean because the tab can legitimately
   * be mounted twice for a frame -- React re-parenting it, a second panel in
   * a later build -- and the second mount's unmount must not stop a poll the
   * first one is still using.
   */
  attach: () => {
    attachCount += 1;
    if (attachCount > 1) return;
    listen();
    // Local writes reach the store through a slot rather than a direct call,
    // so `utils/saveActiveGraph.ts` does not import the git client to
    // announce a save. Registered here and cleared in `detach`, which makes
    // "costs nothing when the tab was never opened" structural rather than a
    // runtime check that everybody has to remember.
    setWorktreeWriteListener(() => useGitStore.getState().noteWorktreeWrite());
    startPoll();
    void get().refresh();
    refreshExpandedRefs();
  },

  detach: () => {
    attachCount = Math.max(0, attachCount - 1);
    if (attachCount > 0) return;
    stopPoll();
    unlisten();
    setWorktreeWriteListener(null);
    cancelWriteDebounce();
    cancelRevisitDebounce();
  },

  /**
   * Read the status once.
   *
   * Every repository state is a 200, so the answer sets `repoState` and the
   * screen follows from it. A rejection leaves `repoState` alone -- on the
   * first read that is still `unknown`, which the tab draws as the loading
   * line until `loadError` turns it into the error line -- because "the
   * server did not answer" is not evidence that the repository changed.
   *
   * The rejection goes through `toStoreError` for the one code whose sentence
   * this side has to write: a status read is git under the server's deadline,
   * so it can come back 504 `timeout` with no number in the body. `loadError`
   * keeps its string shape and its meaning -- the header shows that line in
   * every `repoState`, and it clears on the next read that answers.
   */
  refresh: async () => {
    const seq = (readSeq += 1);
    // True only while nothing has ever been read. A request that never
    // settles leaves this on, which is the honest report; it is never a
    // count of what is in flight, or a hung poll would pin it forever.
    if (get().status === null && get().repoState === 'unknown') set({ loading: true });
    try {
      const answer = await getGitStatus();
      if (seq !== readSeq) return;
      set({
        repo: answer.repo,
        repoState: answer.repo.state,
        status: answer.status,
        loadError: null,
        loading: false,
      });
    } catch (err) {
      if (seq !== readSeq) return;
      set({ loadError: toStoreError(err, 'status').message, loading: false });
    }
  },

  /**
   * Read one of the three reference lists.
   *
   * Three lists rather than one call because they are three git processes on
   * the server and the tab almost never wants all three: the sections load
   * what they show when they open, and every write refreshes the list it can
   * have changed. `null` in the slice means "not read yet", which is not the
   * same as the empty list -- a repository with no remotes is `[]`, and only
   * that answers "there is nothing to publish to".
   *
   * A failure lands in `lastError` rather than being swallowed: the section is
   * open because somebody opened it, and an empty list with no reason given is
   * worse than a line saying git could not be read.
   */
  refreshRefs: async (kind) => {
    const seq = (refReadSeq[kind] += 1);
    try {
      if (kind === 'branches') {
        const answer = await getGitBranches();
        if (seq === refReadSeq.branches) set({ branches: answer });
      } else if (kind === 'remotes') {
        const answer = await getGitRemotes();
        if (seq === refReadSeq.remotes) set({ remotes: answer });
      } else {
        const answer = await getGitStashes();
        if (seq === refReadSeq.stashes) set({ stashes: answer });
      }
    } catch (err) {
      if (seq !== refReadSeq[kind]) return;
      set({ lastError: toStoreError(err, 'read') });
    }
  },

  /**
   * Open or close one section, and remember it.
   *
   * Opening reads the list again even when it was read a minute ago: the
   * cheapest correct answer to "what is on screen now" is the current one,
   * and a section that was collapsed while somebody switched branches at the
   * command line would otherwise open on yesterday's list.
   */
  setSectionOpen: (kind, open) => {
    const sections = { ...get().sections, [kind]: open };
    set({ sections });
    saveSections(sections);
    if (open) void get().refreshRefs(kind);
  },

  /**
   * A project file was just written from inside the app.
   *
   * The save hook in `utils/saveActiveGraph.ts` calls this so the two files
   * a Save produces appear under Changes at once instead of up to fifteen
   * seconds later. It costs nothing when the tab has never been opened,
   * which is the case on almost every save -- and is why the save path needs
   * no knowledge of whether the git tab exists.
   */
  noteWorktreeWrite: () => {
    if (attachCount === 0) return;
    cancelWriteDebounce();
    writeTimer = setTimeout(() => {
      writeTimer = null;
      void get().refresh();
    }, GIT_WRITE_DEBOUNCE_MS);
  },

  /**
   * Say something in the panel's live region.
   *
   * The writes announce themselves through `runOp`, which is where almost
   * every sentence comes from. This is for the other kind: a refusal that
   * never reaches the server at all. The commit chord is the one in this
   * build -- Ctrl/Cmd+Enter with no message, or with an empty index, is a
   * keystroke that would otherwise do nothing and say nothing, because the
   * reason is a `title` on a button the keyboard never went near.
   */
  announce: (message) => set({ liveMessage: message }),

  setCommitMessage: (message) => set({ commitMessage: message }),

  /**
   * Turn "replace the last commit" on or off.
   *
   * It does NOT prefill the message with the last commit's subject, which is
   * what an editor with a log does: this build's store has no log route, and
   * `GitStatus.head` is a commit id, not a subject. The box keeps whatever
   * the user has typed -- including nothing, which the Commit button already
   * refuses.
   */
  setAmend: (amend) => set({ amend }),

  stage: async (paths) => {
    // An empty selection is a no-op, not an error: the client would refuse
    // it (the server answers `{paths: []}` with a 422) and there is nothing
    // to tell the user about a button they did not press.
    if (emptySelection(paths)) return false;
    return runOp('stage', () => gitStage(paths));
  },

  unstage: async (paths) => {
    if (emptySelection(paths)) return false;
    return runOp('unstage', () => gitUnstage(paths));
  },

  /**
   * The one write that destroys, and so the one with `worktree: true`: a
   * tracked file goes back to what the index holds and an untracked one is
   * deleted, either of which can be under an open tab.
   */
  discard: async (paths) => {
    if (emptySelection(paths)) return false;
    return runOp('discard', () => gitDiscard(paths), { worktree: true });
  },

  /**
   * Commit the index, `all` to stage the tracked changes first.
   *
   * The empty message is refused here as well as by the disabled button: a
   * keyboard shortcut reaches this without passing the button at all.
   * `amend` is cleared with the message on success -- it describes the
   * commit that was just made, not the next one.
   *
   * `refs: ['branches']` because a commit MOVES the current branch: its sha,
   * its subject and its ahead count are all in the branch list, and the
   * header's own ahead count comes from the status and updates at once.
   * Without this the two would disagree for up to fifteen seconds.
   */
  commit: async (opts = {}) => {
    const message = get().commitMessage.trim();
    if (message === '') return false;
    const amend = get().amend;
    const ok = await runOp(
      'commit',
      () => gitCommit({ message, all: opts.all ?? false, amend }),
      { refs: ['branches'] },
    );
    if (ok) set({ commitMessage: '', amend: false });
    return ok;
  },

  /**
   * Make the project directory a repository.
   *
   * Followed by a full refresh rather than trusting the mutation's status:
   * `MutationResult` carries the status but not the `repo`, and `repoState`
   * is what decides whether the tab draws the Initialize screen or the
   * panel. Without the refresh the button would work and the screen would
   * not move. The branch list is the other half a repository that did not
   * exist a second ago needs, and `null` there is "not read yet".
   */
  init: async () => {
    const ok = await runOp('init', () => gitInit(), { refs: ['branches'] });
    if (ok) await get().refresh();
    return ok;
  },

  /**
   * Write `user.name` / `user.email` into this repository.
   *
   * Trimmed, because a trailing space in an email is a bug git will keep
   * forever; refused when both halves are empty, because that is the one
   * request `PUT /config` cannot do anything with.
   *
   * A half that was left blank is OMITTED rather than sent empty. The two
   * are different requests: an absent key means "leave that one alone", and
   * an empty string is a value the model refuses with a 400 `invalid_value`.
   * A user filling in only the email -- because the name is already set
   * globally -- is the common case, and sending `name: ""` beside it would
   * refuse the whole write.
   */
  saveIdentity: async ({ name, email }) => {
    const cleanName = name.trim();
    const cleanEmail = email.trim();
    if (cleanName === '' && cleanEmail === '') return false;
    const payload: { name?: string; email?: string } = {};
    if (cleanName !== '') payload.name = cleanName;
    if (cleanEmail !== '') payload.email = cleanEmail;
    const ok = await runOp('identity', () => setGitConfig(payload));
    if (ok) set({ identityFormOpen: false });
    return ok;
  },

  /**
   * Bring the tracking refs up to date, and nothing else.
   *
   * The one network operation that cannot change a file: it moves
   * `refs/remotes/*` and the ahead/behind counts, which is why it refreshes
   * the branch list and is not a worktree op.
   */
  fetch: async () => runOp('fetch', () => gitFetch(), { refs: ['branches'] }),

  /**
   * Fetch, then take the upstream's commits.
   *
   * `ff-only` is what the Pull button sends and what a Sync's middle step
   * does; `merge` is what the "Merge remote changes" follow-up sends after a
   * `diverged` refusal, which is the only way this store writes a merge
   * commit on purpose.
   */
  pull: async (strategy) =>
    runOp('pull', () => gitPull({ strategy }), {
      worktree: true,
      refs: ['branches'],
      refreshStatusOnError: alwaysRefresh,
    }),

  /**
   * Send the current branch to where its upstream says.
   *
   * Never carries a remote: naming one without `set_upstream` is a 400 by the
   * route's own first check, because the two together are the publish this
   * store spells `publish`.
   */
  push: async () =>
    runOp('push', () => gitPush({ setUpstream: false }), {
      refs: ['branches'],
    }),

  /** Pull, then push -- or publish, when there is nothing to pull from. */
  sync: async () =>
    runOp('sync', () => gitSync(), {
      worktree: true,
      refs: ['branches'],
      refreshStatusOnError: alwaysRefresh,
    }),

  /**
   * Push a branch that has no upstream, and record the pairing.
   *
   * The remote is the caller's when it picked one (Task 5 offers an
   * `ActionMenu` when there are several), and the only configured remote when
   * the list has been read and holds exactly one. Otherwise none is sent and
   * the server resolves it: it applies the same "the only one" rule, and
   * refuses with 400 `invalid_value` when several leave it no way to choose.
   * Sending the name we already have is what lets the success toast say where
   * the branch went without reading the remotes again.
   */
  publish: async (remote) => {
    const loadedRemotes = get().remotes;
    const selected = remote
      ?? (loadedRemotes?.length === 1 ? loadedRemotes[0].name : undefined);
    const options = selected === undefined
      ? { setUpstream: true }
      : { remote: selected, setUpstream: true };
    return runOp('publish', () => gitPush(options), {
      refs: ['branches'],
      remote: selected,
    });
  },

  /**
   * Create a branch, and switch to it unless told not to.
   *
   * `worktree: checkout` and not a plain `true`: a branch created without a
   * checkout is a new name for the commit HEAD is already on, and no file
   * moves -- so there is nothing to offer to reload.
   */
  createBranch: async (name, checkout = true, startPoint = null) =>
    runOp('create_branch', () => gitCreateBranch(name, checkout, startPoint), {
      worktree: checkout,
      refs: ['branches'],
    }),

  /**
   * Switch branches, or start tracking a remote one.
   *
   * `name: target` is the fallback for the toast: the server names the branch
   * it landed on in `detail.branch`, and this is what to say if a build
   * without that key ever answers.
   */

  checkout: async (target, kind) =>
    runOp('checkout', () => gitCheckout(target, kind), {
      worktree: true,
      refs: ['branches'],
      name: target,
    }),

  renameBranch: async (name, newName) =>
    runOp('rename_branch', () => gitRenameBranch(name, newName), {
      refs: ['branches'],
    }),

  deleteBranch: async (name, force) =>
    runOp('delete_branch', () => gitDeleteBranch(name, force), {
      refs: ['branches'],
    }),

  addRemote: async (name, url) =>
    runOp('add_remote', () => gitAddRemote(name, url), {
      refs: ['remotes'],
    }),

  setRemoteUrl: async (name, url) =>
    runOp('set_remote_url', () => gitSetRemoteUrl(name, url), {
      refs: ['remotes'],
    }),

  removeRemote: async (name) =>
    runOp('remove_remote', () => gitRemoveRemote(name), {
      refs: ['remotes'],
    }),

  /**
   * Put the working tree on the stash stack.
   *
   * A message nobody typed is sent as `null`, never as `""`: the two are
   * different requests, and the empty string is a 400 `invalid_value`. The
   * prompt is optional and Cancel and an empty box mean the same thing here,
   * which is "let git write its own `WIP on <branch>` subject".
   */
  stashPush: async (message, includeUntracked) => {
    const cleanMessage = message === null || message.trim() === '' ? null : message.trim();
    return runOp(
      'stash_push',
      () => gitStashPush(cleanMessage, includeUntracked),
      { worktree: true, refs: ['stashes'] },
    );
  },

  /**
   * Apply and remove one stash entry.
   *
   * The argument is `StashInfo.index` -- git's own `stash@{N}` -- and never
   * the row's position in the array: the two agree until a drop, after which
   * the position would address a different entry every time.
   */

  stashPop: async (index) =>
    runOp('stash_pop', () => gitStashPop(index), {
      worktree: true,
      refs: ['stashes'],
      refreshStatusOnError: (err) =>
        err instanceof GitApiError && err.code === 'conflict',
    }),

  stashApply: async (index) =>
    runOp('stash_apply', () => gitStashApply(index), {
      worktree: true,
      refs: ['stashes'],
      refreshStatusOnError: (err) =>
        err instanceof GitApiError && err.code === 'conflict',
    }),

  stashDrop: async (index) =>
    runOp('stash_drop', () => gitStashDrop(index), {
      refs: ['stashes'],
    }),

  /** Put the tree back the way it was before the merge started. */
  abortMerge: async () =>
    runOp('abort_merge', () => gitAbortMerge(), { worktree: true }),

  /**
   * Settle one conflicted file.
   *
   * `ours` and `theirs` overwrite it before staging it, which is why this is a
   * worktree op even though `mark` -- the manual resolution the user has
   * already saved -- only stages what is there.
   */
  resolve: async (path, side) =>
    runOp('resolve', () => gitResolve(path, side), { worktree: true }),

  setHideLayout: (hide) => {
    set({ hideLayout: hide });
    saveHideLayout(hide);
  },

  openIdentityForm: () => revealIdentityForm(),

  closeIdentityForm: () => set({ identityFormOpen: false }),

  dismissError: () => set({ lastError: null }),

  /**
   * Offer to reload the open graphs a write put different bytes under.
   *
   * The filter has two halves. `currentGraphFile === <stem>` is the tab
   * showing that graph; `projectOrigin === null || === projectDir` is the
   * tab belonging to THIS project -- a tab stamped with another project's
   * directory is showing a file this repository does not contain, and an
   * unstamped one has never been saved anywhere, so both are left alone.
   *
   * One sticky toast, and the previous one is taken down rather than left to
   * stack: two discards in a row are one situation, and the second toast's
   * count is the true one.
   */
  applyWorktreeChange: (changedPaths) => {
    const bases = new Set<string>();
    for (const path of changedPaths) {
      const base = graphBaseName(path);
      if (base !== null) bases.add(base);
    }
    if (bases.size === 0) return;
    const projectDir = useProjectStore.getState().projectDir;
    const affected: AffectedTab[] = [];
    for (const tab of useTabStore.getState().tabs) {
      const file = tab.currentGraphFile;
      if (file === null || !bases.has(file)) continue;
      if (tab.projectOrigin !== null && tab.projectOrigin !== projectDir) continue;
      affected.push({ id: tab.id, file });
    }
    if (affected.length === 0) return;
    if (changedToastId !== null) useToastStore.getState().removeToast(changedToastId);
    changedToastId = toast(
      t('git.toast.changedOnDisk', { count: affected.length }),
      'warning',
      {
        sticky: true,
        action: {
          label: t('git.toast.reload'),
          onClick: () => {
            // Forgotten, not removed: `ToastContainer`
            // (`components/shared/Toast.tsx`) takes an action toast down as
            // soon as its action has run, so a `removeToast` here would be a
            // second take-down of a toast that is already gone. What this
            // line is for is the HANDLE -- dropping it stops the next
            // "changed on disk" offer from trying to supersede a dead id.
            changedToastId = null;
            void reloadAffected(affected);
          },
        },
      },
    );
  },
}));

/** Test-only: stop the schedulers and put the store back as it loaded. */
export function _resetGitStoreForTesting(): void {
  stopPoll();
  unlisten();
  setWorktreeWriteListener(null);
  cancelWriteDebounce();
  cancelRevisitDebounce();
  attachCount = 0;
  readSeq = 0;
  refReadSeq = { branches: 0, remotes: 0, stashes: 0 };
  identitySeq = 0;
  changedToastId = null;
  useGitStore.setState({
    repoState: 'unknown',
    repo: null,
    status: null,
    identity: null,
    branches: null,
    remotes: null,
    stashes: null,
    sections: loadSections(),
    loading: false,
    loadError: null,
    busyOp: null,
    netOp: null,
    lastError: null,
    liveMessage: '',
    commitMessage: '',
    amend: false,
    hideLayout: loadHideLayout(),
    identityFormOpen: false,
  });
}
