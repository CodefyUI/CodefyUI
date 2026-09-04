import { create } from 'zustand';
import {
  GIT_TIMEOUTS_S,
  GitApiError,
  getGitConfig,
  getGitStatus,
  gitCommit,
  gitDiscard,
  gitInit,
  gitStage,
  gitUnstage,
  setGitConfig,
  type GitErrorCode,
  type GitPathSelection,
  type GitStatus,
  type Identity,
  type MutationResult,
  type RepoInfo,
  type RepoState,
} from '../api/git';
import { GraphMissingError, reloadTabFromDisk } from '../utils/openSavedGraph';
import { setWorktreeWriteListener } from '../utils/worktreeWrite';
import { confirm } from '../utils/dialog';
import { useProjectStore } from './projectStore';
import { useTabStore } from './tabStore';
import { useToastStore, type ToastAction, type ToastType } from './toastStore';
import { useI18n, type TranslationKey } from '../i18n';

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
 *    for the two things that really are failures: a server with no git
 *    service (503) and a server that cannot be reached.
 *
 *  - **`busyOp` is a lock, not a spinner.** The backend serialises mutations
 *    behind one lock and refuses a second with `409 busy`; this store refuses
 *    it a step earlier, so two clicks on Stage cannot race each other into a
 *    409 the user has to read. Reads never take that lock, which is why the
 *    poll runs straight through a commit.
 *
 *  - **Nothing is reloaded from disk without asking.** A discard can put an
 *    older version of a graph a tab is showing back on disk, and the tab has
 *    no unsaved flag to check -- so the store raises one sticky toast with a
 *    Reload button and lets the person holding the unsaved edits decide.
 */

/* ── Vocabulary ─────────────────────────────────────────────────────── */

/**
 * The operations this store runs, in ITS spelling.
 *
 * `status` is the poll; the other six are the writes. Two spellings differ
 * from the wire deliberately:
 *
 *  - the identity write is `set_identity` on the server (it is the word that
 *    comes back in a `busy` refusal's `op` field), and `identity` here,
 *    because the i18n key is `git.op.identity`;
 *  - `status` is an operation here even though `refresh()` never goes through
 *    `runOp`: it is one of the three timeout buckets, and a union that omits
 *    it would make the map below partial for no gain.
 */
export type GitOp =
  | 'status'
  | 'init'
  | 'stage'
  | 'unstage'
  | 'discard'
  | 'commit'
  | 'identity';

/**
 * The i18n key naming an operation, for `git.busy` ("Running {op}...").
 *
 * Exported rather than spelled at the header's call site because the key
 * stem and the wire word are NOT the same string for one of the seven: a
 * component that read `err.op` off a `busy` refusal and pasted it into a key
 * would ask for `git.op.set_identity`, which does not exist. Going through
 * `GitOp` is what makes that impossible.
 */
export function gitOpKey(op: GitOp): `git.op.${GitOp}` {
  return `git.op.${op}`;
}

/** Which of the server's three deadlines an operation runs under. */
type TimeoutBucket = keyof typeof GIT_TIMEOUTS_S;

/**
 * The operation -> deadline map, complete by construction.
 *
 * A 504 arrives as `{code: "timeout"}` and nothing else -- the number of
 * seconds is not in the body -- so `git.error.timeout {seconds}` can only be
 * filled in from this side. `identity` is the WRITE (30 s); reading the
 * config back is the `read` bucket and passes it explicitly.
 */
const OP_TIMEOUT_BUCKET: Record<GitOp, TimeoutBucket> = {
  status: 'status',
  init: 'mutation',
  stage: 'mutation',
  unstage: 'mutation',
  discard: 'mutation',
  commit: 'mutation',
  identity: 'mutation',
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
}

/* ── i18n ───────────────────────────────────────────────────────────── */

/**
 * The keys this store names, listed once.
 *
 * `t` is typed on `TranslationKey` (`keyof typeof en`), and the `git.*`
 * strings land with the rest of the tab's copy rather than here -- so this
 * union is what stands in for the locale file until then: a typo is still a
 * compile error, and the list doubles as the store's half of the copy
 * checklist. `t` already falls back to the key itself when an entry is
 * missing, so a gap costs the raw key on screen and nothing else.
 */
type GitStoreKey =
  | 'git.error.busy'
  | 'git.error.timeout'
  | 'git.group.changes'
  | 'git.group.skipped'
  | 'git.group.staged'
  | 'git.toast.changedOnDisk'
  | 'git.toast.committed'
  | 'git.toast.initialized'
  | 'git.toast.missingOnDisk'
  | 'git.toast.reload'
  | 'git.toast.reloadConfirm';

function t(key: GitStoreKey, vars?: Record<string, string | number>): string {
  return useI18n.getState().t(key as TranslationKey, vars);
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

/** How many mounted views want the poll running. */
let attachCount = 0;
let pollTimer: ReturnType<typeof setInterval> | null = null;
let writeTimer: ReturnType<typeof setTimeout> | null = null;
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

function startPoll(): void {
  if (pollTimer !== null || !pageVisible()) return;
  pollTimer = setInterval(() => {
    void useGitStore.getState().refresh();
  }, GIT_POLL_MS);
}

function stopPoll(): void {
  if (pollTimer === null) return;
  clearInterval(pollTimer);
  pollTimer = null;
}

/**
 * A hidden page polls nothing and a returning one is refreshed at once.
 *
 * Half of what this tab shows is written by something other than this tab --
 * the command line, an editor, another browser tab -- so the moment the page
 * comes back is exactly the moment its status is most likely to be wrong.
 */
function onVisibilityChange(): void {
  if (attachCount === 0) return;
  if (pageVisible()) {
    startPoll();
    void useGitStore.getState().refresh();
  } else {
    stopPoll();
  }
}

function onFocus(): void {
  if (attachCount === 0) return;
  void useGitStore.getState().refresh();
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
 * the component picks the sentence from `code`.
 */
function toStoreError(err: unknown, bucket: TimeoutBucket): GitStoreError {
  if (!(err instanceof GitApiError)) {
    return { code: 'unknown', message: errorMessage(err), hint: null, stderr: null };
  }
  return {
    code: err.code,
    message:
      err.code === 'timeout'
        ? t('git.error.timeout', { seconds: GIT_TIMEOUTS_S[bucket] })
        : err.message,
    hint: err.hint,
    stderr: err.stderr,
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
 */
function groupCounts(status: GitStatus): string {
  const changes = status.unstaged.length + status.untracked.length;
  return `${t('git.group.staged')} ${status.staged.length}, `
    + `${t('git.group.changes')} ${changes}`;
}

/**
 * The sentence an operation ends with, or null when it has nothing to say.
 *
 * Commit and init are the two whose news is not visible in the panel -- a
 * commit id, a repository that now exists -- so they get a toast as well.
 * The identity write returns an identity rather than a status: nothing in
 * the panel moved, and the form redrawing with the saved values is the
 * feedback.
 */
function announcement(op: GitOp, result: MutationResult | Identity): string | null {
  if (!('status' in result)) return null;
  if (op === 'commit') return t('git.toast.committed', { sha: commitSha(result) });
  if (op === 'init') return t('git.toast.initialized');
  return groupCounts(result.status);
}

/**
 * Run one write with the lock, the error mapping and the fresh status.
 *
 * `worktree` is the bit that decides whether the "changed on disk" offer is
 * raised, and it is deliberately NOT `changed_paths.length > 0`: a commit
 * reports every file it swallowed, and every one of those is byte-identical
 * to what the tab already holds. Only an operation that can put DIFFERENT
 * bytes under an open tab sets it -- in this build exactly `discard`.
 *
 * Answers whether the operation succeeded, so a caller can clear its form.
 */
async function runOp(
  op: GitOp,
  call: () => Promise<MutationResult | Identity>,
  opts: { worktree?: boolean } = {},
): Promise<boolean> {
  if (useGitStore.getState().busyOp !== null) {
    // The same sentence the server's own 409 gets, because it is the same
    // fact: one git operation at a time.
    toast(t('git.error.busy'), 'warning');
    return false;
  }
  useGitStore.setState({ busyOp: op, lastError: null });
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
    const said = announcement(op, result);
    if (said !== null) {
      useGitStore.setState({ liveMessage: said });
      if (op === 'commit' || op === 'init') toast(said, 'success');
    }
    if (opts.worktree === true && 'status' in result) {
      useGitStore.getState().applyWorktreeChange(result.changed_paths);
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
    useGitStore.setState({ lastError: toStoreError(err, OP_TIMEOUT_BUCKET[op]) });
    // The one refusal with a form behind it: git will not commit without a
    // name and an email, and asking for them here is the whole fix.
    if (err instanceof GitApiError && err.code === 'identity_missing') {
      revealIdentityForm();
    }
    return false;
  } finally {
    useGitStore.setState({ busyOp: null });
  }
}

/** An explicitly empty selection: a request that could only be refused. */
function emptySelection(paths: GitPathSelection): boolean {
  return paths !== 'all' && paths.length === 0;
}

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
  /** No status has come back yet. Not "a request is in flight". */
  loading: boolean;
  /** The status could not be READ -- a 503 or an unreachable server. */
  loadError: string | null;
  busyOp: GitOp | null;
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
  noteWorktreeWrite: () => void;
  setCommitMessage: (message: string) => void;
  setAmend: (amend: boolean) => void;
  stage: (paths: GitPathSelection) => Promise<boolean>;
  unstage: (paths: GitPathSelection) => Promise<boolean>;
  discard: (paths: GitPathSelection) => Promise<boolean>;
  commit: (opts?: { all?: boolean }) => Promise<boolean>;
  init: () => Promise<boolean>;
  saveIdentity: (identity: { name: string; email: string }) => Promise<boolean>;
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
  loading: false,
  loadError: null,
  busyOp: null,
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
  },

  detach: () => {
    attachCount = Math.max(0, attachCount - 1);
    if (attachCount > 0) return;
    stopPoll();
    unlisten();
    setWorktreeWriteListener(null);
    cancelWriteDebounce();
  },

  /**
   * Read the status once.
   *
   * Every repository state is a 200, so the answer sets `repoState` and the
   * screen follows from it. A rejection leaves `repoState` alone -- on the
   * first read that is still `unknown`, which the tab draws as the loading
   * line until `loadError` turns it into the error line -- because "the
   * server did not answer" is not evidence that the repository changed.
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
      set({ loadError: errorMessage(err), loading: false });
    }
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
   */
  commit: async (opts = {}) => {
    const message = get().commitMessage.trim();
    if (message === '') return false;
    const amend = get().amend;
    const ok = await runOp('commit', () =>
      gitCommit({ message, all: opts.all ?? false, amend }),
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
   * not move.
   */
  init: async () => {
    const ok = await runOp('init', () => gitInit());
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
  attachCount = 0;
  readSeq = 0;
  identitySeq = 0;
  changedToastId = null;
  useGitStore.setState({
    repoState: 'unknown',
    repo: null,
    status: null,
    identity: null,
    loading: false,
    loadError: null,
    busyOp: null,
    lastError: null,
    liveMessage: '',
    commitMessage: '',
    amend: false,
    hideLayout: loadHideLayout(),
    identityFormOpen: false,
  });
}
