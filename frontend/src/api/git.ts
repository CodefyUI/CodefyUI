/**
 * The `/api/git` client — everything the Source Control tab says to git.
 *
 * A sibling of `rest.ts` rather than another thousand lines inside it: the
 * git routes are one feature with one error envelope, and keeping them here
 * means the tab's whole wire contract is a single file to read against
 * `backend/app/core/git/models.py`.
 *
 * Two rules the backend sets and this file follows:
 *
 *  - **The nine GETs are open; mutations are not.**
 *    `auth_guard` (backend/app/main.py) asks for the session token only on
 *    mutating methods, so the reads go through a bare `fetch` like every
 *    other open GET in `rest.ts` and the writes go through `apiFetch`, which
 *    attaches the header. Mutations use POST, PUT and DELETE; the wrapper
 *    covers all three.
 *  - **A refusal is always the same four keys** (`code`, `message`, `hint`,
 *    `stderr`, plus `op` on `busy`), which is what `GitApiError` carries. The
 *    two shapes that are NOT that envelope — FastAPI's 422 list and the
 *    auth guard's 403 string — are folded into it rather than left for every
 *    caller to sniff; see `gitApiError`.
 *
 * Responses are normalized field by field, for the reason `listPacks` and
 * `listPluginCatalog` give: the tab maps over the four status groups on every
 * repaint, so an absent key has to arrive as an empty list rather than as a
 * crash mid-render.
 */

import { apiFetch } from './_auth';
import { ApiError, apiError, errorDetail } from './rest';

const BASE_URL = '/api/git';

/* ── Wire types (mirrors of backend/app/core/git/models.py) ─────────── */

/**
 * What one file in the status is. `untracked` and `conflict` are not git
 * status letters — they are the porcelain-v2 record types `?` and `u` — but
 * they belong in the same union because the tab asks one question of a file
 * ("what chip, what actions") and wants one answer.
 */
export type FileKind =
  | 'modified'
  | 'added'
  | 'deleted'
  | 'renamed'
  | 'copied'
  | 'typechange'
  | 'untracked'
  | 'conflict';

/** Which config file a value was read from. */
export type ConfigScope = 'local' | 'global' | 'system';

/**
 * How far the server got when it looked for a repository. Resolved in this
 * order, first hit wins — so `not_repo` means git is installed and new
 * enough, and only the directory is wrong.
 */
export type RepoState =
  | 'no_project'
  | 'git_missing'
  | 'git_too_old'
  | 'not_repo'
  | 'ready';

/**
 * One file in one of the four status groups.
 *
 * The same path can appear twice in one status — `MM` is staged AND unstaged
 * — as two entries with the same `xy` and different `kind`, which is what
 * lets the tab offer "unstage" and "discard" on the same file at once.
 */
export interface GitFile {
  /** Repository-relative, POSIX separators, never quoted. */
  path: string;
  /** Where a rename or copy came from; null for every other kind. */
  orig_path: string | null;
  kind: FileKind;
  /** git's two porcelain-v2 letters; an untracked entry gets `??`. */
  xy: string;
  /** Similarity percentage of a rename or a copy (git's `R100` / `C75`). */
  score: number | null;
}

/** Everything one `git status --porcelain=v2 --branch` call said. */
export interface GitStatus {
  /** The current branch, or null when HEAD is detached. */
  branch: string | null;
  detached: boolean;
  /** The commit HEAD points at; null on an unborn branch. */
  head: string | null;
  unborn: boolean;
  upstream: string | null;
  /** Both null when there is no upstream, and when the upstream is gone. */
  ahead: number | null;
  behind: number | null;
  upstream_gone: boolean;
  staged: GitFile[];
  unstaged: GitFile[];
  untracked: GitFile[];
  conflicted: GitFile[];
  stash_count: number;
  merge_in_progress: boolean;
  rebase_in_progress: boolean;
}

/** Whether the tab can talk to a repository at all, and why not. */
export interface RepoInfo {
  state: RepoState;
  /** The open project directory, as an absolute path string. */
  project_dir: string | null;
  /** `git --version`'s answer; null means it could not be read. */
  git_version: string | null;
  /** Set only when the project directory sits inside SOME OTHER repository. */
  nested_toplevel: string | null;
}

/**
 * `GET /api/git/status`. Always a 200: "there is no project open" is a screen
 * the tab draws, not a failure it reports — and `status` is null beside every
 * state except `ready`.
 */
export interface StatusResponse {
  repo: RepoInfo;
  status: GitStatus | null;
}

/** Who commits, and which config file each half came from. */
export interface Identity {
  name: string | null;
  email: string | null;
  name_scope: ConfigScope | null;
  email_scope: ConfigScope | null;
}

/** One local branch, including its relationship to an upstream branch. */
export interface BranchInfo {
  name: string;
  sha: string;
  current: boolean;
  upstream: string | null;
  ahead: number | null;
  behind: number | null;
  gone: boolean;
  subject: string;
  committed_at: number;
}

/** One remote-tracking branch, split into remote and branch names. */
export interface RemoteBranchInfo {
  name: string;
  remote: string;
  sha: string;
  subject: string;
  committed_at: number;
}

/** All local and remote-tracking branches. */
export interface BranchesResponse {
  current: string | null;
  detached: boolean;
  local: BranchInfo[];
  remote: RemoteBranchInfo[];
}

/** One configured remote and the URLs git uses to fetch and push. */
export interface RemoteInfo {
  name: string;
  fetch_url: string;
  push_url: string;
}

/** One stash entry. `index` is git's stash index, not the array position. */
export interface StashInfo {
  index: number;
  message: string;
  branch: string | null;
  created_at: number;
}

/* ── History, diffs and one file at a ref ───────────────────────────────
   The four reads below are the only place in this file where the wire's
   snake_case does NOT survive into the app. The status shapes above predate
   the tab and are read by a dozen components as `orig_path` / `stash_count`;
   changing those now would be a rename with no reader asking for it. These
   four are new, nothing outside this file has seen them yet, and the rest of
   the frontend is camelCase -- so the normalizers below are also the place
   the spelling changes, once, rather than at every call site. */

/**
 * One commit, as `git log` printed it.
 *
 * `authoredAt` is epoch SECONDS -- git's own `%at`, and 0 when git could not
 * read the date at all -- because that is what `relativeTime` takes. It is
 * deliberately not a `Date`: the browser formats it, and a normalizer that
 * built one would put a timezone decision in the wire layer.
 *
 * `refs` is `%D` split on `", "`, so an entry looks like `HEAD -> main`,
 * `origin/main` or `tag: v1` -- the whole decoration, not a name.
 */
export interface GitCommit {
  sha: string;
  /** git's own abbreviation, which is what a row shows. */
  short: string;
  /** Every parent; a merge has two or more, a root commit none. */
  parents: string[];
  authorName: string;
  authorEmail: string;
  authoredAt: number;
  refs: string[];
  subject: string;
  /** Everything after the subject line; the empty string for most commits. */
  body: string;
}

/**
 * One page of history.
 *
 * `hasMore` is the server's answer, not a guess from the page size: it asked
 * for one row more than the limit and dropped it. `unborn` is a branch with
 * no commits yet, which is a 200 with an empty page rather than a refusal.
 */
export interface GitLogPage {
  commits: GitCommit[];
  hasMore: boolean;
  unborn: boolean;
}

/**
 * One file a commit touched.
 *
 * Three fields rather than the five a status entry carries: a commit's file
 * list is drawn to be READ and opened, and the two that are missing --
 * porcelain's `xy` letters and a rename score -- say nothing a row shows.
 * `kind` is the same union, so the chip beside a commit's file is the chip
 * beside a status file.
 */
export interface GitCommitFile {
  path: string;
  /** Where a rename or copy came from; null for every other kind. */
  origPath: string | null;
  kind: FileKind;
}

/** Which two sides a diff compares. A fourth word is a 422 from the route. */
export type GitDiffScope = 'worktree' | 'index' | 'commit';

/**
 * One file's change, as a unified patch.
 *
 * `oldRef` / `newRef` are what the two sides ARE (`index` and `worktree`,
 * `HEAD` and `index`, `<sha>^` and `<sha>`), and `oldRef` is null for a root
 * commit, which has no parent to compare against.
 *
 * The `old_text` / `new_text` the route can also answer with are deliberately
 * NOT here: they only arrive with `blobs=1`, which costs two more git reads,
 * and this build's side-by-side view is derived from the patch instead.
 */
export interface GitDiff {
  patch: string;
  /** git printed its "Binary files ... differ" marker rather than a patch. */
  binary: boolean;
  /** The patch was cut at 1 MiB; what is here is the first megabyte. */
  truncated: boolean;
  oldRef: string | null;
  newRef: string | null;
  /** The file does not exist on that side -- an add, or a delete. */
  oldMissing: boolean;
  newMissing: boolean;
}

/**
 * One file's whole contents at one ref.
 *
 * `size` is the size BEFORE any truncation, and it is the field that tells a
 * 2 MiB blob (never read, so `text` is empty and `truncated` is true) apart
 * from a file that really is empty.
 */
export interface GitFileAtRef {
  text: string;
  binary: boolean;
  size: number;
  truncated: boolean;
}

/**
 * What one write left behind.
 *
 * `status` is the fresh status after the operation, so the tab never has to
 * ask twice and can never draw a stale panel after a stage or a commit.
 * `detail` is deliberately open — `{sha, short}` after a commit, `{skipped}`
 * after a whole-tree write, `{scaffold}` after an init — so it is read
 * through narrowing rather than typed per operation.
 */
export interface MutationResult {
  status: GitStatus;
  /** Paths this operation changed, for the tab to reload in an open editor. */
  changed_paths: string[];
  head: string | null;
  detail: Record<string, unknown>;
}

/**
 * What stage / unstage / discard act on: some paths, or the whole tree.
 *
 * A union rather than two arguments because `PathsRequest` accepts exactly
 * one of the two forms — `{paths: [...]}` or `{all: true}` — and refuses
 * both, neither, and the empty list with a 422. `'all'` is spelled out at
 * every call site so "nothing was selected" can never arrive here as an
 * empty array that a `paths.length ? ... : all` line would turn into
 * "everything".
 */
export type GitPathSelection = string[] | 'all';

/* ── Errors ─────────────────────────────────────────────────────────── */

/**
 * Every `code` a refusal can carry.
 *
 * The list is `backend/app/core/git/errors.py`'s `CODES` table, plus
 * `git_service_unavailable` (which the route raises without going through
 * that table, for a server that started without the service), plus the two
 * this file synthesises for the answers that are not the git envelope at
 * all: `invalid` for FastAPI's 422 and `unknown` for everything else.
 *
 * Runtime data as well as a type, so a code from a NEWER server degrades to
 * `unknown` — and therefore to the generic message — instead of reaching a
 * translation lookup that has nothing for it. The raw string is still on the
 * error, under `body.detail.code`, for a bug report to quote.
 */
export const GIT_ERROR_CODES = [
  // Pre-flight: the repository is not in a state where git can be asked.
  'no_project',
  'git_missing',
  'git_too_old',
  'not_repo',
  'busy',
  'merge_in_progress',
  'no_remote',
  // The process itself.
  'timeout',
  // Talking to a remote.
  'auth_required',
  'network',
  'non_fast_forward',
  'remote_rejected',
  'diverged',
  'no_upstream',
  // The host's own git configuration refuses a plain push: `push.default`, or
  // an upstream branch whose name is not this branch's.
  'push_config',
  'remote_exists',
  // The working tree.
  'conflict',
  'dirty_tree',
  'nothing_to_commit',
  'identity_missing',
  'detached_head',
  'branch_exists',
  'branch_not_merged',
  'signing_failed',
  'not_found',
  // Validation.
  'invalid_path',
  'invalid_ref',
  'invalid_url',
  'invalid_value',
  'path_not_in_status',
  'ignored',
  // Everything else git can fail at.
  'git_failed',
  // The server failing to have started, which is not a git failure.
  'git_service_unavailable',
  // Not the git envelope at all: see `gitApiError`.
  'invalid',
  'unknown',
] as const;

export type GitErrorCode = (typeof GIT_ERROR_CODES)[number];

const GIT_ERROR_CODE_SET: ReadonlySet<string> = new Set(GIT_ERROR_CODES);

/**
 * How long the server gives one git process before it stops it, in seconds
 * (`backend/app/core/git/runner.py`: `T_STATUS` / `T_LOCAL` / `T_READ` /
 * `T_NETWORK`).
 *
 * Held here because a 504 arrives as `{code: "timeout"}` and NOTHING else:
 * the number is not in the body, so the sentence the tab shows has to supply
 * it. This client does not abort on these — the server already enforces
 * them, and a second deadline in the browser would answer a slow commit with
 * an `AbortError` that has no code to translate.
 *
 * Three of the four are the server's number exactly. The network one is
 * `T_NETWORK` (120) plus a ten-second grace, and that is deliberate: a
 * network operation is two or three git processes plus the request itself, so
 * the deadline the USER experienced is longer than the one any single process
 * was given. Do not "correct" it to 120.
 */
export const GIT_TIMEOUTS_S = {
  /** `GET /status` — the poll. */
  status: 10,
  /** Every write: init, stage, unstage, discard, commit, identity. */
  mutation: 30,
  /** The other reads: log, diff, file, config, refs and stashes. */
  read: 20,
  /** A remote operation: git's own `T_NETWORK` 120 s, plus the grace above. */
  network: 130,
} as const;

/**
 * A `/api/git` call the server refused.
 *
 * An `ApiError` subclass rather than a bare `Error`, the way `PackApiError`
 * is: `status`, `message` and `body` mean exactly what they mean everywhere
 * else in the app, and the git-specific half is the four keys added beside
 * them. Nothing is lost by the inheritance — the envelope has no field
 * `ApiError` cannot hold — and `errorDetail()` keeps working on it.
 */
export class GitApiError extends ApiError {
  // A field rather than a line in a constructor, so it wins over the base's
  // own assignment (a field initializer runs after `super()`).
  override name = 'GitApiError';
  /** What the tab switches on and translates. */
  readonly code: GitErrorCode;
  /** The one fact the code cannot carry: which file, which branch. */
  readonly hint: string | null;
  /** git's own tail, for when the classification is wrong. */
  readonly stderr: string | null;
  /**
   * Which operation holds one of the server's two locks, so the tab can say
   * what is still running rather than only "wait". The local vocabulary is
   * `init`, `stage`, `unstage`, `discard`, `commit`, `set_identity`,
   * `create_branch`, `checkout`, `rename_branch`, `delete_branch`,
   * `add_remote`, `set_remote_url`, `remove_remote`, `stash_push`,
   * `stash_pop`, `stash_apply`, `stash_drop`, `abort_merge`, `resolve`; the
   * network lane adds `fetch`, `pull`, `push`, `sync` and `publish`.
   *
   * Kept as a string rather than a closed union so a newer server can still
   * name its operation. Store-side display goes through `gitOpKey`, including
   * the `set_identity` wire spelling whose translation key is `identity`.
   */
  readonly op: string | null;

  constructor(
    status: number,
    message: string,
    detail: {
      code: GitErrorCode;
      hint?: string | null;
      stderr?: string | null;
      op?: string | null;
    },
    body: Record<string, unknown> | null = null,
  ) {
    super(status, message, body);
    this.code = detail.code;
    this.hint = detail.hint ?? null;
    this.stderr = detail.stderr ?? null;
    this.op = detail.op ?? null;
  }
}

/** The value when it is a string, and null for anything else (including null). */
function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/** A code the frontend knows, or `unknown` for one from a newer server. */
function asGitErrorCode(value: unknown): GitErrorCode {
  return typeof value === 'string' && GIT_ERROR_CODE_SET.has(value)
    ? (value as GitErrorCode)
    : 'unknown';
}

/**
 * FastAPI's 422 body is a LIST of `{type, loc, msg, input}` entries, and the
 * whole complaint is in the `msg` fields. Joined rather than picked, because
 * a `PathsRequest` that fails two checks has two sentences and showing one
 * of them would describe half the problem.
 */
function validationMessage(detail: readonly unknown[]): string | null {
  const messages = detail
    .map((entry) =>
      entry !== null && typeof entry === 'object' && 'msg' in entry
        ? (entry as { msg: unknown }).msg
        : null,
    )
    .filter((msg): msg is string => typeof msg === 'string');
  return messages.length > 0 ? messages.join('; ') : null;
}

/**
 * Build the error for a refused git request, reading the body once.
 *
 * Three body shapes reach this, and only the first is the git envelope:
 *
 *  - `{detail: {code, message, hint, stderr}}` — a `GitError`, in full.
 *  - `{detail: [{msg, ...}, ...]}` — FastAPI's own 422, from a body that
 *    failed `PathsRequest`/`CommitRequest` validation or carried a key the
 *    model forbids. Code `invalid`, message = every `msg` it listed.
 *  - `{detail: "Missing or invalid X-CodefyUI-Token header"}` — the auth
 *    guard's 403, which is a string. Code `unknown`, message = the string.
 *
 * Anything else (a proxy's HTML, a body that is not JSON) lands on `unknown`
 * with the status text, so a caller never has to handle "no error at all".
 */
export async function gitApiError(res: Response): Promise<GitApiError> {
  // Built through `apiError` so the string / coded / not-JSON reading is the
  // one in `rest.ts` rather than a second copy of it here.
  const base = await apiError(res);
  const coded = errorDetail(base);
  if (coded !== null) {
    return new GitApiError(
      base.status,
      asString(coded.message) ?? base.message,
      {
        code: asGitErrorCode(coded.code),
        hint: asString(coded.hint),
        stderr: asString(coded.stderr),
        op: asString(coded.op),
      },
      base.body,
    );
  }
  // `errorDetail` answers null for an array as well as for a string, so the
  // 422 has to be read off the body itself.
  const detail = base.body?.detail;
  if (Array.isArray(detail)) {
    return new GitApiError(
      base.status,
      validationMessage(detail) ?? base.message,
      { code: 'invalid' },
      base.body,
    );
  }
  return new GitApiError(base.status, base.message, { code: 'unknown' }, base.body);
}

/* ── Normalization ──────────────────────────────────────────────────── */

type RawGitFile = Omit<Partial<GitFile>, 'kind'> & { kind?: string };

type RawGitStatus = Omit<
  Partial<GitStatus>,
  'staged' | 'unstaged' | 'untracked' | 'conflicted'
> & {
  staged?: RawGitFile[];
  unstaged?: RawGitFile[];
  untracked?: RawGitFile[];
  conflicted?: RawGitFile[];
};

type RawRepoInfo = Omit<Partial<RepoInfo>, 'state'> & { state?: string };

type RawIdentity = Omit<Partial<Identity>, 'name_scope' | 'email_scope'> & {
  name_scope?: string | null;
  email_scope?: string | null;
};

type RawBranchInfo = Partial<BranchInfo>;
type RawRemoteBranchInfo = Partial<RemoteBranchInfo>;
type RawBranchesResponse = Omit<Partial<BranchesResponse>, 'local' | 'remote'> & {
  local?: RawBranchInfo[];
  remote?: RawRemoteBranchInfo[];
};
type RawRemoteInfo = Partial<RemoteInfo>;
type RawStashInfo = Partial<StashInfo>;

type RawStatusResponse = {
  repo?: RawRepoInfo;
  status?: RawGitStatus | null;
};

type RawCommitInfo = {
  sha?: string;
  short?: string;
  parents?: string[];
  author_name?: string;
  author_email?: string;
  authored_at?: number;
  refs?: string[];
  subject?: string;
  body?: string;
};

type RawLogResponse = {
  commits?: RawCommitInfo[];
  has_more?: boolean;
  unborn?: boolean;
};

type RawDiffResponse = {
  patch?: string;
  binary?: boolean;
  truncated?: boolean;
  old_ref?: string | null;
  new_ref?: string | null;
  old_missing?: boolean;
  new_missing?: boolean;
};

type RawFileAtRef = Partial<GitFileAtRef>;

type RawMutationResult = Omit<Partial<MutationResult>, 'status'> & {
  status?: RawGitStatus | null;
};

const FILE_KINDS: readonly string[] = [
  'modified',
  'added',
  'deleted',
  'renamed',
  'copied',
  'typechange',
  'untracked',
  'conflict',
];

const REPO_STATES: readonly string[] = [
  'no_project',
  'git_missing',
  'git_too_old',
  'not_repo',
  'ready',
];

const CONFIG_SCOPES: readonly string[] = ['local', 'global', 'system'];

function normalizeFile(raw: RawGitFile): GitFile {
  return {
    path: raw.path ?? '',
    orig_path: raw.orig_path ?? null,
    // A kind this build does not know reads as a plain modification: the row
    // then draws an `M` chip and offers the actions every tracked file has,
    // which is a worse label but not a broken row.
    kind: FILE_KINDS.includes(raw.kind ?? '') ? (raw.kind as FileKind) : 'modified',
    xy: raw.xy ?? '',
    score: raw.score ?? null,
  };
}

function normalizeFiles(raw: RawGitFile[] | undefined): GitFile[] {
  return (raw ?? []).map(normalizeFile);
}

/**
 * One status, field by field.
 *
 * A partly filled object normalizes to "git did not mention it", which is
 * the backend model's own default. A status that is not there AT ALL is a
 * different thing, and only `normalizeMutation` meets it -- see there.
 */
function normalizeStatus(raw: RawGitStatus): GitStatus {
  return {
    branch: raw.branch ?? null,
    detached: raw.detached ?? false,
    head: raw.head ?? null,
    unborn: raw.unborn ?? false,
    upstream: raw.upstream ?? null,
    ahead: raw.ahead ?? null,
    behind: raw.behind ?? null,
    upstream_gone: raw.upstream_gone ?? false,
    staged: normalizeFiles(raw.staged),
    unstaged: normalizeFiles(raw.unstaged),
    untracked: normalizeFiles(raw.untracked),
    conflicted: normalizeFiles(raw.conflicted),
    stash_count: raw.stash_count ?? 0,
    merge_in_progress: raw.merge_in_progress ?? false,
    rebase_in_progress: raw.rebase_in_progress ?? false,
  };
}

function normalizeRepo(raw: RawRepoInfo | undefined): RepoInfo {
  const state = raw?.state ?? '';
  return {
    // An unreadable state is `not_repo`: the tab then offers Initialize
    // Repository, which either works or refuses with a code of its own.
    state: REPO_STATES.includes(state) ? (state as RepoState) : 'not_repo',
    project_dir: raw?.project_dir ?? null,
    git_version: raw?.git_version ?? null,
    nested_toplevel: raw?.nested_toplevel ?? null,
  };
}

function normalizeScope(raw: string | null | undefined): ConfigScope | null {
  return raw !== null && raw !== undefined && CONFIG_SCOPES.includes(raw)
    ? (raw as ConfigScope)
    : null;
}

function normalizeIdentity(raw: RawIdentity): Identity {
  return {
    name: raw.name ?? null,
    email: raw.email ?? null,
    name_scope: normalizeScope(raw.name_scope),
    email_scope: normalizeScope(raw.email_scope),
  };
}

function normalizeBranch(raw: RawBranchInfo): BranchInfo {
  return {
    name: raw.name ?? '',
    sha: raw.sha ?? '',
    current: raw.current ?? false,
    upstream: raw.upstream ?? null,
    ahead: raw.ahead ?? null,
    behind: raw.behind ?? null,
    gone: raw.gone ?? false,
    subject: raw.subject ?? '',
    committed_at: raw.committed_at ?? 0,
  };
}

function normalizeRemoteBranch(raw: RawRemoteBranchInfo): RemoteBranchInfo {
  return {
    name: raw.name ?? '',
    remote: raw.remote ?? '',
    sha: raw.sha ?? '',
    subject: raw.subject ?? '',
    committed_at: raw.committed_at ?? 0,
  };
}

function normalizeBranches(raw: RawBranchesResponse): BranchesResponse {
  return {
    current: raw.current ?? null,
    detached: raw.detached ?? false,
    local: (raw.local ?? []).map(normalizeBranch),
    remote: (raw.remote ?? []).map(normalizeRemoteBranch),
  };
}

function normalizeRemote(raw: RawRemoteInfo): RemoteInfo {
  return {
    name: raw.name ?? '',
    fetch_url: raw.fetch_url ?? '',
    push_url: raw.push_url ?? '',
  };
}

function normalizeStash(raw: RawStashInfo): StashInfo {
  return {
    index: raw.index ?? 0,
    message: raw.message ?? '',
    branch: raw.branch ?? null,
    created_at: raw.created_at ?? 0,
  };
}

/**
 * One commit, field by field and snake_case to camelCase.
 *
 * A row the server could not fill still arrives whole, for the reason every
 * other normalizer here gives: the history list maps over these on every
 * repaint, and a missing `parents` reaching a `.length` mid-render is a blank
 * panel rather than a row with a hole in it.
 */
function normalizeCommit(raw: RawCommitInfo): GitCommit {
  return {
    sha: raw.sha ?? '',
    short: raw.short ?? '',
    parents: raw.parents ?? [],
    authorName: raw.author_name ?? '',
    authorEmail: raw.author_email ?? '',
    authoredAt: raw.authored_at ?? 0,
    refs: raw.refs ?? [],
    subject: raw.subject ?? '',
    body: raw.body ?? '',
  };
}

/**
 * One page of history.
 *
 * A body with no `unborn` in it reads as an unborn branch rather than as a
 * born one with nothing in it: those are the same empty list on screen, and
 * the honest one is the state that says WHY there is nothing to show. A page
 * that really is born always carries at least one commit.
 */
function normalizeLogPage(raw: RawLogResponse): GitLogPage {
  return {
    commits: (raw.commits ?? []).map(normalizeCommit),
    hasMore: raw.has_more ?? false,
    unborn: raw.unborn ?? true,
  };
}

/** One file a commit touched -- the three fields a history row draws. */
function normalizeCommitFile(raw: RawGitFile): GitCommitFile {
  return {
    path: raw.path ?? '',
    origPath: raw.orig_path ?? null,
    // Degraded exactly the way a status file's is: an unknown kind draws an
    // `M` chip, which is a worse label rather than a broken row.
    kind: FILE_KINDS.includes(raw.kind ?? '') ? (raw.kind as FileKind) : 'modified',
  };
}

/**
 * One diff, field by field.
 *
 * The two blob fields are read past rather than carried: they are null unless
 * `blobs=1` was asked for, and this client never asks.
 */
function normalizeDiff(raw: RawDiffResponse): GitDiff {
  return {
    patch: raw.patch ?? '',
    binary: raw.binary ?? false,
    truncated: raw.truncated ?? false,
    oldRef: raw.old_ref ?? null,
    newRef: raw.new_ref ?? null,
    oldMissing: raw.old_missing ?? false,
    newMissing: raw.new_missing ?? false,
  };
}

/** One file at one ref. `size` defaults to 0, never to the text's length. */
function normalizeFileAtRef(raw: RawFileAtRef): GitFileAtRef {
  return {
    text: raw.text ?? '',
    binary: raw.binary ?? false,
    size: raw.size ?? 0,
    truncated: raw.truncated ?? false,
  };
}

/**
 * One write's answer.
 *
 * A missing `status` is a REFUSAL, not a default. The backend's contract is
 * that every mutation answers with the status it left behind -- "a write
 * that succeeds and then cannot be read back is a failed request, not a
 * result with a hole in it" (`models.py`) -- and normalizing the hole would
 * hand the tab an empty status, which reads as a clean repository: the tab
 * would draw away the user's changes and claim the stage they just asked for
 * did nothing.
 */
function normalizeMutation(raw: RawMutationResult): MutationResult {
  if (raw.status === undefined || raw.status === null) {
    throw new GitApiError(502, 'the write was not read back', { code: 'unknown' });
  }
  return {
    status: normalizeStatus(raw.status),
    changed_paths: raw.changed_paths ?? [],
    head: raw.head ?? null,
    detail: raw.detail ?? {},
  };
}

/* ── Reads (open GETs, no token) ────────────────────────────────────── */

/**
 * The repository, and its status when there is one to read.
 *
 * The route answers 200 for every repository state, so "there is no
 * repository" is never a rejection: it arrives as `repo.state` beside a null
 * `status`. What DOES reject is everything that is genuinely a failure — a
 * server with no git service (503), a server that cannot be reached, and git
 * itself failing or being stopped at the deadline (500 `git_failed`, 504
 * `timeout`), because reading the status runs two real git processes under
 * the server's `T_STATUS` and not a lookup.
 */
export async function getGitStatus(): Promise<StatusResponse> {
  const res = await fetch(`${BASE_URL}/status`);
  if (!res.ok) throw await gitApiError(res);
  const data = (await res.json()) as RawStatusResponse;
  return {
    repo: normalizeRepo(data.repo),
    status: data.status ? normalizeStatus(data.status) : null,
  };
}

/**
 * `user.name` / `user.email`, and which config file each came from.
 *
 * The scope is half the answer: it is the difference between "this
 * repository" and "every repository on this machine".
 */
export async function getGitConfig(): Promise<Identity> {
  const res = await fetch(`${BASE_URL}/config`);
  if (!res.ok) throw await gitApiError(res);
  return normalizeIdentity((await res.json()) as RawIdentity);
}

/** Local and remote-tracking branches, including upstream state. */
export async function getGitBranches(): Promise<BranchesResponse> {
  const res = await fetch(`${BASE_URL}/branches`);
  if (!res.ok) throw await gitApiError(res);
  return normalizeBranches((await res.json()) as RawBranchesResponse);
}

/** Configured remotes, with credentials masked by the server. */
export async function getGitRemotes(): Promise<RemoteInfo[]> {
  const res = await fetch(`${BASE_URL}/remotes`);
  if (!res.ok) throw await gitApiError(res);
  const data = (await res.json()) as RawRemoteInfo[];
  return data.map(normalizeRemote);
}

/** The stash stack; callers address entries by `StashInfo.index`. */
export async function getGitStashes(): Promise<StashInfo[]> {
  const res = await fetch(`${BASE_URL}/stashes`);
  if (!res.ok) throw await gitApiError(res);
  const data = (await res.json()) as RawStashInfo[];
  return data.map(normalizeStash);
}

/**
 * One page of history, newest first.
 *
 * Offset paging, because that is what the route takes: `skip` rows in and
 * `limit` rows wide, with `hasMore` decided by the server having read one row
 * more than it returned. The window can DRIFT -- a commit made between two
 * pages shifts every row down one -- which is why the store dedupes by sha
 * when it appends and reloads from page 1 after anything that moves HEAD.
 *
 * The server bounds both numbers (`limit` at 100, `skip` at 2^31-1, because
 * git parses `--skip=` as a signed 32-bit integer) and answers 422 outside
 * them; nothing is clamped here, so a caller that gets one wrong finds out.
 */
export async function getGitLog(skip: number, limit: number): Promise<GitLogPage> {
  const query = new URLSearchParams({ skip: String(skip), limit: String(limit) });
  const res = await fetch(`${BASE_URL}/log?${query.toString()}`);
  if (!res.ok) throw await gitApiError(res);
  return normalizeLogPage((await res.json()) as RawLogResponse);
}

/**
 * The files one commit changed, against its first parent.
 *
 * A bare list rather than an envelope, which is the route's own shape. A sha
 * nothing resolves to is a 404 `not_found`, not an empty list.
 */
export async function getGitCommitFiles(sha: string): Promise<GitCommitFile[]> {
  const res = await fetch(`${BASE_URL}/commits/${encodeURIComponent(sha)}/files`);
  if (!res.ok) throw await gitApiError(res);
  const data = (await res.json()) as RawGitFile[];
  return data.map(normalizeCommitFile);
}

/**
 * One file's change, in one scope.
 *
 * The sha and the scope go together and the route refuses both mismatches: a
 * `commit` scope without a sha is a 400, and so is any other scope WITH one.
 * So an absent sha is sent as an absent key rather than as an empty value,
 * and a caller holding `''` -- which is what a form field with nothing in it
 * hands over -- is treated as having none.
 *
 * `blobs` is never sent. It would make the server read both whole files, and
 * the side-by-side view in this build is derived from the patch instead.
 */
export async function getGitDiff(target: {
  path: string;
  scope: GitDiffScope;
  sha?: string;
}): Promise<GitDiff> {
  const query = new URLSearchParams({ path: target.path, scope: target.scope });
  if (target.sha !== undefined && target.sha !== '') query.set('sha', target.sha);
  const res = await fetch(`${BASE_URL}/diff?${query.toString()}`);
  if (!res.ok) throw await gitApiError(res);
  return normalizeDiff((await res.json()) as RawDiffResponse);
}

/**
 * One whole file, at `HEAD`, `index`, `worktree` or a sha.
 *
 * The refusals are worth knowing before calling: an ignored path is a 403
 * `ignored` at every ref (as is anything `.env`-shaped), a conflicted file
 * has no stage 0 so `index` answers 409, and a blob over 2 MiB comes back
 * unread -- empty text, the real size, `truncated`.
 */
export async function getGitFile(target: {
  path: string;
  ref: string;
}): Promise<GitFileAtRef> {
  const query = new URLSearchParams({ path: target.path, ref: target.ref });
  const res = await fetch(`${BASE_URL}/file?${query.toString()}`);
  if (!res.ok) throw await gitApiError(res);
  return normalizeFileAtRef((await res.json()) as RawFileAtRef);
}

/* ── Writes (the session token, via apiFetch) ───────────────────────── */

type MutationMethod = 'POST' | 'PUT' | 'DELETE';

/**
 * One mutating call, from the URL to the fresh status.
 *
 * Every mutation shares the token, refusal envelope and response shape. The
 * method stays explicit here because branch and remote edits also use PUT and
 * DELETE.
 */
async function mutate(
  path: string,
  body?: unknown,
  method: MutationMethod = 'POST',
): Promise<MutationResult> {
  const res = await apiFetch(
    `${BASE_URL}${path}`,
    body === undefined
      ? { method }
      : {
          method,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
  );
  if (!res.ok) throw await gitApiError(res);
  return normalizeMutation((await res.json()) as RawMutationResult);
}

/**
 * The body `PathsRequest` accepts: the named paths, or the whole tree.
 *
 * Exactly one key, never both, and never `paths: []` -- the model refuses
 * all three of those with a 422. The empty list is the one a caller can
 * reach by accident, from a selection that turned out to be empty, so it is
 * refused HERE rather than a round trip later: a request that cannot succeed
 * should not leave the browser, and the caller gets the same
 * `GitApiError` it already handles instead of a 422 whose English is
 * pydantic's.
 */
function pathsBody(paths: GitPathSelection): { paths: string[] } | { all: true } {
  if (paths !== 'all' && paths.length === 0) {
    throw new GitApiError(400, 'nothing selected', { code: 'invalid' });
  }
  return paths === 'all' ? { all: true } : { paths };
}

/**
 * Make the project directory a repository, with the shared scaffold.
 *
 * No body: there is nothing to choose. Works from `not_repo` INCLUDING the
 * nested case, which is the one screen where it is the only way forward.
 */
export function gitInit(): Promise<MutationResult> {
  return mutate('/init');
}

/**
 * Stage the named paths, or the whole tree.
 *
 * `async` for the empty-selection refusal in `pathsBody`: a client function
 * that can fail should always fail the same way, and a synchronous throw
 * would miss the `.catch()` of a caller that had not awaited yet.
 */
export async function gitStage(paths: GitPathSelection): Promise<MutationResult> {
  return mutate('/stage', pathsBody(paths));
}

/**
 * Take the named paths, or everything, back out of the index.
 *
 * `async` for the reason `gitStage` gives: the empty-selection refusal must
 * arrive as a rejection, never as a synchronous throw.
 */
export async function gitUnstage(paths: GitPathSelection): Promise<MutationResult> {
  return mutate('/unstage', pathsBody(paths));
}

/**
 * Throw away working-tree changes. The one write that destroys — a tracked
 * file is restored from the index and an untracked one is deleted — so every
 * caller asks first.
 *
 * `async` for the reason `gitStage` gives: the empty-selection refusal must
 * arrive as a rejection, never as a synchronous throw.
 */
export async function gitDiscard(paths: GitPathSelection): Promise<MutationResult> {
  return mutate('/discard', pathsBody(paths));
}

/**
 * Commit the index, `all` to stage the tracked changes first, `amend` to
 * replace the previous commit instead of adding one.
 *
 * Both flags are sent explicitly rather than left off: `CommitRequest`
 * defaults them to false, and spelling them out is what keeps a partly
 * filled options object from reaching the wire as `amend: undefined`.
 * `detail.sha` / `detail.short` on the result name the commit this made.
 */
export function gitCommit(options: {
  message: string;
  all?: boolean;
  amend?: boolean;
}): Promise<MutationResult> {
  return mutate('/commit', {
    message: options.message,
    all: options.all ?? false,
    amend: options.amend ?? false,
  });
}

/**
 * Write `user.name` / `user.email` into THIS repository (git's `--local`;
 * the server never writes the machine's global config from a web request).
 *
 * Answers with the identity as it now READS, which is not the same thing as
 * what was written: a name written locally can still sit beside an email
 * that is still global, and that pair is what the tab shows.
 *
 * An omitted field is left alone. Sending neither is a 400 `invalid_value`,
 * so the form checks that before it calls.
 */
export async function setGitConfig(identity: {
  name?: string;
  email?: string;
}): Promise<Identity> {
  const res = await apiFetch(`${BASE_URL}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    // `JSON.stringify` drops an undefined value, so an omitted field is an
    // absent key rather than one the model has to read past. `IdentityRequest`
    // treats a missing half and a null half the same way -- leave it alone --
    // and refuses only the request that carries neither.
    body: JSON.stringify({ name: identity.name, email: identity.email }),
  });
  if (!res.ok) throw await gitApiError(res);
  return normalizeIdentity((await res.json()) as RawIdentity);
}

/* ── Refs, stashes, merges and remotes ───────────────────────────────── */

export type GitCheckoutKind = 'local' | 'remote';
export type GitResolveSide = 'ours' | 'theirs' | 'mark';
export type GitPullStrategy = 'ff-only' | 'merge';

/** Bring one resolved remote's tracking refs up to date. */
export function gitFetch(): Promise<MutationResult> {
  return mutate('/fetch', { remote: null });
}

/** Fetch, then fast-forward or explicitly merge the current upstream. */
export function gitPull(options: { strategy: GitPullStrategy }): Promise<MutationResult> {
  return mutate('/pull', { strategy: options.strategy });
}

/** Push to the upstream, or publish to a selected/resolved remote with `-u`. */
export function gitPush(options: {
  remote?: string;
  setUpstream: boolean;
}): Promise<MutationResult> {
  return mutate('/push', {
    remote: options.remote ?? null,
    set_upstream: options.setUpstream,
  });
}

/** Make the local and remote branch agree using the server's chosen steps. */
export function gitSync(): Promise<MutationResult> {
  return mutate('/sync');
}

/** Create a branch, checking it out by default. */
export function gitCreateBranch(
  name: string,
  checkout = true,
  startPoint: string | null = null,
): Promise<MutationResult> {
  return mutate('/branches', { name, checkout, start_point: startPoint });
}

/** Switch to a local branch, or create one that tracks a remote branch. */
export function gitCheckout(
  target: string,
  kind: GitCheckoutKind,
): Promise<MutationResult> {
  return mutate('/checkout', { target, kind });
}

/** Rename a local branch. */
export function gitRenameBranch(name: string, newName: string): Promise<MutationResult> {
  return mutate(`/branches/${encodeURIComponent(name)}`, { new_name: newName }, 'PUT');
}

/** Delete a local branch, optionally including one not fully merged. */
export function gitDeleteBranch(name: string, force: boolean): Promise<MutationResult> {
  const query = force ? '?force=1' : '';
  return mutate(`/branches/${encodeURIComponent(name)}${query}`, undefined, 'DELETE');
}

/** Add a named remote. */
export function gitAddRemote(name: string, url: string): Promise<MutationResult> {
  return mutate('/remotes', { name, url });
}

/** Replace one remote's URL. */
export function gitSetRemoteUrl(name: string, url: string): Promise<MutationResult> {
  return mutate(`/remotes/${encodeURIComponent(name)}`, { url }, 'PUT');
}

/** Remove one configured remote. */
export function gitRemoveRemote(name: string): Promise<MutationResult> {
  return mutate(`/remotes/${encodeURIComponent(name)}`, undefined, 'DELETE');
}

/** Put the working tree on the stash stack. */
export function gitStashPush(
  message: string | null,
  includeUntracked: boolean,
): Promise<MutationResult> {
  return mutate('/stashes', { message, include_untracked: includeUntracked });
}

/** Apply and remove the stash entry at git's own index. */
export function gitStashPop(index: number): Promise<MutationResult> {
  return mutate(`/stashes/${index}/pop`);
}

/** Apply and retain the stash entry at git's own index. */
export function gitStashApply(index: number): Promise<MutationResult> {
  return mutate(`/stashes/${index}/apply`);
}

/** Remove the stash entry at git's own index. */
export function gitStashDrop(index: number): Promise<MutationResult> {
  return mutate(`/stashes/${index}`, undefined, 'DELETE');
}

/** Restore the working tree from before the in-progress merge. */
export function gitAbortMerge(): Promise<MutationResult> {
  return mutate('/merge/abort');
}

/** Keep a side of one conflict, or stage the user's manual resolution. */
export function gitResolve(path: string, side: GitResolveSide): Promise<MutationResult> {
  return mutate('/resolve', { path, side });
}
