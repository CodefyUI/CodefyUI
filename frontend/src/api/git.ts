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
 *  - **The six GETs are open, the five POSTs and the one PUT are not.**
 *    `auth_guard` (backend/app/main.py) asks for the session token only on
 *    mutating methods, so the reads go through a bare `fetch` like every
 *    other open GET in `rest.ts` and the writes go through `apiFetch`, which
 *    attaches the header. `PUT /api/git/config` is the app's first PUT; the
 *    wrapper already covers it.
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
  'diverged',
  'no_upstream',
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
 * (`backend/app/core/git/runner.py`: `T_STATUS` / `T_LOCAL` / `T_READ`).
 *
 * Held here because a 504 arrives as `{code: "timeout"}` and NOTHING else:
 * the number is not in the body, so the sentence the tab shows has to supply
 * it. This client does not abort on these — the server already enforces
 * them, and a second deadline in the browser would answer a slow commit with
 * an `AbortError` that has no code to translate.
 */
export const GIT_TIMEOUTS_S = {
  /** `GET /status` — the poll. */
  status: 10,
  /** Every write: init, stage, unstage, discard, commit, identity. */
  mutation: 30,
  /** The other reads: log, diff, file, config. */
  read: 20,
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
   * Which operation holds the lock, so the tab can say "wait for the commit"
   * rather than "wait". Only ever set on `busy`, and only a mutation can be
   * refused that way -- reads never take the lock.
   *
   * The server's own vocabulary, which is not the tab's: `init`, `stage`,
   * `unstage`, `discard`, `commit`, `set_identity`. Note the last one --
   * `set_identity`, not `identity` -- so a store mapping it to a translated
   * op name has to spell the wire word, not the key's (`git.op.identity`).
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

type RawStatusResponse = {
  repo?: RawRepoInfo;
  status?: RawGitStatus | null;
};

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
 * The route answers 200 for every repository state, so a rejection here is
 * the server having no git service at all (503) or being unreachable — not
 * "there is no repository", which arrives as `repo.state` beside a null
 * `status`.
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

/* ── Writes (the session token, via apiFetch) ───────────────────────── */

/**
 * One mutating call, from the URL to the fresh status.
 *
 * Shared by all five POSTs because they differ only in the path and the
 * body: the token header, the content type, the refusal envelope and the
 * normalization are the same four lines each of them would otherwise repeat.
 */
async function mutate(path: string, body?: unknown): Promise<MutationResult> {
  const res = await apiFetch(
    `${BASE_URL}${path}`,
    body === undefined
      ? { method: 'POST' }
      : {
          method: 'POST',
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

/** Take the named paths, or everything, back out of the index. */
export async function gitUnstage(paths: GitPathSelection): Promise<MutationResult> {
  return mutate('/unstage', pathsBody(paths));
}

/**
 * Throw away working-tree changes. The one write that destroys — a tracked
 * file is restored from the index and an untracked one is deleted — so every
 * caller asks first.
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
