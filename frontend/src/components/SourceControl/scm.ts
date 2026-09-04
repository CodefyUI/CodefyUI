/**
 * What the Source Control tab CALLS things: the sentence a refusal gets, the
 * name of the operation that is running, and the one-click way out of the two
 * refusals that have one.
 *
 * A file of its own rather than lines inside `ScmHeader.tsx`, which is where
 * this table started. Three callers need the same answers now -- the header's
 * error line, the tab's busy line, and the store, which fills in the one hint
 * a server cannot write -- and a table only one component can see is a table
 * that gets copied into the second component that needs it.
 *
 * Everything here is a pure function of a code, an operation and `t`. Nothing
 * reads the store, so a component can translate a refusal it is holding
 * without asking anybody what is happening right now.
 */

import type { GitErrorCode } from '../../api/git';
import type { TranslationKey } from '../../i18n';
import type { GitAnyOp, GitStoreError } from '../../store/gitStore';

/** Translate with the same signature the store and the i18n hook both use. */
type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

/**
 * The codes that have a sentence of their own.
 *
 * Everything absent from this map falls to `git.error.generic`, which shows
 * git's own words -- and that is deliberately where `invalid` (FastAPI's 422),
 * `unknown` (a code from a newer server, or a body that was not the git
 * envelope) and `git_service_unavailable` land: a fixed sentence for those
 * would replace the only description of the problem that exists.
 */
export const ERROR_KEY: Partial<Record<GitErrorCode, TranslationKey>> = {
  busy: 'git.error.busy',
  nothing_to_commit: 'git.error.nothingToCommit',
  identity_missing: 'git.error.identityMissing',
  detached_head: 'git.error.detachedHead',
  merge_in_progress: 'git.error.mergeInProgress',
  not_repo: 'git.error.notRepo',
  invalid_value: 'git.error.invalid',
  auth_required: 'git.error.authRequired',
  network: 'git.error.network',
  non_fast_forward: 'git.error.nonFastForward',
  diverged: 'git.error.diverged',
  conflict: 'git.error.conflict',
  dirty_tree: 'git.error.dirtyTree',
  no_upstream: 'git.error.noUpstream',
  no_remote: 'git.error.noRemote',
  branch_exists: 'git.error.branchExists',
  signing_failed: 'git.error.signingFailed',
  remote_exists: 'git.error.remoteExists',
  remote_rejected: 'git.error.remoteRejected',
  // No follow-up button: nothing the tab can press changes `push.default` on
  // the server's machine, and git's own sentence -- which the Details
  // disclosure shows -- is the one that names the setting.
  push_config: 'git.error.pushConfig',
};

/**
 * The codes whose recovery step is local, and therefore ours to write.
 *
 * A server hint is English, and it describes what the SERVER saw. That is the
 * right thing to show for the one fact a code cannot carry -- which file,
 * which branch, which step of a sync failed -- so those are passed through
 * verbatim. A credential refusal is the exception: the recovery is two
 * commands on the machine running the server, they are the same two commands
 * every time, and they should be readable in the reader's own language.
 */
export const ERROR_HINT_KEY: Partial<Record<GitErrorCode, TranslationKey>> = {
  auth_required: 'git.error.authRequiredHint',
};

/** The local recovery hint where there is one, otherwise the server's own. */
export function errorHint(
  code: GitErrorCode,
  serverHint: string | null,
  t: Translate,
): string | null {
  const key = ERROR_HINT_KEY[code];
  return key === undefined ? serverHint : t(key);
}

/**
 * The operations whose 400 `invalid_value` means "which remote?".
 *
 * These four reach `resolve_remote` (backend `network.py`) with no remote
 * name: the server takes the one the upstream says, or the only one there is,
 * and refuses with 400 `invalid_value` when a branch with no upstream leaves
 * several to choose between. `fetch` and `pull` (whose first step is a fetch)
 * always send none; `sync` sends none and publishes when there is nothing to
 * pull from; `publish` sends none whenever the caller did not pick one and the
 * remote list is not exactly one long -- which includes the list not having
 * been read yet. What the user is looking at in that moment is the state
 * `no_upstream` describes -- this branch is not published -- and the way out is
 * the same button, whose remote picker is exactly the choice the server could
 * not make on its own.
 *
 * A PLAIN push is NOT in the set, and cannot be: it names no remote either,
 * but Git's own `%(push:remotename)` resolver answers for it
 * (`network._tracked_remote`), which has no ambiguous case -- nothing
 * configured is `no_remote`, and everything else is a destination. The one
 * `invalid_value` the push route raises is for a request that named a remote
 * WITHOUT asking to publish, which this store never sends.
 *
 * Publish is in the set even though it can name a remote, because the refusal
 * a named one gets is a malformed name, and the picker the button opens is
 * what replaces it. So there is no loop either way.
 */
const REMOTE_AMBIGUITY_OPS: ReadonlySet<string> = new Set([
  'fetch',
  'pull',
  'sync',
  'publish',
]);

/** Whether this refusal is that ambiguity rather than a bad value. */
function readsAsNoUpstream(code: GitErrorCode, op: GitAnyOp | null | undefined): boolean {
  return (
    code === 'invalid_value'
    && op !== null
    && op !== undefined
    && REMOTE_AMBIGUITY_OPS.has(op)
  );
}

/**
 * The sentence for one refusal.
 *
 * `timeout` is the exception that is not a lookup: the 504 body carries a code
 * and nothing else, so the store has already written the finished sentence
 * (it is the only place that knows which of the four deadlines applied) and
 * re-mapping the code here would throw that number away.
 */
export function errorSentence(err: GitStoreError, t: Translate): string {
  if (err.code === 'timeout') return err.message;
  if (err.code === 'not_found') return t('git.error.notFound', { what: err.message });
  if (readsAsNoUpstream(err.code, err.op)) return t('git.error.noUpstream');
  const key = ERROR_KEY[err.code];
  return key === undefined ? t('git.error.generic', { message: err.message }) : t(key);
}

/**
 * The i18n key naming an operation, for `git.busy` ("Running {op}...").
 *
 * A function rather than a key spelled at each call site, because the key stem
 * and the wire word are NOT the same string for two of them: a component that
 * read `op` off a `busy` refusal and pasted it into a key would ask for
 * `git.op.set_identity`, which does not exist, and a push the server refused
 * during a publish would be labelled as somebody else's operation. Taking a
 * `GitAnyOp` -- the store's vocabulary, not the server's -- is what makes both
 * impossible.
 */
export function gitOpKey(op: GitAnyOp): `git.op.${GitAnyOp}` {
  return `git.op.${op}`;
}

export type GitErrorFollowUp = 'mergeRemote' | 'publish';

/**
 * The refusal that has a button next to Dismiss, and which button that is.
 *
 * Two of them, and both are a retry the user could not have known to ask for:
 * `diverged` is answered by pulling with `strategy: 'merge'`, and a branch
 * with no upstream is answered by publishing it. `op` is what the refusal was
 * asked of -- see `REMOTE_AMBIGUITY_OPS` -- and may be left off by a caller
 * that has only a code.
 */
export function followUpFor(
  code: GitErrorCode,
  op: GitAnyOp | null = null,
): GitErrorFollowUp | null {
  if (code === 'diverged') return 'mergeRemote';
  if (code === 'no_upstream') return 'publish';
  if (readsAsNoUpstream(code, op)) return 'publish';
  return null;
}

/* ── What a prompt will accept ───────────────────────────────────────────
   Three grammars, all of them NARROWER than the server's and none of them
   the authority: `git check-ref-format` decides a branch name, and
   `paths.validate_remote_name` / `validate_remote_url` decide the other two.
   What they buy is a refusal while the dialog is still open, with the box
   still holding what was typed -- rather than a round trip that closes it and
   answers with a red line the user has to read to find out they left a space
   in. Every rule here has to be one the server also enforces, or the panel
   would refuse a name git would have taken. */

/** The whole alphabet a branch name may be spelled from. */
const BRANCH_CHARS = /^[A-Za-z0-9/._-]+$/;

/** `paths.MAX_BRANCH_NAME` -- a filesystem gives up well before this. */
const MAX_BRANCH_NAME = 255;

/**
 * Whether *name* is a branch name worth sending.
 *
 * Deliberately conservative: git accepts far more than this (a branch may
 * hold a `+`, a unicode letter, a `{`), and every one of those is a name
 * somebody has to type again on a command line one day. The rules that are
 * not taste are git's own -- no component starting with `.` or ending in
 * `.lock` or `.`, no empty component, no `..`, no leading `-` (which git
 * reads as an option rather than a name).
 */
export function isValidBranchName(name: string): boolean {
  if (name === '' || name.length > MAX_BRANCH_NAME) return false;
  if (!BRANCH_CHARS.test(name)) return false;
  if (name.startsWith('-')) return false;
  if (name.includes('..')) return false;
  return name
    .split('/')
    .every(
      (part) =>
        part !== ''
        && !part.startsWith('.')
        && !part.endsWith('.')
        && !part.endsWith('.lock'),
    );
}

/** `paths.REMOTE_NAME_RE`, character for character. */
const REMOTE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

/** Whether *name* is a remote name the server will take. */
export function isValidRemoteName(name: string): boolean {
  return REMOTE_NAME.test(name);
}

/** `paths.SCP_URL_RE` -- `git@github.com:owner/repo.git`, which has no scheme. */
const SCP_URL = /^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:\S+$/;

/** `paths.ALLOWED_URL_SCHEMES`, and the two whose host is checked on its own. */
const URL_SCHEMES = ['https://', 'ssh://', 'file://'];
const HOSTED_SCHEMES = ['https://', 'ssh://'];

/** `paths.MAX_REMOTE_URL`. */
const MAX_REMOTE_URL = 2048;

/**
 * Whether *url* is a remote this server would hand to git.
 *
 * The scheme list is the server's `GIT_ALLOW_PROTOCOL` allowlist, which is
 * why plain `http://` is absent and `ext::`/`fd::` -- the transports whose
 * "URL" is a command line git runs -- can never match it. A hosted URL has to
 * name a host, and neither the host nor the path of an ssh-bound URL may
 * start with `-`, because by the time git is done with one it is `ssh <host>
 * ...` and a host called `-oProxyCommand=...` is an ssh OPTION.
 *
 * `file://` is accepted although the prompt's copy says https or SSH: a bare
 * repository on a shared drive is the remote a classroom actually reaches
 * for, the server takes it, and there is no string here that could say so.
 */
export function isValidRemoteUrl(url: string): boolean {
  if (url === '' || url.length > MAX_REMOTE_URL) return false;
  // Whitespace and the C0 controls, both refused rather than trimmed: a URL
  // with a space in it is either a paste accident or two arguments trying to
  // look like one.
  if (/\s/.test(url) || /[\u0000-\u001f\u007f]/.test(url)) return false;
  if (url.startsWith('-')) return false;

  const lowered = url.toLowerCase();
  if (HOSTED_SCHEMES.some((scheme) => lowered.startsWith(scheme))) {
    const authority = url.split('://', 2)[1].split(/[/?#]/, 1)[0];
    return hasUsableHost(authority);
  }
  if (URL_SCHEMES.some((scheme) => lowered.startsWith(scheme))) return true;
  if (!SCP_URL.test(url)) return false;
  const colon = url.indexOf(':');
  return hasUsableHost(url.slice(0, colon)) && !url.slice(colon + 1).startsWith('-');
}

/** An authority whose host is present and is not an ssh option. */
function hasUsableHost(authority: string): boolean {
  let host = authority.slice(authority.lastIndexOf('@') + 1);
  const colon = host.lastIndexOf(':');
  // A trailing `:<digits>` is a port; a `:` with anything else after it is
  // part of an IPv6 literal, which keeps its brackets and its colons.
  if (colon >= 0 && /^\d+$/.test(host.slice(colon + 1))) host = host.slice(0, colon);
  return host !== '' && !host.startsWith('-');
}

/* ── How long ago ────────────────────────────────────────────────────────── */

/** Each unit and how many seconds of it before the next one takes over. */
const SPANS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
  ['year', 365 * 24 * 60 * 60],
  ['month', 30 * 24 * 60 * 60],
  ['week', 7 * 24 * 60 * 60],
  ['day', 24 * 60 * 60],
  ['hour', 60 * 60],
  ['minute', 60],
  ['second', 1],
];

/**
 * "3 hours ago", in the reader's own language.
 *
 * `Intl.RelativeTimeFormat` rather than a locale table: the words are the
 * platform's, so there is nothing here to translate and nothing to keep in
 * step with a second locale file. The unit is the largest one that fits,
 * which is what makes a stash list scannable -- "2 days ago" beside "an hour
 * ago" says more at a glance than two five-digit minute counts.
 *
 * *at* is epoch SECONDS, which is what git's `%at` answers with -- and 0 when
 * git could not read the reflog's date at all. That case gets an empty string
 * rather than "56 years ago": a row with no date on it says less than a row
 * with the wrong one.
 */
export function relativeTime(at: number, locale: string, now: number = Date.now()): string {
  if (!Number.isFinite(at) || at <= 0) return '';
  // Never the future. A server clock a second or two ahead of the browser's
  // is ordinary, and "in 2 seconds" beside a stash somebody just made is not.
  const ago = Math.max(0, Math.round(now / 1000 - at));
  const [unit, seconds] = SPANS.find(([, size]) => ago >= size) ?? SPANS[SPANS.length - 1];
  return new Intl.RelativeTimeFormat(locale, { numeric: 'auto' }).format(
    -Math.floor(ago / seconds),
    unit,
  );
}
