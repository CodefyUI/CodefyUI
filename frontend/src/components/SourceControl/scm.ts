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
