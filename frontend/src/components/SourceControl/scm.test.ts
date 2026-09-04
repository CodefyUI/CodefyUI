import { describe, expect, it, vi } from 'vitest';
import type { GitErrorCode } from '../../api/git';
import en from '../../i18n/locales/en';
import zhTW from '../../i18n/locales/zh-TW';
import type { GitAnyOp, GitStoreError } from '../../store/gitStore';
import {
  ERROR_HINT_KEY,
  ERROR_KEY,
  errorSentence,
  followUpFor,
  gitOpKey,
} from './scm';

const mappedErrors: Array<[GitErrorCode, string]> = [
  ['busy', 'git.error.busy'],
  ['nothing_to_commit', 'git.error.nothingToCommit'],
  ['identity_missing', 'git.error.identityMissing'],
  ['detached_head', 'git.error.detachedHead'],
  ['merge_in_progress', 'git.error.mergeInProgress'],
  ['not_repo', 'git.error.notRepo'],
  ['invalid_value', 'git.error.invalid'],
  ['auth_required', 'git.error.authRequired'],
  ['network', 'git.error.network'],
  ['non_fast_forward', 'git.error.nonFastForward'],
  ['diverged', 'git.error.diverged'],
  ['conflict', 'git.error.conflict'],
  ['dirty_tree', 'git.error.dirtyTree'],
  ['no_upstream', 'git.error.noUpstream'],
  ['no_remote', 'git.error.noRemote'],
  ['branch_exists', 'git.error.branchExists'],
  ['signing_failed', 'git.error.signingFailed'],
  ['remote_exists', 'git.error.remoteExists'],
  ['remote_rejected', 'git.error.remoteRejected'],
  ['push_config', 'git.error.pushConfig'],
];

/** A translate that shows its key and its arguments, so both are assertable. */
function translator(): (key: string, vars?: Record<string, string | number>) => string {
  return vi.fn((key: string, vars?: Record<string, string | number>) =>
    vars === undefined ? key : `${key}:${JSON.stringify(vars)}`,
  );
}

function err(
  code: GitErrorCode,
  over: { message?: string; op?: GitAnyOp | null } = {},
): GitStoreError {
  return {
    code,
    message: over.message ?? 'wire detail',
    hint: null,
    stderr: null,
    op: over.op ?? null,
  };
}

describe('SCM error copy', () => {
  it.each(mappedErrors)('maps %s to %s', (code, key) => {
    expect(ERROR_KEY[code]).toBe(key);
  });

  it('provides the actionable credential setup hint for auth_required', () => {
    expect(ERROR_HINT_KEY.auth_required).toBe('git.error.authRequiredHint');
  });

  it('carries the push-configuration sentence in both locales, translated', () => {
    // zh-TW is typed `Record<TranslationKey, string>`, so a missing key is a
    // build error rather than a test failure; what a test can still catch is
    // the key left as a copy of the English.
    expect(en['git.error.pushConfig']).toContain('push.default');
    expect(zhTW['git.error.pushConfig']).toContain('push.default');
    expect(zhTW['git.error.pushConfig']).not.toBe(en['git.error.pushConfig']);
  });

  it('offers no recovery button for a push the host configuration refuses', () => {
    // Nothing the tab can press fixes `push.default`: the way out is git's own
    // sentence, which the Details disclosure already shows.
    expect(followUpFor('push_config')).toBeNull();
    expect(followUpFor('push_config', 'push')).toBeNull();
  });

  it('uses mapped, timeout, not-found, and generic sentences without losing details', () => {
    const t = translator();

    expect(errorSentence(err('conflict'), t)).toBe('git.error.conflict');
    expect(errorSentence(err('timeout', { message: 'git stopped after 130s' }), t)).toBe(
      'git stopped after 130s',
    );
    expect(errorSentence(err('not_found', { message: 'lost.graph.json' }), t)).toBe(
      'git.error.notFound:{"what":"lost.graph.json"}',
    );
    expect(errorSentence(err('unknown'), t)).toBe(
      'git.error.generic:{"message":"wire detail"}',
    );
  });

  it.each(['fetch', 'pull', 'sync', 'publish'] as const)(
    'reads an ambiguous remote refusal from %s as an unpublished branch',
    (op) => {
      expect(errorSentence(err('invalid_value', { op }), translator())).toBe(
        'git.error.noUpstream',
      );
    },
  );

  it.each([
    { name: 'a plain push, which sends no remote to be ambiguous', op: 'push' as const },
    { name: 'a local write', op: 'create_branch' as const },
    { name: 'no operation at all', op: null },
  ])('keeps the plain invalid sentence for $name', ({ op }) => {
    expect(errorSentence(err('invalid_value', { op }), translator())).toBe(
      'git.error.invalid',
    );
  });
});

describe('SCM operation labels and follow-ups', () => {
  // Every operation the store can be running, including the two spellings that
  // are the store's rather than the wire's: `identity` (the server calls that
  // write `set_identity`) and `publish` (the server sees a push).
  it.each([
    'status',
    'init',
    'stage',
    'unstage',
    'discard',
    'commit',
    'identity',
    'create_branch',
    'checkout',
    'rename_branch',
    'delete_branch',
    'add_remote',
    'set_remote_url',
    'remove_remote',
    'stash_push',
    'stash_pop',
    'stash_apply',
    'stash_drop',
    'abort_merge',
    'resolve',
    'fetch',
    'pull',
    'push',
    'sync',
    'publish',
  ] as const)('names %s with a git.op key', (op) => {
    expect(gitOpKey(op)).toBe(`git.op.${op}`);
  });

  it('offers only the two ruled inline recovery actions', () => {
    expect(followUpFor('diverged')).toBe('mergeRemote');
    expect(followUpFor('no_upstream')).toBe('publish');
    expect(followUpFor('network')).toBeNull();
    expect(followUpFor('unknown')).toBeNull();
  });

  it.each(['fetch', 'pull', 'sync', 'publish'] as const)(
    'offers Publish after an ambiguous remote refusal from %s',
    (op) => {
      expect(followUpFor('invalid_value', op)).toBe('publish');
    },
  );

  it.each([
    { name: 'a plain push, which goes where the upstream says', op: 'push' as const },
    { name: 'a local write', op: 'add_remote' as const },
    { name: 'a caller holding only a code', op: null },
  ])('offers nothing after an invalid value from $name', ({ op }) => {
    expect(followUpFor('invalid_value', op)).toBeNull();
  });

  it('answers a diverged pull with the merge retry, whatever the op', () => {
    expect(followUpFor('diverged', 'pull')).toBe('mergeRemote');
    expect(followUpFor('no_upstream', 'sync')).toBe('publish');
  });
});
