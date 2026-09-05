import { describe, expect, it, vi } from 'vitest';
import type { GitErrorCode } from '../../api/git';
import en from '../../i18n/locales/en';
import zhTW from '../../i18n/locales/zh-TW';
import type { GitAnyOp, GitStoreError } from '../../store/gitStore';
import {
  ERROR_HINT_KEY,
  ERROR_KEY,
  errorHint,
  errorSentence,
  followUpFor,
  gitOpKey,
  isValidBranchName,
  isValidRemoteName,
  isValidRemoteUrl,
  relativeTime,
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
    expect(errorHint('auth_required', 'the fetch step failed', translator())).toBe(
      'git.error.authRequiredHint',
    );
  });

  // The server writes its hints in English, and the sentence above them is
  // translated -- so "the merge step failed" under 'Local and remote branches
  // have diverged.' was a second line in a language the reader may not have,
  // saying no more than the first one.
  // `auth_required` is the exception, and it is the row above: it has a
  // sentence of its own AND a hint of its own, which replaces the server's.
  it.each(mappedErrors
    .map(([code]) => code)
    .filter((code) => ERROR_HINT_KEY[code] === undefined))(
    'drops the server hint for %s, which has a sentence of its own',
    (code) => {
      expect(errorHint(code, 'the merge step failed', translator())).toBeNull();
    },
  );

  it.each(['timeout', 'not_found'] as const)(
    'drops it for %s too, whose sentence is written here rather than mapped',
    (code) => {
      expect(errorHint(code, 'the push step failed', translator())).toBeNull();
    },
  );

  it.each(['git_failed', 'unknown', 'invalid_path', 'git_service_unavailable'] as const)(
    'keeps it for %s, where the sentence is git\'s own words as well',
    (code) => {
      expect(errorHint(code, 'the merge step failed', translator())).toBe(
        'the merge step failed',
      );
    },
  );

  it('has nothing to show when the server sent no hint either', () => {
    expect(errorHint('git_failed', null, translator())).toBeNull();
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

/*
 * The name grammars are DELIBERATELY narrower than git's, and the server is
 * still the authority: `git check-ref-format` is what a branch name has to
 * pass. What these buy is a name refused while the prompt is still open, with
 * the box still holding what was typed, instead of a round trip that closes it
 * and answers with a red line.
 */
describe('scm: the name a prompt will accept', () => {
  it.each([
    'main',
    'feature/login',
    'release-2.5.0',
    'fix_the_thing',
    'a.b/c-d_e',
    'wip/2026/09',
  ])('accepts %s as a branch name', (name) => {
    expect(isValidBranchName(name)).toBe(true);
  });

  it.each([
    ['', 'nothing at all'],
    ['-start', 'a leading dash, which git reads as an option'],
    ['has space', 'a space'],
    ['two..dots', 'the range operator'],
    ['trailing/', 'a trailing separator'],
    ['work.lock', 'the suffix git reserves for its own lock files'],
    ['a~b', 'a revision operator'],
    ['a^b', 'a revision operator'],
    ['a:b', 'a refspec separator'],
    ['a?b', 'a glob character'],
    ['a*b', 'a glob character'],
    ['a[b', 'a glob character'],
    ['a\\b', 'a backslash'],
    ['a\u0001b', 'a control character'],
    ['//double', 'an empty path component'],
    ['.hidden', 'a component starting with a dot'],
    ['@', 'the shorthand for HEAD'],
  ])('refuses %s -- %s', (name) => {
    expect(isValidBranchName(name)).toBe(false);
  });

  it.each(['origin', 'up-stream', 'a.b_c', 'x0'])(
    'accepts %s as a remote name',
    (name) => {
      expect(isValidRemoteName(name)).toBe(true);
    },
  );

  it.each([
    ['', 'nothing at all'],
    ['-origin', 'a leading dash'],
    ['.origin', 'a leading dot'],
    ['has space', 'a space'],
    ['a/b', 'a separator, which git uses to address a remote branch'],
    ['x'.repeat(65), 'more than sixty-four characters'],
  ])('refuses %s as a remote name -- %s', (name) => {
    expect(isValidRemoteName(name)).toBe(false);
  });

  it.each([
    'https://github.com/owner/repo.git',
    'ssh://git@github.com:22/owner/repo.git',
    'git@github.com:owner/repo.git',
    'user.name@host.example:path/to/repo',
    // The copy says https or SSH; a bare repository on a shared drive is the
    // one a classroom actually reaches for, and git takes it.
    'file:///srv/git/repo.git',
    'file://D:/work/bare.git',
  ])('accepts %s as a remote URL', (url) => {
    expect(isValidRemoteUrl(url)).toBe(true);
  });

  it.each([
    ['', 'nothing at all'],
    ['github.com/owner/repo', 'no scheme and no user@host: form'],
    ['ftp://host/repo.git', 'a protocol git does not speak here'],
    // The server's allowlist is https, ssh and file; plain http is not on it.
    ['http://gitlab.local/team/repo.git', 'a scheme the server refuses'],
    ['https://', 'a scheme with no host'],
    ['git@host', 'a user@host with no path'],
    ['has space://host/repo', 'a space'],
    ['https://host/re po', 'a space in the path'],
    ['ext::sh -c payload', 'the transport that runs a command'],
  ])('refuses %s as a remote URL -- %s', (url) => {
    expect(isValidRemoteUrl(url)).toBe(false);
  });
});

describe('scm: how long ago a stash was made', () => {
  // Epoch SECONDS, which is what `%at` answers with.
  const now = 1_700_000_000_000;
  const at = (secondsAgo: number) => Math.floor(now / 1000) - secondsAgo;

  // `Intl` writes the words; what is pinned is which unit each span picks and
  // how it rounds -- the largest unit that fits, so a stash list is scannable
  // at a glance rather than a column of five-digit minute counts.
  it.each([
    [30, '30 seconds ago'],
    [90, '1 minute ago'],
    [60 * 90, '1 hour ago'],
    [60 * 60 * 30, 'yesterday'],
    [60 * 60 * 24 * 10, 'last week'],
    [60 * 60 * 24 * 70, '2 months ago'],
    [60 * 60 * 24 * 800, '2 years ago'],
    [0, 'now'],
  ])('reads %s seconds back as %s', (ago, said) => {
    expect(relativeTime(at(ago), 'en', now)).toBe(said);
  });

  it('says it in the reader\'s own language', () => {
    expect(relativeTime(at(120), 'zh-TW', now)).not.toBe(
      relativeTime(at(120), 'en', now),
    );
  });

  it('says nothing for a timestamp git could not read', () => {
    // `_timestamp` answers 0 for a reflog it could not parse, and "56 years
    // ago" is worse than no date at all.
    expect(relativeTime(0, 'en', now)).toBe('');
  });

  it('does not report the future for a clock a second out of step', () => {
    expect(relativeTime(at(-2), 'en', now)).toBe(relativeTime(at(0), 'en', now));
  });
});
