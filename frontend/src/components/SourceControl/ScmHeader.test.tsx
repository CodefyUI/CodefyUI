import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ScmHeader } from './ScmHeader';
import { refSectionIds } from './RefSection';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import type { FileKind, GitFile, GitStatus, RemoteInfo } from '../../api/git';
import { prompt } from '../../utils/dialog';

// The stash message is asked for through the in-app prompt, which is a promise
// driven by a modal the header does not draw.
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const asked = vi.mocked(prompt);

/*
 * The tab's header: the title row, the branch line, the busy bar and the error
 * line. It is drawn in every repository state, so what is pinned here is which
 * of those four appear for a given store, and what each one says.
 *
 * The store is faked field by field through `setState` with fresh `vi.fn()`
 * actions -- never `vi.spyOn` on an action read off `getState()`, which keeps a
 * stale object and carries its call history into the next case.
 */

function file(path: string, kind: FileKind = 'modified'): GitFile {
  return { path, orig_path: null, kind, xy: 'M.', score: null };
}

function status(over: Partial<GitStatus> = {}): GitStatus {
  return {
    branch: 'main',
    detached: false,
    head: 'abc1234',
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

function remote(name: string): RemoteInfo {
  return {
    name,
    fetch_url: `https://example.invalid/${name}.git`,
    push_url: `https://example.invalid/${name}.git`,
  };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let refresh: ReturnType<typeof vi.fn<GitActions['refresh']>>;
let setHideLayout: ReturnType<typeof vi.fn<GitActions['setHideLayout']>>;
let openIdentityForm: ReturnType<typeof vi.fn<GitActions['openIdentityForm']>>;
let dismissError: ReturnType<typeof vi.fn<GitActions['dismissError']>>;
let refreshRefs: ReturnType<typeof vi.fn<GitActions['refreshRefs']>>;
let setSectionOpen: ReturnType<typeof vi.fn<GitActions['setSectionOpen']>>;
let doFetch: ReturnType<typeof vi.fn<GitActions['fetch']>>;
let pull: ReturnType<typeof vi.fn<GitActions['pull']>>;
let push: ReturnType<typeof vi.fn<GitActions['push']>>;
let sync: ReturnType<typeof vi.fn<GitActions['sync']>>;
let publish: ReturnType<typeof vi.fn<GitActions['publish']>>;
let stashPush: ReturnType<typeof vi.fn<GitActions['stashPush']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  asked.mockReset();
  asked.mockResolvedValue(null);
  refresh = vi.fn(async () => {});
  setHideLayout = vi.fn();
  openIdentityForm = vi.fn();
  dismissError = vi.fn();
  refreshRefs = vi.fn(async () => {});
  setSectionOpen = vi.fn();
  doFetch = vi.fn(async () => true);
  pull = vi.fn(async () => true);
  push = vi.fn(async () => true);
  sync = vi.fn(async () => true);
  publish = vi.fn(async () => true);
  stashPush = vi.fn(async () => true);
  useGitStore.setState({
    repoState: 'ready',
    status: status(),
    refresh,
    setHideLayout,
    openIdentityForm,
    dismissError,
    refreshRefs,
    setSectionOpen,
    fetch: doFetch,
    pull,
    push,
    sync,
    publish,
    stashPush,
  });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

/** Open the overflow menu and hand back its rows. */
function openMore() {
  fireEvent.click(screen.getByRole('button', { name: 'More actions' }));
  return screen.getByRole('menu', { name: 'More actions' });
}

/** One overflow row, by the label it shows. */
const menuRow = (name: string) => screen.getByRole('menuitem', { name });

describe('ScmHeader: the title row', () => {
  it('shows the tab title and a refresh button that reads the status', () => {
    render(<ScmHeader />);
    expect(screen.getByText('Source Control')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it('offers the five git actions above the three panel ones', () => {
    render(<ScmHeader />);
    const menu = openMore();
    const rows = menu.querySelectorAll('[role="menuitem"], [role="menuitemcheckbox"]');
    expect(rows).toHaveLength(8);
    // What git can be asked to do first, what the panel itself does after.
    expect([...rows].map((row) => row.getAttribute('aria-label') ?? row.textContent))
      .toEqual([
        'Fetch',
        'Pull',
        'Push',
        'Publish Branch',
        'Stash Changes...',
        'Hide layout files',
        'Commit identity...',
        'Setup guide',
      ]);
  });

  it('drops the git actions where there is no repository to ask', () => {
    useGitStore.setState({ repoState: 'not_repo', status: null });
    render(<ScmHeader />);
    const menu = openMore();
    const rows = menu.querySelectorAll('[role="menuitem"], [role="menuitemcheckbox"]');
    expect(rows).toHaveLength(3);
    expect(screen.queryByRole('menuitem', { name: 'Fetch' })).toBeNull();
  });

  it('Hide layout files is a checkbox that reports and flips the setting', () => {
    useGitStore.setState({ hideLayout: true });
    render(<ScmHeader />);
    openMore();
    const row = screen.getByRole('menuitemcheckbox', { name: /Hide layout files/ });
    expect(row.getAttribute('aria-checked')).toBe('true');
    fireEvent.click(row);
    expect(setHideLayout).toHaveBeenCalledWith(false);
  });

  it('appends the hidden count only while the filter is on', () => {
    useGitStore.setState({
      hideLayout: false,
      status: status({
        unstaged: [file('graphs/a.graph.json'), file('layout/a.layout.json')],
        untracked: [file('layout/b.layout.json', 'untracked')],
      }),
    });
    const view = render(<ScmHeader />);
    openMore();
    expect(screen.getByRole('menuitemcheckbox')).toHaveAccessibleName(
      'Hide layout files',
    );

    view.unmount();
    useGitStore.setState({ hideLayout: true });
    render(<ScmHeader />);
    openMore();
    expect(screen.getByRole('menuitemcheckbox')).toHaveAccessibleName(
      'Hide layout files (2 hidden)',
    );
  });

  it('opens the identity form from the menu', () => {
    render(<ScmHeader />);
    openMore();
    fireEvent.click(screen.getByRole('menuitem', { name: 'Commit identity...' }));
    expect(openIdentityForm).toHaveBeenCalledTimes(1);
  });

  it('cannot open the identity form where there is no repository to read', () => {
    // `GET /config` needs a repository; every other state answers it with a
    // refusal, so the row would open an empty form above an error line.
    useGitStore.setState({ repoState: 'not_repo', status: null });
    render(<ScmHeader />);
    openMore();
    const row = screen.getByRole('menuitem', { name: 'Commit identity...' });
    expect(row).toHaveAttribute('aria-disabled', 'true');
    fireEvent.click(row);
    expect(openIdentityForm).not.toHaveBeenCalled();
  });

  it('opens the setup guide in a new tab', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    render(<ScmHeader />);
    openMore();
    fireEvent.click(screen.getByRole('menuitem', { name: 'Setup guide' }));
    expect(open).toHaveBeenCalledWith(
      'https://docs.codefyui.com/usage/project-directories',
      '_blank',
      'noopener,noreferrer',
    );
  });
});

describe('ScmHeader: the branch line', () => {
  it('names the branch and says it is not published', () => {
    render(<ScmHeader />);
    expect(screen.getByText('Branch: main')).toBeTruthy();
    expect(screen.getByText('Not published')).toBeTruthy();
  });

  it('counts what is ahead and behind an upstream', () => {
    useGitStore.setState({
      status: status({ upstream: 'origin/main', ahead: 2, behind: 3 }),
    });
    render(<ScmHeader />);
    expect(screen.getByText('2 to push, 3 to pull')).toBeTruthy();
  });

  it('reports an upstream that no longer exists', () => {
    useGitStore.setState({
      status: status({ upstream: 'origin/gone', upstream_gone: true }),
    });
    render(<ScmHeader />);
    expect(screen.getByText('Upstream deleted')).toBeTruthy();
  });

  it('says HEAD is detached instead of naming a branch', () => {
    useGitStore.setState({ status: status({ branch: null, detached: true }) });
    render(<ScmHeader />);
    expect(screen.getByText('Detached HEAD')).toBeTruthy();
    expect(screen.queryByText(/^Branch:/)).toBeNull();
  });

  it('says a branch has no commits, and nothing about publishing it', () => {
    useGitStore.setState({ status: status({ unborn: true, head: null }) });
    render(<ScmHeader />);
    expect(screen.getByText('No commits yet')).toBeTruthy();
    expect(screen.queryByText('Not published')).toBeNull();
  });

  it('draws no branch line at all when there is no status to read', () => {
    useGitStore.setState({ repoState: 'not_repo', status: null });
    render(<ScmHeader />);
    expect(screen.queryByText(/^Branch:/)).toBeNull();
    expect(screen.getByText('Source Control')).toBeTruthy();
  });
});

describe('ScmHeader: the branch name is what opens the branch list', () => {
  const branchButton = () => screen.getByRole('button', { name: 'Branch: main' });

  it('says what it controls and whether that is open', () => {
    render(<ScmHeader />);
    expect(branchButton().getAttribute('aria-expanded')).toBe('false');
    // The list it names is drawn by another component, which is why the ids
    // are fixed per kind rather than coming from a `useId`.
    expect(branchButton().getAttribute('aria-controls')).toBe(
      refSectionIds('branches').listId,
    );
  });

  it('opens the Branches section through the store, which remembers it', () => {
    render(<ScmHeader />);
    fireEvent.click(branchButton());
    expect(setSectionOpen).toHaveBeenCalledWith('branches', true);
  });

  it('closes it again, so the state it reports is one it can undo', () => {
    useGitStore.setState({ sections: { branches: true, remotes: false, stashes: false } });
    render(<ScmHeader />);
    expect(branchButton().getAttribute('aria-expanded')).toBe('true');
    fireEvent.click(branchButton());
    expect(setSectionOpen).toHaveBeenCalledWith('branches', false);
  });
});

describe('ScmHeader: Sync, Publish and the remote picker', () => {
  const published = () => status({ upstream: 'origin/main', ahead: 1, behind: 0 });

  it('reads the remote list once, because null is not "no remotes"', async () => {
    render(<ScmHeader />);
    await waitFor(() => expect(refreshRefs).toHaveBeenCalledWith('remotes'));
    expect(refreshRefs).toHaveBeenCalledTimes(1);
  });

  it('offers Sync on a branch that has an upstream', () => {
    useGitStore.setState({ status: published(), remotes: [remote('origin')] });
    render(<ScmHeader />);
    fireEvent.click(screen.getByRole('button', { name: 'Sync (pull, then push)' }));
    expect(sync).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: 'Publish Branch' })).toBeNull();
  });

  it('publishes straight to the only remote, and says it once', async () => {
    useGitStore.setState({ remotes: [remote('origin')] });
    render(<ScmHeader />);
    // "Not published" and "Publish Branch" are the same fact; the button is
    // the half that can be acted on.
    expect(screen.queryByText('Not published')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Publish Branch' }));
    await waitFor(() => expect(publish).toHaveBeenCalledWith('origin'));
  });

  it('asks which remote when there are several', () => {
    useGitStore.setState({ remotes: [remote('origin'), remote('backup')] });
    render(<ScmHeader />);
    fireEvent.click(screen.getByRole('button', { name: 'Publish Branch' }));
    expect(screen.getByRole('menu', { name: 'Publish Branch' })).toBeTruthy();
    fireEvent.click(screen.getByRole('menuitem', { name: 'backup' }));
    expect(publish).toHaveBeenCalledWith('backup');
  });

  it('shows nothing to publish to on a repository with no remote', () => {
    useGitStore.setState({ remotes: [] });
    render(<ScmHeader />);
    expect(screen.queryByRole('button', { name: 'Publish Branch' })).toBeNull();
    // The line says the state instead: adding a remote lives under Remotes.
    expect(screen.getByText('Not published')).toBeTruthy();
  });

  it('waits for the list rather than guessing while it is unread', () => {
    render(<ScmHeader />);
    expect(screen.queryByRole('button', { name: 'Publish Branch' })).toBeNull();
    expect(screen.getByText('Not published')).toBeTruthy();
  });

  it('offers neither on a detached HEAD or a branch with no commits', () => {
    useGitStore.setState({
      status: status({ branch: null, detached: true, upstream: 'origin/main' }),
      remotes: [remote('origin')],
    });
    const view = render(<ScmHeader />);
    expect(screen.queryByRole('button', { name: 'Sync (pull, then push)' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Publish Branch' })).toBeNull();

    view.unmount();
    useGitStore.setState({ status: status({ unborn: true, head: null }) });
    render(<ScmHeader />);
    expect(screen.queryByRole('button', { name: 'Publish Branch' })).toBeNull();
  });
});

describe('ScmHeader: what the git actions are refused for', () => {
  /** The reason a row gives for being refused, or null when it can be pressed. */
  function refusal(name: string): string | null {
    const row = menuRow(name);
    return row.getAttribute('aria-disabled') === 'true'
      ? (row.getAttribute('aria-describedby') === null
        ? ''
        : document.getElementById(row.getAttribute('aria-describedby') as string)
          ?.textContent ?? '')
      : null;
  }

  it('refuses all four remote actions with one reason on a repository with no remote', () => {
    useGitStore.setState({ remotes: [] });
    render(<ScmHeader />);
    openMore();
    for (const name of ['Fetch', 'Pull', 'Push', 'Publish Branch']) {
      expect(refusal(name)).toBe('No remote yet.');
    }
  });

  it('refuses Pull and Push on a branch that is not published, leaving Publish', () => {
    useGitStore.setState({ remotes: [remote('origin')] });
    render(<ScmHeader />);
    openMore();
    expect(refusal('Fetch')).toBeNull();
    expect(refusal('Pull')).toBe('Not published');
    expect(refusal('Push')).toBe('Not published');
    expect(refusal('Publish Branch')).toBeNull();
  });

  it('refuses everything but Fetch on a detached HEAD', () => {
    useGitStore.setState({
      status: status({ branch: null, detached: true, upstream: 'origin/main' }),
      remotes: [remote('origin')],
    });
    render(<ScmHeader />);
    openMore();
    expect(refusal('Fetch')).toBeNull();
    for (const name of ['Pull', 'Push', 'Publish Branch']) {
      expect(refusal(name)).toBe('Detached HEAD');
    }
  });

  it('refuses every remote action on a branch with no commits', () => {
    useGitStore.setState({ status: status({ unborn: true, head: null }), remotes: [] });
    render(<ScmHeader />);
    openMore();
    for (const name of ['Fetch', 'Pull', 'Push', 'Publish Branch']) {
      expect(refusal(name)).toBe('No commits yet');
    }
  });

  it('refuses Stash on a clean tree, and while a merge is in progress', () => {
    const view = render(<ScmHeader />);
    openMore();
    expect(refusal('Stash Changes...')).toBe('No changes');
    view.unmount();

    useGitStore.setState({
      status: status({
        merge_in_progress: true,
        conflicted: [file('src/train.py', 'conflict')],
      }),
    });
    render(<ScmHeader />);
    openMore();
    expect(refusal('Stash Changes...')).toBe(
      'Merge in progress: resolve each file, then commit.',
    );
  });

  it('leaves a refused row focusable, so the reason can be reached', () => {
    useGitStore.setState({ remotes: [] });
    render(<ScmHeader />);
    openMore();
    expect(menuRow('Fetch')).not.toBeDisabled();
    fireEvent.click(menuRow('Fetch'));
    expect(doFetch).not.toHaveBeenCalled();
  });
});

describe('ScmHeader: running a git action from the menu', () => {
  beforeEach(() => {
    useGitStore.setState({
      status: status({
        upstream: 'origin/main',
        ahead: 1,
        behind: 1,
        unstaged: [file('src/train.py')],
      }),
      remotes: [remote('origin')],
    });
  });

  it('fetches, pulls and pushes', () => {
    render(<ScmHeader />);
    openMore();
    fireEvent.click(menuRow('Fetch'));
    expect(doFetch).toHaveBeenCalledTimes(1);

    openMore();
    fireEvent.click(menuRow('Pull'));
    // Fast-forward: the merge is what the `diverged` refusal offers next.
    expect(pull).toHaveBeenCalledWith('ff-only');

    openMore();
    fireEvent.click(menuRow('Push'));
    expect(push).toHaveBeenCalledTimes(1);
  });

  it('stashes under the message the prompt asked for, untracked files included', async () => {
    asked.mockResolvedValue('before the demo');
    render(<ScmHeader />);
    openMore();
    fireEvent.click(menuRow('Stash Changes...'));
    await waitFor(() =>
      expect(stashPush).toHaveBeenCalledWith('before the demo', true),
    );
    expect(asked).toHaveBeenCalledWith({ title: 'Stash message (optional)' });
  });

  it('stashes nothing when the prompt was dismissed', async () => {
    render(<ScmHeader />);
    openMore();
    fireEvent.click(menuRow('Stash Changes...'));
    await waitFor(() => expect(asked).toHaveBeenCalledTimes(1));
    expect(stashPush).not.toHaveBeenCalled();
  });
});

describe('ScmHeader: the follow-up beside a refusal', () => {
  const diverged = {
    code: 'diverged' as const,
    message: 'diverged',
    hint: null,
    stderr: null,
    op: 'pull' as const,
  };
  const noUpstream = {
    code: 'no_upstream' as const,
    message: 'no upstream',
    hint: null,
    stderr: null,
    op: 'push' as const,
  };

  it('offers the merge retry after a diverged pull', () => {
    useGitStore.setState({ lastError: diverged });
    render(<ScmHeader />);
    fireEvent.click(screen.getByRole('button', { name: 'Merge remote changes' }));
    expect(pull).toHaveBeenCalledWith('merge');
  });

  it('keeps the follow-up out of the alert it belongs to', () => {
    // An alert re-announces its whole subtree whenever it changes, so a button
    // inside one is read out again on every repaint of the panel behind it.
    useGitStore.setState({ lastError: diverged });
    render(<ScmHeader />);
    const alert = screen.getByRole('alert');
    const button = screen.getByRole('button', { name: 'Merge remote changes' });
    expect(alert.contains(button)).toBe(false);
    expect(alert.textContent).toContain('Local and remote branches have diverged.');
  });

  it('offers Publish after an unpublished-branch refusal', async () => {
    useGitStore.setState({ lastError: noUpstream, remotes: [remote('origin')] });
    render(<ScmHeader />);
    fireEvent.click(screen.getByRole('button', { name: 'Publish Branch' }));
    await waitFor(() => expect(publish).toHaveBeenCalledWith('origin'));
  });

  it('is the only Publish on screen while it is showing', () => {
    useGitStore.setState({ lastError: noUpstream, remotes: [remote('origin')] });
    render(<ScmHeader />);
    expect(screen.getAllByRole('button', { name: 'Publish Branch' })).toHaveLength(1);
  });

  it('opens the same remote picker when there are several', () => {
    useGitStore.setState({
      lastError: noUpstream,
      remotes: [remote('origin'), remote('backup')],
    });
    render(<ScmHeader />);
    fireEvent.click(screen.getByRole('button', { name: 'Publish Branch' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'origin' }));
    expect(publish).toHaveBeenCalledWith('origin');
  });

  it('reads the remote list first when the refusal arrived before it did', async () => {
    // The button must never send back the identical remote-less publish the
    // server just refused, so it resolves the list before it decides.
    refreshRefs = vi.fn(async () => {
      useGitStore.setState({ remotes: [remote('origin')] });
    });
    useGitStore.setState({ lastError: noUpstream, remotes: null, refreshRefs });
    render(<ScmHeader />);
    fireEvent.click(screen.getByRole('button', { name: 'Publish Branch' }));
    await waitFor(() => expect(publish).toHaveBeenCalledWith('origin'));
  });

  it('offers nothing to publish to on a repository with no remote', () => {
    useGitStore.setState({ lastError: noUpstream, remotes: [] });
    render(<ScmHeader />);
    expect(screen.queryByRole('button', { name: 'Publish Branch' })).toBeNull();
    expect(screen.getByRole('alert').textContent).toContain(
      'This branch is not published yet.',
    );
  });

  it('offers nothing for a refusal with no way out of its own', () => {
    useGitStore.setState({
      lastError: { code: 'auth_required', message: 'no', hint: null, stderr: null },
    });
    render(<ScmHeader />);
    expect(screen.queryByRole('button', { name: 'Merge remote changes' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Publish Branch' })).toBeNull();
  });
});

describe('ScmHeader: the busy bar', () => {
  it('names the running operation', () => {
    useGitStore.setState({ busyOp: 'commit' });
    render(<ScmHeader />);
    expect(screen.getByRole('progressbar', { name: 'Running commit...' })).toBeTruthy();
  });

  it('translates the operation name rather than showing the wire word', () => {
    useI18n.setState({ locale: 'zh-TW' });
    useGitStore.setState({ busyOp: 'discard' });
    render(<ScmHeader />);
    expect(screen.getByRole('progressbar', { name: '正在執行 捨棄...' })).toBeTruthy();
  });

  it('names a network operation too, which holds the other lane', () => {
    // Local writes and network operations run in two independent lanes; a bar
    // that watched only the local one left a long fetch looking like nothing
    // was happening.
    useGitStore.setState({ netOp: 'fetch' });
    render(<ScmHeader />);
    expect(screen.getByRole('progressbar', { name: 'Running fetch...' })).toBeTruthy();
  });

  it('names the local operation when both lanes are busy', () => {
    useGitStore.setState({ busyOp: 'stage', netOp: 'pull' });
    render(<ScmHeader />);
    expect(screen.getByRole('progressbar', { name: 'Running stage...' })).toBeTruthy();
  });

  it('has no bar while nothing is running', () => {
    render(<ScmHeader />);
    expect(screen.queryByRole('progressbar')).toBeNull();
  });
});

describe('ScmHeader: the error line', () => {
  it('shows a coded refusal, its hint, and hides stderr behind Details', () => {
    useGitStore.setState({
      lastError: {
        code: 'nothing_to_commit',
        message: 'there is nothing to commit',
        hint: 'stage something first',
        stderr: 'nothing added to commit',
      },
    });
    render(<ScmHeader />);
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toContain('Nothing to commit.');
    expect(alert.textContent).toContain('stage something first');
    expect(screen.queryByText('nothing added to commit')).toBeNull();

    const details = screen.getByRole('button', { name: 'Details' });
    expect(details.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(details);
    const opened = screen.getByText('nothing added to commit');
    expect(details.getAttribute('aria-expanded')).toBe('true');
    // The toggle names what it opens, so the reader it was just announced to
    // can go straight there rather than hunting for what changed.
    expect(opened.id).not.toBe('');
    expect(details.getAttribute('aria-controls')).toBe(opened.id);
  });

  it('offers no Details when git said nothing on stderr', () => {
    useGitStore.setState({
      lastError: { code: 'invalid_value', message: 'no', hint: null, stderr: null },
    });
    render(<ScmHeader />);
    expect(screen.getByRole('alert').textContent).toContain('Invalid value.');
    expect(screen.queryByRole('button', { name: 'Details' })).toBeNull();
  });

  it('dismisses through the store', () => {
    useGitStore.setState({
      lastError: { code: 'git_failed', message: 'boom', hint: null, stderr: null },
    });
    render(<ScmHeader />);
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(dismissError).toHaveBeenCalledTimes(1);
  });

  it('shows a timeout sentence exactly as the store wrote it', () => {
    useGitStore.setState({
      lastError: {
        code: 'timeout',
        message: 'git did not finish within 30s.',
        hint: null,
        stderr: null,
      },
    });
    render(<ScmHeader />);
    expect(screen.getByRole('alert').textContent).toContain(
      'git did not finish within 30s.',
    );
  });

  it('falls back to git\'s own words for a code with no sentence', () => {
    useGitStore.setState({
      lastError: {
        code: 'git_service_unavailable',
        message: 'source control is not available on this server',
        hint: null,
        stderr: null,
      },
    });
    render(<ScmHeader />);
    expect(screen.getByRole('alert').textContent).toContain(
      'git failed: source control is not available on this server',
    );
  });

  it('reports a status that could not be read, even from a working panel', () => {
    // The RULING: a server that stops answering after a good first read must
    // not leave a silent stale panel. `repoState` is still `ready` here.
    useGitStore.setState({ repoState: 'ready', status: status(), loadError: 'Failed to fetch' });
    render(<ScmHeader />);
    expect(screen.getByRole('alert').textContent).toContain(
      'Could not read repository status: Failed to fetch',
    );
    // Nothing to dismiss: `loadError` clears on the next successful read.
    expect(screen.queryByRole('button', { name: 'Dismiss' })).toBeNull();
  });

  it('says it once when a write and the read behind it failed together', () => {
    // A stopped server refuses both with the same words, and the panel used to
    // print both of them, one under the other. The operation's sentence is the
    // one carrying the hint and the Details toggle, so it is the one kept.
    useGitStore.setState({
      loadError: 'Failed to fetch',
      lastError: {
        code: 'unknown',
        message: 'Failed to fetch',
        hint: null,
        stderr: null,
      },
    });
    const view = render(<ScmHeader />);
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toContain('git failed: Failed to fetch');
    expect(alert.textContent).not.toContain('Could not read repository status');
    expect(screen.getByRole('button', { name: 'Dismiss' })).toBeTruthy();

    // Nothing is suppressed for good: the read's own line is what Dismiss
    // leaves behind, and what the next failed poll shows again.
    view.unmount();
    useGitStore.setState({ lastError: null });
    render(<ScmHeader />);
    expect(screen.getByRole('alert').textContent).toContain(
      'Could not read repository status: Failed to fetch',
    );
  });

  it('has no error line when there is nothing wrong', () => {
    render(<ScmHeader />);
    expect(screen.queryByRole('alert')).toBeNull();
  });
});
