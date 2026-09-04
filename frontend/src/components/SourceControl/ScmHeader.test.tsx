import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ScmHeader } from './ScmHeader';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import type { FileKind, GitFile, GitStatus } from '../../api/git';

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

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let refresh: ReturnType<typeof vi.fn<GitActions['refresh']>>;
let setHideLayout: ReturnType<typeof vi.fn<GitActions['setHideLayout']>>;
let openIdentityForm: ReturnType<typeof vi.fn<GitActions['openIdentityForm']>>;
let dismissError: ReturnType<typeof vi.fn<GitActions['dismissError']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  refresh = vi.fn(async () => {});
  setHideLayout = vi.fn();
  openIdentityForm = vi.fn();
  dismissError = vi.fn();
  useGitStore.setState({
    repoState: 'ready',
    status: status(),
    refresh,
    setHideLayout,
    openIdentityForm,
    dismissError,
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

describe('ScmHeader: the title row', () => {
  it('shows the tab title and a refresh button that reads the status', () => {
    render(<ScmHeader />);
    expect(screen.getByText('Source Control')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it('offers exactly the three overflow actions', () => {
    render(<ScmHeader />);
    const menu = openMore();
    const rows = menu.querySelectorAll('[role="menuitem"], [role="menuitemcheckbox"]');
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveAccessibleName('Hide layout files');
    expect(rows[1]).toHaveAccessibleName('Commit identity...');
    expect(rows[2]).toHaveAccessibleName('Setup guide');
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
