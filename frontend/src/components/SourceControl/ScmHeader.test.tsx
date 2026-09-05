import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
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

  it('refreshes every OPEN section\'s list beside the status', async () => {
    // Half of what this tab shows is written somewhere else -- a stash pushed
    // at the command line, a commit made in an editor -- and a hidden tab
    // runs no poll at all. So Refresh read the status and left the Stashes
    // count and the branch list as they were, which is the one button whose
    // whole job is to make the panel true.
    useGitStore.setState({
      sections: { branches: true, remotes: false, stashes: true },
    });
    render(<ScmHeader />);
    refreshRefs.mockClear();

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    expect(refresh).toHaveBeenCalledTimes(1);
    expect(refreshRefs).toHaveBeenCalledWith('branches');
    expect(refreshRefs).toHaveBeenCalledWith('stashes');
    // Not the closed one: nothing is on screen for it to be wrong about.
    expect(refreshRefs).not.toHaveBeenCalledWith('remotes');
  });

  it('reads no list at all when every section is collapsed', () => {
    render(<ScmHeader />);
    refreshRefs.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(refresh).toHaveBeenCalledTimes(1);
    expect(refreshRefs).not.toHaveBeenCalled();
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

  it('asks again on the next status the server answers, when the read failed', async () => {
    // A failed refs read leaves `remotes` null, so a read that is never retried
    // hides Publish for as long as the panel stays open. Every status the poll
    // brings back is evidence the server is answering, and another chance.
    render(<ScmHeader />);
    await waitFor(() => expect(refreshRefs).toHaveBeenCalledTimes(1));
    act(() => {
      useGitStore.setState({ status: status() });
    });
    await waitFor(() => expect(refreshRefs).toHaveBeenCalledTimes(2));
  });

  it('gives up after three refused reads rather than asking once a poll forever', async () => {
    // Nobody pressed anything, so a read that keeps being refused would be
    // asked again on every poll, for the life of the tab, for an answer the
    // server has already refused three times.
    render(<ScmHeader />);
    for (let i = 0; i < 5; i += 1) {
      // Awaited, so each status is a separate poll answer: the read the last
      // one started has come back before the next one lands, which is what
      // the budget counts. Fifteen seconds apart, in the panel.
      // eslint-disable-next-line no-await-in-loop
      await act(async () => {
        useGitStore.setState({ status: status() });
      });
    }
    expect(refreshRefs).toHaveBeenCalledTimes(3);
    await act(async () => {
      useGitStore.setState({ status: status() });
    });
    expect(refreshRefs).toHaveBeenCalledTimes(3);
  });

  it('spends no attempt on a read that is still out', async () => {
    // Two statuses inside one read -- which is what StrictMode's double
    // effect and a slow server both look like -- must not spend two of the
    // three attempts on one question. Without the guard, a server that was
    // restarting while the panel opened had the whole budget gone before it
    // came back.
    let release = () => {};
    const outstanding = new Promise<void>((resolve) => {
      release = resolve;
    });
    refreshRefs = vi.fn(() => outstanding);
    useGitStore.setState({ refreshRefs });
    render(<ScmHeader />);

    act(() => {
      useGitStore.setState({ status: status() });
    });
    act(() => {
      useGitStore.setState({ status: status() });
    });
    expect(refreshRefs).toHaveBeenCalledTimes(1);

    // The read comes back, and only then is another one worth asking for.
    await act(async () => {
      release();
    });
    await act(async () => {
      useGitStore.setState({ status: status() });
    });
    expect(refreshRefs).toHaveBeenCalledTimes(2);
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

  it('opens the Remotes section rather than doing nothing at all', async () => {
    // `remotes` null hides the header button but NOT the menu row, which is
    // drawn enabled -- and a Publish that resolved no remote used to return in
    // silence, forever, on a server that keeps refusing the read. The section
    // holding that refusal is opened instead, which also reads the list again
    // on the way in.
    render(<ScmHeader />);
    openMore();
    fireEvent.click(menuRow('Publish Branch'));
    await waitFor(() => expect(setSectionOpen).toHaveBeenCalledWith('remotes', true));
    expect(publish).not.toHaveBeenCalled();
    // ...and the keyboard goes with it. A section that opens below the fold
    // with focus left on a Publish that did nothing is the same dead end
    // with one more thing on screen. The Remotes heading is drawn by another
    // component, so in here the move lands on the panel's own fallback --
    // which is the branch that proves it was attempted at all.
    await waitFor(() =>
      expect(document.activeElement)
        .toBe(document.querySelector('[data-scm-focus="title"]')));
  });

  it('does the same when the read answers that there is nowhere to publish to', async () => {
    // The row was pressed in the moment before the list landed, and the answer
    // is an empty one. Sending the publish anyway is the 400 the panel must
    // never ask for twice; the Remotes section says why, and carries the Add
    // Remote... that fixes it.
    // The panel's own read on mount leaves the list unread -- which is what
    // keeps the row enabled; the read this press makes is the one that
    // answers, and answers empty.
    let reads = 0;
    refreshRefs = vi.fn(async () => {
      reads += 1;
      if (reads > 1) useGitStore.setState({ remotes: [] });
    });
    useGitStore.setState({ refreshRefs });
    render(<ScmHeader />);
    openMore();
    fireEvent.click(menuRow('Publish Branch'));
    await waitFor(() => expect(setSectionOpen).toHaveBeenCalledWith('remotes', true));
    expect(publish).not.toHaveBeenCalled();
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
    // A detached HEAD has no branch, so no upstream either -- and the reason
    // the rows give is the detachment, which is checked before the missing
    // upstream and is the one a reader can act on.
    useGitStore.setState({
      status: status({ branch: null, detached: true }),
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

  it('drops Publish where the branch line is already offering the picker', () => {
    // With several remotes the row could only send a publish naming none of
    // them, which the server refuses with a 400 -- and the picker that answers
    // that refusal is on the branch line before the row is ever pressed.
    useGitStore.setState({ remotes: [remote('origin'), remote('backup')] });
    render(<ScmHeader />);
    openMore();
    expect(screen.queryByRole('menuitem', { name: 'Publish Branch' })).toBeNull();
    expect(refusal('Fetch')).toBeNull();
  });

  it('keeps the refused Publish row where no picker is drawn to replace it', () => {
    // The row is dropped because the branch line's picker answers the same
    // question -- and that line draws NOTHING on a detached HEAD or a branch
    // with no commits. Dropping the row there took away the one thing on
    // screen that said why publishing is not on offer.
    useGitStore.setState({
      status: status({ branch: null, detached: true }),
      remotes: [remote('origin'), remote('backup')],
    });
    const detached = render(<ScmHeader />);
    openMore();
    expect(refusal('Publish Branch')).toBe('Detached HEAD');
    detached.unmount();

    useGitStore.setState({ status: status({ unborn: true, head: null }) });
    render(<ScmHeader />);
    openMore();
    expect(refusal('Publish Branch')).toBe('No commits yet');
  });

  it('refuses every remote row while the network lane is busy', () => {
    // R11: one network operation at a time. The store answers a second one
    // with a toast, which is a working outcome and not a visible one -- the
    // row that cannot run says so where the pointer already is.
    useGitStore.setState({
      status: status({
        upstream: 'origin/main',
        ahead: 1,
        behind: 0,
        unstaged: [file('src/train.py', 'modified')],
      }),
      remotes: [remote('origin')],
      netOp: 'fetch',
    });
    render(<ScmHeader />);
    openMore();
    for (const name of ['Fetch', 'Pull', 'Push']) {
      expect(refusal(name)).toBe('Running fetch...');
    }
    // The local lane is untouched: a commit during a fetch is allowed on
    // both sides, and so is a stash.
    expect(refusal('Stash Changes...')).toBeNull();
  });

  it('refuses Sync and Publish on the branch line for the same reason', () => {
    // `aria-disabled` and NOT the native attribute -- the rule the rest of the
    // panel already states twice (CommitBox's button, every menu row). A
    // natively disabled button takes no focus and opens no tooltip, so the
    // icon-only Sync would lose its only label at the exact moment it has a
    // reason to give. The press is refused in the handler instead.
    useGitStore.setState({
      status: status({ upstream: 'origin/main', ahead: 1, behind: 0 }),
      remotes: [remote('origin')],
      netOp: 'pull',
    });
    const view = render(<ScmHeader />);
    const syncButton = screen.getByRole('button', { name: 'Sync (pull, then push)' });
    expect(syncButton).toHaveAttribute('aria-disabled', 'true');
    expect(syncButton).not.toBeDisabled();
    // The same sentence the refused menu rows give, in the one place this
    // control can give it.
    expect(syncButton.getAttribute('title')).toBe('Running pull...');
    fireEvent.click(syncButton);
    expect(sync).not.toHaveBeenCalled();
    view.unmount();

    useGitStore.setState({ status: status(), netOp: 'fetch' });
    render(<ScmHeader />);
    const publishButton = screen.getByRole('button', { name: 'Publish Branch' });
    expect(publishButton).toHaveAttribute('aria-disabled', 'true');
    expect(publishButton).not.toBeDisabled();
    expect(publishButton.getAttribute('title')).toBe('Running fetch...');
    fireEvent.click(publishButton);
    expect(publish).not.toHaveBeenCalled();
  });

  it('refuses the Publish picker the same way, trigger and all', () => {
    // Several remotes draw an ActionMenu in place of the button, and its
    // trigger is refused through the same attribute -- `close(true)` puts
    // focus back on it after a remote is chosen, which a natively disabled
    // trigger could not take.
    useGitStore.setState({
      remotes: [remote('origin'), remote('backup')],
      netOp: 'push',
    });
    render(<ScmHeader />);
    const picker = screen.getByRole('button', { name: 'Publish Branch' });
    expect(picker).toHaveAttribute('aria-disabled', 'true');
    expect(picker).not.toBeDisabled();
    expect(picker.getAttribute('title')).toBe('Running push...');
    fireEvent.click(picker);
    expect(screen.queryByRole('menu', { name: 'Publish Branch' })).toBeNull();
  });

  it('keeps focus on the control that was just pressed', () => {
    // `netOp` is set synchronously inside the store's `runOp`, before the
    // request is awaited, so the control the keyboard just pressed is refused
    // on the very next render. Natively disabled it would stop being focusable
    // there, and the browser would drop focus to <body> -- the next Tab
    // restarting at the top of the page, seconds into every sync.
    useGitStore.setState({
      status: status({ upstream: 'origin/main', ahead: 1, behind: 0 }),
      remotes: [remote('origin')],
    });
    render(<ScmHeader />);
    const syncButton = screen.getByRole('button', { name: 'Sync (pull, then push)' });
    act(() => syncButton.focus());
    fireEvent.click(syncButton);
    expect(sync).toHaveBeenCalledTimes(1);
    act(() => {
      useGitStore.setState({ netOp: 'sync' });
    });
    expect(syncButton).toHaveAttribute('aria-disabled', 'true');
    expect(document.activeElement).toBe(syncButton);
    // jsdom neither blurs a disabled element nor refuses `focus()` on one, so
    // the two lines above pass against the native attribute as well. What a
    // test CAN see is the attribute the browser's rule keys off: focus
    // survives here precisely because the refusal is not `disabled`. Measured
    // -- with both attributes set, everything but this line still passed.
    expect(syncButton).not.toBeDisabled();
    act(() => {
      syncButton.blur();
      syncButton.focus();
    });
    expect(document.activeElement).toBe(syncButton);
  });

  it('drops Publish on a branch that already has an upstream', () => {
    // The branch line hides Publish there and offers Sync instead; a row that
    // disagreed would push with --set-upstream on a branch that tracks one.
    useGitStore.setState({
      status: status({ upstream: 'origin/main', ahead: 0, behind: 0 }),
      remotes: [remote('origin')],
    });
    render(<ScmHeader />);
    openMore();
    expect(screen.queryByRole('menuitem', { name: 'Publish Branch' })).toBeNull();
    expect(refusal('Push')).toBeNull();
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

  it.each(['', '   '])(
    'sends no message at all when the box was left at %o',
    async (typed) => {
      // The prompt says the message is optional. An empty STRING is a
      // different thing from no message -- the route refuses it with a 400 --
      // and git writes its own subject for a stash nobody named.
      asked.mockResolvedValue(typed);
      render(<ScmHeader />);
      openMore();
      fireEvent.click(menuRow('Stash Changes...'));
      await waitFor(() => expect(stashPush).toHaveBeenCalledWith(null, true));
    },
  );

  it('trims a message that was typed', async () => {
    asked.mockResolvedValue('  before the demo  ');
    render(<ScmHeader />);
    openMore();
    fireEvent.click(menuRow('Stash Changes...'));
    await waitFor(() =>
      expect(stashPush).toHaveBeenCalledWith('before the demo', true),
    );
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
  it('shows a coded refusal and hides stderr behind Details', () => {
    useGitStore.setState({
      lastError: {
        code: 'nothing_to_commit',
        message: 'there is nothing to commit',
        // Null, and that is the store's doing: a code with a sentence of its
        // own drops the server's English hint (`scm.errorHint`), because two
        // lines in two languages said no more than the first one.
        hint: null,
        stderr: 'nothing added to commit',
      },
    });
    render(<ScmHeader />);
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toContain('Nothing to commit.');

    const details = screen.getByRole('button', { name: 'Details' });
    expect(details.getAttribute('aria-expanded')).toBe('false');
    // MOUNTED while it is closed, and hidden -- which is what makes the
    // `aria-controls` below name something that is really in the document.
    // Collapsed is the state this toggle is in every time an error line
    // appears, and an idref that resolves to nothing gives a reader offering
    // "go to the controlled element" nowhere to go.
    const stderr = screen.getByText('nothing added to commit');
    expect(stderr.hidden).toBe(true);
    expect(stderr.id).not.toBe('');
    expect(details.getAttribute('aria-controls')).toBe(stderr.id);

    fireEvent.click(details);
    expect(details.getAttribute('aria-expanded')).toBe('true');
    expect(screen.getByText('nothing added to commit').hidden).toBe(false);
  });

  it('draws the hint on its own line, under a sentence that is git\'s words', () => {
    // The only refusals that still carry one: `git_failed` and the rest of
    // the generic bucket, where the sentence itself is what the server said
    // and the hint is the same voice continuing.
    useGitStore.setState({
      lastError: {
        code: 'git_failed',
        message: 'git push failed (exit 1)',
        hint: 'the merge step failed',
        stderr: null,
      },
    });
    render(<ScmHeader />);
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toContain('git failed: git push failed (exit 1)');
    const hint = screen.getByText('the merge step failed');
    // Its own block, never run on to the end of the sentence above it.
    expect(hint.textContent).toBe('the merge step failed');
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
