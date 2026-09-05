import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { HistorySection } from './HistorySection';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { useUIStore } from '../../store/uiStore';
import type { GitCommit, GitCommitFile } from '../../api/git';

/*
 * The commit list. Two things here are not taste. The file list is read from
 * the row's own CLICK and never from an effect -- the store caches it only
 * once the read answers, and StrictMode double-invokes effects, so an effect
 * would cost two requests per expand. And the page is READ, never counted:
 * `hasMore` is the server's answer, so Load more is on screen exactly while
 * there is another page.
 */

/** Fixed, so a relative date is a fixed string; `authoredAt` is epoch seconds. */
const NOW = 1_700_000_000_000;
const secondsAgo = (n: number) => Math.floor(NOW / 1000) - n;

function commit(over: Partial<GitCommit> = {}): GitCommit {
  return {
    sha: 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678',
    short: 'a1b2c3d',
    parents: ['0000000000000000000000000000000000000001'],
    authorName: 'Ada',
    authorEmail: 'ada@example.com',
    authoredAt: secondsAgo(2 * 60 * 60),
    refs: [],
    subject: 'Teach the model to count',
    body: '',
    ...over,
  };
}

function commitFile(over: Partial<GitCommitFile> = {}): GitCommitFile {
  return { path: 'graphs/cnn.graph.json', origPath: null, kind: 'modified', ...over };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;
type UIActions = ReturnType<typeof useUIStore.getState>;

let setSectionOpen: ReturnType<typeof vi.fn<GitActions['setSectionOpen']>>;
let loadMoreLog: ReturnType<typeof vi.fn<GitActions['loadMoreLog']>>;
let loadCommitFiles: ReturnType<typeof vi.fn<GitActions['loadCommitFiles']>>;
let announce: ReturnType<typeof vi.fn<GitActions['announce']>>;
let openGitDiff: ReturnType<typeof vi.fn<UIActions['openGitDiff']>>;
let writeText: ReturnType<typeof vi.fn<(text: string) => Promise<void>>>;

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  setSectionOpen = vi.fn();
  loadMoreLog = vi.fn(async () => {});
  loadCommitFiles = vi.fn(async () => {});
  announce = vi.fn();
  openGitDiff = vi.fn();
  // jsdom has no clipboard at all, which is also what a page served over
  // plain http gets -- so the component has to reach it through a promise
  // chain rather than call it directly. See the refusal case below.
  writeText = vi.fn(async () => {});
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText },
    configurable: true,
  });
  useGitStore.setState({
    repoState: 'ready',
    log: { commits: [commit()], hasMore: false, unborn: false, loading: false },
    sections: { branches: false, remotes: false, stashes: false, history: true },
    setSectionOpen,
    loadMoreLog,
    loadCommitFiles,
    announce,
  });
  // Installed through `setState` with a fresh mock, never a spy taken off a
  // `getState()` snapshot.
  useUIStore.setState({ gitDiff: null, openGitDiff });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const section = () => screen.getByRole('region', { name: 'History' });

describe('HistorySection: the rows', () => {
  it('shows the short id, the subject, when it was made and by whom', () => {
    render(<HistorySection />);
    expect(within(section()).getByText('a1b2c3d')).toBeTruthy();
    expect(within(section()).getByText('Teach the model to count')).toBeTruthy();
    expect(within(section()).getByText('2 hours ago Ada')).toBeTruthy();
  });

  it('keeps the id in full where a reader can reach it', () => {
    // Seven characters is what a row has room for and what the next command
    // line takes; the forty are what a bug report needs.
    render(<HistorySection />);
    expect(within(section()).getByText('a1b2c3d').getAttribute('title')).toBe(
      'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678',
    );
  });

  it('answers a pointer with the subject and the meta the narrow row drops', () => {
    render(<HistorySection />);
    const row = within(section()).getByText('a1b2c3d').closest('li');
    expect(row?.getAttribute('title')).toBe('Teach the model to count\n2 hours ago Ada');
  });

  it('counts the commits it is showing', () => {
    useGitStore.setState({
      log: {
        commits: [commit(), commit({ sha: 'b'.repeat(40), short: 'bbbbbbb' })],
        hasMore: false,
        unborn: false,
        loading: false,
      },
    });
    render(<HistorySection />);
    expect(within(section()).getByText('2')).toBeTruthy();
  });

  it('counts nothing until the history has been read', () => {
    // The section is closed on a fresh profile and nothing reads the log until
    // it is opened, so a `?? 0` here would print "History 0" beside a
    // repository with two hundred commits.
    useGitStore.setState({
      log: { commits: [], hasMore: false, unborn: false, loading: false },
    });
    render(<HistorySection />);
    expect(within(section()).queryByText('0')).toBeNull();
  });

  it('draws a header and nothing else on a branch with no commits', () => {
    // No count -- zero is a number about a repository that has none of the
    // thing being counted -- and no sentence: the header line already says
    // "No commits yet" about the same branch.
    useGitStore.setState({
      log: { commits: [], hasMore: true, unborn: true, loading: false },
    });
    render(<HistorySection />);
    expect(section()).toBeTruthy();
    expect(within(section()).queryByText('0')).toBeNull();
    expect(within(section()).queryAllByRole('listitem')).toHaveLength(0);
    expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull();
  });

  it('opens and closes through the store, which is what remembers it', () => {
    render(<HistorySection />);
    fireEvent.click(screen.getByRole('button', { name: 'History' }));
    expect(setSectionOpen).toHaveBeenCalledWith('history', false);
  });
});

describe('HistorySection: expanding a commit', () => {
  const rowToggle = () =>
    within(section()).getByRole('button', { name: /^a1b2c3d/ });

  it('reads the file list from the press, and says the row is expanded', async () => {
    render(<HistorySection />);
    const toggle = rowToggle();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');

    fireEvent.click(toggle);

    await waitFor(() =>
      expect(loadCommitFiles).toHaveBeenCalledWith(
        'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678',
      ),
    );
    expect(loadCommitFiles).toHaveBeenCalledTimes(1);
    expect(rowToggle().getAttribute('aria-expanded')).toBe('true');
  });

  it('keeps the file region in the document, hidden, while the row is closed', () => {
    // The row's `aria-controls` has to name an element that is really there:
    // a reader offered "go to the controlled element" would otherwise have
    // nowhere to go. The stylesheet's `.refSublist[hidden]` is the other half
    // -- an author `display` beats the user agent's own `[hidden]` rule.
    render(<HistorySection />);
    const controls = rowToggle().getAttribute('aria-controls');
    expect(controls).not.toBeNull();
    const region = document.getElementById(controls as string);
    expect(region).not.toBeNull();
    expect(region?.hasAttribute('hidden')).toBe(true);

    fireEvent.click(rowToggle());
    expect(document.getElementById(controls as string)?.hasAttribute('hidden')).toBe(false);
  });

  it('reads nothing at all until a row is pressed', () => {
    // The whole reason the call is in the click handler: an effect would run
    // twice per expand under StrictMode, and once per row on mount.
    render(<HistorySection />);
    expect(loadCommitFiles).not.toHaveBeenCalled();
  });

  it('draws the files it was given, counted, once they land', () => {
    useGitStore.setState({
      commitFiles: {
        'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678': [
          commitFile(),
          commitFile({ path: 'src/train.py', kind: 'added' }),
        ],
      },
    });
    render(<HistorySection />);
    fireEvent.click(rowToggle());
    expect(within(section()).getByText('2 file(s)')).toBeTruthy();
    expect(
      within(section()).getByRole('button', { name: 'Open changes cnn.graph.json' }),
    ).toBeTruthy();
  });

  it('opens a file against the commit it was changed in', () => {
    useGitStore.setState({
      commitFiles: { 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678': [commitFile()] },
    });
    render(<HistorySection />);
    fireEvent.click(rowToggle());
    fireEvent.click(
      within(section()).getByRole('button', { name: 'Open changes cnn.graph.json' }),
    );
    expect(openGitDiff).toHaveBeenCalledWith({
      path: 'graphs/cnn.graph.json',
      scope: 'commit',
      sha: 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678',
    });
  });

  it('shows one commit at a time', () => {
    // A page is thirty rows deep, and a second file list under the first one
    // pushes both off screen.
    const other = commit({ sha: 'b'.repeat(40), short: 'bbbbbbb', subject: 'Second' });
    useGitStore.setState({
      log: { commits: [commit(), other], hasMore: false, unborn: false, loading: false },
    });
    render(<HistorySection />);
    fireEvent.click(rowToggle());
    fireEvent.click(within(section()).getByRole('button', { name: /^bbbbbbb/ }));

    expect(rowToggle().getAttribute('aria-expanded')).toBe('false');
    expect(
      within(section()).getByRole('button', { name: /^bbbbbbb/ }).getAttribute('aria-expanded'),
    ).toBe('true');
  });

  it('closes a row that is pressed again, and reads nothing a second time', () => {
    render(<HistorySection />);
    fireEvent.click(rowToggle());
    fireEvent.click(rowToggle());
    expect(rowToggle().getAttribute('aria-expanded')).toBe('false');
    expect(loadCommitFiles).toHaveBeenCalledTimes(1);
  });
});

describe('HistorySection: the next page', () => {
  it('is absent while the server says this is the whole history', () => {
    // `hasMore` is the server's own answer -- it read one row more than it
    // returned -- and never a guess from the page size.
    render(<HistorySection />);
    expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull();
  });

  it('asks the store for the page after the one on screen', () => {
    useGitStore.setState({
      log: { commits: [commit()], hasMore: true, unborn: false, loading: false },
    });
    render(<HistorySection />);
    fireEvent.click(screen.getByRole('button', { name: 'Load more' }));
    expect(loadMoreLog).toHaveBeenCalledTimes(1);
  });

  it('refuses a second press while the page is still out', () => {
    // `aria-disabled` and a guarded handler, never the native attribute: the
    // press itself is what sets `loading`, so the button the keyboard is on
    // would stop being focusable on the very next render.
    useGitStore.setState({
      log: { commits: [commit()], hasMore: true, unborn: false, loading: true },
    });
    render(<HistorySection />);
    const more = screen.getByRole('button', { name: 'Load more' });
    expect(more.getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(more);
    expect(loadMoreLog).not.toHaveBeenCalled();
  });
});

describe('HistorySection: what the section says when a read fails', () => {
  it('reports it inside the section, naming the list it is about', () => {
    // Not on the header's error line: that one belongs to the operation the
    // user pressed a button for, and a history read is the panel's own.
    useGitStore.setState({
      historyError: {
        code: 'git_failed',
        message: 'Failed to fetch',
        hint: null,
        stderr: null,
        op: null,
      },
    });
    render(<HistorySection />);
    expect(
      within(section()).getByText('Could not read History: Failed to fetch'),
    ).toBeTruthy();
  });

  it('keeps the page that is already on screen', () => {
    useGitStore.setState({
      historyError: {
        code: 'git_failed',
        message: 'Failed to fetch',
        hint: null,
        stderr: null,
        op: null,
      },
    });
    render(<HistorySection />);
    expect(within(section()).getByText('Teach the model to count')).toBeTruthy();
  });
});

describe('HistorySection: copying the commit id', () => {
  it('copies the whole id and says so', async () => {
    render(<HistorySection />);
    fireEvent.click(
      within(section()).getByRole('button', { name: 'Copy commit id a1b2c3d' }),
    );
    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith('a1b2c3d4e5f60718293a4b5c6d7e8f9012345678'),
    );
    await waitFor(() => expect(announce).toHaveBeenCalledWith('Copied to clipboard.'));
  });

  it('says so when there is no clipboard to copy to', async () => {
    // `navigator.clipboard` is absent outside a secure context -- a LAN
    // address over plain http, for one -- and the row must not die of it.
    Object.defineProperty(navigator, 'clipboard', {
      value: undefined,
      configurable: true,
    });
    render(<HistorySection />);
    fireEvent.click(
      within(section()).getByRole('button', { name: 'Copy commit id a1b2c3d' }),
    );
    await waitFor(() =>
      expect(announce).toHaveBeenCalledWith(
        'Could not copy. Select the text and copy it by hand.',
      ),
    );
  });

  it('repeats the same action inside one compact menu, for a 180px panel', () => {
    render(<HistorySection />);
    fireEvent.click(screen.getByRole('button', { name: 'More actions a1b2c3d' }));
    const menu = screen.getByRole('menu', { name: 'More actions a1b2c3d' });
    expect(within(menu).getByRole('menuitem', { name: 'Copy commit id' })).toBeTruthy();
  });
});
