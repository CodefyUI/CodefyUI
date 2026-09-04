import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { ChangeGroup } from './ChangeGroup';
import { SCM_FOCUS } from './ScmHeader';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { confirm } from '../../utils/dialog';
import type { FileKind, GitFile, GitStatus } from '../../api/git';

/*
 * One group: its heading, its count, its collapse state and the actions that
 * act on every row at once. The whole-tree ones are the ones with teeth --
 * "Discard All" throws away files the current filter is not even showing, so
 * what the question it asks reports is a correctness property, not copy.
 */
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const askedConfirm = vi.mocked(confirm);

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

let stage: ReturnType<typeof vi.fn<GitActions['stage']>>;
let unstage: ReturnType<typeof vi.fn<GitActions['unstage']>>;
let discard: ReturnType<typeof vi.fn<GitActions['discard']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  askedConfirm.mockReset();
  askedConfirm.mockResolvedValue(true);
  stage = vi.fn(async () => true);
  unstage = vi.fn(async () => true);
  discard = vi.fn(async () => true);
  useGitStore.setState({ repoState: 'ready', status: status(), stage, unstage, discard });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

describe('ChangeGroup: structure', () => {
  it('is a section named by its own heading, holding one list', () => {
    render(<ChangeGroup kind="changes" files={[file('a.py'), file('b.py')]} />);
    const section = screen.getByRole('region', { name: 'Changes' });
    expect(within(section).getAllByRole('listitem')).toHaveLength(2);
  });

  it('counts the rows it is showing', () => {
    render(<ChangeGroup kind="staged" files={[file('a.py')]} />);
    const header = screen.getByRole('button', { name: 'Staged Changes' });
    expect(header.parentElement?.textContent).toContain('1');
  });

  it('collapses and reopens from its heading', () => {
    render(<ChangeGroup kind="changes" files={[file('a.py')]} />);
    const header = screen.getByRole('button', { name: 'Changes' });
    expect(header.getAttribute('aria-expanded')).toBe('true');
    fireEvent.click(header);
    expect(header.getAttribute('aria-expanded')).toBe('false');
    expect(screen.queryAllByRole('listitem')).toHaveLength(0);
    fireEvent.click(header);
    expect(screen.getAllByRole('listitem')).toHaveLength(1);
  });

  it('names each group from the locale table', () => {
    const view = render(<ChangeGroup kind="merge" files={[file('a.py', 'conflict')]} />);
    expect(screen.getByRole('button', { name: 'Merge Changes' })).toBeTruthy();
    view.unmount();
    render(<ChangeGroup kind="staged" files={[]} />);
    expect(screen.getByRole('button', { name: 'Staged Changes' })).toBeTruthy();
  });
});

describe('ChangeGroup: the group actions', () => {
  it('stages the whole tree from the Changes group', async () => {
    render(<ChangeGroup kind="changes" files={[file('a.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Stage All' }));
    await waitFor(() => expect(stage).toHaveBeenCalledWith('all'));
  });

  it('unstages the whole index from the Staged group', async () => {
    render(<ChangeGroup kind="staged" files={[file('a.py')]} />);
    expect(screen.queryByRole('button', { name: 'Stage All' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Unstage All' }));
    await waitFor(() => expect(unstage).toHaveBeenCalledWith('all'));
  });

  it('leaves the Merge group without a single group action', () => {
    render(<ChangeGroup kind="merge" files={[file('a.py', 'conflict')]} />);
    expect(screen.queryByRole('button', { name: 'Stage All' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Unstage All' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Discard All Changes' })).toBeNull();
  });

  it('asks before a whole-tree discard, counting what is NOT on screen too', async () => {
    // The filter is hiding one layout file, and `discard("all")` will delete
    // it anyway: the question has to report the status, not the view.
    useGitStore.setState({
      status: status({
        unstaged: [file('a.py'), file('b.py'), file('layout/a.layout.json')],
        untracked: [file('c.py', 'untracked'), file('d.py', 'untracked')],
      }),
    });
    render(
      <ChangeGroup
        kind="changes"
        files={[file('a.py'), file('b.py'), file('c.py', 'untracked'), file('d.py', 'untracked')]}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Discard All Changes' }));
    await waitFor(() => expect(discard).toHaveBeenCalledWith('all'));
    expect(askedConfirm).toHaveBeenCalledWith({
      title:
        'Discard 3 changed file(s) and delete 2 untracked file(s)? This cannot be undone.',
      confirmText: 'Discard',
      variant: 'danger',
    });
  });

  it('discards nothing when the question is answered no', async () => {
    askedConfirm.mockResolvedValue(false);
    render(<ChangeGroup kind="changes" files={[file('a.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Discard All Changes' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalled());
    expect(discard).not.toHaveBeenCalled();
  });
});

describe('ChangeGroup: focus after a row leaves', () => {
  it('moves focus to the heading when a row action lands', async () => {
    render(<ChangeGroup kind="changes" files={[file('a.py')]} />);
    // A row action names its file; the group's own is "Stage All".
    fireEvent.click(screen.getByRole('button', { name: 'Stage a.py' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Changes' })),
    );
  });

  it('moves focus to the heading after a group action too', async () => {
    render(<ChangeGroup kind="staged" files={[file('a.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Unstage All' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole('button', { name: 'Staged Changes' }),
      ),
    );
  });

  it('falls back to the message box when the heading itself is gone', async () => {
    // Discarding the last change empties the panel down to "No changes", so
    // the group that owns the heading unmounts in the same beat. Without a
    // fallback focus lands on the document body and the next Tab starts from
    // the top of the page.
    useGitStore.setState({
      status: status({ unstaged: [file('a.py')] }),
      discard: vi.fn(async () => {
        useGitStore.setState({ status: status() });
        return true;
      }),
    });
    render(<Panel files={[file('a.py')]} />);
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes a.py' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByLabelText('message')),
    );
  });
});

/**
 * The panel in miniature: the Changes group is drawn only while there is
 * something in it, which is exactly how `SourceControlTab` draws it.
 */
function Panel({ files }: { files: GitFile[] }) {
  const current = useGitStore((s) => s.status);
  const empty =
    (current?.unstaged.length ?? 0) + (current?.untracked.length ?? 0) === 0;
  return (
    <>
      <textarea data-scm-focus={SCM_FOCUS.commit} aria-label="message" />
      {!empty && <ChangeGroup kind="changes" files={files} />}
    </>
  );
}
