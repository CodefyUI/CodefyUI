import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { FileRow, type ChangeGroupKind } from './FileRow';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { confirm } from '../../utils/dialog';
import type { FileKind, GitFile } from '../../api/git';

/*
 * One file. The interesting half is which buttons a row is allowed to have:
 * the backend refuses a discard of a conflicted path outright (400
 * `path_not_in_status`), so a Discard button there would be a button that only
 * ever produces an error line.
 *
 * The discard confirmation is an in-app modal driven by a promise; mocking the
 * helper keeps these cases about the ROW's decisions.
 */
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const askedConfirm = vi.mocked(confirm);

function file(over: Partial<GitFile> = {}): GitFile {
  return { path: 'src/model.py', orig_path: null, kind: 'modified', xy: 'M.', score: null, ...over };
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
  useGitStore.setState({ stage, unstage, discard });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

function draw(f: GitFile, group: ChangeGroupKind, onActed?: () => void) {
  return render(
    <ul>
      <FileRow file={f} group={group} onActed={onActed} />
    </ul>,
  );
}

describe('FileRow: what the row says', () => {
  const cases: [FileKind, string, string][] = [
    ['modified', 'M', 'Modified'],
    ['added', 'A', 'Added'],
    ['deleted', 'D', 'Deleted'],
    ['renamed', 'R', 'Renamed'],
    ['copied', 'C', 'Copied'],
    ['typechange', 'T', 'Type changed'],
    ['untracked', 'U', 'Untracked'],
    ['conflict', '!', 'Conflict'],
  ];

  it.each(cases)('%s wears the letter %s named %s', (kind, letter, name) => {
    draw(file({ kind }), 'changes');
    const chip = screen.getByRole('img', { name });
    expect(chip.textContent).toBe(letter);
  });

  it('shows the basename beside a dimmed directory, and the path in full on hover', () => {
    draw(file({ path: 'graphs/deep/cnn.graph.json' }), 'changes');
    expect(screen.getByText('cnn.graph.json')).toBeTruthy();
    expect(screen.getByText('graphs/deep')).toBeTruthy();
    // On the ROW: a `title` on the disabled button inside it would never open.
    expect(screen.getByRole('listitem').getAttribute('title')).toBe(
      'graphs/deep/cnn.graph.json',
    );
  });

  it('shows a file at the repository root with no directory half', () => {
    draw(file({ path: 'README.md' }), 'changes');
    expect(screen.getByText('README.md')).toBeTruthy();
  });

  it('shows a rename as one path arrowing into the other', () => {
    draw(
      file({ path: 'src/model.py', orig_path: 'src/net.py', kind: 'renamed' }),
      'staged',
    );
    expect(screen.getByText('net.py -> model.py')).toBeTruthy();
    expect(screen.getByRole('listitem').getAttribute('title')).toBe(
      'src/net.py -> src/model.py',
    );
  });

  it('leaves the row button inert, carrying only the full path', () => {
    draw(file(), 'changes');
    const open = screen.getByRole('button', { name: /model\.py/ });
    expect(open).toBeDisabled();
    // The button carries no tooltip of its own -- a disabled element opens
    // none in Chrome, so the path is the row's.
    expect(open.getAttribute('title')).toBeNull();
    expect(screen.getByRole('listitem').getAttribute('title')).toBe('src/model.py');
  });
});

describe('FileRow: the actions a group allows', () => {
  it('a staged row can only be taken back out of the index', () => {
    draw(file(), 'staged');
    expect(screen.queryByRole('button', { name: 'Stage' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Discard Changes' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Unstage' }));
    expect(unstage).toHaveBeenCalledWith(['src/model.py']);
  });

  it('a changed row can be staged or discarded', () => {
    draw(file(), 'changes');
    expect(screen.queryByRole('button', { name: 'Unstage' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Stage' }));
    expect(stage).toHaveBeenCalledWith(['src/model.py']);
    expect(screen.getByRole('button', { name: 'Discard Changes' })).toBeTruthy();
  });

  it('a conflicted row offers no way to discard it', () => {
    // `discard_paths` builds its list from `unstaged` + `untracked`, and a
    // conflicted path is in neither: the server answers 400
    // `path_not_in_status`. Staging IS how a resolution is marked, so that
    // one stays.
    draw(file({ kind: 'conflict', xy: 'UU' }), 'merge');
    expect(screen.queryByRole('button', { name: 'Discard Changes' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Unstage' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Stage' }));
    expect(stage).toHaveBeenCalledWith(['src/model.py']);
  });
});

describe('FileRow: discarding asks first', () => {
  it('asks about a tracked file with the reversible wording', async () => {
    draw(file(), 'changes');
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes' }));
    await waitFor(() => expect(discard).toHaveBeenCalledWith(['src/model.py']));
    expect(askedConfirm).toHaveBeenCalledWith({
      title: 'Discard changes to src/model.py?',
      confirmText: 'Discard',
      variant: 'danger',
    });
  });

  it('warns that an untracked file cannot be recovered', async () => {
    draw(file({ path: 'notes.txt', kind: 'untracked', xy: '??' }), 'changes');
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes' }));
    await waitFor(() => expect(discard).toHaveBeenCalledWith(['notes.txt']));
    expect(askedConfirm.mock.calls[0][0].title).toBe(
      'Delete notes.txt? It is not tracked by git and cannot be recovered.',
    );
  });

  it('does nothing at all when the question is answered no', async () => {
    askedConfirm.mockResolvedValue(false);
    draw(file(), 'changes');
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalled());
    expect(discard).not.toHaveBeenCalled();
  });
});

describe('FileRow: focus after the row is gone', () => {
  it('tells the group when an action landed', async () => {
    const acted = vi.fn();
    draw(file(), 'changes', acted);
    fireEvent.click(screen.getByRole('button', { name: 'Stage' }));
    await waitFor(() => expect(acted).toHaveBeenCalledTimes(1));
  });

  it('stays put when the action was refused, because the row still exists', async () => {
    const acted = vi.fn();
    useGitStore.setState({ stage: vi.fn(async () => false) });
    draw(file(), 'changes', acted);
    fireEvent.click(screen.getByRole('button', { name: 'Stage' }));
    await waitFor(() => expect(useGitStore.getState().stage).toHaveBeenCalled());
    expect(acted).not.toHaveBeenCalled();
  });
});
