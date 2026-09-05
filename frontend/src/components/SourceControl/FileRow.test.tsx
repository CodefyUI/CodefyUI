import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { displayPath, FileRow, fileRowLabel, type ChangeGroupKind } from './FileRow';
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

let opened: ReturnType<typeof vi.fn<(file: GitFile) => void>>;

function draw(f: GitFile, group: ChangeGroupKind, onActed?: () => void) {
  opened = vi.fn();
  return render(
    <ul>
      <FileRow file={f} group={group} onActed={onActed} onOpen={opened} />
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
    // On the ROW, which is what the chip and the action buttons are over as
    // well -- and the button inside it carries no `title` of its own, so this
    // one opens over the name too.
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

  it('opens the change from the name, naming the file it will show', () => {
    draw(file(), 'changes');
    const open = screen.getByRole('button', { name: 'Open changes model.py' });
    expect(open).not.toBeDisabled();
    // No tooltip of its own: the button fills most of the row, and the path
    // in full is the answer a pointer over a truncated name is after -- so
    // the row's `title` is the one that has to open here.
    expect(open.getAttribute('title')).toBeNull();
    expect(screen.getByRole('listitem').getAttribute('title')).toBe('src/model.py');

    fireEvent.click(open);
    expect(opened).toHaveBeenCalledTimes(1);
    expect(opened).toHaveBeenCalledWith(file());
  });

  it('names the open button after a rename the way the row shows it', () => {
    draw(file({ path: 'src/model.py', orig_path: 'src/net.py', kind: 'renamed' }), 'staged');
    expect(
      screen.getByRole('button', { name: 'Open changes net.py -> model.py' }),
    ).toBeTruthy();
  });
});

describe('FileRow: the two spellings of a rename source', () => {
  /*
   * A status entry carries `orig_path` (the server's own snake_case) and a
   * commit's file carries `origPath`, because they come from two different
   * routes. Both are one fact, and these helpers take both -- a cast at the
   * call site would let the other shape through silently and draw a rename as
   * one path.
   */
  it('reads a status entry\'s orig_path', () => {
    const entry = { path: 'src/model.py', orig_path: 'src/net.py' };
    expect(displayPath(entry)).toBe('src/net.py -> src/model.py');
    expect(fileRowLabel(entry)).toBe('net.py -> model.py');
  });

  it('reads a commit file\'s origPath', () => {
    const entry = { path: 'src/model.py', origPath: 'src/net.py' };
    expect(displayPath(entry)).toBe('src/net.py -> src/model.py');
    expect(fileRowLabel(entry)).toBe('net.py -> model.py');
  });

  it('treats an absent source the same as a null one, in either spelling', () => {
    // `=== null` was the old test, and `undefined` is what the OTHER shape
    // leaves behind for a file that was not renamed.
    expect(displayPath({ path: 'a.py' })).toBe('a.py');
    expect(displayPath({ path: 'a.py', orig_path: null })).toBe('a.py');
    expect(displayPath({ path: 'a.py', origPath: null })).toBe('a.py');
    expect(fileRowLabel({ path: 'src/a.py', origPath: undefined })).toBe('a.py');
  });
});

describe('FileRow: the actions a group allows', () => {
  it('a staged row can only be taken back out of the index', () => {
    draw(file(), 'staged');
    expect(screen.queryByRole('button', { name: 'Stage model.py' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Discard Changes model.py' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Unstage model.py' }));
    expect(unstage).toHaveBeenCalledWith(['src/model.py']);
  });

  it('a changed row can be staged or discarded', () => {
    draw(file(), 'changes');
    expect(screen.queryByRole('button', { name: 'Unstage model.py' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Stage model.py' }));
    expect(stage).toHaveBeenCalledWith(['src/model.py']);
    expect(screen.getByRole('button', { name: 'Discard Changes model.py' })).toBeTruthy();
  });

  it('a conflicted row offers no way to discard it', () => {
    // `discard_paths` builds its list from `unstaged` + `untracked`, and a
    // conflicted path is in neither: the server answers 400
    // `path_not_in_status`. The status keeps conflicts in a list of their own,
    // so `MergeGroup` is where one is normally drawn -- this is the guard for
    // a conflict that turns up somewhere else.
    draw(file({ kind: 'conflict', xy: 'UU' }), 'changes');
    expect(screen.queryByRole('button', { name: 'Discard Changes model.py' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Unstage model.py' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Stage model.py' }));
    expect(stage).toHaveBeenCalledWith(['src/model.py']);
  });
});

describe('FileRow: the collapsed actions stay focusable', () => {
  it('holds its buttons in a rowActions box that nothing takes out of the tab order', () => {
    // The focusability contract behind the CSS. `.rowActions` is collapsed to
    // zero width so the file name gets that column back, and the ONLY things
    // that open it again are `:hover` and `:focus-within` -- so focus has to
    // be able to arrive in the first place. jsdom has no layout, so what is
    // pinned here is the half that is not layout: the class that does the
    // collapsing, and buttons that a keyboard can still reach.
    draw(file(), 'changes');
    const actions = screen.getByRole('button', { name: 'Stage model.py' }).parentElement;
    expect(actions).not.toBeNull();
    expect(actions?.className).toContain('rowActions');

    const buttons = within(actions as HTMLElement).getAllByRole('button');
    expect(buttons).toHaveLength(2);
    for (const button of buttons) {
      expect(button.tagName).toBe('BUTTON');
      expect(button).not.toBeDisabled();
      expect(button.hasAttribute('hidden')).toBe(false);
      expect(button.getAttribute('tabindex')).toBeNull();
      expect(button.getAttribute('aria-hidden')).toBeNull();
    }
  });
});

describe('FileRow: what a screen reader hears on the buttons', () => {
  it('names the file in the verb, and keeps the tooltip bare', () => {
    draw(file(), 'changes');
    const stageButton = screen.getByRole('button', { name: 'Stage model.py' });
    // The tooltip is for a reader who can see which row it is over, so it
    // stays the short word the design asks for.
    expect(stageButton.getAttribute('title')).toBe('Stage');
    expect(
      screen.getByRole('button', { name: 'Discard Changes model.py' }).getAttribute('title'),
    ).toBe('Discard Changes');
  });

  it('gives two rows two different names, which is the whole point', () => {
    // Twenty rows deep, every one of them with a Stage button: a name that is
    // only the verb is twenty identical buttons to anybody not looking at the
    // screen.
    render(
      <ul>
        <FileRow file={file({ path: 'graphs/a.graph.json' })} group="changes" onOpen={vi.fn()} />
        <FileRow file={file({ path: 'graphs/b.graph.json' })} group="changes" onOpen={vi.fn()} />
      </ul>,
    );
    const names = screen
      .getAllByRole('button', { name: /^Stage / })
      .map((button) => button.getAttribute('aria-label'));
    expect(names).toEqual(['Stage a.graph.json', 'Stage b.graph.json']);
  });

  it('names a rename the way the row shows it', () => {
    draw(file({ path: 'src/model.py', orig_path: 'src/net.py', kind: 'renamed' }), 'staged');
    expect(
      screen.getByRole('button', { name: 'Unstage net.py -> model.py' }),
    ).toBeTruthy();
  });
});

describe('FileRow: discarding asks first', () => {
  it('asks about a tracked file with the reversible wording', async () => {
    draw(file(), 'changes');
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes model.py' }));
    await waitFor(() => expect(discard).toHaveBeenCalledWith(['src/model.py']));
    expect(askedConfirm).toHaveBeenCalledWith({
      title: 'Discard changes to src/model.py?',
      confirmText: 'Discard',
      variant: 'danger',
    });
  });

  it('warns that an untracked file cannot be recovered', async () => {
    draw(file({ path: 'notes.txt', kind: 'untracked', xy: '??' }), 'changes');
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes notes.txt' }));
    await waitFor(() => expect(discard).toHaveBeenCalledWith(['notes.txt']));
    expect(askedConfirm.mock.calls[0][0].title).toBe(
      'Delete notes.txt? It is not tracked by git and cannot be recovered.',
    );
  });

  it('does nothing at all when the question is answered no', async () => {
    askedConfirm.mockResolvedValue(false);
    draw(file(), 'changes');
    fireEvent.click(screen.getByRole('button', { name: 'Discard Changes model.py' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalled());
    expect(discard).not.toHaveBeenCalled();
  });
});

describe('FileRow: focus after the row is gone', () => {
  it('tells the group when an action landed', async () => {
    const acted = vi.fn();
    draw(file(), 'changes', acted);
    fireEvent.click(screen.getByRole('button', { name: 'Stage model.py' }));
    await waitFor(() => expect(acted).toHaveBeenCalledTimes(1));
  });

  it('stays put when the action was refused, because the row still exists', async () => {
    const acted = vi.fn();
    useGitStore.setState({ stage: vi.fn(async () => false) });
    draw(file(), 'changes', acted);
    fireEvent.click(screen.getByRole('button', { name: 'Stage model.py' }));
    await waitFor(() => expect(useGitStore.getState().stage).toHaveBeenCalled());
    expect(acted).not.toHaveBeenCalled();
  });
});
