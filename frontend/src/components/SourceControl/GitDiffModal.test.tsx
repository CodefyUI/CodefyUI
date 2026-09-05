import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, within } from '@testing-library/react';
import { GitDiffModal } from './GitDiffModal';
import { useI18n } from '../../i18n';
import { useDialogStore } from '../../store/dialogStore';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { useUIStore, type GitDiffTarget } from '../../store/uiStore';
import type { GitCommit, GitDiff, GitFileAtRef } from '../../api/git';

/*
 * The diff window: what it asks for, what it draws, and what it does with the
 * Escape key.
 *
 * `api/git` is stubbed at the module -- the two reads here are open GETs, and
 * a real one would leave a fetch in flight for every case in this file.
 */
vi.mock('../../api/git', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/git')>();
  return { ...actual, getGitDiff: vi.fn(), getGitFile: vi.fn() };
});

const { getGitDiff, getGitFile, GitApiError } = await import('../../api/git');
const readDiff = vi.mocked(getGitDiff);
const readFile = vi.mocked(getGitFile);

const PATCH = [
  'diff --git a/src/model.py b/src/model.py',
  'index 1111111..2222222 100644',
  '--- a/src/model.py',
  '+++ b/src/model.py',
  '@@ -1,3 +1,3 @@',
  ' import torch',
  '-hidden = 16',
  '+hidden = 32',
  ' print(hidden)',
  '',
].join('\n');

// What git really answers for an UNMERGED path: the combined format, two
// prefix columns and a `@@@` header. Captured from a real conflict, because
// this is the exact patch a conflicted row opens and a two-way fixture in its
// place would let a parser that cannot read one pass this file.
const CONFLICT_PATCH = [
  'diff --cc src/model.py',
  'index f797e30,ffafd7c..0000000',
  '--- a/src/model.py',
  '+++ b/src/model.py',
  '@@@ -1,3 -1,3 +1,7 @@@',
  '  import torch',
  '++<<<<<<< HEAD',
  ' +hidden = 32',
  '++=======',
  '+ hidden = 64',
  '++>>>>>>> feature',
  '  print(hidden)',
  '',
].join('\n');

function diff(over: Partial<GitDiff> = {}): GitDiff {
  return {
    patch: PATCH,
    binary: false,
    truncated: false,
    oldRef: 'index',
    newRef: 'worktree',
    oldMissing: false,
    newMissing: false,
    ...over,
  };
}

function fileAtRef(text: string, over: Partial<GitFileAtRef> = {}): GitFileAtRef {
  return { text, binary: false, size: text.length, truncated: false, ...over };
}

function commit(over: Partial<GitCommit> = {}): GitCommit {
  return {
    sha: 'a'.repeat(40),
    short: 'aaaaaaa',
    parents: ['b'.repeat(40)],
    authorName: 'Ada',
    authorEmail: 'ada@example.com',
    authoredAt: 1_700_000_000,
    refs: [],
    subject: 'Widen the layer',
    body: '',
    ...over,
  };
}

/** The two halves of a graph, as they are written on disk. */
const GRAPH_BEFORE = JSON.stringify({
  format_version: 1,
  nodes: [{ id: 'linear', type: 'Linear', data: { params: { out_features: 16 } } }],
  edges: [],
});
const GRAPH_AFTER = JSON.stringify({
  format_version: 1,
  nodes: [{ id: 'linear', type: 'Linear', data: { params: { out_features: 32 } } }],
  edges: [],
});

/** Open the modal on one target and let its reads settle. */
async function open(target: GitDiffTarget) {
  const view = render(<GitDiffModal />);
  await act(async () => {
    useUIStore.getState().openGitDiff(target);
  });
  return view;
}

const escape = () => act(() => {
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
});

const dialog = () => screen.getByRole('dialog');

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useDialogStore.setState({ active: null });
  _resetGitStoreForTesting();
  useUIStore.setState({
    gitDiff: null,
    packCenterOpen: false,
    pluginCenterOpen: false,
    shortcutsModalOpen: false,
  });
  readDiff.mockReset();
  readFile.mockReset();
  readDiff.mockResolvedValue(diff());
  readFile.mockResolvedValue(fileAtRef(''));
});

afterEach(() => {
  act(() => {
    useUIStore.setState({ gitDiff: null });
  });
  vi.restoreAllMocks();
});

describe('GitDiffModal: what it opens on', () => {
  it('draws nothing at all until a row asks for a diff', () => {
    render(<GitDiffModal />);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(readDiff).not.toHaveBeenCalled();
  });

  it('says it is loading the CHANGES, in its own words', async () => {
    // Not the toolbar's "Loading...", which belongs to the graph list: a
    // borrowed string is a string nobody can change for this window.
    readDiff.mockReturnValue(new Promise<GitDiff>(() => {}));
    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(within(dialog()).getByText('Loading changes...')).toBeTruthy();
  });

  it('names the file it is showing and which two sides those are', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(within(dialog()).getByText('Changes: src/model.py')).toBeTruthy();
    expect(within(dialog()).getByText('Unstaged changes')).toBeTruthy();
  });

  it('asks for the scope the row was in', async () => {
    await open({ path: 'src/model.py', scope: 'index' });
    expect(readDiff).toHaveBeenCalledWith({
      path: 'src/model.py',
      scope: 'index',
      sha: undefined,
    });
    expect(within(dialog()).getByText('Staged changes')).toBeTruthy();
  });

  it('carries the commit a history row was opened from', async () => {
    await open({ path: 'src/model.py', scope: 'commit', sha: 'a'.repeat(40) });
    expect(readDiff).toHaveBeenCalledWith({
      path: 'src/model.py',
      scope: 'commit',
      sha: 'a'.repeat(40),
    });
    // The short form in the header: forty characters is not a heading.
    expect(within(dialog()).getByText('Commit aaaaaaa')).toBeTruthy();
  });

  it('shows the next file it is asked for and nothing of the last one', async () => {
    // The body is keyed by target, so a second row REMOUNTS it. Without that
    // key the graph summary -- which is state, and which only a successful
    // pair of graph reads ever writes -- would still be on screen above the
    // new file's patch, saying what changed in a graph nobody is looking at.
    readFile.mockImplementation(async ({ ref }) =>
      fileAtRef(ref === 'worktree' ? GRAPH_AFTER : GRAPH_BEFORE));
    render(<GitDiffModal />);
    await act(async () => {
      useUIStore.getState().openGitDiff({ path: 'graphs/net.graph.json', scope: 'worktree' });
    });
    expect(within(dialog()).getByText('linear: out_features 16 -> 32')).toBeTruthy();

    await act(async () => {
      useUIStore.getState().openGitDiff({ path: 'src/model.py', scope: 'worktree' });
    });
    expect(within(dialog()).getByText('Changes: src/model.py')).toBeTruthy();
    expect(within(dialog()).queryByText('linear: out_features 16 -> 32')).toBeNull();
  });
});

describe('GitDiffModal: the patch', () => {
  it('draws every line of the hunk, with both line numbers', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    const body = dialog();
    expect(within(body).getByText('@@ -1,3 +1,3 @@')).toBeTruthy();
    expect(within(body).getByText('import torch')).toBeTruthy();
    expect(within(body).getByText('hidden = 16')).toBeTruthy();
    expect(within(body).getByText('hidden = 32')).toBeTruthy();
  });

  it('says which lines were added and removed without using colour alone', async () => {
    // The sign is drawn text, not a class: a reader who cannot tell a green
    // row from a red one still has the one character that says which is which.
    await open({ path: 'src/model.py', scope: 'worktree' });
    const removed = within(dialog()).getByText('hidden = 16').closest('[data-kind]');
    const added = within(dialog()).getByText('hidden = 32').closest('[data-kind]');
    expect(removed?.getAttribute('data-kind')).toBe('del');
    expect(added?.getAttribute('data-kind')).toBe('add');
    expect(removed?.textContent).toContain('-');
    expect(added?.textContent).toContain('+');
  });

  it('says a binary file has no text diff', async () => {
    readDiff.mockResolvedValue(diff({ binary: true, patch: 'Binary files differ\n' }));
    await open({ path: 'assets/logo.png', scope: 'worktree' });
    expect(within(dialog()).getByText('Binary file; no text diff.')).toBeTruthy();
  });

  it('says a patch was cut, in either view', async () => {
    readDiff.mockResolvedValue(diff({ truncated: true }));
    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(within(dialog()).getByText('Diff truncated at 1024 KB.')).toBeTruthy();
    fireEvent.click(screen.getByRole('radio', { name: 'Side by side' }));
    expect(within(dialog()).getByText('Diff truncated at 1024 KB.')).toBeTruthy();
  });

  it('says so when the two sides are the same', async () => {
    readDiff.mockResolvedValue(diff({ patch: '' }));
    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(within(dialog()).getByText('No changes')).toBeTruthy();
  });

  it('draws the conflict markers of an unmerged path', async () => {
    // The one file a diff matters most for, and the one git answers in its
    // combined format. Read as a two-way patch it opens no hunk at all, and
    // the window said "No changes" over a file full of markers.
    readDiff.mockResolvedValue(diff({ patch: CONFLICT_PATCH }));
    await open({ path: 'src/model.py', scope: 'worktree', conflicted: true });

    const body = dialog();
    expect(within(body).queryByText('No changes')).toBeNull();
    expect(within(body).getByText('@@@ -1,3 -1,3 +1,7 @@@')).toBeTruthy();
    expect(within(body).getByText('<<<<<<< HEAD')).toBeTruthy();
    expect(within(body).getByText('hidden = 32')).toBeTruthy();
    expect(within(body).getByText('hidden = 64')).toBeTruthy();
    expect(within(body).getByText('>>>>>>> feature')).toBeTruthy();
  });

  it('shows a patch it cannot read as the text git wrote', async () => {
    // "No changes" is what GIT said, never what this parser managed to read:
    // bytes that yield no hunks are a shape this build does not know, and
    // summarising them as no change is a claim about the file.
    readDiff.mockResolvedValue(diff({ patch: 'some shape from a later git\n' }));
    await open({ path: 'src/model.py', scope: 'worktree' });

    expect(within(dialog()).queryByText('No changes')).toBeNull();
    expect(within(dialog()).getByText(/some shape from a later git/)).toBeTruthy();
  });
});

describe('GitDiffModal: the two views', () => {
  it('names the choice, not only its two answers', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(within(dialog()).getByRole('radiogroup', { name: 'View' })).toBeTruthy();
  });

  it('is one tab stop, on whichever view is chosen', async () => {
    // A radio group is ONE stop: Tab leaves it rather than walking to the
    // answer nobody picked.
    await open({ path: 'src/model.py', scope: 'worktree' });
    const unified = screen.getByRole('radio', { name: 'Unified' });
    const split = screen.getByRole('radio', { name: 'Side by side' });
    expect(unified.tabIndex).toBe(0);
    expect(split.tabIndex).toBe(-1);

    fireEvent.click(split);
    expect(unified.tabIndex).toBe(-1);
    expect(split.tabIndex).toBe(0);
  });

  it.each([
    ['ArrowRight', 'Side by side'],
    ['ArrowDown', 'Side by side'],
    ['ArrowLeft', 'Side by side'],
    ['ArrowUp', 'Side by side'],
  ])('moves and chooses on %s, which the role promises', async (key, expected) => {
    // Two answers and the group wraps, so either direction lands on the other
    // one. A reader who hears "radio button" reaches for these keys.
    await open({ path: 'src/model.py', scope: 'worktree' });
    const group = within(dialog()).getByRole('radiogroup');
    fireEvent.keyDown(group, { key });

    const chosen = screen.getByRole('radio', { name: expected });
    expect(chosen.getAttribute('aria-checked')).toBe('true');
    // Focus follows the choice, or the next arrow press would move from the
    // radio the reader left behind.
    expect(document.activeElement).toBe(chosen);
  });

  it('leaves every other key to the window', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    const group = within(dialog()).getByRole('radiogroup');
    fireEvent.keyDown(group, { key: 'Home' });
    expect(
      screen.getByRole('radio', { name: 'Unified' }).getAttribute('aria-checked'),
    ).toBe('true');
  });

  it.each([
    ['while the read is still out', () => {
      readDiff.mockReturnValue(new Promise<GitDiff>(() => {}));
    }],
    ['over a refusal', () => {
      readDiff.mockRejectedValue(new GitApiError(404, 'no such path', { code: 'not_found' }));
    }],
    ['over a binary file', () => {
      readDiff.mockResolvedValue(diff({ binary: true, patch: 'Binary files differ\n' }));
    }],
    ['over no changes at all', () => {
      readDiff.mockResolvedValue(diff({ patch: '' }));
    }],
  ])('offers no view switch %s', async (_name, arrange) => {
    // There is one sentence on screen in each of these, and pressing either
    // radio would change nothing at all.
    arrange();
    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(within(dialog()).queryByRole('radiogroup')).toBeNull();
  });

  it('offers one choice of two, starting on the unified one', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    const group = within(dialog()).getByRole('radiogroup');
    expect(
      within(group).getByRole('radio', { name: 'Unified' }).getAttribute('aria-checked'),
    ).toBe('true');
    expect(
      within(group).getByRole('radio', { name: 'Side by side' }).getAttribute('aria-checked'),
    ).toBe('false');
  });

  it('pairs the removed and added runs when asked for two columns', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    fireEvent.click(screen.getByRole('radio', { name: 'Side by side' }));
    expect(
      screen.getByRole('radio', { name: 'Side by side' }).getAttribute('aria-checked'),
    ).toBe('true');
    // One row holds both halves of the change.
    const row = within(dialog()).getByText('hidden = 16').closest('[data-row]');
    expect(row?.textContent).toContain('hidden = 32');
  });

  it('offers no side-by-side view of a conflicted file', async () => {
    // Its INDEX copy does not exist until it is settled -- there is no stage
    // 0 -- so one of the two columns could only be a guess. The patch is the
    // combined one git really answers for an unmerged path, so the case is
    // about the file kind and not about a fixture that resembles it.
    readDiff.mockResolvedValue(diff({ patch: CONFLICT_PATCH }));
    await open({ path: 'src/model.py', scope: 'worktree', conflicted: true });
    expect(within(dialog()).queryByRole('radiogroup')).toBeNull();
    expect(within(dialog()).getByText('hidden = 32')).toBeTruthy();
  });
});

describe('GitDiffModal: what changed in the graph', () => {
  it('reads both sides of a graph file and says what moved in it', async () => {
    readFile.mockImplementation(async ({ ref }) =>
      fileAtRef(ref === 'worktree' ? GRAPH_AFTER : GRAPH_BEFORE));
    await open({ path: 'graphs/net.graph.json', scope: 'worktree' });

    expect(readFile).toHaveBeenCalledWith({ path: 'graphs/net.graph.json', ref: 'index' });
    expect(readFile).toHaveBeenCalledWith({ path: 'graphs/net.graph.json', ref: 'worktree' });
    expect(
      within(dialog()).getByText('linear: out_features 16 -> 32'),
    ).toBeTruthy();
  });

  it('reads a commit\'s old side at the parent, which is a ref that exists', async () => {
    // The diff response says `<sha>^`, and `GET /file` cannot be asked for
    // that: its grammar is HEAD, index, worktree or a bare commit id. The
    // parent's own id is in the log the row was drawn from.
    useGitStore.setState({
      log: { commits: [commit()], hasMore: false, unborn: false, loading: false },
    });
    readDiff.mockResolvedValue(diff({
      oldRef: `${'a'.repeat(40)}^`,
      newRef: 'a'.repeat(40),
    }));
    readFile.mockImplementation(async ({ ref }) =>
      fileAtRef(ref === 'a'.repeat(40) ? GRAPH_AFTER : GRAPH_BEFORE));

    await open({ path: 'graphs/net.graph.json', scope: 'commit', sha: 'a'.repeat(40) });

    expect(readFile).toHaveBeenCalledWith({
      path: 'graphs/net.graph.json',
      ref: 'b'.repeat(40),
    });
    expect(within(dialog()).getByText('linear: out_features 16 -> 32')).toBeTruthy();
  });

  it('says nothing about a graph whose old side it cannot name', async () => {
    // No log to resolve the parent from: a summary built against a missing
    // old side would report every node in the file as added.
    readDiff.mockResolvedValue(diff({
      oldRef: `${'a'.repeat(40)}^`,
      newRef: 'a'.repeat(40),
    }));
    await open({ path: 'graphs/net.graph.json', scope: 'commit', sha: 'a'.repeat(40) });
    expect(readFile).not.toHaveBeenCalled();
    expect(within(dialog()).queryByText('No logic change')).toBeNull();
  });

  it('says nothing about a graph one of whose sides came back unread', async () => {
    // Over the 2 MiB cap, `GET /file` answers with the real size, `truncated`
    // and NO text -- which parses as broken JSON and would report "could not
    // parse" about a graph that is perfectly well formed.
    readFile.mockImplementation(async ({ ref }) =>
      (ref === 'worktree'
        ? fileAtRef('', { truncated: true, size: 3_000_000 })
        : fileAtRef(GRAPH_BEFORE)));
    await open({ path: 'graphs/net.graph.json', scope: 'worktree' });
    expect(within(dialog()).queryByText('Could not parse as a graph')).toBeNull();
    expect(within(dialog()).queryByText('No logic change')).toBeNull();
  });

  it('says nothing about a graph whose side came back as bytes', async () => {
    // `GET /file` answers a binary blob with `binary` and no text. Treated as
    // text it is an empty string, which parses as broken JSON and would
    // report "could not parse" about a file that is not a graph at all.
    readFile.mockResolvedValue(fileAtRef('', { binary: true }));
    await open({ path: 'graphs/net.graph.json', scope: 'worktree' });
    expect(within(dialog()).queryByText('Could not parse as a graph')).toBeNull();
    expect(within(dialog()).queryByText('No logic change')).toBeNull();
  });

  it('takes a side that is not there as an empty one', async () => {
    // The commonest commit case: a graph ADDED in that commit has no parent
    // side, and `GET /file` answers 404. That is an ANSWER -- "everything in
    // this file is new" -- and the summary is what says so.
    readFile.mockImplementation(async ({ ref }) => {
      if (ref === 'index') {
        throw new GitApiError(404, 'no such path', { code: 'not_found' });
      }
      return fileAtRef(GRAPH_AFTER);
    });
    await open({ path: 'graphs/net.graph.json', scope: 'worktree' });
    expect(within(dialog()).getByText('1 node(s) added')).toBeTruthy();
  });

  it('says nothing about a graph whose side was refused for any other reason', async () => {
    // Only 404 means "not there". A 403 means the read did not happen, and a
    // summary built on the one side that answered would report every node in
    // the file as added.
    readFile.mockImplementation(async ({ ref }) => {
      if (ref === 'index') {
        throw new GitApiError(403, 'path is ignored', { code: 'ignored' });
      }
      return fileAtRef(GRAPH_AFTER);
    });
    await open({ path: 'graphs/net.graph.json', scope: 'worktree' });
    expect(within(dialog()).queryByText('1 node(s) added')).toBeNull();
    expect(within(dialog()).queryByText('Could not parse as a graph')).toBeNull();
    // And the patch itself is unaffected: the diff read succeeded.
    expect(within(dialog()).getByText('hidden = 32')).toBeTruthy();
  });

  it('says when a graph file cannot be read as a graph', async () => {
    readFile.mockResolvedValue(fileAtRef('{ not json'));
    await open({ path: 'graphs/net.graph.json', scope: 'worktree' });
    expect(within(dialog()).getByText('Could not parse as a graph')).toBeTruthy();
  });

  it('says when only the text of a graph moved', async () => {
    readFile.mockImplementation(async ({ ref }) =>
      fileAtRef(ref === 'worktree'
        ? JSON.stringify({ nodes: [], edges: [] })
        : '{"edges": [], "nodes": []}'));
    await open({ path: 'graphs/net.graph.json', scope: 'worktree' });
    expect(within(dialog()).getByText('No logic change')).toBeTruthy();
  });

  it('asks for no whole file at all for anything that is not a graph', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(readFile).not.toHaveBeenCalled();
  });
});

describe('GitDiffModal: a read that was refused', () => {
  it('shows one sentence, with git\'s own words behind a disclosure', async () => {
    readDiff.mockRejectedValue(
      new GitApiError(403, 'path is ignored', {
        code: 'ignored',
        stderr: 'fatal: the tail git wrote',
      }),
    );
    await open({ path: '.env', scope: 'worktree' });

    expect(within(dialog()).getByText('This file is ignored by git.')).toBeTruthy();
    const details = within(dialog()).getByRole('button', { name: 'Details' });
    expect(details.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(details);
    expect(details.getAttribute('aria-expanded')).toBe('true');
    expect(within(dialog()).getByText('fatal: the tail git wrote')).toBeTruthy();
  });

  it('announces the refusal, which nothing else in the window would', async () => {
    // The window opens on a loading line and swaps it for this one. A
    // `role="dialog"` body is not a live region, so without an alert the
    // reader hears "Loading changes..." and then silence.
    readDiff.mockRejectedValue(
      new GitApiError(403, 'path is ignored', {
        code: 'ignored',
        stderr: 'fatal: the tail git wrote',
      }),
    );
    await open({ path: '.env', scope: 'worktree' });

    const alert = within(dialog()).getByRole('alert');
    expect(alert.textContent).toBe('This file is ignored by git.');
    // The sentence ALONE: an alert re-announces on any change in its subtree,
    // so opening the stderr must not read the whole refusal out again.
    expect(within(alert).queryByRole('button')).toBeNull();
    expect(alert.textContent).not.toContain('fatal: the tail git wrote');
  });

  it('keeps the refusal to itself, off the panel\'s error line', async () => {
    // R12: a read refusal belongs to the surface that asked for it. The
    // header's line is the one an operation the user pressed a button for
    // writes, and replacing it would take that refusal off screen.
    readDiff.mockRejectedValue(
      new GitApiError(404, 'no such path', { code: 'not_found' }),
    );
    await open({ path: 'src/gone.py', scope: 'worktree' });
    expect(useGitStore.getState().lastError).toBeNull();
    expect(within(dialog()).getByText('Not found: no such path')).toBeTruthy();
  });
});

describe('GitDiffModal: the key and the focus', () => {
  it('closes on Escape', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    escape();
    expect(useUIStore.getState().gitDiff).toBeNull();
  });

  it('stands down while a dialog is open on top of it', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    useDialogStore.setState({ active: { kind: 'confirm', title: 'Sure?' } });
    escape();
    expect(useUIStore.getState().gitDiff).not.toBeNull();
  });

  it('closes from the button that says so', async () => {
    await open({ path: 'src/model.py', scope: 'worktree' });
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(useUIStore.getState().gitDiff).toBeNull();
  });

  it('takes focus in and hands it back to whatever opened it', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();

    await open({ path: 'src/model.py', scope: 'worktree' });
    expect(dialog().contains(document.activeElement)).toBe(true);

    await act(async () => {
      useUIStore.getState().closeGitDiff();
    });
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });

  it('lands on the panel when the row that opened it has gone', async () => {
    // The fifteen-second poll re-renders the file groups, and a row whose key
    // changes -- staged or settled from elsewhere while the window was open --
    // unmounts the button that had focus. Falling to `<body>` would start the
    // next Tab at the top of the page.
    const panel = document.createElement('div');
    panel.setAttribute('data-scm-focus', 'title');
    panel.tabIndex = -1;
    document.body.appendChild(panel);
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();

    await open({ path: 'src/model.py', scope: 'worktree' });
    opener.remove();
    await act(async () => {
      useUIStore.getState().closeGitDiff();
    });

    expect(document.activeElement).toBe(panel);
    panel.remove();
  });
});
