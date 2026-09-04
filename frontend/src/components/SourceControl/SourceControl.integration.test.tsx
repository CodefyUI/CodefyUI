import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import * as gitApi from '../../api/git';
import type {
  FileKind,
  GitFile,
  GitPathSelection,
  GitStatus,
  Identity,
  MutationResult,
  StatusResponse,
} from '../../api/git';

/**
 * The whole tab, against a git that answers.
 *
 * Every other file in this directory fakes the store and asks what one
 * component draws. This one keeps the REAL store and fakes the wire instead,
 * so the things that only exist BETWEEN the pieces are what get pinned: a
 * save that reaches the panel through a debounce, a row that moves from one
 * group to another because a mutation answered with a new status, a refusal
 * that opens a form four components away, a discard that offers to reload the
 * tab it changed.
 *
 * The fake is a small working tree rather than a list of canned answers:
 * `gitStage` really moves a file into the index and `gitCommit` really empties
 * it, so "the row is in Staged now" is the store applying the status the
 * server sent and not a fixture that says so.
 */
vi.mock('../../api/git', async (importOriginal) => {
  const actual = await importOriginal<typeof gitApi>();
  return {
    ...actual,
    getGitStatus: vi.fn(),
    getGitConfig: vi.fn(),
    gitInit: vi.fn(),
    gitStage: vi.fn(),
    gitUnstage: vi.fn(),
    gitDiscard: vi.fn(),
    gitCommit: vi.fn(),
    setGitConfig: vi.fn(),
  };
});

// The two danger questions -- discard, and reload-from-disk -- are in-app
// modals driven by a promise. Mocking the helper keeps these cases about what
// the panel asks and what it does with the answer.
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

// `GraphMissingError` is a real class the store narrows on, so only the reload
// itself is stubbed -- the real one fetches a graph and installs it into a tab.
vi.mock('../../utils/openSavedGraph', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../utils/openSavedGraph')>();
  return { ...actual, reloadTabFromDisk: vi.fn(async () => false) };
});

import { SourceControlTab } from './SourceControlTab';
import { GIT_WRITE_DEBOUNCE_MS, _resetGitStoreForTesting } from '../../store/gitStore';
import { GraphMissingError, reloadTabFromDisk } from '../../utils/openSavedGraph';
import { announceWorktreeWrite } from '../../utils/worktreeWrite';
import { confirm } from '../../utils/dialog';
import { useProjectStore } from '../../store/projectStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import { MOD_LABEL } from '../../utils/platform';

const api = vi.mocked(gitApi);
const askedConfirm = vi.mocked(confirm);
const reloadMock = vi.mocked(reloadTabFromDisk);

const PROJECT_DIR = 'D:/work/demo';
/** The pair one Save writes, which is the whole point of the save hook. */
const GRAPH = 'graphs/demo.graph.json';
const LAYOUT = 'layout/demo.layout.json';

/* ── a working tree the fake client operates on ──────────────────────── */

interface Worktree {
  staged: GitFile[];
  unstaged: GitFile[];
  untracked: GitFile[];
  conflicted: GitFile[];
}

let tree: Worktree;
let head: string | null;

function gitFile(path: string, kind: FileKind = 'modified'): GitFile {
  return { path, orig_path: null, kind, xy: kind === 'untracked' ? '??' : '.M', score: null };
}

function currentStatus(): GitStatus {
  return {
    branch: 'main',
    detached: false,
    head,
    unborn: false,
    upstream: null,
    ahead: null,
    behind: null,
    upstream_gone: false,
    staged: [...tree.staged],
    unstaged: [...tree.unstaged],
    untracked: [...tree.untracked],
    conflicted: [...tree.conflicted],
    stash_count: 0,
    merge_in_progress: false,
    rebase_in_progress: false,
  };
}

function statusAnswer(): StatusResponse {
  return {
    repo: {
      state: 'ready',
      project_dir: PROJECT_DIR,
      git_version: '2.45.0',
      nested_toplevel: null,
    },
    status: currentStatus(),
  };
}

/** What a write answers with: the fresh status, and what it touched. */
function landed(over: Partial<MutationResult> = {}): MutationResult {
  return { status: currentStatus(), changed_paths: [], head, detail: {}, ...over };
}

/** Take the selection out of the working-tree groups and hand it back. */
function takeFromWorktree(paths: GitPathSelection): GitFile[] {
  const taken: GitFile[] = [];
  for (const group of ['unstaged', 'untracked'] as const) {
    tree[group] = tree[group].filter((file) => {
      if (paths !== 'all' && !paths.includes(file.path)) return true;
      taken.push(file);
      return false;
    });
  }
  return taken;
}

/** A refusal built by the REAL reader, from a body off the wire. */
function refusal(httpStatus: number, body: unknown): Promise<gitApi.GitApiError> {
  const response = {
    ok: false,
    status: httpStatus,
    statusText: 'mock',
    json: async () => body,
  } as unknown as Response;
  return gitApi.gitApiError(response);
}

function coded(
  httpStatus: number,
  code: string,
  message: string,
): Promise<gitApi.GitApiError> {
  return refusal(httpStatus, { detail: { code, message, hint: null, stderr: null } });
}

function identity(over: Partial<Identity> = {}): Identity {
  return { name: null, email: null, name_scope: null, email_scope: null, ...over };
}

/* ── harness ─────────────────────────────────────────────────────────── */

const toasts = () => useToastStore.getState().toasts;
const messageBox = () =>
  screen.getByPlaceholderText(`Message (${MOD_LABEL}+Enter to commit)`);
const group = (name: string) => screen.getByRole('region', { name });

/** The row showing *name*, so an action can be pressed on that row alone. */
function row(name: string): HTMLElement {
  const li = screen.getByText(name).closest('li');
  if (li === null) throw new Error(`no file row named ${name}`);
  return li;
}

/** Open a tab bound to *file*, stamped with *origin*, and answer its id. */
function openTab(name: string, file: string, origin: string | null): string {
  useTabStore.getState().addTab(name);
  const id = useTabStore.getState().activeTabId;
  useTabStore.getState().setCurrentGraphFile(file);
  useTabStore.getState().stampActiveTabProject(origin);
  return id;
}

/** Mount the panel and let its first status read land. */
async function openPanel(): Promise<void> {
  await act(async () => {
    render(<SourceControlTab />);
    await vi.advanceTimersByTimeAsync(0);
  });
}

/** Advance the clock, flushing whatever the panel does in response. */
async function settle(ms = 0): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

/** Long enough for the frame the focus return waits for. */
const A_FRAME = 50;

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  useProjectStore.setState({ projectDir: PROJECT_DIR, projectName: 'demo', loaded: true });
  useTabStore.setState({ tabs: [], activeTabId: '' });
  _resetGitStoreForTesting();

  tree = { staged: [], unstaged: [], untracked: [], conflicted: [] };
  head = 'abc1234def';

  api.getGitStatus.mockImplementation(async () => statusAnswer());
  api.getGitConfig.mockImplementation(async () => identity());
  api.setGitConfig.mockImplementation(async () => identity());
  api.gitStage.mockImplementation(async (paths) => {
    const moved = takeFromWorktree(paths);
    // git records a staged untracked file as an addition.
    tree.staged.push(
      ...moved.map((file): GitFile =>
        file.kind === 'untracked' ? { ...file, kind: 'added', xy: 'A.' } : file,
      ),
    );
    return landed({ changed_paths: moved.map((file) => file.path) });
  });
  api.gitUnstage.mockImplementation(async (paths) => {
    const moved: GitFile[] = [];
    tree.staged = tree.staged.filter((file) => {
      if (paths !== 'all' && !paths.includes(file.path)) return true;
      moved.push(file);
      return false;
    });
    for (const file of moved) {
      if (file.kind === 'added') tree.untracked.push({ ...file, kind: 'untracked', xy: '??' });
      else tree.unstaged.push(file);
    }
    return landed({ changed_paths: moved.map((file) => file.path) });
  });
  api.gitDiscard.mockImplementation(async (paths) => {
    const gone = takeFromWorktree(paths);
    return landed({ changed_paths: gone.map((file) => file.path) });
  });
  api.gitCommit.mockImplementation(async ({ all, amend }) => {
    if (all === true) {
      tree.staged.push(
        ...takeFromWorktree('all').map((file): GitFile =>
          file.kind === 'untracked' ? { ...file, kind: 'added', xy: 'A.' } : file,
        ),
      );
    }
    const committed = tree.staged.map((file) => file.path);
    tree.staged = [];
    if (amend !== true) head = 'c0ffee1234567';
    return landed({
      changed_paths: committed,
      detail: { sha: 'c0ffee1234567', short: 'c0ffee1' },
    });
  });

  askedConfirm.mockResolvedValue(true);
  reloadMock.mockResolvedValue(false);
});

afterEach(() => {
  // Unmount first: the panel's own effect calls `detach()`, which stops the
  // poll and unregisters the save hook while the fake clock is still running.
  cleanup();
  _resetGitStoreForTesting();
  useToastStore.setState({ toasts: [] });
  useTabStore.setState({ tabs: [], activeTabId: '' });
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  vi.useRealTimers();
  vi.clearAllMocks();
});

/* ── the everyday flow ───────────────────────────────────────────────── */

describe('Source Control: a save, a stage and a commit', () => {
  it('shows the two files a save wrote, and not before the debounce has run', async () => {
    await openPanel();
    expect(screen.getByText('No changes')).toBeTruthy();
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);

    // What `saveActiveGraph` does after a successful project save: both halves
    // are on disk, and whoever is watching the worktree is told so.
    tree.unstaged.push(gitFile(GRAPH), gitFile(LAYOUT));
    act(() => {
      announceWorktreeWrite();
    });

    // Not yet. A Save writes two files and three saves in a row are one
    // intention, so the panel coalesces the burst instead of reading the
    // status once per file.
    await settle(GIT_WRITE_DEBOUNCE_MS - 1);
    expect(api.getGitStatus).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('demo.graph.json')).toBeNull();

    await settle(1);
    expect(api.getGitStatus).toHaveBeenCalledTimes(2);
    const changes = group('Changes');
    expect(within(changes).getByText('demo.graph.json')).toBeTruthy();
    expect(within(changes).getByText('demo.layout.json')).toBeTruthy();
  });

  it('moves one file into Staged when its own row stages it', async () => {
    tree.unstaged.push(gitFile(GRAPH), gitFile(LAYOUT));
    await openPanel();

    fireEvent.click(within(row('demo.graph.json')).getByRole('button', { name: 'Stage' }));
    await settle(A_FRAME);

    expect(api.gitStage).toHaveBeenCalledWith([GRAPH]);
    expect(within(group('Staged Changes')).getByText('demo.graph.json')).toBeTruthy();
    const changes = group('Changes');
    expect(within(changes).queryByText('demo.graph.json')).toBeNull();
    expect(within(changes).getByText('demo.layout.json')).toBeTruthy();
  });

  it('commits the index from the message box with the chord, and empties the panel', async () => {
    tree.unstaged.push(gitFile(GRAPH), gitFile(LAYOUT));
    await openPanel();

    fireEvent.click(within(group('Changes')).getByRole('button', { name: 'Stage All' }));
    await settle(A_FRAME);
    expect(within(group('Staged Changes')).getAllByRole('listitem')).toHaveLength(2);

    fireEvent.change(messageBox(), { target: { value: 'save the demo graph' } });
    // Ctrl and Cmd are one chord to the handler; only the label differs.
    fireEvent.keyDown(messageBox(), { key: 'Enter', ctrlKey: true });
    await settle();

    expect(api.gitCommit).toHaveBeenCalledWith({
      message: 'save the demo graph',
      all: false,
      amend: false,
    });
    expect(toasts().map((toast) => toast.message)).toContain('Committed c0ffee1');
    expect(screen.getByText('No changes')).toBeTruthy();
    expect(screen.queryByRole('region', { name: 'Staged Changes' })).toBeNull();
    // The message went with the commit it described.
    expect(messageBox()).toHaveValue('');
  });

  it('lets a message-only amend through: it is rewriting a subject', async () => {
    // Nothing staged, on purpose. An amend that only rewords the last commit
    // is the commonest reason to amend at all.
    tree.unstaged.push(gitFile('src/train.py'));
    await openPanel();

    fireEvent.click(screen.getByRole('button', { name: 'Commit options' }));
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Amend Last Commit' }));
    fireEvent.keyDown(screen.getByRole('menu', { name: 'Commit options' }), {
      key: 'Escape',
    });
    expect(screen.getByText('Amending')).toBeTruthy();

    fireEvent.change(messageBox(), { target: { value: 'a better subject' } });
    const button = screen.getByRole('button', { name: 'Commit' });
    expect(button).toHaveAttribute('aria-disabled', 'false');
    fireEvent.click(button);
    await settle();

    expect(api.gitCommit).toHaveBeenCalledWith({
      message: 'a better subject',
      all: false,
      amend: true,
    });
    expect(toasts().map((toast) => toast.message)).toContain('Committed c0ffee1');
  });
});

/* ── a discard under an open graph ───────────────────────────────────── */

describe('Source Control: a discard under an open graph', () => {
  async function discardTheGraph(): Promise<void> {
    fireEvent.click(
      within(row('demo.graph.json')).getByRole('button', { name: 'Discard Changes' }),
    );
    await settle(A_FRAME);
  }

  async function takeTheOffer(): Promise<void> {
    const offer = toasts()[0].action;
    expect(offer?.label).toBe('Reload');
    await act(async () => {
      offer?.onClick();
      await vi.advanceTimersByTimeAsync(0);
    });
  }

  it('offers to reload the tab it changed, and reloads it when the offer is taken', async () => {
    const tabId = openTab('demo', 'demo', PROJECT_DIR);
    tree.unstaged.push(gitFile(GRAPH));
    await openPanel();

    await discardTheGraph();
    expect(askedConfirm).toHaveBeenCalledWith({
      title: 'Discard changes to graphs/demo.graph.json?',
      confirmText: 'Discard',
      variant: 'danger',
    });
    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe('1 open graph(s) changed on disk.');
    expect(toasts()[0].type).toBe('warning');

    // Sticky: the one thing standing between an open tab and a silently older
    // graph does not vanish while the reader is looking at the canvas.
    await settle(10_000);
    expect(toasts()).toHaveLength(1);

    await takeTheOffer();
    expect(askedConfirm).toHaveBeenLastCalledWith({
      title: 'Reload 1 graph(s) from disk? Unsaved edits in those tabs are lost.',
      variant: 'danger',
    });
    expect(reloadMock).toHaveBeenCalledWith(tabId, 'demo');
  });

  it('reloads nothing when the reload question is answered no', async () => {
    openTab('demo', 'demo', PROJECT_DIR);
    tree.unstaged.push(gitFile(GRAPH));
    await openPanel();

    await discardTheGraph();
    askedConfirm.mockResolvedValue(false);
    await takeTheOffer();
    expect(reloadMock).not.toHaveBeenCalled();
  });

  it('keeps the tab when the graph is gone on this branch', async () => {
    const tabId = openTab('demo', 'demo', PROJECT_DIR);
    reloadMock.mockRejectedValue(new GraphMissingError('demo'));
    tree.unstaged.push(gitFile(GRAPH));
    await openPanel();

    await discardTheGraph();
    await takeTheOffer();

    expect(toasts().map((toast) => toast.message)).toContain(
      'demo no longer exists on this branch.',
    );
    // The tab keeps what it is showing, which is now the only copy of it.
    expect(useTabStore.getState().getTab(tabId)).toBeDefined();
  });

  it('parks focus in the message box when the discarded row was the last one', async () => {
    // No tab is bound to this file, so the panel empties and nothing else
    // happens: the group that owned the focused button unmounts in the same
    // beat, and without a fallback focus falls to the document body.
    tree.unstaged.push(gitFile('src/train.py'));
    await openPanel();

    fireEvent.click(
      within(row('train.py')).getByRole('button', { name: 'Discard Changes' }),
    );
    await settle(A_FRAME);

    expect(screen.getByText('No changes')).toBeTruthy();
    expect(document.activeElement).toBe(messageBox());
  });
});

/* ── what a refusal does to the panel ────────────────────────────────── */

describe('Source Control: refusals', () => {
  it('opens the identity form, and saves only the half that was filled in', async () => {
    tree.staged.push(gitFile(GRAPH));
    api.gitCommit.mockRejectedValue(
      await coded(409, 'identity_missing', 'Please tell me who you are'),
    );
    api.setGitConfig.mockImplementation(async () =>
      identity({ email: 'ada@example.com', email_scope: 'local' }),
    );
    await openPanel();

    fireEvent.change(messageBox(), { target: { value: 'first commit' } });
    fireEvent.click(screen.getByRole('button', { name: 'Commit' }));
    await settle();

    expect(screen.getByText('Commit identity')).toBeTruthy();
    expect(screen.getByRole('alert').textContent).toContain(
      'Set your name and email before committing.',
    );

    fireEvent.change(screen.getByLabelText('Email'), {
      target: { value: 'ada@example.com' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    await settle();

    // Only the half that was filled in: an absent key means "leave that one
    // alone", and an empty string is a value the server refuses outright.
    expect(api.setGitConfig).toHaveBeenCalledWith({ email: 'ada@example.com' });
    // Nothing retries the commit. The form closes and the user presses Commit
    // again, which is the only moment they can still change their mind.
    expect(api.gitCommit).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Commit identity')).toBeNull();
  });

  it('says another git action is running, and puts nothing in the error line', async () => {
    tree.unstaged.push(gitFile(GRAPH));
    api.gitStage.mockRejectedValue(await coded(409, 'busy', 'a commit is running'));
    await openPanel();

    fireEvent.click(within(row('demo.graph.json')).getByRole('button', { name: 'Stage' }));
    await settle(A_FRAME);

    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe('Another git action is still running.');
    expect(toasts()[0].type).toBe('warning');
    // It is over in a second and the button is still there to press again, so
    // there is nothing for the header to keep saying.
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('shows the timeout sentence the store wrote, with the deadline that applied', async () => {
    tree.unstaged.push(gitFile(GRAPH));
    api.gitStage.mockRejectedValue(await coded(504, 'timeout', 'git took too long'));
    await openPanel();

    fireEvent.click(within(row('demo.graph.json')).getByRole('button', { name: 'Stage' }));
    await settle(A_FRAME);

    // The 504 body carries a code and nothing else, so the number is the
    // store's -- 30s, the write bucket -- and the header shows it as written.
    expect(screen.getByRole('alert').textContent).toContain(
      'git did not finish within 30s.',
    );
    expect(screen.getByRole('alert').textContent).not.toContain('git took too long');
  });

  it('keeps the panel and reports a status that stopped answering', async () => {
    tree.unstaged.push(gitFile(GRAPH));
    await openPanel();
    expect(within(group('Changes')).getByText('demo.graph.json')).toBeTruthy();

    api.getGitStatus.mockRejectedValue(new Error('Failed to fetch'));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await settle();

    expect(screen.getByRole('alert').textContent).toContain(
      'Could not read repository status: Failed to fetch',
    );
    // The panel stays. The last status that answered is stale, but it is also
    // the best information there is, and a blank tab says less than that.
    expect(within(group('Changes')).getByText('demo.graph.json')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Commit' })).toBeTruthy();
  });
});

/* ── the density pass, as far as jsdom can take it ───────────────────── */

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (...parts: string[]): string => readFileSync(join(HERE, ...parts), 'utf8');

/** Blank the comments, keeping every line and column where it was. */
const uncommented = (css: string): string =>
  css.replace(/\/\*[\s\S]*?\*\//g, (block) => block.replace(/[^\n]/g, ' '));

const panelCss = uncommented(read('SourceControl.module.css'));
const shellCss = uncommented(read('..', 'Sidebar', 'NodePalette.module.css'));
const tokensCss = read('..', '..', 'styles', 'tokens.css');

interface Declaration {
  line: number;
  property: string;
  value: string;
}

/** Every `property: value;` in the file. Neither stylesheet wraps one. */
function declarations(css: string): Declaration[] {
  const found: Declaration[] = [];
  css.split('\n').forEach((text, index) => {
    const match = /^\s*([a-z-]+)\s*:\s*([^;]+);/.exec(text);
    if (match !== null) {
      found.push({ line: index + 1, property: match[1], value: match[2].trim() });
    }
  });
  return found;
}

const COLOUR_LITERAL = /#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\s*\(/;

const COLOUR_PROPERTY =
  /^(?:color|background|background-color|border|border-(?:top|right|bottom|left|color)|outline|outline-color|box-shadow|fill|stroke|caret-color|accent-color|text-decoration-color)$/;

/** A value that draws no colour at all, so it needs no token. */
const NO_COLOUR = /^(?:0|none|inherit|currentcolor|initial|unset)$|\btransparent$/;

const fontSizes = (css: string): Declaration[] =>
  declarations(css).filter((one) => one.property === 'font-size');

/**
 * The width pass, which jsdom cannot do.
 *
 * A panel is 180 / 250 / 520 px wide and nothing in this environment has a
 * width at all, so "does it fit" belongs to the Chrome pass. What CAN be
 * checked here is the two rules that decide whether it fits honestly:
 * `tokens.css` is the only source of colour, and density comes from padding
 * rather than from shrinking the text -- both of which are properties of the
 * stylesheet as text.
 */
describe('SourceControl.module.css: colour and type come from the tokens', () => {
  it('names no colour of its own', () => {
    const offenders = panelCss
      .split('\n')
      .map((text, index) => ({ line: index + 1, text: text.trim() }))
      .filter((one) => COLOUR_LITERAL.test(one.text));
    expect(offenders).toEqual([]);
  });

  it('takes every colour from a token, or draws none at all', () => {
    const offenders = declarations(panelCss).filter(
      (one) =>
        COLOUR_PROPERTY.test(one.property)
        && !one.value.includes('var(--')
        && !NO_COLOUR.test(one.value.toLowerCase()),
    );
    expect(offenders).toEqual([]);
  });

  it('sizes its text from the scale, never with a length of its own', () => {
    const offenders = fontSizes(panelCss).filter(
      (one) => !/^var\(--fs-[a-z0-9]+\)$/.test(one.value),
    );
    expect(offenders).toEqual([]);
  });

  it('goes no smaller than the sidebar it is drawn inside', () => {
    const scale = new Map<string, number>();
    for (const match of tokensCss.matchAll(/--(fs-[a-z0-9]+):\s*([\d.]+)rem;/g)) {
      scale.set(match[1], Number(match[2]));
    }
    const smallest = (css: string): number =>
      Math.min(
        ...fontSizes(css).map((one) => {
          const name = /^var\(--(fs-[a-z0-9]+)\)$/.exec(one.value)?.[1];
          const rem = name === undefined ? undefined : scale.get(name);
          if (rem === undefined) throw new Error(`not a type-scale step: ${one.value}`);
          return rem;
        }),
      );
    // Density is padding's job. A tab that reached for a smaller step than the
    // panel around it uses would be buying its rows with the reader's eyes.
    expect(smallest(panelCss)).toBeGreaterThanOrEqual(smallest(shellCss));
  });
});
