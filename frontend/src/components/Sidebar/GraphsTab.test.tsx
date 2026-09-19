import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import { GraphsTab, graphMatches, sortGraphs } from './GraphsTab';
import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useProjectStore } from '../../store/projectStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import {
  announceGraphsWrite,
  getGraphsWriteListener,
  setGraphsWriteListener,
} from '../../utils/graphsWrite';
import { setWorktreeWriteListener } from '../../utils/worktreeWrite';
import { confirm, prompt } from '../../utils/dialog';
import { importFile } from '../../utils/importGraphFile';
import { saveActiveGraph } from '../../utils/saveActiveGraph';
import * as rest from '../../api/rest';
import type { SavedGraphSummary } from '../../api/rest';
import type { NodeData } from '../../types';

// The list, rename and delete routes are stubbed, and the read one row click
// makes is `fetch` (see `stubGraphRead`). Everything between the click and
// the new tab -- `readSavedGraphDocument`, `resolveSavedGraph`,
// `loadGraphDocumentInto` -- runs for real, because "a row click opens the
// graph in a tab of its own" is a fact about the tab store, not about which
// function was called.
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return {
    ...actual,
    listGraphs: vi.fn(),
    deleteGraph: vi.fn(),
    renameGraph: vi.fn(),
  };
});
vi.mock('../../utils/dialog', () => ({ confirm: vi.fn(), prompt: vi.fn() }));
vi.mock('../../utils/importGraphFile', () => ({ importFile: vi.fn() }));
vi.mock('../../utils/saveActiveGraph', () => ({ saveActiveGraph: vi.fn() }));

const mockedRest = vi.mocked(rest);
const mockedConfirm = vi.mocked(confirm);
const mockedPrompt = vi.mocked(prompt);
const mockedImport = vi.mocked(importFile);
const mockedSaveAs = vi.mocked(saveActiveGraph);

const NOW_SECONDS = Math.floor(Date.now() / 1000);

function graph(overrides: Partial<SavedGraphSummary> = {}): SavedGraphSummary {
  return { name: 'alpha', file: 'alpha', modified: NOW_SECONDS, ...overrides };
}

/**
 * Answer the read a row click makes.
 *
 * `readSavedGraphDocument` goes to `fetch` rather than to `rest.loadGraph`,
 * because it is the only reader that can tell a deleted file (404) apart
 * from a broken server -- so stubbing the rest client would not reach it.
 */
function stubGraphRead(body: unknown = { nodes: [], edges: [] }, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: 'mock',
    json: async () => body,
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/**
 * The same read, but this test decides when it answers.
 *
 * The window a double click falls into is between the click and the read
 * coming back, and `stubGraphRead`'s already-resolved promise closes it
 * before a second click can land -- so a bug that puts two tabs on one file
 * passes against it. Holding the read open is the only way to put both
 * clicks inside one window on purpose.
 */
function deferredGraphRead(body: unknown = { nodes: [], edges: [] }) {
  let open: () => void = () => {};
  const gate = new Promise<void>((resolve) => { open = resolve; });
  const fetchMock = vi.fn().mockImplementation(async () => {
    await gate;
    return { ok: true, status: 200, statusText: 'mock', json: async () => body };
  });
  vi.stubGlobal('fetch', fetchMock);
  return { fetchMock, answer: () => { open(); } };
}

/** One node on the canvas, which is the work a careless open would cost. */
function someNode(): Node<NodeData> {
  return {
    id: 'n1',
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: 'Linear', type: 'Linear', params: {} },
  } as Node<NodeData>;
}

function freshTab() {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  useTabStore.getState().addTab('Tab 1');
}

const activeTab = () => useTabStore.getState().getActiveTab();
const tabById = (id: string) => useTabStore.getState().tabs.find((tb) => tb.id === id)!;

/**
 * A list read this test decides the answer to, and when.
 *
 * Every mutation refetches, so two reads are in flight together as a matter
 * of course; the ones that matter here are the pairs that answer out of
 * order.
 */
function deferredList() {
  let settle: (rows: SavedGraphSummary[]) => void = () => {};
  const promise = new Promise<SavedGraphSummary[]>((resolve) => { settle = resolve; });
  return { promise, settle: (rows: SavedGraphSummary[]) => { settle(rows); } };
}

/** Let a settled read reach the component, or be discarded by it. */
async function flush() {
  await act(async () => {
    await new Promise((resolve) => { setTimeout(resolve, 0); });
  });
}

/** The rows on screen, top to bottom, by the name each one shows. */
function rowNames(): string[] {
  return screen
    .getAllByRole('button', { name: /^Open / })
    .map((el) => (el.getAttribute('aria-label') ?? '').replace(/^Open /, ''));
}

/** Open one row's overflow menu and hand back the menu element. */
function openRowMenu(name: string): HTMLElement {
  fireEvent.click(screen.getByRole('button', { name: `More actions ${name}` }));
  return screen.getByRole('menu', { name: `More actions ${name}` });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: true });
  localStorage.clear();
  freshTab();
  vi.clearAllMocks();
  mockedRest.listGraphs.mockResolvedValue([]);
  mockedRest.deleteGraph.mockResolvedValue({});
  mockedRest.renameGraph.mockResolvedValue({});
  mockedConfirm.mockResolvedValue(true);
  mockedPrompt.mockResolvedValue(null);
  mockedImport.mockResolvedValue(true);
  mockedSaveAs.mockResolvedValue(undefined);
});

afterEach(() => {
  setWorktreeWriteListener(null);
  // Both signal modules are module-level singletons, so a listener left
  // installed by an unmounted panel would still be holding that render's
  // `load` when the next test announces a write.
  setGraphsWriteListener(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('sortGraphs', () => {
  it('puts the most recently modified first', () => {
    const sorted = sortGraphs([
      graph({ name: 'old', file: 'old', modified: 100 }),
      graph({ name: 'new', file: 'new', modified: 300 }),
      graph({ name: 'mid', file: 'mid', modified: 200 }),
    ]);
    expect(sorted.map((g) => g.name)).toEqual(['new', 'mid', 'old']);
  });

  it('falls back to the name when nothing is dated', () => {
    const sorted = sortGraphs([
      graph({ name: 'zeta', file: 'zeta', modified: undefined }),
      graph({ name: 'alpha', file: 'alpha', modified: undefined }),
    ]);
    expect(sorted.map((g) => g.name)).toEqual(['alpha', 'zeta']);
  });

  it('leads with the dated rows when only some carry a timestamp', () => {
    const sorted = sortGraphs([
      graph({ name: 'undated', file: 'undated', modified: undefined }),
      graph({ name: 'dated', file: 'dated', modified: 100 }),
    ]);
    expect(sorted.map((g) => g.name)).toEqual(['dated', 'undated']);
  });
});

describe('graphMatches', () => {
  it('matches the display name and the sanitized file stem', () => {
    const g = graph({ name: 'My Graph', file: 'My_Graph' });
    expect(graphMatches(g, 'my graph')).toBe(true);
    expect(graphMatches(g, 'my_graph')).toBe(true);
    expect(graphMatches(g, 'other')).toBe(false);
  });
});

describe('GraphsTab list', () => {
  it('renders the graphs most recently modified first', async () => {
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'oldest', file: 'oldest', modified: NOW_SECONDS - 7200 }),
      graph({ name: 'newest', file: 'newest', modified: NOW_SECONDS - 60 }),
      graph({ name: 'middle', file: 'middle', modified: NOW_SECONDS - 3600 }),
    ]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open newest' });
    expect(rowNames()).toEqual(['newest', 'middle', 'oldest']);
  });

  it('shows when each graph was last written', async () => {
    mockedRest.listGraphs.mockResolvedValue([
      graph({ modified: NOW_SECONDS - 3600 }),
    ]);
    render(<GraphsTab />);
    expect(await screen.findByText('Modified 1 hour ago')).toBeTruthy();
  });

  it('marks the row the active tab is bound to', async () => {
    useTabStore.getState().setCurrentGraphFile('beta', 'beta');
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'alpha', file: 'alpha' }),
      graph({ name: 'beta', file: 'beta' }),
    ]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open beta' });
    // Exactly one row carries it, and it is the bound one.
    const marks = screen.getAllByText('Current');
    expect(marks).toHaveLength(1);
    expect(marks[0].closest('li')?.textContent).toContain('beta');
  });

  it('says which row is bound to a reader who cannot see the chip', async () => {
    useTabStore.getState().setCurrentGraphFile('beta', 'beta');
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'alpha', file: 'alpha' }),
      graph({ name: 'beta', file: 'beta' }),
    ]);
    render(<GraphsTab />);
    const bound = await screen.findByRole('button', { name: 'Open beta' });
    const chip = screen.getByText('Current');
    // Inside the row button, the chip was content that button's `aria-label`
    // replaced -- on screen and announced to nobody.
    expect(chip.closest('button')).toBeNull();
    expect(chip.closest('li')).toBe(bound.closest('li'));
    expect(bound).toHaveAccessibleDescription(/Current/);
    expect(screen.getByRole('button', { name: 'Open alpha' }))
      .not.toHaveAccessibleDescription(/Current/);
  });

  it('spends two characters on the chip in zh-TW, as the branch list does', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    // Twice this wide left about 59px for the name at the 180px sidebar
    // floor. `git.branch.current` is the same two characters.
    expect(await screen.findByText('目前')).toBeTruthy();
  });

  it('filters on the name and on the file stem', async () => {
    mockedRest.listGraphs.mockResolvedValue([
      graph({ name: 'My Graph', file: 'My_Graph', modified: NOW_SECONDS }),
      graph({ name: 'classifier', file: 'classifier', modified: NOW_SECONDS - 10 }),
    ]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open My Graph' });
    const search = screen.getByPlaceholderText('Search saved graphs...');

    fireEvent.change(search, { target: { value: 'classi' } });
    expect(rowNames()).toEqual(['classifier']);

    // The sanitized stem is what the file is actually called, and someone who
    // typed it meant the graph whose display name still has the space in it.
    fireEvent.change(search, { target: { value: 'my_graph' } });
    expect(rowNames()).toEqual(['My Graph']);

    fireEvent.change(search, { target: { value: 'nothing' } });
    expect(screen.getByText('No matching graphs')).toBeTruthy();
  });

  it('does not echo the query back into the no-match line', async () => {
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    // One unbroken token, which is what a paste is: repeating it into a
    // centred message inside a panel that clips leaves it nowhere to wrap.
    const pasted = 'z'.repeat(200);
    fireEvent.change(screen.getByPlaceholderText('Search saved graphs...'), {
      target: { value: pasted },
    });
    expect(screen.getByText('No matching graphs')).toBeTruthy();
    expect(screen.queryByText(pasted, { exact: false })).toBeNull();
  });

  it('refreshing re-reads the list', async () => {
    mockedRest.listGraphs
      .mockResolvedValueOnce([graph({ name: 'before', file: 'before' })])
      .mockResolvedValueOnce([graph({ name: 'after', file: 'after' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open before' });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh the graph list' }));
    expect(await screen.findByRole('button', { name: 'Open after' })).toBeTruthy();
  });

  it('shows one line and the import action when there is nothing saved', async () => {
    render(<GraphsTab />);
    expect(await screen.findByText('No saved graphs')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Import...' })).toBeTruthy();
    // Nothing else: no list, no explanation of what a saved graph is.
    expect(screen.queryByRole('list')).toBeNull();
  });

  it('reports a failed list read and retries it', async () => {
    mockedRest.listGraphs.mockRejectedValueOnce(new Error('server down'));
    mockedRest.listGraphs.mockResolvedValueOnce([graph({ name: 'back', file: 'back' })]);
    render(<GraphsTab />);
    expect(await screen.findByText('Failed to load graphs: server down')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByRole('button', { name: 'Open back' })).toBeTruthy();
  });
});

describe('GraphsTab opening', () => {
  it('a row click opens the graph in a tab of its own', async () => {
    const fetchMock = stubGraphRead({ nodes: [], edges: [], description: 'from disk' });
    useTabStore.getState().setNodes([someNode()]);
    const wasActive = useTabStore.getState().activeTabId;
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    expect(fetchMock).toHaveBeenCalledWith('/api/graph/load/alpha');
    const opened = activeTab();
    expect(opened.id).not.toBe(wasActive);
    expect(opened.name).toBe('alpha');
    expect(opened.description).toBe('from disk');
    // Bound, so Save writes straight back over the file that was opened.
    expect(opened.currentGraphFile).toBe('alpha');
  });

  it('binds the new tab to the file AND to the name the file carries', async () => {
    // The stem is what addresses the file; the name is what an in-place save
    // writes back INTO it. Opened with the stem alone, the first press of the
    // toolbar's save icon renamed "My Graph" to "My_Graph" for good.
    stubGraphRead({ nodes: [], edges: [], name: 'My Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'My Graph', file: 'My_Graph' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open My Graph' }));
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    const opened = activeTab();
    expect(opened.currentGraphFile).toBe('My_Graph');
    expect(opened.currentGraphName).toBe('My Graph');
  });

  it('leaves the canvas the user was working in exactly as it was', async () => {
    stubGraphRead();
    useTabStore.getState().setNodes([someNode()]);
    useTabStore.getState().setCurrentGraphFile('other', 'other');
    const wasActive = useTabStore.getState().activeTabId;
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    expect(tabById(wasActive).nodes).toHaveLength(1);
    expect(tabById(wasActive).currentGraphFile).toBe('other');
  });

  it('never asks to replace anything, because it replaces nothing', async () => {
    stubGraphRead();
    useTabStore.getState().setNodes([someNode()]);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    // A row the user scrolled past costs them nothing, so there is no
    // question in the way of the row they aimed at.
    expect(mockedConfirm).not.toHaveBeenCalled();
  });

  it('raises the tab that already holds the graph instead of opening a second one', async () => {
    const fetchMock = stubGraphRead();
    // Tab 1 opened alpha and has unsaved work in it; the user is in tab 2.
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    useTabStore.getState().setNodes([someNode()]);
    const holder = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useTabStore.getState().activeTabId).toBe(holder);
    });
    // Two tabs bound to one file is two Saves overwriting each other.
    expect(useTabStore.getState().tabs).toHaveLength(2);
    // And the file is not re-read into it either: the tab may be holding
    // edits that are not on disk yet, and nobody asked for them to go.
    expect(fetchMock).not.toHaveBeenCalled();
    expect(tabById(holder).nodes).toHaveLength(1);
  });

  it('switching to a background tab says nothing, because the canvas already has', async () => {
    stubGraphRead();
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    const holder = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useTabStore.getState().activeTabId).toBe(holder);
    });
    await flush();
    // A whole different canvas is now in front of the user. A toast on top
    // of that is one more thing to read for no fact the screen did not give.
    expect(useToastStore.getState().toasts).toHaveLength(0);
  });

  it('clicking the row the active tab already holds says so, and does nothing else', async () => {
    const fetchMock = stubGraphRead();
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    const wasActive = useTabStore.getState().activeTabId;
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    // Raising the tab you are already on moves nothing at all, and a click
    // that produces no response gets made again.
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: '"alpha" is already open in this tab.',
        type: 'info',
      });
    });
    await flush();
    expect(useTabStore.getState().tabs).toHaveLength(1);
    expect(useTabStore.getState().activeTabId).toBe(wasActive);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('two clicks inside one read open one tab, not two', async () => {
    const read = deferredGraphRead();
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    const row = await screen.findByRole('button', { name: 'Open alpha' });
    // Both land before the read answers, so neither can see a tab bound to
    // the file: the first has not created one yet.
    fireEvent.click(row);
    fireEvent.click(row);
    read.answer();
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    await flush();
    // Two tabs on one file is two Saves silently overwriting each other,
    // and the toolbar's Save no longer stops to ask.
    expect(useTabStore.getState().tabs.filter((tb) => tb.currentGraphFile === 'alpha'))
      .toHaveLength(1);
    expect(useTabStore.getState().tabs).toHaveLength(2);
    // The second click is turned away at the door rather than answered with
    // a second read of the same file.
    expect(read.fetchMock).toHaveBeenCalledTimes(1);
  });

  it('raises a tab bound to the file DURING the read instead of duplicating it', async () => {
    const read = deferredGraphRead();
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));

    // While the read is in the air, something that is not this row binds a
    // tab to the same file -- an import, a Save As, a Source Control reload.
    // The in-flight mark only knows about clicks on the row.
    act(() => {
      useTabStore.getState().addTab('Tab 2');
      useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    });
    const raced = useTabStore.getState().activeTabId;
    // And the user has moved on again, so raising that tab is a real switch.
    act(() => { useTabStore.getState().addTab('Tab 3'); });

    read.answer();
    await waitFor(() => {
      expect(useTabStore.getState().activeTabId).toBe(raced);
    });
    await flush();
    expect(useTabStore.getState().tabs.filter((tb) => tb.currentGraphFile === 'alpha'))
      .toHaveLength(1);
    expect(useTabStore.getState().tabs).toHaveLength(3);
  });

  it('settles a race onto the tab in front WITHOUT announcing it', async () => {
    const read = deferredGraphRead();
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));

    // While the read is in the air, the tab in FRONT becomes bound to the
    // same file. In the field this is the panel remounting mid-read when the
    // user switches sidebar tabs: the in-flight Set dies with the instance,
    // the remounted panel issues a second read, and the losing continuation
    // arrives to find the tab the winner has just created.
    act(() => { useTabStore.getState().setCurrentGraphFile('alpha', 'alpha'); });
    const wasActive = useTabStore.getState().activeTabId;

    read.answer();
    await flush();

    // One click, one answer. The post-read check is a race being resolved,
    // not a reply to the click: said out loud it reports "alpha is already
    // open in this tab" to somebody who pressed the row exactly once. The
    // pre-read check is the one that answers a click, and it still speaks --
    // see 'clicking the row the active tab already holds says so'.
    expect(useToastStore.getState().toasts).toEqual([]);
    expect(useTabStore.getState().tabs).toHaveLength(1);
    expect(useTabStore.getState().activeTabId).toBe(wasActive);
  });

  it('a failed read leaves the row clickable, so trying again tries again', async () => {
    const fetchMock = stubGraphRead({}, 500);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    const row = await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(row);
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({ type: 'error' });
    });
    // The in-flight mark is cleared in a `finally`: left behind by the throw,
    // it would make this row refuse every click for the rest of the session,
    // with nothing on screen to say why.
    fireEvent.click(row);
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
  });

  it('a failed read says so and leaves no empty tab behind', async () => {
    stubGraphRead({}, 500);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({ type: 'error' });
    });
    expect(useToastStore.getState().toasts[0].message).toMatch(/Load failed/);
    // The tab is created only once the read has answered, so a file that
    // cannot be read leaves nothing for the user to close.
    expect(useTabStore.getState().tabs).toHaveLength(1);
  });

  it('opens a file written by a newer build read-only, and says so', async () => {
    stubGraphRead({ nodes: [], edges: [], format_version: 99 });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    expect(activeTab().readOnly).toBe(true);
    expect(useToastStore.getState().toasts.some(
      (toast) => toast.type === 'warning' && toast.message.includes('v99'),
    )).toBe(true);
  });

  it('stamps the open project onto the new tab', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    stubGraphRead();
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    fireEvent.click(await screen.findByRole('button', { name: 'Open alpha' }));
    await waitFor(() => {
      expect(useTabStore.getState().tabs).toHaveLength(2);
    });
    // The stamp is what the Source Control tab's affected-tab filter reads:
    // without it this tab sits outside every reload offer the project makes.
    expect(activeTab().projectOrigin).toBe('/proj');
  });
});

describe('GraphsTab row menu', () => {
  it('offers rename and delete, and no second way to open the graph', async () => {
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    const menu = openRowMenu('alpha');
    // The row itself is the only door in, so a menu entry that opened the
    // graph some OTHER way would be a second answer to a settled question.
    expect(within(menu).getAllByRole('menuitem').map((item) => item.textContent))
      .toEqual(['Rename', 'Delete']);
  });
});

describe('GraphsTab rename', () => {
  it('renames the file, re-reads the list and says so', async () => {
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({
      message: 'Graph renamed', name: 'Renamed Graph', file: 'Renamed_Graph',
    });
    mockedRest.listGraphs
      .mockResolvedValueOnce([graph({ name: 'alpha', file: 'alpha' })])
      .mockResolvedValueOnce([graph({ name: 'Renamed Graph', file: 'Renamed_Graph' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(mockedRest.renameGraph).toHaveBeenCalledWith('alpha', 'Renamed Graph');
    });
    expect(await screen.findByRole('button', { name: 'Open Renamed Graph' })).toBeTruthy();
    expect(useToastStore.getState().toasts[0]).toMatchObject({
      message: 'Renamed to "Renamed Graph".', type: 'success',
    });
  });

  it('moves the active tab\'s binding onto the new file', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({
      message: 'Graph renamed', name: 'Renamed Graph', file: 'Renamed_Graph',
    });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBe('Renamed_Graph');
    });
  });

  it('moves the binding of a tab that is not the one in front of the user', async () => {
    // Tab 1 opened alpha; the user is now in tab 2, which never saw it.
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    const background = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Renamed_Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    // Left on the old name, tab 1's next Save would write the graph to disk
    // a second time, under a file the user renamed away.
    await waitFor(() => {
      expect(tabById(background).currentGraphFile).toBe('Renamed_Graph');
    });
    expect(activeTab().currentGraphFile).toBeNull();
  });

  it('moves the bound NAME too, so the next save writes the new title', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Renamed_Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    // Moved onto the new file but left holding the old name, the tab writes
    // "alpha" back into `Renamed_Graph.json` the next time Save is pressed --
    // so the graph the user just renamed renames itself back, silently.
    await waitFor(() => {
      expect(activeTab().currentGraphName).toBe('Renamed Graph');
    });
    expect(activeTab().currentGraphFile).toBe('Renamed_Graph');
  });

  it('leaves a binding on a different file alone', async () => {
    useTabStore.getState().setCurrentGraphFile('beta', 'beta');
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Renamed_Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(mockedRest.renameGraph).toHaveBeenCalled();
    });
    expect(activeTab().currentGraphFile).toBe('beta');
  });

  it('cancelling the prompt writes nothing', async () => {
    mockedPrompt.mockResolvedValue(null);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(mockedPrompt).toHaveBeenCalled();
    });
    expect(mockedRest.renameGraph).not.toHaveBeenCalled();
  });

  it('reports a refused rename and keeps the list as it was', async () => {
    mockedPrompt.mockResolvedValue('beta');
    mockedRest.renameGraph.mockRejectedValue(new Error("Graph 'beta' already exists"));
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: "Rename failed: Graph 'beta' already exists", type: 'error',
      });
    });
    expect(screen.getByRole('button', { name: 'Open alpha' })).toBeTruthy();
  });

  /**
   * The state a rename can collide with, arranged.
   *
   * Tab A is bound to `Alpha`, a file the list no longer carries: the app
   * leaves bindings on files that are gone ON PURPOSE -- a Source Control
   * discard, checkout or stash pop reloads the affected tabs, and a tab
   * whose file has vanished keeps its binding so it goes on showing what it
   * holds. Tab C is bound to `Gamma`, is the tab in front of the user, and
   * is the row about to be renamed onto the name tab A still owns.
   */
  function gammaRenamedOntoAlpha(): { stale: string; renamed: string } {
    useTabStore.getState().setCurrentGraphFile('Alpha', 'Alpha');
    const stale = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab C');
    useTabStore.getState().setCurrentGraphFile('Gamma', 'Gamma');
    const renamed = useTabStore.getState().activeTabId;
    mockedPrompt.mockResolvedValue('Alpha');
    // The server allows it: its only guard is that the destination file does
    // not exist, and `Alpha` is the one that went missing under tab A.
    mockedRest.renameGraph.mockResolvedValue({ file: 'Alpha' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'Gamma', file: 'Gamma' })]);
    return { stale, renamed };
  }

  it('clears the binding of a tab that already held the new name', async () => {
    const { stale, renamed } = gammaRenamedOntoAlpha();
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open Gamma' });
    fireEvent.click(within(openRowMenu('Gamma')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(tabById(renamed).currentGraphFile).toBe('Alpha');
    });
    // The renamed graph keeps the name the user typed, as every other
    // rename does.
    expect(tabById(renamed).currentGraphName).toBe('Alpha');
    // Tab A's graph is still on screen; it simply has nowhere to save back
    // to, so its next Save asks for a name instead of writing over the graph
    // this rename just produced.
    expect(tabById(stale).currentGraphFile).toBeNull();
    expect(tabById(stale).currentGraphName).toBeNull();
  });

  it('leaves exactly one tab bound to the file after such a rename', async () => {
    gammaRenamedOntoAlpha();
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open Gamma' });
    fireEvent.click(within(openRowMenu('Gamma')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(mockedRest.renameGraph).toHaveBeenCalledWith('Gamma', 'Alpha');
    });
    // Two tabs on one physical file is the shape whose next one-click Save
    // destroys the other's work -- and it hides, because the panel draws one
    // row for the file and raises whichever tab comes first in the array.
    await waitFor(() => {
      expect(
        useTabStore.getState().tabs.filter((tb) => tb.currentGraphFile === 'Alpha'),
      ).toHaveLength(1);
    });
  });

  it('a rename that lands back on the same file leaves the tab bound', async () => {
    // "a b" and "a.b" both sanitize to `a_b`, as does any case-only rename on
    // a case-insensitive disk: the destination the clear above looks for is
    // held by the very tab the rebind is about to move, so clearing it would
    // unbind the graph the user just renamed.
    useTabStore.getState().setCurrentGraphFile('a_b', 'a b');
    const bound = useTabStore.getState().activeTabId;
    mockedPrompt.mockResolvedValue('a.b');
    mockedRest.renameGraph.mockResolvedValue({ file: 'a_b' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'a b', file: 'a_b' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open a b' });
    fireEvent.click(within(openRowMenu('a b')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(tabById(bound).currentGraphName).toBe('a.b');
    });
    expect(tabById(bound).currentGraphFile).toBe('a_b');
  });

  it('a rename onto a name no tab holds clears nobody', async () => {
    useTabStore.getState().setCurrentGraphFile('beta', 'beta');
    const other = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    mockedPrompt.mockResolvedValue('Delta');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Delta' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBe('Delta');
    });
    // The clear is aimed at the destination alone. A tab bound to some other
    // graph has nothing to do with this rename and must keep both halves of
    // its binding.
    expect(tabById(other).currentGraphFile).toBe('beta');
    expect(tabById(other).currentGraphName).toBe('beta');
  });

  it('a refused rename moves nothing and clears nothing', async () => {
    const { stale, renamed } = gammaRenamedOntoAlpha();
    mockedRest.renameGraph.mockRejectedValue(new Error("Graph 'Alpha' already exists"));
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open Gamma' });
    fireEvent.click(within(openRowMenu('Gamma')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({ type: 'error' });
    });
    // Nothing on disk moved, so nothing in the store may move either -- the
    // clear belongs after the server has answered, not before it is asked.
    expect(tabById(stale).currentGraphFile).toBe('Alpha');
    expect(tabById(stale).currentGraphName).toBe('Alpha');
    expect(tabById(renamed).currentGraphFile).toBe('Gamma');
  });

  it('project mode tells the Source Control tab about the write', async () => {
    useProjectStore.setState({ projectDir: '/proj', projectName: 'proj', loaded: true });
    const wrote = vi.fn();
    setWorktreeWriteListener(wrote);
    mockedPrompt.mockResolvedValue('Renamed Graph');
    mockedRest.renameGraph.mockResolvedValue({ file: 'Renamed_Graph' });
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Rename' }));
    await waitFor(() => {
      expect(wrote).toHaveBeenCalled();
    });
  });
});

describe('GraphsTab delete', () => {
  it('asks, deletes, re-reads the list and says so', async () => {
    mockedRest.listGraphs
      .mockResolvedValueOnce([graph({ name: 'alpha', file: 'alpha' })])
      .mockResolvedValueOnce([]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(mockedConfirm).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Delete "alpha"? The file is removed from disk.',
          variant: 'danger',
        }),
      );
    });
    await waitFor(() => {
      expect(mockedRest.deleteGraph).toHaveBeenCalledWith('alpha');
    });
    expect(await screen.findByText('No saved graphs')).toBeTruthy();
    expect(useToastStore.getState().toasts[0]).toMatchObject({
      message: 'Deleted "alpha".', type: 'success',
    });
  });

  it('cancelling deletes nothing', async () => {
    mockedConfirm.mockResolvedValue(false);
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(mockedConfirm).toHaveBeenCalled();
    });
    expect(mockedRest.deleteGraph).not.toHaveBeenCalled();
  });

  it('clears the active tab\'s binding when its own file is the one deleted', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'Alpha Graph');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(activeTab().currentGraphFile).toBeNull();
    });
    // Both halves. There is no graph on disk left for the name to name, and a
    // tab holding one is a tab whose next Save has to decide what it means.
    expect(activeTab().currentGraphName).toBeNull();
  });

  it('clears the binding of a tab that is not the one in front of the user', async () => {
    useTabStore.getState().setCurrentGraphFile('alpha', 'alpha');
    const background = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    // Otherwise a Ctrl+S in tab 1 recreates the file that was just deleted,
    // with no prompt and no collision check.
    await waitFor(() => {
      expect(tabById(background).currentGraphFile).toBeNull();
    });
  });

  it('reports a failed delete', async () => {
    mockedRest.deleteGraph.mockRejectedValue(new Error("Graph 'alpha' not found"));
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: "Delete failed: Graph 'alpha' not found", type: 'error',
      });
    });
  });
});

describe('GraphsTab refetching', () => {
  it('discards a list read that lands after a newer one', async () => {
    const stale = deferredList();
    mockedRest.listGraphs
      // Mount.
      .mockResolvedValueOnce([
        graph({ name: 'alpha', file: 'alpha' }),
        graph({ name: 'beta', file: 'beta', modified: NOW_SECONDS - 60 }),
      ])
      // The Save As refetch, still in flight when the delete happens.
      .mockImplementationOnce(() => stale.promise)
      // The delete's own refetch, which is the one that is true.
      .mockResolvedValueOnce([graph({ name: 'beta', file: 'beta' })]);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });

    fireEvent.click(screen.getByRole('button', { name: 'Save as...' }));
    await waitFor(() => {
      expect(mockedRest.listGraphs).toHaveBeenCalledTimes(2);
    });
    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(rowNames()).toEqual(['beta']);
    });

    // The read that started BEFORE the delete answers last, carrying a list
    // the deleted row is still in.
    stale.settle([
      graph({ name: 'alpha', file: 'alpha' }),
      graph({ name: 'beta', file: 'beta', modified: NOW_SECONDS - 60 }),
    ]);
    await flush();
    expect(rowNames()).toEqual(['beta']);
  });

  it('keeps the list on screen while a delete refetches, and blanks it for the refresh button', async () => {
    const afterDelete = deferredList();
    const afterRefresh = deferredList();
    mockedRest.listGraphs
      .mockResolvedValueOnce([
        graph({ name: 'alpha', file: 'alpha' }),
        graph({ name: 'beta', file: 'beta', modified: NOW_SECONDS - 60 }),
      ])
      .mockImplementationOnce(() => afterDelete.promise)
      .mockImplementationOnce(() => afterRefresh.promise);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });

    fireEvent.click(within(openRowMenu('alpha')).getByRole('menuitem', { name: 'Delete' }));
    await waitFor(() => {
      expect(mockedRest.listGraphs).toHaveBeenCalledTimes(2);
    });
    // One row went; the rest of the list -- and the reader's place in it --
    // stays where it was.
    expect(screen.queryByText('Loading graphs...')).toBeNull();
    expect(screen.getByRole('button', { name: 'Open beta' })).toBeTruthy();
    afterDelete.settle([graph({ name: 'beta', file: 'beta' })]);
    await flush();

    // The refresh button is a read the user asked for, and says so.
    fireEvent.click(screen.getByRole('button', { name: 'Refresh the graph list' }));
    expect(screen.getByText('Loading graphs...')).toBeTruthy();
    afterRefresh.settle([graph({ name: 'beta', file: 'beta' })]);
    await flush();
    expect(rowNames()).toEqual(['beta']);
  });
});

describe('GraphsTab live refresh', () => {
  it('a save announced from elsewhere re-reads the list, quietly', async () => {
    const afterSave = deferredList();
    mockedRest.listGraphs
      .mockResolvedValueOnce([graph({ name: 'alpha', file: 'alpha' })])
      .mockImplementationOnce(() => afterSave.promise);
    render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });

    // What the toolbar's Save does once the file is written. The panel and
    // the toolbar are on screen together, so the row has to appear without
    // the user going looking for the refresh button.
    act(() => { announceGraphsWrite(); });
    await waitFor(() => {
      expect(mockedRest.listGraphs).toHaveBeenCalledTimes(2);
    });
    // Quiet: the list stays readable while the new one is on its way.
    expect(screen.queryByText('Loading graphs...')).toBeNull();
    expect(screen.getByRole('button', { name: 'Open alpha' })).toBeTruthy();

    afterSave.settle([
      graph({ name: 'alpha', file: 'alpha' }),
      graph({ name: 'saved', file: 'saved' }),
    ]);
    await flush();
    expect(rowNames()).toContain('saved');
  });

  it('stops listening once the panel is gone', async () => {
    mockedRest.listGraphs.mockResolvedValue([graph({ name: 'alpha', file: 'alpha' })]);
    const { unmount } = render(<GraphsTab />);
    await screen.findByRole('button', { name: 'Open alpha' });
    unmount();

    expect(getGraphsWriteListener()).toBeNull();
    announceGraphsWrite();
    await flush();
    // A read that answers into an unmounted panel sets state on nothing.
    expect(mockedRest.listGraphs).toHaveBeenCalledTimes(1);
  });
});

describe('GraphsTab import and save', () => {
  it('the footer button proxies the click to the hidden file input, which takes both formats', async () => {
    const { container } = render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput.accept).toBe('.json,.cduiworkspace');
    const clickSpy = vi.spyOn(fileInput, 'click').mockImplementation(() => {});
    const button = screen.getByRole('button', { name: 'Import...' });
    // One button for two formats, so its title is where they are named.
    expect(button.title).toBe('A graph (.json) or a workspace (.cduiworkspace)');
    fireEvent.click(button);
    expect(clickSpy).toHaveBeenCalled();
  });

  it('a picked file goes to importFile, whichever format it is, and the input is cleared', async () => {
    const { container } = render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    const graphFile = new File(['{"nodes":[]}'], 'graph.json', { type: 'application/json' });
    fireEvent.change(fileInput, { target: { files: [graphFile] } });
    expect(mockedImport).toHaveBeenLastCalledWith(graphFile);
    // Cleared now, not after the read, so picking the SAME file again fires.
    expect(fileInput.value).toBe('');

    const workspaceFile = new File(['{"format":"codefyui-workspace"}'], 'w.cduiworkspace');
    fireEvent.change(fileInput, { target: { files: [workspaceFile] } });
    expect(mockedImport).toHaveBeenLastCalledWith(workspaceFile);
    expect(mockedImport).toHaveBeenCalledTimes(2);
  });

  it('picking nothing imports nothing', async () => {
    const { container } = render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [] } });
    expect(mockedImport).not.toHaveBeenCalled();
  });

  it('save-as runs the toolbar\'s Save As and re-reads the list', async () => {
    render(<GraphsTab />);
    await screen.findByText('No saved graphs');
    fireEvent.click(screen.getByRole('button', { name: 'Save as...' }));
    await waitFor(() => {
      expect(mockedSaveAs).toHaveBeenCalledWith({ saveAs: true });
    });
    await waitFor(() => {
      expect(mockedRest.listGraphs).toHaveBeenCalledTimes(2);
    });
  });
});
