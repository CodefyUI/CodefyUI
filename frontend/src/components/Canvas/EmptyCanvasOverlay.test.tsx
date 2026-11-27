import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react';
import { EmptyCanvasOverlay } from './EmptyCanvasOverlay';
import { useTabStore, type TabState } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { _resetPluginStoreForTesting, usePluginStore } from '../../store/pluginStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import * as utils from '../../utils';
import type { ExampleSummary } from '../../api/rest';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return {
    ...actual,
    listExamples: vi.fn(),
    listExampleTags: vi.fn(),
    loadExample: vi.fn(),
  };
});

vi.mock('../../utils', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../utils')>();
  return {
    ...actual,
    resolveSerializedNodes: vi.fn(() => [{ id: 'n1' } as any]),
    resolveSerializedEdges: vi.fn(() => [{ id: 'e1' } as any]),
  };
});

const mockedRest = vi.mocked(rest);
const mockedUtils = vi.mocked(utils);

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;

/**
 * Put the active tab inside a block, one level down, with nothing on that
 * level: the state a block is in after every node inside it was deleted.
 *
 * The frame holds the level the block was entered from, here the default
 * tab's own empty one. Asserted rather than annotated: what a frame records
 * is the store's business, and the overlay reads only how deep the stack is.
 */
function enterAnEmptyBlock() {
  const base = ORIGINAL_TABS[0];
  const frame = {
    subgraphId: 'sg-block',
    instanceId: 'blk',
    nodes: base.nodes,
    edges: base.edges,
    presets: base.presets,
    undoStack: base.undoStack,
    redoStack: base.redoStack,
    selectedNodeId: base.selectedNodeId,
    subgraphs: base.subgraphs,
    segmentGroups: base.segmentGroups,
    activeSegment: base.activeSegment,
  } as TabState['subgraphStack'][number];
  useTabStore.setState({
    tabs: [{ ...base, id: 'tab-block', nodes: [], edges: [], subgraphStack: [frame] }],
    activeTabId: 'tab-block',
  });
}

function ex(overrides: Partial<ExampleSummary> = {}): ExampleSummary {
  return {
    name: 'Example',
    description: 'short desc',
    category: 'Usage_Example',
    path: '/examples/foo.json',
    node_count: 3,
    edge_count: 2,
    source: 'builtin',
    ...overrides,
  };
}

/** Stand-ins for the canvas's drop handlers, which FlowCanvas hands the overlay (#526). */
function dropHandlers() {
  return { onDragOver: vi.fn(), onDrop: vi.fn() };
}

function renderOverlay(handlers = dropHandlers(), onDismiss?: () => void) {
  return render(<EmptyCanvasOverlay {...handlers} onDismiss={onDismiss} />);
}

/** A main-tag chip, by the start of its name (the count follows it). */
const tab = (name: RegExp) => screen.getByRole('tab', { name });

/** The section titles of the All view, in order. */
const sectionTitles = () =>
  [...document.querySelectorAll('[class*="sectionTitle"]')]
    .filter((el) => !el.className.includes('subsectionTitle'))
    .map((el) => el.textContent);

/**
 * Names of the example cards in DOM order. Filtered by class rather than
 * taken from every button, because the overlay also carries the
 * browse-the-gallery button (core#128), which is not a card. The "Blank
 * graph" card that always leads is left out; its own cases assert it.
 */
function cardNames(): (string | null | undefined)[] {
  return screen
    .getAllByRole('button')
    .filter((b) => b.className.includes('presetCard') && !b.className.includes('blankCard'))
    .map((b) => b.querySelector('span')?.textContent);
}

describe('EmptyCanvasOverlay', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useNodeDefStore.setState({ definitions: [], presets: [] });
    // An empty pack catalog unless a case seeds one, which is the state the
    // overlay renders in before the boot fetch lands.
    _resetPluginStoreForTesting();
    useToastStore.setState({ toasts: [] });
    mockedRest.listExamples.mockReset();
    mockedRest.loadExample.mockReset();
    // An older server: no tags route, so the order comes from the fallback.
    mockedRest.listExampleTags.mockReset();
    mockedRest.listExampleTags.mockRejectedValue(new Error('404'));
    // No chip remembered from an earlier case.
    window.localStorage.clear();
    mockedUtils.resolveSerializedNodes.mockClear();
    mockedUtils.resolveSerializedEdges.mockClear();
  });

  afterEach(() => {
    // Unmounted before the tabs go back: an overlay still on screen would
    // re-render out of the block and mount the gallery after its test ended.
    cleanup();
    vi.restoreAllMocks();
    useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
  });

  it('shows the loading hint before examples resolve, then hides it', async () => {
    let resolveList!: (v: ExampleSummary[]) => void;
    mockedRest.listExamples.mockReturnValue(
      new Promise<ExampleSummary[]>((res) => {
        resolveList = res;
      }),
    );
    renderOverlay();
    // title/subtitle always render
    expect(screen.getByText('Build your first deep learning model')).toBeInTheDocument();
    expect(screen.getByText('Pick an example')).toBeInTheDocument();
    // loading state visible
    expect(screen.getByText('Loading examples...')).toBeInTheDocument();

    resolveList([]);
    await waitFor(() => expect(screen.queryByText('Loading examples...')).toBeNull());
    // The overlay ends with the cards. "or drag a node from the left palette"
    // pointed at the node palette, which is open beside it carrying its own
    // pinned footer saying the same thing -- and which, unlike this overlay,
    // is still there once the canvas stops being empty.
    expect(screen.queryByText(/drag a node/i)).toBeNull();
  });

  it('falls back to an empty list when listExamples rejects', async () => {
    mockedRest.listExamples.mockRejectedValue(new Error('boom'));
    renderOverlay();
    await waitFor(() => expect(screen.queryByText('Loading examples...')).toBeNull());
    // No cards, but the heading and its one instruction still show.
    expect(screen.getByText('Build your first deep learning model')).toBeInTheDocument();
    expect(screen.getByText('Pick an example')).toBeInTheDocument();
  });

  it('opens on Quick start, with the main tags as chips and the node counts', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train MLP', category: 'Usage_Example', node_count: 5, main_tag: 'quickstart' }),
      ex({
        name: 'ResNet',
        category: 'Model_Architecture',
        path: '/a/resnet.json',
        main_tag: 'architectures',
      }),
    ]);
    renderOverlay();

    await waitFor(() => expect(screen.getByText('Train MLP')).toBeInTheDocument());
    expect(screen.getAllByRole('tab').map((el) => el.textContent)).toEqual([
      'All2', 'Quick start1', 'Architectures1',
    ]);
    expect(tab(/^Quick start/)).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', tab(/^Quick start/).id);
    // One main tag: its cards only, no section title (the chip says it).
    expect(cardNames()).toEqual(['Train MLP']);
    expect(sectionTitles()).toEqual([]);
    // The folder ("Usage Example") is no longer printed on a card.
    expect(screen.queryByText('Usage Example')).toBeNull();
    expect(screen.getByText('5 nodes')).toBeInTheDocument();
  });

  it('prints at most two tags on a card, in the UI language, never the main tag', async () => {
    act(() => useI18n.setState({ locale: 'zh-TW' }));
    mockedRest.listExamples.mockResolvedValue([
      ex({
        name: 'Train MLP',
        main_tag: 'quickstart',
        tags: ['quickstart', 'vision', 'cnn', 'augmentation', 'mnist'],
      }),
    ]);
    renderOverlay();

    await waitFor(() => expect(screen.getByText('Train MLP')).toBeInTheDocument());
    const chips = [...document.querySelectorAll('[class*="difficultyBadge"]')].map((el) => el.textContent);
    expect(chips).toEqual(['視覺', 'CNN']);
    const more = document.querySelector('[class*="moreTags"]')!;
    expect(more.textContent).toBe('+2');
    expect(more.getAttribute('title')).toContain('資料增強');
    expect(screen.queryByText('使用範例')).toBeNull();
  });

  it('switches with the chips and the arrow keys, and remembers the choice for the next tab', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train CNN', path: 'q/1', main_tag: 'quickstart' }),
      ex({ name: 'Train GPT', path: 't/1', main_tag: 'training' }),
      ex({ name: 'Dropout', path: 'p/1', main_tag: 'papers' }),
    ]);
    const { unmount } = renderOverlay();
    await waitFor(() => expect(screen.getByText('Train CNN')).toBeInTheDocument());

    fireEvent.click(tab(/^Training/));
    expect(cardNames()).toEqual(['Train GPT']);
    expect(tab(/^Training/)).toHaveAttribute('tabindex', '0');
    expect(tab(/^Quick start/)).toHaveAttribute('tabindex', '-1');

    fireEvent.keyDown(tab(/^Training/), { key: 'ArrowRight' });
    expect(tab(/^Papers/)).toHaveAttribute('aria-selected', 'true');
    expect(document.activeElement).toBe(tab(/^Papers/));
    expect(cardNames()).toEqual(['Dropout']);
    // Wraps around: right of the last chip is All.
    fireEvent.keyDown(tab(/^Papers/), { key: 'ArrowRight' });
    expect(tab(/^All/)).toHaveAttribute('aria-selected', 'true');
    fireEvent.keyDown(tab(/^All/), { key: 'ArrowLeft' });
    expect(tab(/^Papers/)).toHaveAttribute('aria-selected', 'true');
    unmount();

    // The next new tab opens on the chip chosen last.
    renderOverlay();
    await waitFor(() => expect(screen.getByText('Dropout')).toBeInTheDocument());
    expect(tab(/^Papers/)).toHaveAttribute('aria-selected', 'true');
    expect(cardNames()).toEqual(['Dropout']);
  });

  it('falls back to Quick start when the remembered chip has no examples, or storage is unreadable', async () => {
    window.localStorage.setItem('codefyui.examples.mainTag', 'deploy');
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train CNN', path: 'q/1', main_tag: 'quickstart' }),
      ex({ name: 'Train GPT', path: 't/1', main_tag: 'training' }),
    ]);
    const { unmount } = renderOverlay();
    await waitFor(() => expect(screen.getByText('Train CNN')).toBeInTheDocument());
    expect(tab(/^Quick start/)).toHaveAttribute('aria-selected', 'true');
    unmount();

    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    renderOverlay();
    await waitFor(() => expect(screen.getByText('Train CNN')).toBeInTheDocument());
    expect(tab(/^Quick start/)).toHaveAttribute('aria-selected', 'true');
    // Choosing still works; it is only not remembered.
    fireEvent.click(tab(/^Training/));
    expect(cardNames()).toEqual(['Train GPT']);
    expect(getItem).toHaveBeenCalled();
    expect(setItem).toHaveBeenCalled();
  });

  it('leads with a Blank graph card that dismisses the overlay', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Train CNN', main_tag: 'quickstart' })]);
    const onDismiss = vi.fn();
    renderOverlay(dropHandlers(), onDismiss);
    await waitFor(() => expect(screen.getByText('Train CNN')).toBeInTheDocument());

    const cards = screen.getAllByRole('button').filter((b) => b.className.includes('presetCard'));
    expect(cards[0]).toHaveTextContent('Blank graph');
    fireEvent.click(cards[0]);
    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(mockedRest.loadExample).not.toHaveBeenCalled();
  });

  it('still offers the Blank graph card when there are no examples', async () => {
    mockedRest.listExamples.mockResolvedValue([]);
    renderOverlay();
    expect(await screen.findByText('Blank graph')).toBeInTheDocument();
    expect(screen.queryByRole('tablist')).toBeNull();
  });

  it('renders the main tags in gallery order under All, papers and courses after the practical ones', async () => {
    // Deliberately shuffled: the order comes from the main-tag order, not
    // from where the backend happened to put each example.
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Misc', category: 'Something_Else', path: 'X/misc' }),
      ex({ name: 'Plugin Demo', category: 'Classical', path: 'plugin:c2/Classical/Foo', source: 'plugin:c2' }),
      ex({ name: 'Lesson', category: 'Classical', path: 'plugin:c3/Lesson', source: 'plugin:c3', main_tag: 'courses' }),
      ex({ name: 'ResNet Arch', category: 'Model_Architecture', path: 'm/resnet', main_tag: 'architectures', family: 'CNN' }),
      ex({ name: 'Iris', category: 'Classical', path: 'Classical/Iris', main_tag: 'concepts' }),
      ex({ name: 'Analogy', category: 'LLM', path: 'LLM/Analogy', main_tag: 'llm' }),
      ex({ name: 'Dropout', category: 'Papers', path: 'Papers/Dropout', main_tag: 'papers' }),
      ex({ name: 'Confusion', path: 'Eval/Confusion', main_tag: 'evaluation' }),
      ex({ name: 'Train GPT', category: 'Usage_Example', path: 'Usage_Example/GPT', main_tag: 'training' }),
      ex({ name: 'Train CNN', category: 'Usage_Example', path: 'Usage_Example/CNN', main_tag: 'quickstart' }),
    ]);
    renderOverlay();
    await waitFor(() => expect(screen.getByText('Train CNN')).toBeInTheDocument());
    fireEvent.click(tab(/^All/));

    expect(sectionTitles()).toEqual([
      'Quick start',
      'Training',
      'LLM & RAG',
      'Evaluation',
      'Architectures',
      'Concepts',
      'Papers',
      'Courses',
      'Plugin packs',
      'Other',
    ]);
    expect(cardNames()).toEqual([
      'Train CNN',
      'Train GPT',
      'Analogy',
      'Confusion',
      'ResNet Arch',
      'Iris',
      'Dropout',
      'Lesson',
      'Plugin Demo',
      'Misc',
    ]);
  });

  it('orders a section by the declared order, nulls last, ties as listed', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Unordered', path: 'c/unordered', main_tag: 'concepts' }),
      ex({ name: 'Third', path: 'c/third', main_tag: 'concepts', order: 3 }),
      ex({ name: 'First', path: 'c/first', main_tag: 'concepts', order: 1 }),
      ex({ name: 'Second', path: 'c/second', main_tag: 'concepts', order: 2 }),
    ]);
    renderOverlay();
    await waitFor(() => expect(screen.getByText('First')).toBeInTheDocument());
    expect(cardNames()).toEqual(['First', 'Second', 'Third', 'Unordered']);
  });

  it('gives the architecture families and the packs a sub-header', async () => {
    usePluginStore.setState({ byId: { c2: { id: 'c2', name: 'Chapter 2' } } as never });
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'ResNet', category: 'Model_Architecture', path: 'm/resnet', main_tag: 'architectures', family: 'CNN' }),
      ex({ name: 'LSTM', category: 'Model_Architecture', path: 'm/lstm', main_tag: 'architectures', family: 'RNN' }),
      ex({ name: 'Lesson', category: 'Classical', path: 'plugin:c2/Lesson', source: 'plugin:c2' }),
    ]);
    renderOverlay();
    // No Quick start examples: the overlay opens on All.
    await waitFor(() => expect(screen.getByText('ResNet')).toBeInTheDocument());
    expect(tab(/^All/)).toHaveAttribute('aria-selected', 'true');

    const subheads = [...document.querySelectorAll('[class*="subsectionTitle"]')].map(
      (el) => el.textContent,
    );
    expect(subheads).toEqual(['CNN', 'RNN', 'Chapter 2']);
    // One section title above the two families, not one per family.
    expect(sectionTitles()).toEqual(['Architectures', 'Plugin packs']);
  });

  it('files a legacy entry that sends only `section` under that section', async () => {
    // An older server, or a pack written before tags: no main_tag at all.
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Old Pack Demo', path: 'plugin:c2/Old', source: 'plugin:c2', section: 'training' }),
      ex({ name: 'Old Builtin', path: 'u/old', section: 'quickstart' }),
    ]);
    renderOverlay();
    await waitFor(() => expect(screen.getByText('Old Builtin')).toBeInTheDocument());
    fireEvent.click(tab(/^Training/));
    expect(cardNames()).toEqual(['Old Pack Demo']);
  });

  it('puts a built-in that declares no main tag under Other', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Misc Demo', category: 'Something_Else', path: '/x/misc.json' }),
    ]);
    renderOverlay();
    await waitFor(() => expect(screen.getByText('Misc Demo')).toBeInTheDocument());
    expect(sectionTitles()).toEqual(['Other']);
    expect(screen.getAllByRole('tab').map((el) => el.textContent)).toEqual(['All1', 'Other1']);
  });

  it('truncates descriptions longer than 40 columns', async () => {
    // 40, not the 80 this used to cut at: a description is allowed to be 40
    // columns wide in the first place (`MAX_DESCRIPTION_COLUMNS`, backend and
    // zh-TW), so a wider cut could only ever fire on text that already broke
    // the rule -- which is a third-party pack's description, never one of
    // ours.
    const long = 'x'.repeat(120);
    mockedRest.listExamples.mockResolvedValue([ex({ description: long })]);
    renderOverlay();
    await waitFor(() => expect(screen.getByText('Example')).toBeInTheDocument());
    expect(screen.getByText(`${'x'.repeat(40)}...`)).toBeInTheDocument();
  });

  it('leaves a description that fits the 40 columns uncut', async () => {
    // The other side of the cut, and the one that holds for everything this
    // repo ships: at exactly the cap the card shows the line whole, with no
    // ellipsis. Without this, narrowing the cut again would go unnoticed
    // until a reader found a sentence ending in "...".
    const exact = 'y'.repeat(40);
    mockedRest.listExamples.mockResolvedValue([ex({ description: exact })]);
    renderOverlay();
    expect(await screen.findByText(exact)).toBeInTheDocument();
  });

  it('keeps the whole description reachable as a tooltip (core#305)', async () => {
    // The cut only fires on a description that ran past the 40-column rule,
    // and that is exactly when the tooltip earns its place: this card is the
    // one place a description appears with no way to read the rest. The
    // sidebar's gallery tab already does this.
    const long = `${'x'.repeat(40)} and the part nobody could read`;
    mockedRest.listExamples.mockResolvedValue([ex({ description: long })]);
    renderOverlay();
    const shown = await screen.findByText(`${'x'.repeat(40)}...`);
    expect(shown).toHaveAttribute('title', long);
  });

  it('does not put a tooltip on a description that is already whole', async () => {
    // A native tooltip repeating the text under the cursor is noise.
    mockedRest.listExamples.mockResolvedValue([ex({ description: 'short desc' })]);
    renderOverlay();
    const shown = await screen.findByText('short desc');
    expect(shown).not.toHaveAttribute('title');
  });

  it('applies and clears hover styles on a card', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Hover Me' })]);
    renderOverlay();
    const card = (await screen.findByText('Hover Me')).closest('button') as HTMLButtonElement;

    // Hover styling now writes the CSS custom property directly instead of
    // a resolved literal (was '#D4A017' / '#3a3a3a'). jsdom does not
    // evaluate var() references on `.style`, so the raw token string is
    // what's reported here — and there's nothing to import instead: only
    // the semantic data palettes are mirrored into theme.ts as TS
    // constants (see theme.ts's file comment), not chrome tokens like
    // --status-preset / --border-base.
    fireEvent.mouseEnter(card);
    expect(card.style.borderColor).toBe('var(--status-preset)');
    // rgba(224,169,43,0.15) is --status-preset's own rgb() (#e0a92b) — no
    // glow token is paired with it, so this stays a hand-tuned literal at
    // the hue (see the component's own comment).
    expect(card.style.boxShadow).toBe('0 4px 16px rgba(224, 169, 43, 0.15)');

    fireEvent.mouseLeave(card);
    expect(card.style.borderColor).toBe('var(--border-base)');
    expect(card.style.boxShadow).toBe('none');
  });

  it('loads an example: resolves nodes/edges, merges new presets, and renames the tab', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Loadable' })]);
    mockedRest.loadExample.mockResolvedValue({
      name: '  My Model  ',
      nodes: [{ id: 'a' }],
      edges: [{ id: 'e' }],
      presets: [{ preset_name: 'NewPreset', nodes: [], edges: [] }],
    });
    // Existing preset to exercise the "already present, skip" branch.
    useNodeDefStore.setState({ definitions: [], presets: [{ preset_name: 'Existing' } as any] });

    // One spy, because the gallery path installs the whole document through
    // one store action now (#200 items 4 and 8) instead of sequencing
    // setNodes/setEdges/renameTab itself.
    const loadGraphDocument = vi.fn();
    useTabStore.setState({ loadGraphDocument });

    renderOverlay();
    fireEvent.click(await screen.findByText('Loadable'));

    await waitFor(() => expect(loadGraphDocument).toHaveBeenCalled());
    expect(mockedRest.loadExample).toHaveBeenCalledWith('/examples/foo.json');
    expect(loadGraphDocument).toHaveBeenCalledWith(
      expect.objectContaining({
        nodes: [{ id: 'n1' }],
        edges: [{ id: 'e1' }],
        // trimmed example name, carried on the document rather than applied
        // by a separate renameTab
        name: 'My Model',
      }),
    );
    // new preset merged into the store (Existing + NewPreset)
    await waitFor(() =>
      expect(useNodeDefStore.getState().presets.map((p) => p.preset_name)).toEqual([
        'Existing',
        'NewPreset',
      ]),
    );
  });

  it('skips a preset that already exists by name', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Dup' })]);
    mockedRest.loadExample.mockResolvedValue({
      name: 'Dup Model',
      nodes: [{ id: 'a' }],
      edges: [],
      presets: [{ preset_name: 'Shared' }],
    });
    useNodeDefStore.setState({ definitions: [], presets: [{ preset_name: 'Shared' } as any] });
    useTabStore.setState({ loadGraphDocument: vi.fn() });

    renderOverlay();
    fireEvent.click(await screen.findByText('Dup'));

    await waitFor(() =>
      // still just one "Shared" — the duplicate was not pushed
      expect(useNodeDefStore.getState().presets.filter((p) => p.preset_name === 'Shared')).toHaveLength(1),
    );
  });

  it('handles missing nodes/edges/presets and a blank name without renaming or merging', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Bare' })]);
    // No nodes/edges/presets keys; name is blank whitespace.
    mockedRest.loadExample.mockResolvedValue({ name: '   ' });

    const loadGraphDocument = vi.fn();
    useTabStore.setState({ loadGraphDocument });
    const before = useNodeDefStore.getState().presets;

    renderOverlay();
    fireEvent.click(await screen.findByText('Bare'));

    await waitFor(() => expect(loadGraphDocument).toHaveBeenCalled());
    // resolveSerializedNodes/Edges were called with [] fallbacks; the edges
    // resolver also receives the resolved nodes for per-data-type coloring.
    // The trailing [] is the example's own subgraph definitions (core#137):
    // an instance node's rendered ports come from its definition's
    // interface, so the resolver has to be given them.
    expect(mockedUtils.resolveSerializedNodes).toHaveBeenCalledWith([], [], expect.any(Array), []);
    expect(mockedUtils.resolveSerializedEdges).toHaveBeenCalledWith([], [{ id: 'n1' }]);
    // blank name => the document carries none, so the tab keeps its label
    expect(loadGraphDocument).toHaveBeenCalledWith(
      expect.objectContaining({ name: null }),
    );
    // no imported presets => store unchanged
    expect(useNodeDefStore.getState().presets).toBe(before);
  });

  it('treats a non-array presets field as no presets', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'WeirdPresets' })]);
    mockedRest.loadExample.mockResolvedValue({
      name: 'X',
      nodes: [],
      edges: [],
      presets: 'not-an-array',
    });
    useTabStore.setState({ loadGraphDocument: vi.fn() });
    const before = useNodeDefStore.getState().presets;

    renderOverlay();
    fireEvent.click(await screen.findByText('WeirdPresets'));

    await waitFor(() => expect(mockedUtils.resolveSerializedNodes).toHaveBeenCalled());
    // Non-array => importedPresets = [] => store untouched.
    expect(useNodeDefStore.getState().presets).toBe(before);
  });

  it('shows an error toast when loadExample throws', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Broken' })]);
    mockedRest.loadExample.mockRejectedValue(new Error('load failed'));
    const addToast = vi.fn();
    useToastStore.setState({ addToast });
    useTabStore.setState({ loadGraphDocument: vi.fn() });

    renderOverlay();
    fireEvent.click(await screen.findByText('Broken'));

    await waitFor(() => expect(addToast).toHaveBeenCalledWith('Failed to load example', 'error'));
  });

  it("gives a drag over the gallery, and a drop on it, to the canvas's handlers (#526)", async () => {
    // The gallery covers most of an empty canvas and is drawn beside
    // <ReactFlow>, not inside it. Without the canvas's handlers nothing
    // cancelled a dragover here, so the browser refused the drop: a node
    // dragged into the middle of a new tab was not added.
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Drop Target' })]);
    const handlers = dropHandlers();
    renderOverlay(handlers);
    const card = (await screen.findByText('Drop Target')).closest('button')!;

    fireEvent.dragOver(card);
    fireEvent.drop(card);
    expect(handlers.onDragOver).toHaveBeenCalledTimes(1);
    expect(handlers.onDrop).toHaveBeenCalledTimes(1);

    // Anywhere on the panel, not only on a card.
    fireEvent.dragOver(screen.getByText('Build your first deep learning model'));
    fireEvent.drop(screen.getByText('Pick an example'));
    expect(handlers.onDragOver).toHaveBeenCalledTimes(2);
    expect(handlers.onDrop).toHaveBeenCalledTimes(2);
  });

  it('cancels the dragenter onto each card as well (#526)', async () => {
    // On the frame the pointer crosses into a new element Chrome fires only a
    // dragenter, and an uncancelled one makes that frame's drop effect
    // "none". A drag over this grid of cards crosses one element after another.
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Drop Target' })]);
    const handlers = dropHandlers();
    // What the canvas's own dragover handler does first.
    handlers.onDragOver.mockImplementation((event: React.DragEvent) => event.preventDefault());
    renderOverlay(handlers);
    const card = (await screen.findByText('Drop Target')).closest('button')!;

    expect(fireEvent.dragEnter(card)).toBe(false);
    expect(fireEvent.dragEnter(screen.getByText('Drop Target'))).toBe(false);
    expect(handlers.onDragOver).toHaveBeenCalledTimes(2);
  });

  it('opens an example once when its card is double-clicked (#622)', async () => {
    // Both clicks land while the example is still loading. The first to
    // arrive filled the empty tab; the second found it filled and opened the
    // example again in a new one.
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Twice' })]);
    let arrive!: (payload: unknown) => void;
    mockedRest.loadExample.mockReturnValue(
      new Promise((resolve) => {
        arrive = resolve;
      }),
    );
    const loadGraphDocument = vi.fn();
    useTabStore.setState({ loadGraphDocument });
    renderOverlay();
    const card = (await screen.findByText('Twice')).closest('button')!;

    fireEvent.click(card);
    fireEvent.click(card);
    await act(async () => arrive({ name: 'Twice', nodes: [], edges: [] }));

    expect(mockedRest.loadExample).toHaveBeenCalledTimes(1);
    expect(loadGraphDocument).toHaveBeenCalledTimes(1);
  });

  describe('a level inside a block (#622)', () => {
    it('says the block is empty instead of showing the gallery', async () => {
      // Listed, so a gallery that did render would show a card rather than
      // fail on the missing promise.
      mockedRest.listExamples.mockResolvedValue([ex({ name: 'Starter' })]);
      enterAnEmptyBlock();
      renderOverlay();

      expect(screen.getByText('This subgraph is empty')).toBeInTheDocument();
      expect(
        screen.getByText('Back, above, returns to the graph around it.'),
      ).toBeInTheDocument();
      // A starter picked here would open in another tab: nothing of the
      // gallery belongs inside a block, not its heading, not the button to
      // the full gallery, not a card -- and nothing is fetched for it.
      await act(async () => {});
      expect(screen.queryByText('Build your first deep learning model')).toBeNull();
      expect(screen.queryByText('Starter')).toBeNull();
      expect(screen.queryAllByRole('button')).toHaveLength(0);
      expect(mockedRest.listExamples).not.toHaveBeenCalled();
    });

    it('hands a drag over the note, and a drop on it, to the canvas', () => {
      // A node dragged in from the palette lands inside the block.
      enterAnEmptyBlock();
      const handlers = dropHandlers();
      renderOverlay(handlers);

      fireEvent.dragOver(screen.getByText('This subgraph is empty'));
      fireEvent.drop(screen.getByText('Back, above, returns to the graph around it.'));
      expect(handlers.onDragOver).toHaveBeenCalledTimes(1);
      expect(handlers.onDrop).toHaveBeenCalledTimes(1);
    });

    it('in Traditional Chinese, names the block and the way back as the breadcrumb does', () => {
      useI18n.setState({ locale: 'zh-TW' });
      enterAnEmptyBlock();
      renderOverlay();

      expect(screen.getByText('這個子圖是空的')).toBeInTheDocument();
      expect(screen.getByText('按上方的「返回」回到外層。')).toBeInTheDocument();
    });

    it('leaves the gallery on an empty top level', async () => {
      mockedRest.listExamples.mockResolvedValue([ex({ name: 'Starter' })]);
      renderOverlay();

      expect(await screen.findByText('Starter')).toBeInTheDocument();
      expect(screen.getByText('Build your first deep learning model')).toBeInTheDocument();
      expect(screen.queryByText('This subgraph is empty')).toBeNull();
      expect(mockedRest.listExamples).toHaveBeenCalledTimes(1);
    });
  });
});
