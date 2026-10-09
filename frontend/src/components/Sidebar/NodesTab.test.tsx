import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within, act, waitFor } from '@testing-library/react';
import { NodesTab } from './NodesTab';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import { _resetPluginStoreForTesting, usePluginStore } from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { resolveExample } from '../../utils/openExample';
import type { PackSummary, PluginCatalogEntry } from '../../api/rest';
import { pluginEntry as catalogEntry } from '../../test/pluginEntry';
import type { NodeDefinition, PresetDefinition } from '../../types';

/*
 * The node-library behaviours that used to live in NodePalette.test.tsx,
 * migrated with #126 when the palette became rail + tabs. Everything here
 * asserts the SAME contract as before the split (search, ordering, accordions,
 * drag payload, tooltips, beginner mode) — only the component boundary moved.
 * What is genuinely new (the jump index, expand-all/collapse-all) is in its own
 * block at the bottom.
 */

function def(
  node_name: string,
  category: string,
  description = `${node_name} desc`,
): NodeDefinition {
  return {
    node_name,
    category,
    description,
    inputs: [],
    outputs: [],
    params: [],
  };
}

function preset(
  preset_name: string,
  category: string,
  over: Partial<PresetDefinition> = {},
): PresetDefinition {
  return {
    preset_name,
    category,
    description: `${preset_name} desc`,
    tags: ['beginner'],
    nodes: [],
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
    ...over,
  };
}

/** Seed the node-def store. This tab is a pure consumer — it never fetches. */
function seedStore(opts: {
  categorized?: Record<string, NodeDefinition[]>;
  presets?: PresetDefinition[];
  loading?: boolean;
  error?: string | null;
}) {
  const definitions = Object.values(opts.categorized ?? {}).flat();
  useNodeDefStore.setState({
    definitions,
    categorized: opts.categorized ?? {},
    // Written on every seed, not merged: the store outlives a test, so presets
    // left behind by one case must not still be on screen in the next.
    presets: opts.presets ?? [],
    loading: opts.loading ?? false,
    error: opts.error ?? null,
  });
}

/** The category headers, in render order, by their visible name. */
function categoryNames(container: HTMLElement): string[] {
  // aria-expanded is the accordion header's own attribute — the list toolbar
  // and the jump index carry an aria-label instead.
  return Array.from(container.querySelectorAll('button[aria-expanded]')).map(
    (header) => header.children[1].textContent ?? '',
  );
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ tooltipsEnabled: true, beginnerMode: false });
  // Every case that does not seed a catalog runs against an empty, unloaded
  // one — the base install, where no palette row says anything about packs
  // and nothing in the library came from a plugin.
  _resetPackStoreForTesting();
  _resetPluginStoreForTesting();
  seedStore({ categorized: {} });
  // A fresh mock per test, set through the store rather than vi.spyOn: set()
  // clones the state object, so a spy would outlive restoreAllMocks() and carry
  // its call history into the next test.
  useNodeDefStore.setState({ fetchDefinitions: vi.fn().mockResolvedValue(undefined) });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('NodesTab', () => {
  it('renders the title, search box, and footer hint', () => {
    render(<NodesTab />);
    expect(screen.getByText('Nodes')).toBeTruthy();
    expect(screen.getByPlaceholderText('Search nodes...')).toBeTruthy();
    expect(screen.getByText('Drag nodes onto the canvas')).toBeTruthy();
  });

  it('shows the loading state', () => {
    seedStore({ loading: true });
    render(<NodesTab />);
    expect(screen.getByText('Loading nodes...')).toBeTruthy();
  });

  it('shows the error state and retries on click', () => {
    seedStore({ error: 'network down' });
    const refetch = useNodeDefStore.getState().fetchDefinitions as ReturnType<
      typeof vi.fn
    >;
    render(<NodesTab />);
    expect(screen.getByText('Failed to load nodes: network down')).toBeTruthy();
    fireEvent.click(screen.getByText('Retry'));
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  // The catalog belongs to the whole app, and this tab mounts only while it is
  // the open one — so starting the load is the shell's job, not this tab's.
  it('never starts a catalog load itself, even with an empty store', () => {
    seedStore({ categorized: {} });
    const fetchDefinitions = useNodeDefStore.getState().fetchDefinitions as ReturnType<
      typeof vi.fn
    >;
    render(<NodesTab />);
    expect(fetchDefinitions).not.toHaveBeenCalled();
  });

  it('shows the empty state when there are no nodes and no search', () => {
    seedStore({ categorized: {} });
    render(<NodesTab />);
    expect(screen.getByText('No nodes available')).toBeTruthy();
  });

  it('renders categories in CATEGORY_ORDER, then unknown categories sorted', () => {
    seedStore({
      categorized: {
        Zebra: [def('ZNode', 'Zebra')], // unknown → sorted after ordered ones
        CNN: [def('Conv2d', 'CNN')], // ordered
        Data: [def('Dataset', 'Data')], // ordered (earlier)
        Apple: [def('ANode', 'Apple')], // unknown → sorted
      },
    });
    const { container } = render(<NodesTab />);
    const categoryNames = Array.from(
      container.querySelectorAll('button span'),
    )
      .map((s) => s.textContent)
      .filter((t) => ['Data', 'CNN', 'Apple', 'Zebra'].includes(t ?? ''));
    // Data and CNN (ordered) come before the alphabetical unknowns Apple, Zebra.
    expect(categoryNames).toEqual(['Data', 'CNN', 'Apple', 'Zebra']);
  });

  it('expands and collapses a category section', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodesTab />);
    // Expanded by default → node visible.
    expect(screen.getByText('Conv2d')).toBeTruthy();
    const header = screen.getByText('CNN').closest('button')!;
    // Collapse.
    fireEvent.click(header);
    expect(screen.queryByText('Conv2d')).toBeNull();
    // Chevron flips to collapsed glyph.
    expect(within(header).getByText('▸')).toBeTruthy();
    expect(header.getAttribute('aria-expanded')).toBe('false');
    // Re-expand.
    fireEvent.click(header);
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(within(header).getByText('▾')).toBeTruthy();
    expect(header.getAttribute('aria-expanded')).toBe('true');
  });

  it('shows the category count', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN'), def('MaxPool', 'CNN')] },
    });
    render(<NodesTab />);
    const header = screen.getByText('CNN').closest('button')!;
    expect(within(header).getByText('2')).toBeTruthy();
  });

  // ── Search ───────────────────────────────────────────────────────────────

  it('filters nodes by name', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN'), def('MaxPool', 'CNN')] },
    });
    render(<NodesTab />);
    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'conv' },
    });
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.queryByText('MaxPool')).toBeNull();
  });

  it('filters nodes by description', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN', 'a convolution layer')] },
    });
    render(<NodesTab />);
    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'convolution' },
    });
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('filters nodes by details, which the row does not show', () => {
    // The summary is one line now and the library a node wraps lives in
    // `details`. A search that stopped matching there would have bought the
    // shorter row by making the node harder to find.
    seedStore({
      categorized: {
        CNN: [{ ...def('Conv2d', 'CNN', 'slides a kernel over the input'), details: 'Wraps nn.Conv2d.' }],
      },
    });
    render(<NodesTab />);
    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'nn.conv2d' },
    });
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('drops a category whose nodes all filter out', () => {
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN')],
        Data: [def('Dataset', 'Data')],
      },
    });
    render(<NodesTab />);
    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'conv' },
    });
    expect(screen.getByText('CNN')).toBeTruthy();
    expect(screen.queryByText('Data')).toBeNull();
  });

  it('shows the no-match message when a search matches nothing', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodesTab />);
    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'zzzzz' },
    });
    expect(screen.getByText('No matching nodes')).toBeTruthy();
  });

  // ── Beginner mode ──────────────────────────────────────────────────────────

  it('beginner mode hides non-beginner categories', () => {
    useUIStore.setState({ beginnerMode: true });
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN')], // beginner
        Transformer: [def('Attention', 'Transformer')], // not beginner
      },
    });
    render(<NodesTab />);
    expect(screen.getByText('CNN')).toBeTruthy();
    expect(screen.queryByText('Transformer')).toBeNull();
  });

  // ── NodeItem drag + tooltip + hover ──────────────────────────────────────────

  it('node drag start sets the codefyui-node dataTransfer payload', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodesTab />);
    const item = screen.getByText('Conv2d').closest('div')!.parentElement!;
    const setData = vi.fn();
    fireEvent.dragStart(item, {
      dataTransfer: { setData, effectAllowed: '' },
    });
    expect(setData).toHaveBeenCalledWith(
      'application/codefyui-node',
      'Conv2d',
    );
  });

  it('hovering a node sets a hover background and shows a tooltip portal', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN', 'tip text')] } });
    render(<NodesTab />);
    const nameEl = screen.getByText('Conv2d');
    const item = nameEl.parentElement as HTMLElement;

    fireEvent.mouseEnter(item);
    // Hover background applied. Asserted as the token rather than a resolved
    // colour: what matters is that hovering fills the row from the shared
    // surface ramp, not which grey the ramp currently happens to hold.
    expect(item.style.background).toBe('var(--surface-hover)');
    // Tooltip portal renders the description (appears twice: inline + tooltip).
    const tips = screen.getAllByText('tip text');
    expect(tips.length).toBeGreaterThanOrEqual(2);

    fireEvent.mouseLeave(item);
    expect(item.style.background).toBe('transparent');
  });

  it('does not show a tooltip when tooltips are disabled', () => {
    useUIStore.setState({ tooltipsEnabled: false });
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN', 'tip text')] } });
    render(<NodesTab />);
    const item = screen.getByText('Conv2d').parentElement as HTMLElement;
    fireEvent.mouseEnter(item);
    // Only the inline description remains (no portal duplicate).
    expect(screen.getAllByText('tip text')).toHaveLength(1);
  });

  it('does not render a description block when a node has no description', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN', '')] } });
    render(<NodesTab />);
    const item = screen.getByText('Conv2d').parentElement as HTMLElement;
    fireEvent.mouseEnter(item);
    // No tooltip because desc is empty.
    expect(item.style.background).toBe('var(--surface-hover)');
    expect(screen.queryByText('Conv2d desc')).toBeNull();
  });

  it('shows the summary and never the details, inline or in the tooltip', () => {
    // The whole reason `details` exists: this list is one line deep, and the
    // long half belongs to the config panel and the Docs tab.
    seedStore({
      categorized: {
        CNN: [{
          ...def('Conv2d', 'CNN', 'slides a learned kernel over the input'),
          details: 'Wraps nn.Conv2d. Padding defaults to 0, so the output shrinks.',
        }],
      },
    });
    render(<NodesTab />);
    const item = screen.getByText('Conv2d').parentElement as HTMLElement;
    expect(screen.getByText('slides a learned kernel over the input')).toBeTruthy();
    expect(screen.queryByText(/nn\.Conv2d/)).toBeNull();

    fireEvent.mouseEnter(item);
    expect(screen.queryByText(/nn\.Conv2d/)).toBeNull();
  });

  it('translates node descriptions via i18n when locale is non-English', () => {
    // zh-TW with no node translation falls back to the English description.
    // Use a node name that has no zh-TW entry so `tn` returns the fallback.
    act(() => useI18n.setState({ locale: 'zh-TW' }));
    seedStore({
      categorized: { CNN: [def('TotallyMadeUpNodeXYZ', 'CNN', 'english fallback')] },
    });
    render(<NodesTab />);
    expect(screen.getByText('english fallback')).toBeTruthy();
  });

  // ── New in #126: expand-all / collapse-all + jump index ─────────────────────

  it('collapses and expands every category at once', () => {
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN')],
        Data: [def('Dataset', 'Data')],
      },
    });
    render(<NodesTab />);
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.getByText('Dataset')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Collapse all' }));
    expect(screen.queryByText('Conv2d')).toBeNull();
    expect(screen.queryByText('Dataset')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Expand all' }));
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.getByText('Dataset')).toBeTruthy();
  });

  it('hides the toolbar and jump index while there is only one category', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodesTab />);
    expect(screen.queryByRole('button', { name: 'Collapse all' })).toBeNull();
    expect(screen.queryByRole('navigation', { name: 'Jump to category' })).toBeNull();
  });

  it('the jump index scrolls a category into view and expands it if collapsed', () => {
    const scrollIntoView = vi.fn();
    // setup.ts stubs HTMLElement.prototype.scrollIntoView (jsdom has no layout);
    // spy on that same stub to observe the call.
    vi.spyOn(HTMLElement.prototype, 'scrollIntoView').mockImplementation(scrollIntoView);
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN')],
        Data: [def('Dataset', 'Data')],
      },
    });
    render(<NodesTab />);

    // Collapse Data, then jump to it from the index.
    fireEvent.click(screen.getByText('Data').closest('button')!);
    expect(screen.queryByText('Dataset')).toBeNull();

    const index = screen.getByRole('navigation', { name: 'Jump to category' });
    fireEvent.click(within(index).getByRole('button', { name: 'Data' }));

    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'start' });
    // The jump re-expanded the section it scrolled to.
    expect(screen.getByText('Dataset')).toBeTruthy();
  });

  it('jumping to an already-expanded category leaves it expanded', () => {
    vi.spyOn(HTMLElement.prototype, 'scrollIntoView').mockImplementation(vi.fn());
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN')],
        Data: [def('Dataset', 'Data')],
      },
    });
    render(<NodesTab />);
    const index = screen.getByRole('navigation', { name: 'Jump to category' });
    fireEvent.click(within(index).getByRole('button', { name: 'CNN' }));
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('a category that appears after a collapse-all starts expanded', () => {
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN')],
        Data: [def('Dataset', 'Data')],
      },
    });
    render(<NodesTab />);
    fireEvent.click(screen.getByRole('button', { name: 'Collapse all' }));

    // A reload brings in a category that was not part of the collapse-all.
    act(() => {
      seedStore({
        categorized: {
          CNN: [def('Conv2d', 'CNN')],
          Data: [def('Dataset', 'Data')],
          RNN: [def('LSTM', 'RNN')],
        },
      });
    });
    expect(screen.getByText('LSTM')).toBeTruthy();
    expect(screen.queryByText('Conv2d')).toBeNull();
  });
});

// ── "Needs pack" badge (PR 2, F8) ─────────────────────────────────────────

function packSummary(over: Partial<PackSummary> & { id: string }): PackSummary {
  return {
    title: over.id,
    description: '',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: false,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...over,
  };
}

/** Put a catalog in the store, the way a finished `refresh()` would. */
function seedPacks(...packs: PackSummary[]) {
  usePackStore.setState({
    loaded: true,
    unsupported: false,
    packs,
    byId: Object.fromEntries(packs.map((pack) => [pack.id, pack])),
  });
}

const wordVectors = (over: Partial<PackSummary> = {}) =>
  packSummary({ id: 'word-vectors', title: 'Word vectors', ...over });

const packedDef = (): NodeDefinition => ({
  ...def('WordVectorLookup', 'LLM', 'looks a word up in a vector table'),
  requires_pack: 'word-vectors',
});

// The pack is named from this build's own copy, not from the server title
// above: the Package Center this sentence sends the reader to says the same.
const PALETTE_SENTENCE =
  'Needs the Word vectors (GloVe) pack — drag it now, install from the Package Center.';

describe('NodesTab — needs-pack badge', () => {
  it('shows a Needs pack badge for a node whose pack is missing and keeps it draggable', () => {
    seedPacks(wordVectors());
    seedStore({ categorized: { LLM: [packedDef()] } });
    render(<NodesTab />);

    const badge = screen.getByText('Needs pack');
    // The whole sentence is the badge's accessible name; the visible label
    // is the two-word chip. NO native tooltip, because the portal tooltip
    // below renders the same sentence on the same hover, and two copies of
    // it — one of them a browser tooltip a second late — read as two
    // different messages.
    expect(badge.getAttribute('aria-label')).toBe(PALETTE_SENTENCE);
    expect(badge).not.toHaveAttribute('title');

    // The badge is a note on the row, not a gate. Dragging a node whose pack
    // is missing is exactly how a learner gets to the point of installing it,
    // so the payload the canvas receives must be the one it always was.
    const item = screen.getByText('WordVectorLookup').parentElement as HTMLElement;
    expect(item.contains(badge)).toBe(true);
    expect(item.getAttribute('draggable')).toBe('true');
    const setData = vi.fn();
    fireEvent.dragStart(item, { dataTransfer: { setData, effectAllowed: '' } });
    expect(setData).toHaveBeenCalledTimes(1);
    expect(setData).toHaveBeenCalledWith('application/codefyui-node', 'WordVectorLookup');

    // The same sentence again in the hover tooltip, which is where the
    // description is read before the drag.
    fireEvent.mouseEnter(item);
    expect(screen.getByText(PALETTE_SENTENCE)).toBeTruthy();
  });

  it('keeps the native tooltip when the portal one is switched off', () => {
    // With tooltips off the badge is the only place the sentence can live,
    // so the native title is what a hover has to reach.
    useUIStore.setState({ tooltipsEnabled: false });
    seedPacks(wordVectors());
    seedStore({ categorized: { LLM: [packedDef()] } });
    render(<NodesTab />);

    const badge = screen.getByText('Needs pack');
    expect(badge.getAttribute('title')).toBe(PALETTE_SENTENCE);
    expect(badge.getAttribute('aria-label')).toBe(PALETTE_SENTENCE);

    fireEvent.mouseEnter(screen.getByText('WordVectorLookup').parentElement as HTMLElement);
    expect(screen.queryAllByText(PALETTE_SENTENCE)).toHaveLength(0);
  });

  it('omits the badge when the pack is usable or the catalog is unsupported', () => {
    seedPacks(wordVectors({ pip_ready: true, usable: true, status: 'installed' }));
    seedStore({ categorized: { LLM: [packedDef()] } });
    render(<NodesTab />);
    expect(screen.queryByText('Needs pack')).toBeNull();

    // A server with no Package Center at all: it cannot answer, so it says
    // nothing rather than badging every pack-backed node in the library.
    act(() => {
      usePackStore.setState({
        unsupported: true,
        byId: { 'word-vectors': wordVectors() },
      });
    });
    expect(screen.queryByText('Needs pack')).toBeNull();

    // Same during boot, before the catalog answers.
    act(() => {
      usePackStore.setState({ unsupported: false, loaded: false });
    });
    expect(screen.queryByText('Needs pack')).toBeNull();

    // Counterfactual: the very same row DOES badge once a loaded catalog
    // says the pack is not installed.
    act(() => {
      usePackStore.setState({ loaded: true });
    });
    expect(screen.getByText('Needs pack')).toBeTruthy();
  });

  it('leaves a node with no pack requirement untouched', () => {
    seedPacks(wordVectors());
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodesTab />);
    expect(screen.queryByText('Needs pack')).toBeNull();
  });
});

// ── Plugin provenance (P-F3) ──────────────────────────────────────────────

/** An installed, enabled third-party plugin off GitHub: the row these cases name. */
const pluginEntry = (over: Partial<PluginCatalogEntry> & { id: string }): PluginCatalogEntry =>
  catalogEntry({
    kind: 'github',
    official: false,
    status: 'installed',
    source_kind: 'github_url',
    enabled: true,
    ...over,
  });

const EDU_NAME = 'EDU - hands-on teaching nodes';
const EDU_LINE = `From plugin: ${EDU_NAME}`;

/** Put a catalog in the plugin store, the way a finished `refresh()` would. */
function seedPlugins(...entries: PluginCatalogEntry[]) {
  usePluginStore.setState({
    loaded: true,
    unsupported: false,
    plugins: entries,
    byId: Object.fromEntries(entries.map((entry) => [entry.id, entry])),
  });
}

const eduEntry = () => pluginEntry({ id: 'edu', name: EDU_NAME });

const eduDef = (description = 'drops rows a predicate rejects'): NodeDefinition => ({
  ...def('edu:FilterRows', 'Data', description),
  provider: 'plugin:edu',
});

/** The portal tooltip, found from a line inside it rather than by class. */
const tooltipAround = (line: HTMLElement) => line.parentElement as HTMLElement;

describe('NodesTab — plugin provenance', () => {
  it('names the plugin on the last line of the tooltip, after the description', () => {
    seedPlugins(eduEntry());
    seedStore({ categorized: { Data: [eduDef()] } });
    render(<NodesTab />);

    const item = screen.getByText('edu:FilterRows').parentElement as HTMLElement;
    fireEvent.mouseEnter(item);

    const line = screen.getByText(EDU_LINE);
    const tooltip = tooltipAround(line);
    // Title, description, provenance — provenance last, because it is the
    // least of the three things a reader wants before the drag.
    expect(Array.from(tooltip.children).map((child) => child.textContent)).toEqual([
      'edu:FilterRows',
      'drops rows a predicate rejects',
      EDU_LINE,
    ]);
  });

  it('shows the id until the catalog answers', () => {
    // Three ordinary states leave `byId` empty: before the boot fetch lands,
    // on a server with no Plugin Center, and after a network error. The line
    // still says something true, and sharpens when the catalog arrives.
    seedStore({ categorized: { Data: [eduDef()] } });
    render(<NodesTab />);

    fireEvent.mouseEnter(screen.getByText('edu:FilterRows').parentElement as HTMLElement);
    expect(screen.getByText('From plugin: edu')).toBeTruthy();

    act(() => seedPlugins(eduEntry()));
    expect(screen.getByText(EDU_LINE)).toBeTruthy();
    expect(screen.queryByText('From plugin: edu')).toBeNull();
  });

  it('says nothing for a built-in, a custom node, or a server that sends no provider', () => {
    seedPlugins(eduEntry());
    seedStore({
      categorized: {
        CNN: [
          { ...def('Conv2d', 'CNN'), provider: 'builtin' },
          { ...def('MyLayer', 'CNN'), provider: 'custom' },
          def('Dropout', 'CNN'),
        ],
      },
    });
    render(<NodesTab />);

    for (const name of ['Conv2d', 'MyLayer', 'Dropout']) {
      // Grabbed before the hover: once the tooltip is up the name is on screen
      // twice, in the row and in the tooltip's title.
      const item = screen.getByText(name).parentElement as HTMLElement;
      fireEvent.mouseEnter(item);
      expect(screen.queryByText(/^From plugin: /)).toBeNull();
      fireEvent.mouseLeave(item);
    }
  });

  it('earns a tooltip for a plugin node with no description and no pack sentence', () => {
    // The old gate showed NO tooltip at all when a node had neither, so the
    // one line worth reading about such a node could never appear.
    seedPlugins(eduEntry());
    seedStore({ categorized: { Data: [eduDef('')] } });
    render(<NodesTab />);

    const item = screen.getByText('edu:FilterRows').parentElement as HTMLElement;
    fireEvent.mouseEnter(item);

    const line = screen.getByText(EDU_LINE);
    expect(Array.from(tooltipAround(line).children).map((c) => c.textContent)).toEqual([
      'edu:FilterRows',
      EDU_LINE,
    ]);
  });

  it('puts the provenance under the pack sentence, not over it', () => {
    seedPacks(wordVectors());
    seedPlugins(eduEntry());
    seedStore({
      categorized: {
        LLM: [{ ...packedDef(), node_name: 'edu:WordVectorLookup', provider: 'plugin:edu' }],
      },
    });
    render(<NodesTab />);

    fireEvent.mouseEnter(screen.getByText('edu:WordVectorLookup').parentElement as HTMLElement);
    const line = screen.getByText(EDU_LINE);
    // Muted, not the pack sentence's warning colour — its own class, or a
    // reader is told where a node came from in the colour of a problem.
    expect(line.className).not.toBe(screen.getByText(PALETTE_SENTENCE).className);
    expect(Array.from(tooltipAround(line).children).map((c) => c.textContent)).toEqual([
      'edu:WordVectorLookup',
      'looks a word up in a vector table',
      PALETTE_SENTENCE,
      EDU_LINE,
    ]);
  });

  it('finds a plugin node by the plugin display name', () => {
    // Searching the id already worked — plugin node names are qualified, so
    // `edu` matches `edu:FilterRows` through the name. What is new is the
    // human name, which appears in no field of the definition.
    seedPlugins(eduEntry());
    seedStore({
      categorized: {
        Data: [eduDef()],
        CNN: [def('Conv2d', 'CNN')],
      },
    });
    render(<NodesTab />);

    const search = screen.getByPlaceholderText('Search nodes...');
    fireEvent.change(search, { target: { value: 'hands-on' } });
    expect(screen.getByText('edu:FilterRows')).toBeTruthy();
    expect(screen.queryByText('Conv2d')).toBeNull();

    // Case-insensitive, like the two fields it joins.
    fireEvent.change(search, { target: { value: 'TEACHING' } });
    expect(screen.getByText('edu:FilterRows')).toBeTruthy();

    // A built-in stays unreachable by a plugin name.
    fireEvent.change(search, { target: { value: 'conv' } });
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.queryByText('edu:FilterRows')).toBeNull();
  });

  it('re-filters when the catalog lands mid-search', () => {
    seedStore({ categorized: { Data: [eduDef()] } });
    render(<NodesTab />);

    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'hands-on' },
    });
    expect(screen.getByText('No matching nodes')).toBeTruthy();

    act(() => seedPlugins(eduEntry()));
    expect(screen.getByText('edu:FilterRows')).toBeTruthy();
  });
});

// ── Presets as a pinned group (graphs panel wave) ─────────────────────────

describe('NodesTab — presets group', () => {
  it('renders every preset in one group pinned after the node categories', () => {
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN')],
        Zebra: [def('ZNode', 'Zebra')], // unknown → last of the node categories
      },
      // Two different backend categories, one of them ordered ahead of CNN:
      // the group is pinned by position, not sorted in with the rest.
      presets: [preset('LeNet', 'CNN'), preset('Tabular', 'Data')],
    });
    const { container } = render(<NodesTab />);

    expect(categoryNames(container)).toEqual(['CNN', 'Zebra', 'Presets']);
    expect(screen.getByText('LeNet')).toBeTruthy();
    expect(screen.getByText('Tabular')).toBeTruthy();
    // Counted like any other section, so the header says how many there are.
    expect(within(screen.getByText('Presets').closest('button')!).getByText('2')).toBeTruthy();
  });

  it('drags a preset with the preset payload, not the node one', () => {
    // The whole point of keeping presets reachable: the drop target reads
    // `application/codefyui-preset`, and a preset dragged from here has to
    // carry it exactly as it did from the tab presets used to have.
    seedStore({ presets: [preset('LeNet', 'CNN')] });
    render(<NodesTab />);

    // name → header row → the draggable item.
    const item = screen.getByText('LeNet').parentElement!.parentElement!;
    const setData = vi.fn();
    fireEvent.dragStart(item, { dataTransfer: { setData, effectAllowed: '' } });
    expect(setData).toHaveBeenCalledWith('application/codefyui-preset', 'LeNet');
  });

  it('finds a preset by name, and says so when only a preset matches', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN')] },
      presets: [preset('LeNet', 'CNN'), preset('Tabular', 'Data')],
    });
    const { container } = render(<NodesTab />);

    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'lenet' },
    });
    // Every node category filters out; the preset group survives on its own.
    expect(categoryNames(container)).toEqual(['Presets']);
    expect(screen.getByText('LeNet')).toBeTruthy();
    expect(screen.queryByText('Tabular')).toBeNull();
    expect(screen.queryByText('No matching nodes')).toBeNull();
  });

  it('skips a preset list entry with fields missing, and a search does not trip on it', () => {
    // The readers keep such an entry the document's own (#541); this tab reads
    // every field of a row, so it lists only whole presets.
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN')] },
      presets: [{ preset_name: 'Broken' } as unknown as PresetDefinition, preset('LeNet', 'CNN')],
    });
    render(<NodesTab />);
    expect(screen.getByText('LeNet')).toBeTruthy();
    expect(screen.queryByText('Broken')).toBeNull();

    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'lenet' },
    });
    expect(screen.getByText('LeNet')).toBeTruthy();
  });

  it('still draws after a document brings a preset the backend cannot read', () => {
    seedStore({ presets: [] });
    resolveExample({ nodes: [], edges: [], presets: [{ preset_name: 'Broken' }, preset('LeNet', 'CNN')] });
    render(<NodesTab />);
    expect(screen.getByText('LeNet')).toBeTruthy();
    expect(screen.queryByText('Broken')).toBeNull();
  });

  it('shows no preset group when there are no presets, or none matches', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] }, presets: [] });
    const { container } = render(<NodesTab />);
    expect(categoryNames(container)).toEqual(['CNN']);

    // A search that no preset answers drops the group rather than leaving an
    // empty section with a zero count under the node list.
    act(() => {
      seedStore({
        categorized: { CNN: [def('Conv2d', 'CNN')] },
        presets: [preset('LeNet', 'CNN')],
      });
    });
    expect(categoryNames(container)).toEqual(['CNN', 'Presets']);

    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'conv' },
    });
    expect(categoryNames(container)).toEqual(['CNN']);
  });

  it('beginner mode hides a preset whose category it hides in the node list', () => {
    useUIStore.setState({ beginnerMode: true });
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN')] },
      presets: [preset('LeNet', 'CNN'), preset('Attention', 'Transformer')],
    });
    render(<NodesTab />);
    expect(screen.getByText('LeNet')).toBeTruthy();
    expect(screen.queryByText('Attention')).toBeNull();
  });

  // ── PresetItem: difficulty, node count, hover ────────────────────────────
  // Migrated from PresetsTab's own test when the component moved into this
  // file. A preset row looks the same here as it did on the retired tab.

  it('shows the preset difficulty badge and node count', () => {
    seedStore({
      presets: [
        preset('LeNet', 'CNN', {
          tags: ['intermediate'],
          nodes: [
            { id: 'a', type: 'Linear', params: {} },
            { id: 'b', type: 'ReLU', params: {} },
          ],
        }),
      ],
    });
    render(<NodesTab />);
    expect(screen.getByText('intermediate')).toBeTruthy();
    expect(screen.getByText('2 nodes')).toBeTruthy();
  });

  it.each<['en' | 'zh-TW', string, string]>([
    ['en', 'beginner', 'beginner'],
    ['en', 'intermediate', 'intermediate'],
    ['en', 'advanced', 'advanced'],
    ['zh-TW', 'beginner', '入門'],
    ['zh-TW', 'intermediate', '中級'],
    ['zh-TW', 'advanced', '進階'],
  ])('in %s, a preset tagged %s shows the badge "%s"', (locale, tag, label) => {
    useI18n.setState({ locale });
    // Not the first tag: the badge is the first tag that is a difficulty.
    seedStore({ presets: [preset('LeNet', 'CNN', { tags: ['vision', tag] })] });
    render(<NodesTab />);
    const badge = screen.getByText('LeNet').nextElementSibling as HTMLElement;
    expect(badge.textContent).toBe(label);
    // Coloured by the tag, whatever the language says.
    expect(badge.style.color).not.toBe('');
  });

  // #623: Export as Subgraph writes no tags, and the row used to fall back to
  // an English "beginner" -- a level nobody chose.
  it.each<[string, string[]]>([
    ['no tags', []],
    ['no difficulty tag', ['vision']],
    // Every object has a `constructor`; it is not a difficulty.
    ['only an inherited key', ['constructor']],
  ])('shows no difficulty badge for a preset with %s', (_case, tags) => {
    seedStore({ presets: [preset('LeNet', 'CNN', { tags })] });
    render(<NodesTab />);
    // The name is all the row's header holds.
    expect(screen.getByText('LeNet').nextElementSibling).toBeNull();
    expect(screen.queryByText('beginner')).toBeNull();
  });

  it('finds a preset by the word on its badge, in the UI language', () => {
    seedStore({
      presets: [
        preset('LSTM Sequence', 'RNN', { tags: ['intermediate', 'rnn'] }),
        preset('LeNet', 'CNN', { tags: ['beginner'] }),
      ],
    });
    render(<NodesTab />);
    // By role: the placeholder is translated.
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '中級' } });
    expect(screen.queryByText('LSTM Sequence')).toBeNull();

    act(() => useI18n.setState({ locale: 'zh-TW' }));
    expect(screen.getByText('LSTM Sequence')).toBeTruthy();
    expect(screen.queryByText('LeNet')).toBeNull();
  });

  it('hovering a preset toggles its hover background', () => {
    seedStore({ presets: [preset('LeNet', 'CNN')] });
    render(<NodesTab />);
    const item = screen.getByText('LeNet').parentElement!.parentElement!;
    fireEvent.mouseEnter(item);
    expect(item.style.background).toContain('rgba(212, 160, 23');
    fireEvent.mouseLeave(item);
    expect(item.style.background).toBe('transparent');
  });

  it('translates the node count for a non-English locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    seedStore({
      presets: [
        preset('LeNet', 'CNN', {
          nodes: [
            { id: 'a', type: 'Linear', params: {} },
            { id: 'b', type: 'ReLU', params: {} },
          ],
        }),
      ],
    });
    render(<NodesTab />);
    expect(screen.getByText('2 個節點')).toBeTruthy();
  });
});

// ── Ranked search ─────────────────────────────────────────────────────────
// A search used to filter only, so the rows kept the catalog order and the
// node named exactly what was typed could be the last of fifteen ("linear":
// most matches mention a linear layer in their descriptions). The groups stay;
// the order inside and between them is the shared ranking of nodeSearch.ts.

/** The row names among `names`, in the order the list renders them. */
function rowsNamed(...names: string[]): string[] {
  return screen.getAllByText((content) => names.includes(content)).map((el) => el.textContent ?? '');
}

function search(value: string) {
  // By role, not placeholder: the placeholder is translated in the zh-TW cases.
  fireEvent.change(screen.getByRole('textbox'), { target: { value } });
}

describe('NodesTab — ranked search', () => {
  it('lists the node named exactly what was typed first, its category ahead of description matches', () => {
    seedStore({
      categorized: {
        A: [def('DQN', 'A', 'Q-network with a linear head')],
        B: [def('Linear', 'B', 'fully connected layer')],
      },
    });
    const { container } = render(<NodesTab />);
    expect(categoryNames(container)).toEqual(['A', 'B']);

    search('linear');
    expect(categoryNames(container)).toEqual(['B', 'A']);
    expect(rowsNamed('DQN', 'Linear')).toEqual(['Linear', 'DQN']);
  });

  it('sorts the matches inside a category: exact, prefix, word, substring, then description', () => {
    seedStore({
      categorized: {
        Utility: [
          def('DQN', 'Utility', 'a linear head'),
          def('Bilinear', 'Utility', 'blends two inputs'),
          def('SparseLinear', 'Utility', 'a sparse layer'),
          def('LinearRegression', 'Utility', 'least squares'),
          def('Linear', 'Utility', 'fully connected layer'),
        ],
      },
    });
    render(<NodesTab />);
    search('linear');
    expect(rowsNamed('DQN', 'Bilinear', 'SparseLinear', 'LinearRegression', 'Linear')).toEqual([
      'Linear',
      'LinearRegression',
      'SparseLinear',
      'Bilinear',
      'DQN',
    ]);
  });

  it('orders categories by their best match: tier, then the shorter name, then the usual order', () => {
    seedStore({
      categorized: {
        CNN: [def('ConvA', 'CNN')],
        Data: [def('ConvB', 'Data'), def('LinearModel', 'Data')],
        Utility: [def('LinearNet', 'Utility')],
      },
    });
    const { container } = render(<NodesTab />);

    // Both prefix matches, Utility's the shorter name: ahead of Data, which
    // comes first in the usual order.
    search('linear');
    expect(categoryNames(container)).toEqual(['Utility', 'Data']);

    // Equally good (prefix, same length): the usual order decides.
    search('conv');
    expect(categoryNames(container)).toEqual(['Data', 'CNN']);
  });

  it('ranks the presets by the same rule and keeps their group last', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN', 'the layer LeNet starts with')] },
      presets: [preset('LeNetDeep', 'CNN'), preset('LeNet', 'CNN')],
    });
    const { container } = render(<NodesTab />);
    search('lenet');
    // Conv2d matches only through its description, and its group still comes
    // first: presets stay pinned below the node categories.
    expect(categoryNames(container)).toEqual(['CNN', 'Presets']);
    expect(rowsNamed('LeNetDeep', 'LeNet')).toEqual(['LeNet', 'LeNetDeep']);
  });

  it("takes a plugin node's name without its prefix as an exact match", () => {
    seedStore({ categorized: { Data: [def('FilterRowsByMask', 'Data'), eduDef()] } });
    render(<NodesTab />);
    search('filterrows');
    expect(rowsNamed('FilterRowsByMask', 'edu:FilterRows')).toEqual(['edu:FilterRows', 'FilterRowsByMask']);
  });

  it('finds a node by its Chinese description in the Chinese UI, and by English too', () => {
    // Conv2d is in nodeLocales/zh-TW.ts; its summary there is about a 卷積核.
    useI18n.setState({ locale: 'zh-TW' });
    seedStore({
      categorized: {
        CNN: [
          def('Conv2d', 'CNN', 'slides a learned kernel over the input'),
          def('Linear', 'CNN', 'fully connected layer'),
        ],
      },
    });
    render(<NodesTab />);

    search('卷積');
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.queryByText('Linear')).toBeNull();

    search('kernel');
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('does not search the Chinese text in the English UI', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN', 'slides a learned kernel over the input')] } });
    render(<NodesTab />);
    search('卷積');
    expect(screen.getByText('No matching nodes')).toBeTruthy();
  });

  it('runs the search again when the UI language changes', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN', 'slides a learned kernel over the input')] } });
    render(<NodesTab />);
    search('卷積');
    expect(screen.getByText('No matching nodes')).toBeTruthy();

    act(() => useI18n.setState({ locale: 'zh-TW' }));
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('keeps what beginner mode hides out of a search, even an exact match', () => {
    useUIStore.setState({ beginnerMode: true });
    seedStore({
      categorized: {
        CNN: [def('Conv2d', 'CNN', 'a linear filter')],
        Utility: [def('Linear', 'Utility')], // not a beginner category
      },
      presets: [preset('LinearBlock', 'Transformer'), preset('LinearNet', 'CNN')],
    });
    render(<NodesTab />);
    search('linear');
    expect(screen.queryByText('Linear')).toBeNull();
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.queryByText('LinearBlock')).toBeNull();
    expect(screen.getByText('LinearNet')).toBeTruthy();
  });
});

// ── Formulas in descriptions (#598) ───────────────────────────────────────
// Six node descriptions carry inline LaTeX (Linear's "$y = xW^T + b$"). The
// card's tooltip and the config panel typeset it; the palette row and its
// tooltip used to print the dollars and backslashes as they are.

describe('NodesTab — formulas in descriptions', () => {
  it('typesets a formula in the row and in its tooltip, never its source', async () => {
    seedStore({ categorized: { Utility: [def('Linear', 'Utility', 'Layer: $y = x$')] } });
    render(<NodesTab />);
    const item = screen.getByText('Linear').parentElement as HTMLElement;

    // Inside the row's own description element, so its two-line clamp holds.
    const row = item.querySelector('[class*="nodeItemDesc"]') as HTMLElement;
    await waitFor(() => expect(row.querySelector('.katex')).toBeTruthy());
    expect(row.textContent).toContain('Layer: ');
    expect(row.textContent).not.toContain('$');

    fireEvent.mouseEnter(item);
    // Name, then the tooltip's title: the portal comes after the list.
    const tooltip = screen.getAllByText('Linear')[1].parentElement as HTMLElement;
    const tip = tooltip.querySelector('[class*="nodeTooltipDesc"]') as HTMLElement;
    await waitFor(() => expect(tip.querySelector('.katex')).toBeTruthy());
    expect(tip.textContent).toContain('Layer: ');
    expect(tip.textContent).not.toContain('$');
  });

  it('still searches the source text of a formula', async () => {
    seedStore({
      categorized: {
        Utility: [def('Linear', 'Utility', 'Layer: $y = x$'), def('Conv2d', 'Utility')],
      },
    });
    render(<NodesTab />);
    search('y = x');
    expect(screen.queryByText('Conv2d')).toBeNull();
    const item = screen.getByText('Linear').parentElement as HTMLElement;
    await waitFor(() => expect(item.querySelector('.katex')).toBeTruthy());
  });
});
