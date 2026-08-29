import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within, act } from '@testing-library/react';
import { NodesTab } from './NodesTab';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import type { PackSummary } from '../../api/rest';
import type { NodeDefinition } from '../../types';

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

/** Seed the node-def store. This tab is a pure consumer — it never fetches. */
function seedStore(opts: {
  categorized?: Record<string, NodeDefinition[]>;
  loading?: boolean;
  error?: string | null;
}) {
  const definitions = Object.values(opts.categorized ?? {}).flat();
  useNodeDefStore.setState({
    definitions,
    categorized: opts.categorized ?? {},
    loading: opts.loading ?? false,
    error: opts.error ?? null,
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ tooltipsEnabled: true, beginnerMode: false });
  // Every case that does not seed a catalog runs against an empty, unloaded
  // one — the base install, where no palette row says anything about packs.
  _resetPackStoreForTesting();
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
  'Needs the Word vectors (GloVe) pack. You can drag it now and install the pack from '
  + 'the Package Center.';

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
