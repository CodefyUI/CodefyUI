import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { EmptyCanvasOverlay } from './EmptyCanvasOverlay';
import { useTabStore } from '../../store/tabStore';
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

/**
 * Names of the example cards in DOM order. Filtered by class rather than
 * taken from every button, because the overlay also carries the
 * browse-the-gallery button (core#128), which is not a card.
 */
function cardNames(): (string | null | undefined)[] {
  return screen
    .getAllByRole('button')
    .filter((b) => b.className.includes('presetCard'))
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
    mockedUtils.resolveSerializedNodes.mockClear();
    mockedUtils.resolveSerializedEdges.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows the loading hint before examples resolve, then hides it', async () => {
    let resolveList!: (v: ExampleSummary[]) => void;
    mockedRest.listExamples.mockReturnValue(
      new Promise<ExampleSummary[]>((res) => {
        resolveList = res;
      }),
    );
    render(<EmptyCanvasOverlay />);
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
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.queryByText('Loading examples...')).toBeNull());
    // No cards, but the heading and its one instruction still show.
    expect(screen.getByText('Build your first deep learning model')).toBeInTheDocument();
    expect(screen.getByText('Pick an example')).toBeInTheDocument();
  });

  it('renders the declared section, a known-category badge, and node counts', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train MLP', category: 'Usage_Example', node_count: 5, section: 'quickstart' }),
      ex({
        name: 'ResNet',
        category: 'Model_Architecture',
        path: '/a/resnet.json',
        section: 'architectures',
      }),
    ]);
    render(<EmptyCanvasOverlay />);

    await waitFor(() => expect(screen.getByText('Train MLP')).toBeInTheDocument());
    expect(screen.getByText('Quick Start')).toBeInTheDocument();
    expect(screen.getByText('Model Architectures')).toBeInTheDocument();
    // category label has underscores replaced by spaces
    expect(screen.getByText('Usage Example')).toBeInTheDocument();
    expect(screen.getByText('Model Architecture')).toBeInTheDocument();
    // node count line
    expect(screen.getByText('5 nodes')).toBeInTheDocument();
  });

  it('renders the sections in the contract order, whatever order the server listed', async () => {
    // Deliberately shuffled: the order comes from the gallery metadata each
    // example carries, not from where the backend happened to put it.
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Misc', category: 'Something_Else', path: 'X/misc' }),
      ex({ name: 'Plugin Demo', category: 'Classical', path: 'plugin:c2/Classical/Foo', source: 'plugin:c2' }),
      ex({ name: 'ResNet Arch', category: 'Model_Architecture', path: 'm/resnet', section: 'architectures', family: 'CNN' }),
      ex({ name: 'Iris', category: 'Classical', path: 'Classical/Iris', section: 'concepts' }),
      ex({ name: 'Analogy', category: 'LLM', path: 'LLM/Analogy', section: 'llm' }),
      ex({ name: 'Train GPT', category: 'Usage_Example', path: 'Usage_Example/GPT', section: 'training' }),
      ex({ name: 'Train CNN', category: 'Usage_Example', path: 'Usage_Example/CNN', section: 'quickstart' }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('Train CNN')).toBeInTheDocument());

    const body = document.body.textContent ?? '';
    const positions = [
      'Quick Start',
      'Training',
      'LLM and RAG',
      'Concepts',
      'Model Architectures',
      'Plugin Packs',
      'Other',
    ].map((title) => {
      const idx = body.indexOf(title);
      expect(idx, title).toBeGreaterThanOrEqual(0);
      return idx;
    });
    expect(positions).toEqual([...positions].sort((a, b) => a - b));

    expect(cardNames()).toEqual([
      'Train CNN',
      'Train GPT',
      'Analogy',
      'Iris',
      'ResNet Arch',
      'Plugin Demo',
      'Misc',
    ]);
  });

  it('orders a section by the declared order, nulls last, ties as listed', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Unordered', path: 'c/unordered', section: 'concepts' }),
      ex({ name: 'Third', path: 'c/third', section: 'concepts', order: 3 }),
      ex({ name: 'First', path: 'c/first', section: 'concepts', order: 1 }),
      ex({ name: 'Second', path: 'c/second', section: 'concepts', order: 2 }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('First')).toBeInTheDocument());
    expect(cardNames()).toEqual(['First', 'Second', 'Third', 'Unordered']);
  });

  it('gives the architecture families and the packs a sub-header', async () => {
    usePluginStore.setState({ byId: { c2: { id: 'c2', name: 'Chapter 2' } } as never });
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'ResNet', category: 'Model_Architecture', path: 'm/resnet', section: 'architectures', family: 'CNN' }),
      ex({ name: 'LSTM', category: 'Model_Architecture', path: 'm/lstm', section: 'architectures', family: 'RNN' }),
      ex({ name: 'Lesson', category: 'Classical', path: 'plugin:c2/Lesson', source: 'plugin:c2' }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('ResNet')).toBeInTheDocument());

    const subheads = [...document.querySelectorAll('[class*="subsectionTitle"]')].map(
      (el) => el.textContent,
    );
    expect(subheads).toEqual(['CNN', 'RNN', 'Chapter 2']);
    // One section title above the two families, not one per family.
    expect(screen.getAllByText('Model Architectures')).toHaveLength(1);
  });

  it('leaves the chip off a card whose sub-header already says it', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'ResNet', category: 'Model_Architecture', path: 'm/resnet', section: 'architectures', family: 'CNN' }),
      ex({ name: 'Iris', category: 'Classical', path: 'c/iris', section: 'concepts' }),
      ex({ name: 'Tiny CNN', category: 'Usage_Example', path: 'u/tiny', section: 'training', family: 'CNN' }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('ResNet')).toBeInTheDocument());

    const chips = [...document.querySelectorAll('[class*="difficultyBadge"]')].map(
      (el) => el.textContent,
    );
    // Training, then concepts, then the architectures. ResNet sits under a
    // "CNN" sub-header, so a "CNN" chip on it would say the same thing twice;
    // Tiny CNN has no sub-header above it, so its family chip stays.
    expect(chips).toEqual(['CNN', 'Classical']);
    // The node count is what is left in that card's footer.
    expect(screen.getByText('ResNet').closest('button')).toHaveTextContent('3 nodes');
  });

  it('puts a built-in that declares no section under Other', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Misc Demo', category: 'Something_Else', path: '/x/misc.json' }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('Misc Demo')).toBeInTheDocument());
    // Unknown category still renders its label (replaced underscores).
    expect(screen.getByText('Something Else')).toBeInTheDocument();
    expect(screen.getByText('Other')).toBeInTheDocument();
    expect(screen.queryByText('Quick Start')).toBeNull();
    expect(screen.queryByText('Model Architectures')).toBeNull();
  });

  it('truncates descriptions longer than 40 columns', async () => {
    // 40, not the 80 this used to cut at: a description is allowed to be 40
    // columns wide in the first place (`MAX_DESCRIPTION_COLUMNS`, backend and
    // zh-TW), so a wider cut could only ever fire on text that already broke
    // the rule -- which is a third-party pack's description, never one of
    // ours.
    const long = 'x'.repeat(120);
    mockedRest.listExamples.mockResolvedValue([ex({ description: long })]);
    render(<EmptyCanvasOverlay />);
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
    render(<EmptyCanvasOverlay />);
    expect(await screen.findByText(exact)).toBeInTheDocument();
  });

  it('keeps the whole description reachable as a tooltip (core#305)', async () => {
    // The cut only fires on a description that ran past the 40-column rule,
    // and that is exactly when the tooltip earns its place: this card is the
    // one place a description appears with no way to read the rest. The
    // sidebar's gallery tab already does this.
    const long = `${'x'.repeat(40)} and the part nobody could read`;
    mockedRest.listExamples.mockResolvedValue([ex({ description: long })]);
    render(<EmptyCanvasOverlay />);
    const shown = await screen.findByText(`${'x'.repeat(40)}...`);
    expect(shown).toHaveAttribute('title', long);
  });

  it('does not put a tooltip on a description that is already whole', async () => {
    // A native tooltip repeating the text under the cursor is noise.
    mockedRest.listExamples.mockResolvedValue([ex({ description: 'short desc' })]);
    render(<EmptyCanvasOverlay />);
    const shown = await screen.findByText('short desc');
    expect(shown).not.toHaveAttribute('title');
  });

  it('applies and clears hover styles on a card', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Hover Me' })]);
    render(<EmptyCanvasOverlay />);
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
      presets: [{ preset_name: 'NewPreset' }],
    });
    // Existing preset to exercise the "already present, skip" branch.
    useNodeDefStore.setState({ definitions: [], presets: [{ preset_name: 'Existing' } as any] });

    // One spy, because the gallery path installs the whole document through
    // one store action now (#200 items 4 and 8) instead of sequencing
    // setNodes/setEdges/renameTab itself.
    const loadGraphDocument = vi.fn();
    useTabStore.setState({ loadGraphDocument });

    render(<EmptyCanvasOverlay />);
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

    render(<EmptyCanvasOverlay />);
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

    render(<EmptyCanvasOverlay />);
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

    render(<EmptyCanvasOverlay />);
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

    render(<EmptyCanvasOverlay />);
    fireEvent.click(await screen.findByText('Broken'));

    await waitFor(() => expect(addToast).toHaveBeenCalledWith('Failed to load example', 'error'));
  });
});
