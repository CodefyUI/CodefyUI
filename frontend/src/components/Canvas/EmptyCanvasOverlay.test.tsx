import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { EmptyCanvasOverlay } from './EmptyCanvasOverlay';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
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
    expect(screen.getByText('Pick an example to get started quickly')).toBeInTheDocument();
    // loading state visible
    expect(screen.getByText('Loading examples...')).toBeInTheDocument();

    resolveList([]);
    await waitFor(() => expect(screen.queryByText('Loading examples...')).toBeNull());
    // hint always present at the bottom
    expect(screen.getByText('or drag a node from the left palette')).toBeInTheDocument();
  });

  it('falls back to an empty list when listExamples rejects', async () => {
    mockedRest.listExamples.mockRejectedValue(new Error('boom'));
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.queryByText('Loading examples...')).toBeNull());
    // No cards, but the static hint still shows.
    expect(screen.getByText('or drag a node from the left palette')).toBeInTheDocument();
  });

  it('renders grouped sections, a known-category badge, and node counts', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train MLP', category: 'Usage_Example', node_count: 5 }),
      ex({ name: 'ResNet', category: 'Model_Architecture', path: '/a/resnet.json' }),
    ]);
    render(<EmptyCanvasOverlay />);

    await waitFor(() => expect(screen.getByText('Train MLP')).toBeInTheDocument());
    // Unpinned Usage_Example lands in Advanced; Model_Architecture in its own section.
    expect(screen.getByText('Advanced Examples')).toBeInTheDocument();
    expect(screen.getByText('Model Architectures')).toBeInTheDocument();
    // category label has underscores replaced by spaces
    expect(screen.getByText('Usage Example')).toBeInTheDocument();
    expect(screen.getByText('Model Architecture')).toBeInTheDocument();
    // node count line
    expect(screen.getByText('5 nodes')).toBeInTheDocument();
  });

  it('pins Quick Start by path and renders sections in order: quickstart, advanced, plugin, architectures', async () => {
    // Deliberately shuffled relative to the pinned order to prove path pinning
    // (the backend returns alphabetical-by-path, not curated order).
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'ResNet Arch', category: 'Model_Architecture', path: 'Model_Architecture/ResNet-SkipConnection-CNN' }),
      ex({ name: 'Api Fn', category: 'Usage_Example', path: 'Usage_Example/Api-Function' }),
      ex({ name: 'Inference CNN', category: 'Usage_Example', path: 'Usage_Example/CNN-MNIST/InferenceCNN-MNIST' }),
      ex({ name: 'Train CNN', category: 'Usage_Example', path: 'Usage_Example/CNN-MNIST/TrainCNN-MNIST' }),
      ex({ name: 'Train GPT', category: 'Usage_Example', path: 'Usage_Example/GPT-Mini/TrainGPT-Mini' }),
      ex({ name: 'Analogy', category: 'LLM', path: 'LLM/Word-Embedding-Analogy' }),
      // Plugin example whose folder shares a builtin category name — must go
      // to the plugin section, not Advanced.
      ex({ name: 'Plugin Demo', category: 'Classical', path: 'plugin:c2/Classical/Foo' }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('Train CNN')).toBeInTheDocument());

    // All four section titles are present and in document order.
    const body = document.body.textContent ?? '';
    const positions = [
      'Quick Start',
      'Advanced Examples',
      'Plugin Examples',
      'Model Architectures',
    ].map((title) => {
      const idx = body.indexOf(title);
      expect(idx, title).toBeGreaterThanOrEqual(0);
      return idx;
    });
    expect(positions).toEqual([...positions].sort((a, b) => a - b));

    // Global card order: pinned trio first (in pinned order, not list order),
    // then Advanced (LLM before leftover Usage_Example), then plugin, then
    // architectures last.
    expect(cardNames()).toEqual([
      'Train CNN',
      'Inference CNN',
      'Api Fn',
      'Analogy',
      'Train GPT',
      'Plugin Demo',
      'ResNet Arch',
    ]);
  });

  it('orders Advanced by category order, then per-path priority within a category', async () => {
    // Backend alphabetical order within Diffusion is Forward, Mini-UNet, Toy;
    // the per-path priority hoists Toy-Sampling above Mini-UNet-Compact.
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Iris', category: 'Classical', path: 'Classical/Iris-Sklearn-KNN' }),
      ex({ name: 'Forward', category: 'Diffusion', path: 'Diffusion/Forward-Process' }),
      ex({ name: 'MiniUNet', category: 'Diffusion', path: 'Diffusion/Mini-UNet-Compact' }),
      ex({ name: 'ToySampling', category: 'Diffusion', path: 'Diffusion/Toy-Sampling' }),
      ex({ name: 'Analogy', category: 'LLM', path: 'LLM/Word-Embedding-Analogy' }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('Analogy')).toBeInTheDocument());
    expect(cardNames()).toEqual(['Analogy', 'Forward', 'ToySampling', 'MiniUNet', 'Iris']);
  });

  it('renders unknown categories in the plugin section with the fallback badge colour', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Misc Demo', category: 'Something_Else', path: '/x/misc.json' }),
    ]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('Misc Demo')).toBeInTheDocument());
    // Unknown category still renders its label (replaced underscores).
    expect(screen.getByText('Something Else')).toBeInTheDocument();
    // It lands under the generic plugin/other section, not the curated ones.
    expect(screen.getByText('Plugin Examples')).toBeInTheDocument();
    expect(screen.queryByText('Quick Start')).toBeNull();
    expect(screen.queryByText('Advanced Examples')).toBeNull();
    expect(screen.queryByText('Model Architectures')).toBeNull();
  });

  it('truncates descriptions longer than 80 characters', async () => {
    const long = 'x'.repeat(120);
    mockedRest.listExamples.mockResolvedValue([ex({ description: long })]);
    render(<EmptyCanvasOverlay />);
    await waitFor(() => expect(screen.getByText('Example')).toBeInTheDocument());
    expect(screen.getByText(`${'x'.repeat(80)}...`)).toBeInTheDocument();
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
