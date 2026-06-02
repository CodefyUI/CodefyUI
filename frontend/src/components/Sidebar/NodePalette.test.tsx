import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within, act } from '@testing-library/react';
import { NodePalette } from './NodePalette';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import type { NodeDefinition, PresetDefinition } from '../../types';

// ── Builders ─────────────────────────────────────────────────────────────────

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
  tags: string[] = ['beginner'],
  description = `${preset_name} desc`,
): PresetDefinition {
  return {
    preset_name,
    category,
    description,
    tags,
    nodes: [
      { id: 'a', type: 'Linear', params: {} },
      { id: 'b', type: 'ReLU', params: {} },
    ],
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
  };
}

/** Seed the node-def store and mark it loaded so the auto-fetch effect is a no-op. */
function seedStore(opts: {
  categorized?: Record<string, NodeDefinition[]>;
  presetCategorized?: Record<string, PresetDefinition[]>;
  loading?: boolean;
  error?: string | null;
}) {
  const definitions = Object.values(opts.categorized ?? {}).flat();
  useNodeDefStore.setState({
    definitions,
    categorized: opts.categorized ?? {},
    presets: Object.values(opts.presetCategorized ?? {}).flat(),
    presetCategorized: opts.presetCategorized ?? {},
    loading: opts.loading ?? false,
    error: opts.error ?? null,
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ tooltipsEnabled: true, beginnerMode: false });
  // Prevent the auto-fetch effect from hitting the network: pretend loaded.
  seedStore({ categorized: {}, presetCategorized: {} });
  // Also stub fetchDefinitions defensively in case any path triggers it.
  vi.spyOn(useNodeDefStore.getState(), 'fetchDefinitions').mockResolvedValue(undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('NodePalette', () => {
  it('renders the title, search box, and footer hint', () => {
    render(<NodePalette />);
    expect(screen.getByText('Nodes')).toBeTruthy();
    expect(screen.getByPlaceholderText('Search nodes...')).toBeTruthy();
    expect(screen.getByText('Drag nodes onto the canvas')).toBeTruthy();
  });

  it('shows the loading state', () => {
    seedStore({ loading: true });
    render(<NodePalette />);
    expect(screen.getByText('Loading nodes...')).toBeTruthy();
  });

  it('shows the error state and retries on click', () => {
    seedStore({ error: 'network down' });
    const refetch = useNodeDefStore.getState().fetchDefinitions as ReturnType<
      typeof vi.fn
    >;
    render(<NodePalette />);
    expect(screen.getByText('Failed to load nodes: network down')).toBeTruthy();
    fireEvent.click(screen.getByText('Retry'));
    expect(refetch).toHaveBeenCalled();
  });

  it('shows the empty state when there are no nodes and no search', () => {
    seedStore({ categorized: {}, presetCategorized: {} });
    render(<NodePalette />);
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
    const { container } = render(<NodePalette />);
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
    render(<NodePalette />);
    // Expanded by default → node visible.
    expect(screen.getByText('Conv2d')).toBeTruthy();
    const header = screen.getByText('CNN').closest('button')!;
    // Collapse.
    fireEvent.click(header);
    expect(screen.queryByText('Conv2d')).toBeNull();
    // Chevron flips to collapsed glyph.
    expect(within(header).getByText('▸')).toBeTruthy();
    // Re-expand.
    fireEvent.click(header);
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(within(header).getByText('▾')).toBeTruthy();
  });

  it('shows the category count = presets + nodes', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN'), def('MaxPool', 'CNN')] },
      presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] },
    });
    render(<NodePalette />);
    const header = screen.getByText('CNN').closest('button')!;
    expect(within(header).getByText('3')).toBeTruthy();
  });

  it('renders preset and node sub-headers only when a category has both', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN')] },
      presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] },
    });
    render(<NodePalette />);
    expect(screen.getByText('Composite')).toBeTruthy();
    expect(screen.getByText('Basic')).toBeTruthy();
    expect(screen.getByText('CNNBlock')).toBeTruthy();
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('does not render sub-headers when a category has only nodes', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodePalette />);
    expect(screen.queryByText('Composite')).toBeNull();
    expect(screen.queryByText('Basic')).toBeNull();
  });

  // ── Search ───────────────────────────────────────────────────────────────

  it('filters nodes by name', () => {
    seedStore({
      categorized: { CNN: [def('Conv2d', 'CNN'), def('MaxPool', 'CNN')] },
    });
    render(<NodePalette />);
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
    render(<NodePalette />);
    fireEvent.change(screen.getByPlaceholderText('Search nodes...'), {
      target: { value: 'convolution' },
    });
    expect(screen.getByText('Conv2d')).toBeTruthy();
  });

  it('filters presets by name, description, and tags', () => {
    seedStore({
      presetCategorized: {
        CNN: [
          preset('AlphaNet', 'CNN', ['advanced'], 'a deep net'),
          preset('BetaNet', 'CNN', ['beginner'], 'shallow'),
        ],
      },
    });
    render(<NodePalette />);
    const input = screen.getByPlaceholderText('Search nodes...');

    // By name.
    fireEvent.change(input, { target: { value: 'alpha' } });
    expect(screen.getByText('AlphaNet')).toBeTruthy();
    expect(screen.queryByText('BetaNet')).toBeNull();

    // By description.
    fireEvent.change(input, { target: { value: 'shallow' } });
    expect(screen.getByText('BetaNet')).toBeTruthy();
    expect(screen.queryByText('AlphaNet')).toBeNull();

    // By tag.
    fireEvent.change(input, { target: { value: 'advanced' } });
    expect(screen.getByText('AlphaNet')).toBeTruthy();
    expect(screen.queryByText('BetaNet')).toBeNull();
  });

  it('shows the no-match message when a search matches nothing', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodePalette />);
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
    render(<NodePalette />);
    expect(screen.getByText('CNN')).toBeTruthy();
    expect(screen.queryByText('Transformer')).toBeNull();
  });

  // ── NodeItem drag + tooltip + hover ──────────────────────────────────────────

  it('node drag start sets the codefyui-node dataTransfer payload', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN')] } });
    render(<NodePalette />);
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
    render(<NodePalette />);
    const nameEl = screen.getByText('Conv2d');
    const item = nameEl.parentElement as HTMLElement;

    fireEvent.mouseEnter(item);
    // Hover background applied (jsdom normalizes to rgb).
    expect(item.style.background).toBe('rgb(42, 42, 42)');
    // Tooltip portal renders the description (appears twice: inline + tooltip).
    const tips = screen.getAllByText('tip text');
    expect(tips.length).toBeGreaterThanOrEqual(2);

    fireEvent.mouseLeave(item);
    expect(item.style.background).toBe('transparent');
  });

  it('does not show a tooltip when tooltips are disabled', () => {
    useUIStore.setState({ tooltipsEnabled: false });
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN', 'tip text')] } });
    render(<NodePalette />);
    const item = screen.getByText('Conv2d').parentElement as HTMLElement;
    fireEvent.mouseEnter(item);
    // Only the inline description remains (no portal duplicate).
    expect(screen.getAllByText('tip text')).toHaveLength(1);
  });

  it('does not render a description block when a node has no description', () => {
    seedStore({ categorized: { CNN: [def('Conv2d', 'CNN', '')] } });
    render(<NodePalette />);
    const item = screen.getByText('Conv2d').parentElement as HTMLElement;
    fireEvent.mouseEnter(item);
    // No tooltip because desc is empty.
    expect(item.style.background).toBe('rgb(42, 42, 42)');
    expect(screen.queryByText('Conv2d desc')).toBeNull();
  });

  // ── PresetItem drag + difficulty + hover ──────────────────────────────────────

  it('preset drag start sets the codefyui-preset dataTransfer payload', () => {
    seedStore({ presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] } });
    render(<NodePalette />);
    const item = screen.getByText('CNNBlock').closest('div')!.parentElement!
      .parentElement!;
    const setData = vi.fn();
    fireEvent.dragStart(item, { dataTransfer: { setData, effectAllowed: '' } });
    expect(setData).toHaveBeenCalledWith(
      'application/codefyui-preset',
      'CNNBlock',
    );
  });

  it('shows the preset difficulty badge and node count', () => {
    seedStore({
      presetCategorized: { CNN: [preset('CNNBlock', 'CNN', ['intermediate'])] },
    });
    render(<NodePalette />);
    expect(screen.getByText('intermediate')).toBeTruthy();
    expect(screen.getByText('2 nodes')).toBeTruthy();
  });

  it('defaults preset difficulty to beginner when no difficulty tag present', () => {
    seedStore({
      presetCategorized: { CNN: [preset('CNNBlock', 'CNN', ['vision'])] },
    });
    render(<NodePalette />);
    expect(screen.getByText('beginner')).toBeTruthy();
  });

  it('hovering a preset toggles its hover background', () => {
    seedStore({ presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] } });
    render(<NodePalette />);
    const item = screen.getByText('CNNBlock').closest('div')!.parentElement!
      .parentElement!;
    fireEvent.mouseEnter(item);
    expect(item.style.background).toContain('rgba(212, 160, 23');
    fireEvent.mouseLeave(item);
    expect(item.style.background).toBe('transparent');
  });

  it('translates node descriptions via i18n when locale is non-English', () => {
    // zh-TW with no node translation falls back to the English description.
    // Use a node name that has no zh-TW entry so `tn` returns the fallback.
    act(() => useI18n.setState({ locale: 'zh-TW' }));
    seedStore({
      categorized: { CNN: [def('TotallyMadeUpNodeXYZ', 'CNN', 'english fallback')] },
    });
    render(<NodePalette />);
    expect(screen.getByText('english fallback')).toBeTruthy();
  });
});
