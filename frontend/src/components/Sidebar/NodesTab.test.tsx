import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within, act } from '@testing-library/react';
import { NodesTab } from './NodesTab';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
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

/** Seed the node-def store and mark it loaded so the auto-fetch effect is a no-op. */
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
  // Prevent the auto-fetch effect from hitting the network: pretend loaded.
  seedStore({ categorized: {} });
  // Also stub fetchDefinitions defensively in case any path triggers it.
  vi.spyOn(useNodeDefStore.getState(), 'fetchDefinitions').mockResolvedValue(undefined);
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
    expect(refetch).toHaveBeenCalled();
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
    expect(item.style.background).toBe('rgb(42, 42, 42)');
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
