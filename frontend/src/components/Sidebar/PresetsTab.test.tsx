import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { PresetsTab } from './PresetsTab';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import type { PresetDefinition } from '../../types';

/*
 * Presets had their own item component and drag payload before #126, but were
 * rendered inside the node list under a "Composite" sub-header. The item-level
 * behaviours below are migrated verbatim from NodePalette.test.tsx; what is new
 * is that they now live in a tab of their own, with their own search and their
 * own category counts.
 */

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

function seedStore(opts: {
  presetCategorized?: Record<string, PresetDefinition[]>;
  loading?: boolean;
}) {
  useNodeDefStore.setState({
    presets: Object.values(opts.presetCategorized ?? {}).flat(),
    presetCategorized: opts.presetCategorized ?? {},
    loading: opts.loading ?? false,
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ tooltipsEnabled: true, beginnerMode: false });
  seedStore({ presetCategorized: {} });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('PresetsTab', () => {
  it('renders the title, search box, and footer hint', () => {
    render(<PresetsTab />);
    expect(screen.getByText('Presets')).toBeTruthy();
    expect(screen.getByPlaceholderText('Search presets...')).toBeTruthy();
    expect(screen.getByText('Drag presets onto the canvas')).toBeTruthy();
  });

  it('shows the loading state', () => {
    seedStore({ loading: true });
    render(<PresetsTab />);
    expect(screen.getByText('Loading nodes...')).toBeTruthy();
  });

  it('shows the empty state when there are no presets', () => {
    render(<PresetsTab />);
    expect(screen.getByText('No presets available')).toBeTruthy();
  });

  it('groups presets by category with a per-category count', () => {
    seedStore({
      presetCategorized: {
        CNN: [preset('CNNBlock', 'CNN'), preset('ResBlock', 'CNN')],
        Data: [preset('Loader', 'Data')],
      },
    });
    render(<PresetsTab />);
    expect(within(screen.getByText('CNN').closest('button')!).getByText('2')).toBeTruthy();
    expect(within(screen.getByText('Data').closest('button')!).getByText('1')).toBeTruthy();
  });

  it('orders categories the same way the Nodes tab does', () => {
    seedStore({
      presetCategorized: {
        Zebra: [preset('ZBlock', 'Zebra')],
        CNN: [preset('CNNBlock', 'CNN')],
        Data: [preset('Loader', 'Data')],
      },
    });
    const { container } = render(<PresetsTab />);
    const names = Array.from(container.querySelectorAll('button span'))
      .map((s) => s.textContent)
      .filter((t) => ['Data', 'CNN', 'Zebra'].includes(t ?? ''));
    expect(names).toEqual(['Data', 'CNN', 'Zebra']);
  });

  it('does not render node-library items (presets are split out cleanly)', () => {
    // A category that has both kinds server-side must show only presets here.
    useNodeDefStore.setState({
      categorized: { CNN: [{ node_name: 'Conv2d', category: 'CNN', description: '', inputs: [], outputs: [], params: [] }] },
    });
    seedStore({ presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] } });
    render(<PresetsTab />);
    expect(screen.getByText('CNNBlock')).toBeTruthy();
    expect(screen.queryByText('Conv2d')).toBeNull();
    // The Composite/Basic sub-headers that used to separate them are gone.
    expect(screen.queryByText('Composite')).toBeNull();
    expect(screen.queryByText('Basic')).toBeNull();
  });

  // ── Search ────────────────────────────────────────────────────────────────

  it('filters presets by name, description, and tags', () => {
    seedStore({
      presetCategorized: {
        CNN: [
          preset('AlphaNet', 'CNN', ['advanced'], 'a deep net'),
          preset('BetaNet', 'CNN', ['beginner'], 'shallow'),
        ],
      },
    });
    render(<PresetsTab />);
    const input = screen.getByPlaceholderText('Search presets...');

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
    seedStore({ presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] } });
    render(<PresetsTab />);
    fireEvent.change(screen.getByPlaceholderText('Search presets...'), {
      target: { value: 'zzzzz' },
    });
    expect(screen.getByText('No matching presets')).toBeTruthy();
  });

  // ── Beginner mode ─────────────────────────────────────────────────────────

  it('beginner mode hides non-beginner categories', () => {
    useUIStore.setState({ beginnerMode: true });
    seedStore({
      presetCategorized: {
        CNN: [preset('CNNBlock', 'CNN')],
        Transformer: [preset('AttnBlock', 'Transformer')],
      },
    });
    render(<PresetsTab />);
    expect(screen.getByText('CNN')).toBeTruthy();
    expect(screen.queryByText('Transformer')).toBeNull();
  });

  // ── PresetItem drag + difficulty + hover ──────────────────────────────────

  it('preset drag start sets the codefyui-preset dataTransfer payload', () => {
    seedStore({ presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] } });
    render(<PresetsTab />);
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
    render(<PresetsTab />);
    expect(screen.getByText('intermediate')).toBeTruthy();
    expect(screen.getByText('2 nodes')).toBeTruthy();
  });

  it('defaults preset difficulty to beginner when no difficulty tag present', () => {
    seedStore({
      presetCategorized: { CNN: [preset('CNNBlock', 'CNN', ['vision'])] },
    });
    render(<PresetsTab />);
    expect(screen.getByText('beginner')).toBeTruthy();
  });

  it('hovering a preset toggles its hover background', () => {
    seedStore({ presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] } });
    render(<PresetsTab />);
    const item = screen.getByText('CNNBlock').closest('div')!.parentElement!
      .parentElement!;
    fireEvent.mouseEnter(item);
    expect(item.style.background).toContain('rgba(212, 160, 23');
    fireEvent.mouseLeave(item);
    expect(item.style.background).toBe('transparent');
  });

  it('translates the node count for a non-English locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    seedStore({ presetCategorized: { CNN: [preset('CNNBlock', 'CNN')] } });
    render(<PresetsTab />);
    expect(screen.getByText('2 個節點')).toBeTruthy();
  });
});
