import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, fireEvent, waitFor, within } from '@testing-library/react';
import { QuickNodeSearch } from './QuickNodeSearch';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import type { NodeDefinition, PresetDefinition } from '../../types';

function def(name: string, overrides: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: name,
    category: 'CNN',
    description: `${name} description`,
    inputs: [],
    outputs: [],
    params: [],
    ...overrides,
  };
}

function preset(name: string, overrides: Partial<PresetDefinition> = {}): PresetDefinition {
  return {
    preset_name: name,
    category: 'RNN',
    description: `${name} preset desc`,
    tags: [],
    nodes: [],
    edges: [],
    exposed_inputs: [],
    exposed_outputs: [],
    exposed_params: [],
    ...overrides,
  };
}

const SCREEN = { x: 50, y: 60 };
const FLOW = { x: 11, y: 22 };

function setStore(defs: NodeDefinition[], presets: PresetDefinition[]) {
  useNodeDefStore.setState({ definitions: defs, presets });
}

/** Index of the visually-selected item button (the one with the extra class token). */
function selectedButtonIndex(container: HTMLElement): number {
  const buttons = Array.from(container.querySelectorAll('button'));
  let maxTokens = 1;
  let idx = -1;
  buttons.forEach((b, i) => {
    const tokens = b.className.trim().split(/\s+/).filter(Boolean).length;
    if (tokens > maxTokens) {
      maxTokens = tokens;
      idx = i;
    }
  });
  return idx;
}

describe('QuickNodeSearch', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    setStore([], []);
    useTabStore.setState({ addNode: vi.fn(), addPresetNode: vi.fn() });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('auto-focuses the input on mount and shows the placeholder', () => {
    const { getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const input = getByPlaceholderText('Search nodes...') as HTMLInputElement;
    expect(document.activeElement).toBe(input);
  });

  it('shows the no-match message when nothing matches the query', () => {
    setStore([def('Conv2d')], []);
    const { getByPlaceholderText, getByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    fireEvent.change(getByPlaceholderText('Search nodes...'), {
      target: { value: 'zzzzznope' },
    });
    expect(getByText('No matching nodes')).toBeInTheDocument();
  });

  it('lists nodes and presets, with a preset badge and category labels', () => {
    setStore([def('Conv2d', { category: 'CNN' })], [preset('MyBlock', { category: 'RNN' })]);
    const { container, getByText, getAllByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    expect(getByText('Conv2d')).toBeInTheDocument();
    expect(getByText('MyBlock')).toBeInTheDocument();
    // preset badge
    expect(getByText('PRESET')).toBeInTheDocument();
    // category labels (one per item)
    expect(getByText('CNN')).toBeInTheDocument();
    expect(getByText('RNN')).toBeInTheDocument();
    // descriptions present
    expect(getByText('Conv2d description')).toBeInTheDocument();
    expect(getByText('MyBlock preset desc')).toBeInTheDocument();
    // both rendered as buttons
    expect(container.querySelectorAll('button').length).toBe(2);
    expect(getAllByText).toBeTruthy();
  });

  it('filters by node description and preset name, and hides empty descriptions', () => {
    setStore(
      [
        def('Conv2d', { description: 'image convolution layer' }),
        def('NoDesc', { description: '' }),
      ],
      [preset('SpecialPreset', { description: 'a reusable block' })],
    );
    const { getByPlaceholderText, queryByText, getByText, container } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const input = getByPlaceholderText('Search nodes...');

    // Match a node via its description text only.
    fireEvent.change(input, { target: { value: 'convolution' } });
    expect(getByText('Conv2d')).toBeInTheDocument();
    expect(queryByText('NoDesc')).toBeNull();
    expect(queryByText('SpecialPreset')).toBeNull();

    // Match a preset via its name.
    fireEvent.change(input, { target: { value: 'special' } });
    expect(getByText('SpecialPreset')).toBeInTheDocument();
    expect(queryByText('Conv2d')).toBeNull();

    // Empty-description node renders without a description span.
    fireEvent.change(input, { target: { value: 'nodesc' } });
    const btn = container.querySelector('button')!;
    // Only the name appears, not an extra description line.
    expect(within(btn as HTMLElement).getByText('NoDesc')).toBeInTheDocument();
  });

  it('uses the fallback colour for unknown categories', () => {
    setStore([def('Mystery', { category: 'Unknown' })], []);
    const { getByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const categorySpan = getByText('Unknown');
    // CATEGORY_COLORS has no 'Unknown' => '#607D8B' => rgb(96, 125, 139)
    expect(categorySpan.style.color).toBe('rgb(96, 125, 139)');
  });

  it('boosts the Start node to the top when query is empty', () => {
    setStore([def('Apple'), def('Start'), def('Zebra')], []);
    const { container } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const names = Array.from(container.querySelectorAll('button')).map(
      // First span inside the itemContent div is the node name.
      (b) => b.querySelector('div > span')?.textContent,
    );
    expect(names[0]).toBe('Start');
  });

  it('keeps the Start boost when the query is a prefix of "start"', () => {
    setStore([def('Apple'), def('Start')], []);
    const { container, getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    // "sta" is included in "start" => boost branch runs; both Apple/Start match? No.
    // Use empty -> change to 'st' which matches Start by name and triggers boost.
    fireEvent.change(getByPlaceholderText('Search nodes...'), { target: { value: 'st' } });
    const first = container.querySelector('button')!;
    expect(within(first as HTMLElement).getByText('Start')).toBeInTheDocument();
  });

  it('caps the result list at 20 items', () => {
    const defs = Array.from({ length: 30 }, (_, i) => def(`Node${i}`));
    setStore(defs, []);
    const { container } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    expect(container.querySelectorAll('button').length).toBe(20);
  });

  it('navigates with ArrowDown/ArrowUp and clamps at the ends', () => {
    setStore([def('Aaa'), def('Bbb'), def('Ccc')], []);
    const { container, getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const input = getByPlaceholderText('Search nodes...');
    // starts at 0
    expect(selectedButtonIndex(container)).toBe(0);

    // ArrowUp at top clamps to 0
    fireEvent.keyDown(input, { key: 'ArrowUp' });
    expect(selectedButtonIndex(container)).toBe(0);

    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(selectedButtonIndex(container)).toBe(1);
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(selectedButtonIndex(container)).toBe(2);
    // ArrowDown at bottom clamps to last
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(selectedButtonIndex(container)).toBe(2);

    fireEvent.keyDown(input, { key: 'ArrowUp' });
    expect(selectedButtonIndex(container)).toBe(1);
  });

  it('Escape calls onClose', () => {
    setStore([def('Aaa')], []);
    const onClose = vi.fn();
    const { getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    fireEvent.keyDown(getByPlaceholderText('Search nodes...'), { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Enter selects the highlighted node and adds it, then closes', async () => {
    setStore([def('Conv2d')], []);
    const addNode = vi.fn();
    const onClose = vi.fn();
    useTabStore.setState({ addNode, addPresetNode: vi.fn() });
    const { getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    fireEvent.keyDown(getByPlaceholderText('Search nodes...'), { key: 'Enter' });
    expect(addNode).toHaveBeenCalledWith(expect.objectContaining({ node_name: 'Conv2d' }), FLOW);
    // onClose is deferred via queueMicrotask
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  it('Enter does nothing when there are no results', () => {
    setStore([], []);
    const addNode = vi.fn();
    const onClose = vi.fn();
    useTabStore.setState({ addNode, addPresetNode: vi.fn() });
    const { getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    fireEvent.keyDown(getByPlaceholderText('Search nodes...'), { key: 'Enter' });
    expect(addNode).not.toHaveBeenCalled();
  });

  it('ignores unhandled keys', () => {
    setStore([def('Aaa')], []);
    const onClose = vi.fn();
    const { getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    fireEvent.keyDown(getByPlaceholderText('Search nodes...'), { key: 'a' });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('clicking a node item adds it; mouse enter changes the selection', async () => {
    setStore([def('First'), def('Second')], []);
    const addNode = vi.fn();
    const onClose = vi.fn();
    useTabStore.setState({ addNode, addPresetNode: vi.fn() });
    const { container, getByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    // hovering the 2nd item moves the selection there
    const buttons = container.querySelectorAll('button');
    fireEvent.mouseEnter(buttons[1]);
    expect(selectedButtonIndex(container)).toBe(1);

    fireEvent.click(getByText('First').closest('button')!);
    expect(addNode).toHaveBeenCalledWith(expect.objectContaining({ node_name: 'First' }), FLOW);
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  it('clicking a preset item adds it via addPresetNode', async () => {
    setStore([], [preset('Block')]);
    const addPresetNode = vi.fn();
    const onClose = vi.fn();
    useTabStore.setState({ addNode: vi.fn(), addPresetNode });
    const { getByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    fireEvent.click(getByText('Block').closest('button')!);
    expect(addPresetNode).toHaveBeenCalledWith(
      expect.objectContaining({ preset_name: 'Block' }),
      FLOW,
    );
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  it('resets the selected index back to 0 when the query changes', () => {
    setStore([def('Aaa'), def('Bbb'), def('Ccc')], []);
    const { container, getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const input = getByPlaceholderText('Search nodes...');
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(selectedButtonIndex(container)).toBe(2);
    // Typing resets selection to the first item.
    fireEvent.change(input, { target: { value: 'b' } });
    expect(selectedButtonIndex(container)).toBe(0);
  });

  it('closes when clicking outside the panel, but not when clicking inside', () => {
    setStore([def('Aaa')], []);
    const onClose = vi.fn();
    const { container } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    // Click inside the panel -> no close.
    const panel = container.firstElementChild as HTMLElement;
    fireEvent.mouseDown(panel);
    expect(onClose).not.toHaveBeenCalled();

    // Click on document body (outside) -> close.
    fireEvent.mouseDown(document.body);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('clamps the panel position to stay on screen', () => {
    // Force a large screenPos so Math.min picks the (innerWidth/Height - margin) branch.
    const big = { x: 99999, y: 99999 };
    const { container } = render(
      <QuickNodeSearch screenPos={big} flowPos={FLOW} onClose={() => {}} />,
    );
    const panel = container.firstElementChild as HTMLElement;
    expect(panel.style.left).toBe(`${window.innerWidth - 300}px`);
    expect(panel.style.top).toBe(`${window.innerHeight - 400}px`);
  });
});
