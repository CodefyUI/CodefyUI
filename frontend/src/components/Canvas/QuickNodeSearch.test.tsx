import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, fireEvent, waitFor, within } from '@testing-library/react';
import { QuickNodeSearch } from './QuickNodeSearch';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { _resetPluginStoreForTesting, usePluginStore } from '../../store/pluginStore';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import type { PluginCatalogEntry } from '../../api/rest';
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

function pluginEntry(over: Partial<PluginCatalogEntry> & { id: string }): PluginCatalogEntry {
  return {
    name: over.id,
    description: '',
    kind: 'github',
    official: false,
    status: 'installed',
    source_kind: 'github_url',
    source: over.id,
    repo: null,
    ref: null,
    sha: null,
    url: null,
    homepage: '',
    version: null,
    installed_at: null,
    enabled: true,
    chapters: [],
    lessons: [],
    tags: [],
    nodes: [],
    node_count: 0,
    capabilities: [],
    trusted_modules: [],
    python_deps: {},
    has_frontend: false,
    consent_required: false,
    frontend_entry: null,
    job: null,
    ...over,
  };
}

const eduEntry = () => pluginEntry({ id: 'edu', name: 'EDU - hands-on teaching nodes' });

/** Put a catalog in the plugin store, the way a finished `refresh()` would. */
function seedPlugins(...entries: PluginCatalogEntry[]) {
  usePluginStore.setState({
    loaded: true,
    unsupported: false,
    plugins: entries,
    byId: Object.fromEntries(entries.map((entry) => [entry.id, entry])),
  });
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
    // Every case that does not seed one runs against an empty plugin catalog:
    // the base install, where nothing in the library came from a plugin.
    _resetPluginStoreForTesting();
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
    // The label itself is no longer tinted: a hue drawn on a tint of itself
    // cannot reach 4.5:1, so the label takes the shared text tier and the hue
    // identifies the category through the dot instead.
    expect(categorySpan.style.color).toBe('');
    // CATEGORY_COLORS has no 'Unknown', so it falls back to the Utility hue
    // every other unknown-category node uses -- not to a stale literal.
    const dot = categorySpan.closest('button')?.querySelector('span');
    expect(dot).toBeTruthy();
    expect((dot as HTMLElement).style.background).toBe('rgb(128, 151, 162)');
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

  // ── Dismissal on a surface that owns the pointer ─────────────────────────
  // React Flow's pane is dragged by d3-zoom, whose mousedown handler calls
  // `nopropagation(event)` so it can own panning. A bubble-phase document
  // listener never sees that event, which is why clicking the canvas used to
  // leave the palette open; and because clicking also blurred the input, the
  // Escape key that lived on the input stopped working too, leaving no way
  // out at all.

  /** A stand-in for the React Flow pane: swallows mousedown like d3-zoom. */
  function paneThatSwallowsMouseDown(): HTMLElement {
    const pane = document.createElement('div');
    pane.addEventListener('mousedown', (e) => e.stopImmediatePropagation());
    document.body.appendChild(pane);
    return pane;
  }

  it('closes on a click that the surface underneath stops propagating', () => {
    setStore([def('Aaa')], []);
    const onClose = vi.fn();
    render(<QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />);

    const pane = paneThatSwallowsMouseDown();
    fireEvent.mouseDown(pane);
    expect(onClose).toHaveBeenCalledTimes(1);
    pane.remove();
  });

  it('Escape still closes after the input has lost focus', () => {
    setStore([def('Aaa')], []);
    const onClose = vi.fn();
    const { getByPlaceholderText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    // What clicking the canvas does to the palette: the input is no longer
    // where keys are routed.
    (getByPlaceholderText('Search nodes...') as HTMLInputElement).blur();
    fireEvent.keyDown(document.body, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('a key other than Escape does not close it', () => {
    setStore([def('Aaa')], []);
    const onClose = vi.fn();
    render(<QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />);
    fireEvent.keyDown(document.body, { key: 'a' });
    fireEvent.keyDown(document.body, { key: 'Enter' });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('stops listening once unmounted', () => {
    setStore([def('Aaa')], []);
    const onClose = vi.fn();
    const { unmount } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={onClose} />,
    );
    unmount();
    fireEvent.mouseDown(document.body);
    fireEvent.keyDown(document.body, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
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

  // ── Plugin provenance (P-F3) ────────────────────────────────────────────

  it('finds a plugin node by the plugin display name', () => {
    // The same third field the palette search gained: the plugin as the
    // Plugin Center names it, which appears in no field of a definition.
    seedPlugins(eduEntry());
    setStore(
      [def('edu:FilterRows', { description: 'drops rows a predicate rejects', provider: 'plugin:edu' }),
        def('Conv2d')],
      [],
    );
    const { getByPlaceholderText, getByText, queryByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const input = getByPlaceholderText('Search nodes...');

    fireEvent.change(input, { target: { value: 'hands-on' } });
    expect(getByText('edu:FilterRows')).toBeInTheDocument();
    expect(queryByText('Conv2d')).toBeNull();

    // Case-insensitive, like the two fields it joins.
    fireEvent.change(input, { target: { value: 'TEACHING' } });
    expect(getByText('edu:FilterRows')).toBeInTheDocument();
  });

  it('matches the plugin id while the catalog has not answered', () => {
    // The id here is deliberately NOT a substring of the node name or of the
    // description: `edu:FilterRows` is contributed by the plugin `teach`, so
    // the query below reaches this row through nothing but the third clause
    // falling back to the id of a catalog that has not answered yet. (An id
    // that also spells the node's prefix would pass with the clause deleted.)
    setStore(
      [
        def('edu:FilterRows', {
          description: 'drops rows a predicate rejects',
          provider: 'plugin:teach',
        }),
        def('Conv2d'),
      ],
      [],
    );
    const { getByPlaceholderText, getByText, queryByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const input = getByPlaceholderText('Search nodes...');

    fireEvent.change(input, { target: { value: 'teach' } });
    expect(getByText('edu:FilterRows')).toBeInTheDocument();
    expect(queryByText('Conv2d')).toBeNull();

    // The other half of "has not answered": the display name lives only in
    // the catalog, so with an empty index a query that matches nothing but
    // that name finds nothing. Seeded, it is the case above this one.
    fireEvent.change(input, { target: { value: 'hands-on' } });
    expect(getByText('No matching nodes')).toBeInTheDocument();
    expect(queryByText('edu:FilterRows')).toBeNull();
  });

  it('leaves built-ins, custom nodes and presets unreachable by a plugin name', () => {
    seedPlugins(eduEntry());
    setStore(
      [def('Conv2d', { provider: 'builtin' }), def('MyLayer', { provider: 'custom' })],
      [preset('MyBlock')],
    );
    const { getByPlaceholderText, getByText, queryByText } = render(
      <QuickNodeSearch screenPos={SCREEN} flowPos={FLOW} onClose={() => {}} />,
    );
    const input = getByPlaceholderText('Search nodes...');

    fireEvent.change(input, { target: { value: 'hands-on' } });
    expect(getByText('No matching nodes')).toBeInTheDocument();

    // Counterfactual: the same three are still found by their own fields.
    fireEvent.change(input, { target: { value: 'my' } });
    expect(getByText('MyLayer')).toBeInTheDocument();
    expect(getByText('MyBlock')).toBeInTheDocument();
    expect(queryByText('Conv2d')).toBeNull();
  });
});
