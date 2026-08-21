import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TemplatesTab, groupExamplesByCategory } from './TemplatesTab';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';

// Only the two network calls are stubbed; `insertExample` runs for real so
// the "an example joins the canvas the same way the gallery inserts one"
// contract is exercised end to end rather than mocked away.
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, listExamples: vi.fn(), loadExample: vi.fn() };
});

const mockedRest = vi.mocked(rest);

const activeTab = () => useTabStore.getState().getActiveTab();

/** A serialized example node, in the shape `/api/examples/load` returns. */
function rawNode(id: string) {
  return { id, type: 'Dropout', position: { x: 0, y: 0 }, data: { params: {} } };
}

/** A single empty tab named "Tab 1", so insertion assertions start from zero. */
function freshTab() {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
  useTabStore.getState().addTab('Tab 1');
}

function ex(overrides: Partial<ExampleSummary> = {}): ExampleSummary {
  return {
    name: 'Example',
    description: 'short desc',
    category: 'Usage_Example',
    path: 'Usage_Example/Foo',
    node_count: 3,
    edge_count: 2,
    source: 'builtin',
    ...overrides,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  mockedRest.listExamples.mockReset();
  mockedRest.loadExample.mockReset();
  mockedRest.listExamples.mockResolvedValue([]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('groupExamplesByCategory', () => {
  it('puts usage examples first, architectures last, and the rest alphabetically', () => {
    const groups = groupExamplesByCategory([
      ex({ category: 'Model_Architecture', path: 'a' }),
      ex({ category: 'RNN', path: 'b' }),
      ex({ category: 'Usage_Example', path: 'c' }),
      ex({ category: 'Diffusion', path: 'd' }),
    ]);
    expect(groups.map((g) => g.category)).toEqual([
      'Usage_Example',
      'Diffusion',
      'RNN',
      'Model_Architecture',
    ]);
  });

  it('keeps the backend order of examples within a category', () => {
    const groups = groupExamplesByCategory([
      ex({ name: 'First', path: '1' }),
      ex({ name: 'Second', path: '2' }),
    ]);
    expect(groups[0].items.map((i) => i.name)).toEqual(['First', 'Second']);
  });
});

describe('TemplatesTab', () => {
  it('shows the loading state, then the header, search box and hint', async () => {
    render(<TemplatesTab />);
    expect(screen.getByText('Loading examples...')).toBeTruthy();
    await waitFor(() => expect(screen.queryByText('Loading examples...')).toBeNull());
    expect(screen.getByText('Templates')).toBeTruthy();
    expect(screen.getByPlaceholderText('Search examples...')).toBeTruthy();
    expect(
      screen.getByText('Drag an example onto the canvas, or click to add it'),
    ).toBeTruthy();
  });

  it('lists examples grouped by category, with the node count', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train CNN', category: 'Usage_Example', path: 'u/1', node_count: 7 }),
      ex({ name: 'ResNet', category: 'Model_Architecture', path: 'm/1' }),
    ]);
    render(<TemplatesTab />);
    await screen.findByText('Train CNN');
    // Category label has its underscores replaced for display.
    expect(screen.getByText('Usage Example')).toBeTruthy();
    expect(screen.getByText('Model Architecture')).toBeTruthy();
    expect(screen.getByText('7 nodes')).toBeTruthy();
  });

  it('shows the empty state when the backend has no examples', async () => {
    render(<TemplatesTab />);
    await screen.findByText('No examples available');
  });

  it('filters by name, description, and category', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train CNN', description: 'mnist digits', category: 'Usage_Example', path: 'u/1' }),
      ex({ name: 'Sampler', description: 'toy denoiser', category: 'Diffusion', path: 'd/1' }),
    ]);
    render(<TemplatesTab />);
    await screen.findByText('Train CNN');
    const input = screen.getByPlaceholderText('Search examples...');

    fireEvent.change(input, { target: { value: 'train' } });
    expect(screen.getByText('Train CNN')).toBeTruthy();
    expect(screen.queryByText('Sampler')).toBeNull();

    fireEvent.change(input, { target: { value: 'denoiser' } });
    expect(screen.getByText('Sampler')).toBeTruthy();

    fireEvent.change(input, { target: { value: 'diffusion' } });
    expect(screen.getByText('Sampler')).toBeTruthy();
    expect(screen.queryByText('Train CNN')).toBeNull();

    fireEvent.change(input, { target: { value: 'zzzz' } });
    expect(screen.getByText('No matching examples')).toBeTruthy();
  });

  it('shows the error state and retries on click', async () => {
    mockedRest.listExamples.mockRejectedValueOnce(new Error('server down'));
    render(<TemplatesTab />);
    await screen.findByText('Failed to load examples: server down');

    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Back online' })]);
    fireEvent.click(screen.getByText('Retry'));
    await screen.findByText('Back online');
  });

  it('re-fetches from the refresh button', async () => {
    render(<TemplatesTab />);
    await screen.findByText('No examples available');
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Newly installed' })]);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await screen.findByText('Newly installed');
  });

  // -- #348: an example JOINS the canvas; it never replaces it --

  it('clicking an example inserts it into the canvas instead of replacing it', async () => {
    freshTab();
    useTabStore.getState().setNodes([
      { id: 'mine', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'MINE', type: 'K', params: {} } },
    ] as never);
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Loadable', path: 'Usage_Example/Loadable' }),
    ]);
    mockedRest.loadExample.mockResolvedValue({
      name: '  My Model  ',
      nodes: [rawNode('a'), rawNode('b')],
      edges: [],
    });

    render(<TemplatesTab />);
    fireEvent.click(await screen.findByText('Loadable'));

    await waitFor(() => expect(activeTab().nodes).toHaveLength(3));
    expect(mockedRest.loadExample).toHaveBeenCalledWith('Usage_Example/Loadable');
    // The user's own node is still there, still under its own id.
    expect(activeTab().nodes.find((n) => n.id === 'mine')!.data.label).toBe('MINE');
    // ...and the example's name did NOT take over the tab, because the tab
    // still holds the user's graph.
    expect(activeTab().name).toBe('Tab 1');
  });

  it('a click is one undo step away from the canvas that was there', async () => {
    freshTab();
    useTabStore.getState().setNodes([
      { id: 'mine', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'MINE', type: 'K', params: {} } },
    ] as never);
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Loadable' })]);
    mockedRest.loadExample.mockResolvedValue({
      nodes: [rawNode('a'), rawNode('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'tensor', targetHandle: 'tensor' }],
    });

    render(<TemplatesTab />);
    fireEvent.click(await screen.findByText('Loadable'));
    await waitFor(() => expect(activeTab().nodes).toHaveLength(3));

    useTabStore.getState().undo();
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['mine']);
    expect(activeTab().edges).toHaveLength(0);
  });

  it('inserts from the keyboard, so the drag is an enhancement and not the way in', async () => {
    freshTab();
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Reachable' })]);
    mockedRest.loadExample.mockResolvedValue({ nodes: [rawNode('a')], edges: [] });

    render(<TemplatesTab />);
    const item = (await screen.findByRole('button', { name: /Reachable/ }));
    fireEvent.keyDown(item, { key: 'Enter' });

    await waitFor(() => expect(activeTab().nodes).toHaveLength(1));
  });

  it('makes every example draggable, carrying its path', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Draggable', path: 'Usage_Example/Drag' }),
    ]);
    render(<TemplatesTab />);
    const item = (await screen.findByText('Draggable')).closest('[draggable]') as HTMLElement;
    expect(item).toBeTruthy();

    const setData = vi.fn();
    const dt: any = { setData, effectAllowed: '' };
    fireEvent.dragStart(item, { dataTransfer: dt });

    expect(setData).toHaveBeenCalledWith(
      'application/codefyui-example',
      'Usage_Example/Drag',
    );
    expect(dt.effectAllowed).toBe('move');
  });

  it('surfaces a load failure as a toast and leaves the graph alone', async () => {
    freshTab();
    useTabStore.getState().setNodes([
      { id: 'mine', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'MINE', type: 'K', params: {} } },
    ] as never);
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Broken' })]);
    mockedRest.loadExample.mockRejectedValue(new Error('nope'));
    const addToast = vi.fn();
    useToastStore.setState({ addToast });

    render(<TemplatesTab />);
    fireEvent.click(await screen.findByText('Broken'));

    await waitFor(() =>
      expect(addToast).toHaveBeenCalledWith('Failed to load example', 'error'),
    );
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['mine']);
  });

  it('offers a jump index across example categories', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'A', category: 'Usage_Example', path: 'u/1' }),
      ex({ name: 'B', category: 'Diffusion', path: 'd/1' }),
    ]);
    render(<TemplatesTab />);
    await screen.findByText('A');
    const index = screen.getByRole('navigation', { name: 'Jump to category' });
    expect(within(index).getByRole('button', { name: 'Usage Example' })).toBeTruthy();
    expect(within(index).getByRole('button', { name: 'Diffusion' })).toBeTruthy();
  });

  it('gives an unknown (plugin-defined) category the fallback accent colour', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Plugin Demo', category: 'Something_Else', path: 'plugin:c2/x' }),
    ]);
    render(<TemplatesTab />);
    const header = (await screen.findByText('Something Else')).closest('button')!;
    // #FF9800 — EXAMPLE_CATEGORY_FALLBACK, normalized by jsdom.
    expect(header.style.borderBottom).toContain('rgb(255, 152, 0)');
  });

  it('renders an example with no description without an empty description line', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Terse', description: '' })]);
    const { container } = render(<TemplatesTab />);
    await screen.findByText('Terse');
    expect(container.querySelectorAll('[class*="exampleDesc"]')).toHaveLength(0);
  });
});
