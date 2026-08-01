import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TemplateGalleryModal, exampleSourceLabel } from './TemplateGalleryModal';
import { useDialogStore } from '../../store/dialogStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';

// Only the two network calls are stubbed. `openExampleInNewTab` and
// `insertExample` run for real against the real tab store, so the two actions
// are exercised end to end rather than asserted as "a function was called".
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, listExamples: vi.fn(), loadExample: vi.fn() };
});

const mockedRest = vi.mocked(rest);
const store = () => useTabStore.getState();
const activeTab = () => useTabStore.getState().getActiveTab();

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

function raw(id: string) {
  return { id, type: 'Dropout', position: { x: 0, y: 0 }, data: { params: {} } };
}

/** The card list. Scoped because the detail pane repeats the chosen name. */
const grid = () => screen.getByRole('region', { name: 'Template list' });
const detail = () => screen.getByRole('complementary', { name: 'Template details' });

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({ templateGalleryOpen: true });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
  mockedRest.listExamples.mockReset();
  mockedRest.loadExample.mockReset();
  mockedRest.listExamples.mockResolvedValue([]);
});

afterEach(() => {
  useUIStore.setState({ templateGalleryOpen: false, shortcutsModalOpen: false });
  useDialogStore.setState({ active: null });
  vi.restoreAllMocks();
});

describe('exampleSourceLabel', () => {
  it('names the plugin pack, and nothing for a builtin', () => {
    expect(exampleSourceLabel(ex({ source: 'plugin:c2' }))).toBe('c2');
    expect(exampleSourceLabel(ex({ source: 'builtin' }))).toBeNull();
  });
});

describe('TemplateGalleryModal', () => {
  it('renders nothing while closed', () => {
    useUIStore.setState({ templateGalleryOpen: false });
    const { container } = render(<TemplateGalleryModal />);
    expect(container.firstChild).toBeNull();
    expect(mockedRest.listExamples).not.toHaveBeenCalled();
  });

  it('is reachable with a populated canvas and lists examples by category', async () => {
    // The whole point of the modal: before it, a canvas with nodes on it had
    // no way back to the examples at all.
    store().setNodes([
      { id: 'mine', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'K', params: {} } },
    ] as never);
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'ResNet', category: 'Model_Architecture', path: 'm/1' }),
      ex({ name: 'Train CNN', category: 'Usage_Example', path: 'u/1', node_count: 7 }),
    ]);

    render(<TemplateGalleryModal />);
    await screen.findByText('Train CNN');

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    // Category labels have their underscores replaced for display, and appear
    // in the shared order (usage examples first, architectures last).
    const listed = grid().textContent ?? '';
    expect(listed.indexOf('Usage Example')).toBeLessThan(listed.indexOf('Model Architecture'));
    expect(within(grid()).getByText('ResNet')).toBeInTheDocument();
    // The canvas keeps its own graph while the gallery is up.
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['mine']);
  });

  it('shows the loading, empty, and error states', async () => {
    render(<TemplateGalleryModal />);
    expect(screen.getByText('Loading examples...')).toBeInTheDocument();
    await screen.findByText('No examples available');
  });

  it('offers a retry after a failed list', async () => {
    mockedRest.listExamples.mockRejectedValueOnce(new Error('server down'));
    render(<TemplateGalleryModal />);
    await screen.findByText('Failed to load examples: server down');

    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Back online' })]);
    fireEvent.click(screen.getByText('Retry'));
    await waitFor(() =>
      expect(within(grid()).getByText('Back online')).toBeInTheDocument(),
    );
  });

  it('filters by name, description, category and source pack', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train CNN', description: 'mnist digits', category: 'Usage_Example', path: 'u/1' }),
      ex({ name: 'Sampler', description: 'toy denoiser', category: 'Diffusion', path: 'd/1', source: 'plugin:c4' }),
    ]);
    render(<TemplateGalleryModal />);
    await waitFor(() => expect(within(grid()).getByText('Train CNN')).toBeInTheDocument());
    const input = screen.getByPlaceholderText('Search templates...');

    fireEvent.change(input, { target: { value: 'denoiser' } });
    expect(within(grid()).getByText('Sampler')).toBeInTheDocument();
    expect(within(grid()).queryByText('Train CNN')).toBeNull();

    fireEvent.change(input, { target: { value: 'c4' } });
    expect(within(grid()).getByText('Sampler')).toBeInTheDocument();

    fireEvent.change(input, { target: { value: 'zzzz' } });
    expect(screen.getByText('No matching examples')).toBeInTheDocument();
  });

  it('describes the first example until another is chosen', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'First', description: 'the first one', path: 'a', node_count: 4, edge_count: 3 }),
      ex({ name: 'Second', description: 'the second one', path: 'b', source: 'plugin:c2' }),
    ]);
    render(<TemplateGalleryModal />);
    await waitFor(() =>
      expect(within(detail()).getByText('the first one')).toBeInTheDocument(),
    );

    expect(within(detail()).getByText('First')).toBeInTheDocument();
    expect(within(detail()).getByText('4 nodes')).toBeInTheDocument();
    expect(within(detail()).getByText('3 connections')).toBeInTheDocument();
    expect(within(detail()).getByText('Built-in example')).toBeInTheDocument();

    fireEvent.click(within(grid()).getByText('Second'));
    expect(within(detail()).getByText('the second one')).toBeInTheDocument();
    expect(within(detail()).getByText('From plugin pack "c2"')).toBeInTheDocument();
  });

  it('falls back to a placeholder for a template with no description', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Terse', description: '' })]);
    render(<TemplateGalleryModal />);
    await screen.findByText('This template ships no description.');
    expect(within(grid()).getByText('Terse')).toBeInTheDocument();
  });

  it('opens the chosen template in a new tab, leaving the current graph alone', async () => {
    store().setNodes([
      { id: 'mine', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'K', params: {} } },
    ] as never);
    const firstTabId = store().activeTabId;
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Template', path: 'u/1' })]);
    mockedRest.loadExample.mockResolvedValue({ name: 'Template', nodes: [raw('a')], edges: [] });

    render(<TemplateGalleryModal />);
    await waitFor(() => expect(within(grid()).getByText('Template')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Open in new tab'));

    await waitFor(() => expect(store().tabs).toHaveLength(2));
    expect(mockedRest.loadExample).toHaveBeenCalledWith('u/1');
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['a']);
    expect(store().getTab(firstTabId)!.nodes.map((n) => n.id)).toEqual(['mine']);
    // The modal gets out of the way once it has done its job.
    expect(useUIStore.getState().templateGalleryOpen).toBe(false);
  });

  it('double-clicking a card opens it in a new tab too', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Template', path: 'u/1' })]);
    mockedRest.loadExample.mockResolvedValue({ nodes: [raw('a')], edges: [] });

    render(<TemplateGalleryModal />);
    await waitFor(() => expect(within(grid()).getByText('Template')).toBeInTheDocument());
    const card = within(grid()).getByText('Template');
    // Two rapid double-clicks: the second must not start a second load, or
    // the user gets two tabs (and one of them after the modal has closed).
    fireEvent.doubleClick(card);
    fireEvent.doubleClick(card);

    await waitFor(() => expect(store().tabs).toHaveLength(2));
    expect(mockedRest.loadExample).toHaveBeenCalledTimes(1);
    expect(mockedRest.loadExample).toHaveBeenCalledWith('u/1');
  });

  it('inserts into the current canvas with remapped ids', async () => {
    // Canvas and template share the id "a": inserting must not replace the
    // user's node with the template's.
    store().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'MINE', type: 'K', params: {} } },
    ] as never);
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Template', path: 'u/1' })]);
    mockedRest.loadExample.mockResolvedValue({
      name: 'Template',
      nodes: [raw('a'), raw('b')],
      edges: [{ id: 'e1', source: 'a', target: 'b', sourceHandle: 'tensor', targetHandle: 'tensor' }],
    });

    render(<TemplateGalleryModal />);
    await waitFor(() => expect(within(grid()).getByText('Template')).toBeInTheDocument());
    fireEvent.click(screen.getByText('Insert into this canvas'));

    await waitFor(() => expect(activeTab().nodes).toHaveLength(3));
    expect(store().tabs).toHaveLength(1);
    expect(activeTab().nodes.find((n) => n.id === 'a')!.data.label).toBe('MINE');
    expect(new Set(activeTab().nodes.map((n) => n.id)).size).toBe(3);
    expect(useUIStore.getState().templateGalleryOpen).toBe(false);
  });

  it('closes on Escape', async () => {
    render(<TemplateGalleryModal />);
    await screen.findByText('No examples available');
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().templateGalleryOpen).toBe(false);
  });

  it('leaves Escape alone while something is stacked above it', async () => {
    render(<TemplateGalleryModal />);
    await screen.findByText('No examples available');

    // The shortcuts modal (`?`) renders over whatever is showing and has no
    // Escape handler of its own — closing the gallery underneath it would
    // dismiss the surface the user is NOT looking at.
    useUIStore.setState({ shortcutsModalOpen: true });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().templateGalleryOpen).toBe(true);
    useUIStore.setState({ shortcutsModalOpen: false });

    // Same for a confirm/prompt dialog.
    useDialogStore.setState({ active: { kind: 'confirm', title: 'x' } as never });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().templateGalleryOpen).toBe(true);
    useDialogStore.setState({ active: null });

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().templateGalleryOpen).toBe(false);
  });

  it('closes on a backdrop press but not on a press inside the surface', async () => {
    render(<TemplateGalleryModal />);
    await screen.findByText('No examples available');
    const dialog = screen.getByRole('dialog');

    fireEvent.mouseDown(dialog);
    expect(useUIStore.getState().templateGalleryOpen).toBe(true);

    fireEvent.mouseDown(dialog.parentElement!);
    expect(useUIStore.getState().templateGalleryOpen).toBe(false);
  });

  it('closes on the close button', async () => {
    render(<TemplateGalleryModal />);
    await screen.findByText('No examples available');
    fireEvent.click(screen.getByRole('button', { name: 'Close template gallery' }));
    expect(useUIStore.getState().templateGalleryOpen).toBe(false);
  });

  it('says so when nothing is selectable', async () => {
    render(<TemplateGalleryModal />);
    await screen.findByText('No examples available');
    expect(screen.getByText('Select a template to see what it contains')).toBeInTheDocument();
  });
});
