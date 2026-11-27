import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { TemplateGalleryModal } from './TemplateGalleryModal';
import { useDialogStore } from '../../store/dialogStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { _resetPluginStoreForTesting, usePluginStore } from '../../store/pluginStore';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import type { ExampleSummary, PluginCatalogEntry } from '../../api/rest';
import { pluginEntry as catalogEntry } from '../../test/pluginEntry';

// Only the two network calls are stubbed. `openExampleInNewTab` and
// `insertExample` run for real against the real tab store, so the two actions
// are exercised end to end rather than asserted as "a function was called".
vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, listExamples: vi.fn(), listExampleTags: vi.fn(), loadExample: vi.fn() };
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

/** An installed, enabled third-party plugin off GitHub: the row these cases name. */
const pluginEntry = (over: Partial<PluginCatalogEntry> & { id: string }): PluginCatalogEntry =>
  catalogEntry({
    kind: 'github',
    official: false,
    status: 'installed',
    source_kind: 'github_url',
    enabled: true,
    ...over,
  });

/** The card list. Scoped because the detail pane repeats the chosen name. */
const grid = () => screen.getByRole('region', { name: 'Template list' });
const detail = () => screen.getByRole('complementary', { name: 'Template details' });

/** Each section heading, without the item count its last child carries. */
const headings = () =>
  [...grid().querySelectorAll('h3')].map((h) =>
    [...h.childNodes]
      .filter((n) => n.nodeType === Node.TEXT_NODE)
      .map((n) => n.textContent)
      .join(''),
  );

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useNodeDefStore.setState({ definitions: [], presets: [] });
  // An empty plugin catalog unless a case seeds one: the state the modal
  // renders in before the boot fetch lands, where the id stands in.
  _resetPluginStoreForTesting();
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({ templateGalleryOpen: true });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
  mockedRest.listExamples.mockReset();
  mockedRest.loadExample.mockReset();
  mockedRest.listExamples.mockResolvedValue([]);
  // No tags route unless a case gives one: the order and counts then come
  // from the list itself, as against an older server.
  mockedRest.listExampleTags.mockReset();
  mockedRest.listExampleTags.mockRejectedValue(new Error('404'));
});

afterEach(() => {
  // Unmount first: this hook runs before setup.ts's cleanup, and a store
  // write with the tree still mounted is a render outside act() (#505).
  cleanup();
  useUIStore.setState({
    templateGalleryOpen: false,
    shortcutsModalOpen: false,
    packCenterOpen: false,
  });
  useDialogStore.setState({ active: null });
  vi.restoreAllMocks();
});

// The pane's built-in/plugin gate is `utils/provider`'s `pluginNameOf`, and
// the two unit cases the modal used to hold for a prefix test of its own live
// in `utils/provider.test.ts` beside it. What stays here is what only a render
// can answer: which sentence the detail pane shows.

describe('TemplateGalleryModal', () => {
  it('renders nothing while closed', () => {
    useUIStore.setState({ templateGalleryOpen: false });
    const { container } = render(<TemplateGalleryModal />);
    expect(container.firstChild).toBeNull();
    expect(mockedRest.listExamples).not.toHaveBeenCalled();
  });

  it('is reachable with a populated canvas and lists examples by section', async () => {
    // The whole point of the modal: before it, a canvas with nodes on it had
    // no way back to the examples at all.
    store().setNodes([
      { id: 'mine', type: 'baseNode', position: { x: 0, y: 0 }, data: { label: 'A', type: 'K', params: {} } },
    ] as never);
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'ResNet', category: 'Model_Architecture', path: 'm/1', section: 'architectures' }),
      ex({
        name: 'Train CNN',
        category: 'Usage_Example',
        path: 'u/1',
        node_count: 7,
        section: 'quickstart',
      }),
    ]);

    render(<TemplateGalleryModal />);
    // In the grid: the detail pane names the preselected first card too.
    await within(grid()).findByText('Train CNN');

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    // The same sections, in the same order, as the sidebar's Templates tab and
    // the empty-canvas overlay (#141).
    expect(headings()).toEqual(['Quick start', 'Architectures']);
    expect(within(grid()).getByText('ResNet')).toBeInTheDocument();
    // The canvas keeps its own graph while the gallery is up.
    expect(activeTab().nodes.map((n) => n.id)).toEqual(['mine']);
  });

  it('heads an architecture family and a pack with its own name', async () => {
    usePluginStore.setState({
      loaded: true,
      byId: { c2: pluginEntry({ id: 'c2', name: 'Chapter 2' }) },
    });
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'ResNet', category: 'Model_Architecture', path: 'm/1', section: 'architectures', family: 'CNN' }),
      ex({ name: 'LSTM', category: 'Model_Architecture', path: 'm/2', section: 'architectures', family: 'RNN' }),
      ex({ name: 'Lesson', category: 'Classical', path: 'plugin:c2/1', source: 'plugin:c2' }),
    ]);

    render(<TemplateGalleryModal />);
    await waitFor(() => expect(within(grid()).getByText('ResNet')).toBeInTheDocument());
    expect(headings()).toEqual(['CNN', 'RNN', 'Chapter 2']);
  });

  it('prints at most two free tags on a card, never the main tag, and +n for the rest', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({
        name: 'Augment',
        path: 'u/1',
        main_tag: 'training',
        tags: ['training', 'vision', 'cnn', 'augmentation', 'mnist'],
      }),
      ex({ name: 'Iris', category: 'Classical', path: 'c/1', main_tag: 'data', tags: ['data', 'tabular'] }),
    ]);

    render(<TemplateGalleryModal />);
    await waitFor(() => expect(within(grid()).getByText('Augment')).toBeInTheDocument());

    const chips = [...grid().querySelectorAll('[class*="cardChip"]')].map((el) => el.textContent);
    // The folder ("Usage Example") is no longer on the card: the tags are.
    expect(chips).toEqual(['Vision', 'CNN', 'Tabular']);
    const more = grid().querySelector('[class*="cardMore"]')!;
    expect(more.textContent).toBe('+2');
    expect(more.getAttribute('title')).toBe('2 more tags: Augmentation, MNIST');
    // The detail pane still names the category of whatever is chosen — the
    // first card shown, until a card is clicked (#678) — because it is the one
    // place with room for the example's own provenance.
    expect(within(detail()).getByText('Usage Example')).toBeInTheDocument();
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
    expect(screen.getByText('No examples match these filters')).toBeInTheDocument();
  });

  // #684: the chip and the detail pane named the category in English in the
  // Chinese UI. A search matches it in both languages.
  it('names the category in the UI language in the detail pane and finds it by either name', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train CNN', category: 'Usage_Example', path: 'u/1', section: 'training' }),
      ex({ name: 'Iris', category: 'Classical', path: 'c/1', section: 'concepts' }),
    ]);
    const { unmount } = render(<TemplateGalleryModal />);
    const zhGrid = () => screen.getByRole('region', { name: '範例清單' });
    await waitFor(() => expect(within(zhGrid()).getByText('Train CNN')).toBeInTheDocument());

    expect(within(screen.getByRole('complementary', { name: '範例詳情' })).getByText('使用範例')).toBeInTheDocument();

    const input = screen.getByPlaceholderText('搜尋範例...');
    fireEvent.change(input, { target: { value: '使用範例' } });
    expect(within(zhGrid()).getByText('Train CNN')).toBeInTheDocument();
    expect(within(zhGrid()).queryByText('Iris')).toBeNull();
    fireEvent.change(input, { target: { value: 'usage' } });
    expect(within(zhGrid()).getByText('Train CNN')).toBeInTheDocument();
    expect(within(zhGrid()).queryByText('Iris')).toBeNull();
    // Unmounted here, so closing the gallery in afterEach re-renders nothing
    // outside act.
    unmount();
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

  it('preselects the first card shown, in display order, before and after a search', async () => {
    // #678: the API lists an architecture before the quick-start examples,
    // but the gallery renders Quick Start first. The pane, the pressed card
    // and "Open in new tab" all have to agree with the card at the top.
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Two moons', description: 'six classifiers', category: 'Model_Architecture', path: 'm/moons', section: 'architectures' }),
      ex({ name: 'Inference CNN on MNIST', description: 'mnist inference', path: 'u/infer', section: 'quickstart' }),
      ex({ name: 'Train CNN on MNIST', description: 'mnist training', path: 'u/train', section: 'quickstart', order: 0 }),
    ]);
    mockedRest.loadExample.mockResolvedValue({ name: 'Train CNN on MNIST', nodes: [raw('a')], edges: [] });
    render(<TemplateGalleryModal />);
    await within(grid()).findByText('Two moons');

    const cards = () => within(grid()).getAllByRole('button');
    const pressed = () =>
      cards().filter((c) => c.getAttribute('aria-pressed') === 'true');
    const names = () => cards().map((c) => c.querySelector('span')!.textContent);

    // Display order: Quick Start (order 0 first), then the architecture.
    expect(names()).toEqual(['Train CNN on MNIST', 'Inference CNN on MNIST', 'Two moons']);
    expect(pressed()).toEqual([cards()[0]]);
    expect(within(detail()).getByText('mnist training')).toBeInTheDocument();
    expect(within(detail()).queryByText('six classifiers')).toBeNull();

    // After a search the fallback is the first card of the filtered list as
    // rendered, which is not the first match in API order.
    fireEvent.change(screen.getByPlaceholderText('Search templates...'), {
      target: { value: 'mnist' },
    });
    expect(names()).toEqual(['Train CNN on MNIST', 'Inference CNN on MNIST']);
    expect(pressed()).toEqual([cards()[0]]);
    expect(within(detail()).getByText('mnist training')).toBeInTheDocument();

    // "Open in new tab" opens the example the pane describes.
    fireEvent.click(screen.getByText('Open in new tab'));
    await waitFor(() => expect(store().tabs).toHaveLength(2));
    expect(mockedRest.loadExample).toHaveBeenCalledWith('u/train');
  });

  it('names the plugin the catalog knows, and the id until it answers', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Lesson', description: 'a lesson graph', path: 'e/1', source: 'plugin:edu' }),
    ]);
    render(<TemplateGalleryModal />);
    await waitFor(() =>
      expect(within(detail()).getByText('a lesson graph')).toBeInTheDocument(),
    );

    // Before the catalog answers the id stands in — still a true sentence.
    expect(within(detail()).getByText('From plugin pack "edu"')).toBeInTheDocument();

    // The catalog lands: the same pane names the plugin as the Plugin Center
    // does, without a reload and without the search behind it changing.
    act(() => {
      usePluginStore.setState({
        loaded: true,
        byId: { edu: pluginEntry({ id: 'edu', name: 'EDU - hands-on teaching nodes' }) },
      });
    });
    expect(
      within(detail()).getByText('From plugin pack "EDU - hands-on teaching nodes"'),
    ).toBeInTheDocument();
    expect(within(detail()).queryByText('From plugin pack "edu"')).toBeNull();
  });

  it('still calls a source that names no plugin a built-in', async () => {
    // `plugin:` with nothing after it names nothing, and the pane said
    // "Built-in example" for it before the catalog was consulted at all.
    // Resolving a name must not turn that into an empty pair of quotes.
    const older = ex({ name: 'Older', description: 'no source at all', path: 'o/2' });
    // The wire type makes `source` optional: a frontend built from source can
    // meet a backend that omits it, and that example is a built-in too.
    delete older.source;
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Odd', description: 'a malformed source', path: 'o/1', source: 'plugin:' }),
      older,
    ]);
    render(<TemplateGalleryModal />);
    await waitFor(() =>
      expect(within(detail()).getByText('a malformed source')).toBeInTheDocument(),
    );
    expect(within(detail()).getByText('Built-in example')).toBeInTheDocument();

    fireEvent.click(within(grid()).getByText('Older'));
    expect(within(detail()).getByText('no source at all')).toBeInTheDocument();
    expect(within(detail()).getByText('Built-in example')).toBeInTheDocument();
  });

  it('falls back to a placeholder for a template with no description', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Terse', description: '' })]);
    render(<TemplateGalleryModal />);
    await screen.findByText('No description.');
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

    // A confirm/prompt dialog: closing the gallery underneath it would
    // dismiss the surface the user is NOT looking at. (The shortcuts sheet
    // stops its own Escape; `ShortcutsModal.test.tsx` covers that.)
    useDialogStore.setState({ active: { kind: 'confirm', title: 'x' } as never });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().templateGalleryOpen).toBe(true);
    useDialogStore.setState({ active: null });

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().templateGalleryOpen).toBe(false);
  });

  it('leaves Escape alone while the Package Center is open', async () => {
    render(<TemplateGalleryModal />);
    await screen.findByText('No examples available');

    // Same rule as the shortcuts modal above: the Package Center renders over
    // this one, and Escape belongs to the surface the user is looking at.
    useUIStore.setState({ packCenterOpen: true });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().templateGalleryOpen).toBe(true);

    useUIStore.setState({ packCenterOpen: false });
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

  // ── Phase 3: main-tag chips, the tag filter, the empty state ──

  describe('filters', () => {
    // Two tags on three examples (above the threshold), two below it.
    const catalog = () => [
      ex({ name: 'Paper', path: 'p/1', main_tag: 'papers', tags: ['papers', 'tabular'] }),
      ex({ name: 'Train A', path: 't/1', main_tag: 'training', tags: ['training', 'vision', 'cnn'] }),
      ex({ name: 'Train B', path: 't/2', main_tag: 'training', tags: ['training', 'vision'] }),
      ex({ name: 'Eval', path: 'e/1', main_tag: 'evaluation', tags: ['evaluation', 'vision', 'cnn', 'nlp'] }),
      ex({ name: 'Chat', path: 'l/1', main_tag: 'llm', tags: ['llm', 'nlp', 'cnn'] }),
      ex({ name: 'Lesson', path: 'plugin:c2/1', source: 'plugin:c2', main_tag: 'courses', tags: ['courses'] }),
    ];
    const tabs = () => within(screen.getByRole('tablist', { name: 'Example categories' })).getAllByRole('tab');
    const tab = (name: RegExp) => screen.getByRole('tab', { name });
    const cardNames = () =>
      within(grid())
        .queryAllByRole('button')
        .filter((b) => b.hasAttribute('aria-pressed'))
        .map((c) => c.querySelector('span')!.textContent);

    it('opens on All, with the practical categories before papers and courses', async () => {
      mockedRest.listExamples.mockResolvedValue(catalog());
      render(<TemplateGalleryModal />);
      await within(grid()).findByText('Paper');

      expect(tabs().map((el) => el.getAttribute('aria-selected'))[0]).toBe('true');
      // Each chip is its name and its count, in text.
      expect(tabs().map((el) => el.textContent)).toEqual([
        'All6', 'Training2', 'LLM & RAG1', 'Evaluation1', 'Papers1', 'Courses1',
      ]);
      expect(headings()).toEqual(['Training', 'LLM & RAG', 'Evaluation', 'Papers', 'c2']);
    });

    it('follows the main-tag order the server sends', async () => {
      mockedRest.listExamples.mockResolvedValue(catalog());
      mockedRest.listExampleTags.mockResolvedValue({
        // Deliberately not the built-in order: the server's wins.
        main_tags: [
          { id: 'evaluation', order: 1, count: 1 },
          { id: 'training', order: 2, count: 2 },
          { id: 'llm', order: 3, count: 1 },
          { id: 'papers', order: 4, count: 1 },
          { id: 'courses', order: 5, count: 1 },
        ],
        tags: [],
      });
      render(<TemplateGalleryModal />);
      await within(grid()).findByText('Paper');
      await waitFor(() => expect(headings()[0]).toBe('Evaluation'));
      expect(headings()).toEqual(['Evaluation', 'Training', 'LLM & RAG', 'Papers', 'c2']);
    });

    it('narrows to one main tag, and the arrow keys move along the chips', async () => {
      mockedRest.listExamples.mockResolvedValue(catalog());
      render(<TemplateGalleryModal />);
      await within(grid()).findByText('Paper');

      fireEvent.click(tab(/^Training/));
      expect(tab(/^Training/)).toHaveAttribute('aria-selected', 'true');
      expect(tab(/^Training/)).toHaveAttribute('tabindex', '0');
      expect(tab(/^All/)).toHaveAttribute('tabindex', '-1');
      expect(cardNames()).toEqual(['Train A', 'Train B']);

      fireEvent.keyDown(tab(/^Training/), { key: 'ArrowRight' });
      expect(tab(/^LLM/)).toHaveAttribute('aria-selected', 'true');
      expect(document.activeElement).toBe(tab(/^LLM/));
      expect(cardNames()).toEqual(['Chat']);

      fireEvent.keyDown(tab(/^LLM/), { key: 'Home' });
      expect(tab(/^All/)).toHaveAttribute('aria-selected', 'true');
      expect(cardNames()).toHaveLength(6);
    });

    it('offers only tags on three or more examples until More is pressed', async () => {
      mockedRest.listExamples.mockResolvedValue(catalog());
      render(<TemplateGalleryModal />);
      await within(grid()).findByText('Paper');

      const toggle = screen.getByRole('button', { name: 'Filter by tag' });
      expect(toggle).toHaveAttribute('aria-expanded', 'false');
      fireEvent.click(toggle);
      expect(toggle).toHaveAttribute('aria-expanded', 'true');

      const group = () => screen.getByRole('group', { name: 'Filter by tag' });
      const offered = () =>
        within(group())
          .getAllByRole('button')
          .filter((b) => b.hasAttribute('aria-pressed'))
          .map((b) => b.getAttribute('aria-label'));
      // Most used first; a tie in id order.
      expect(offered()).toEqual(['CNN, 3 examples', 'Vision, 3 examples']);

      fireEvent.click(within(group()).getByRole('button', { name: 'More tags' }));
      expect(offered()).toEqual([
        'CNN, 3 examples', 'Vision, 3 examples', 'NLP, 2 examples', 'Tabular, 1 examples',
      ]);
      fireEvent.click(within(group()).getByRole('button', { name: 'Fewer tags' }));
      expect(offered()).toHaveLength(2);
    });

    it('combines selected tags as an intersection, with the main tag and the search', async () => {
      mockedRest.listExamples.mockResolvedValue(catalog());
      render(<TemplateGalleryModal />);
      await within(grid()).findByText('Paper');
      fireEvent.click(screen.getByRole('button', { name: 'Filter by tag' }));

      fireEvent.click(screen.getByRole('button', { name: 'Vision, 3 examples' }));
      expect(cardNames()).toEqual(['Train A', 'Train B', 'Eval']);
      fireEvent.click(screen.getByRole('button', { name: 'CNN, 3 examples' }));
      expect(screen.getByRole('button', { name: 'CNN, 3 examples' })).toHaveAttribute('aria-pressed', 'true');
      expect(cardNames()).toEqual(['Train A', 'Eval']);
      // The toggle counts the selection, so a collapsed row still says so.
      expect(screen.getByRole('button', { name: 'Filter by tag (2)' })).toBeInTheDocument();

      fireEvent.click(tab(/^Evaluation/));
      expect(cardNames()).toEqual(['Eval']);
      fireEvent.change(screen.getByPlaceholderText('Search templates...'), { target: { value: 'train' } });
      expect(cardNames()).toEqual([]);
    });

    it('says which filters emptied the list, and clears them all at once', async () => {
      mockedRest.listExamples.mockResolvedValue(catalog());
      render(<TemplateGalleryModal />);
      await within(grid()).findByText('Paper');
      fireEvent.click(screen.getByRole('button', { name: 'Filter by tag' }));
      fireEvent.click(screen.getByRole('button', { name: 'Vision, 3 examples' }));
      fireEvent.click(tab(/^Papers/));
      fireEvent.change(screen.getByPlaceholderText('Search templates...'), { target: { value: 'zz' } });

      expect(within(grid()).getByText('No examples match these filters')).toBeInTheDocument();
      expect(within(grid()).getByText('Category: Papers')).toBeInTheDocument();
      expect(within(grid()).getByText('Tags: Vision')).toBeInTheDocument();
      expect(within(grid()).getByText('Search: "zz"')).toBeInTheDocument();

      fireEvent.click(within(grid()).getByRole('button', { name: 'Clear filters' }));
      expect(cardNames()).toHaveLength(6);
      expect(tab(/^All/)).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByPlaceholderText('Search templates...')).toHaveValue('');
      expect(screen.getByRole('button', { name: 'Vision, 3 examples' })).toHaveAttribute('aria-pressed', 'false');
    });

    it('lists every tag in the detail pane, and a tag there is a filter', async () => {
      mockedRest.listExamples.mockResolvedValue(catalog());
      render(<TemplateGalleryModal />);
      await within(grid()).findByText('Paper');
      fireEvent.click(within(grid()).getByText('Eval'));

      const tagButtons = () =>
        within(detail())
          .getAllByRole('button')
          .filter((b) => b.title.startsWith('Show'));
      // The main tag first, then all three free tags (the card shows two).
      expect(tagButtons().map((b) => b.textContent)).toEqual(['Evaluation', 'Vision', 'CNN', 'NLP']);

      fireEvent.click(within(detail()).getByRole('button', { name: 'NLP' }));
      // The filter row opens on the applied tag, a rare one included.
      expect(screen.getByRole('button', { name: 'NLP, 2 examples' })).toHaveAttribute('aria-pressed', 'true');
      expect(cardNames()).toEqual(['Chat', 'Eval']);

      fireEvent.click(within(detail()).getByRole('button', { name: 'Evaluation' }));
      expect(tab(/^Evaluation/)).toHaveAttribute('aria-selected', 'true');
      expect(cardNames()).toEqual(['Eval']);
    });

    it('finds an example by a tag translation', async () => {
      useI18n.setState({ locale: 'zh-TW' });
      mockedRest.listExamples.mockResolvedValue(catalog());
      const { unmount } = render(<TemplateGalleryModal />);
      const zhGrid = () => screen.getByRole('region', { name: '範例清單' });
      await within(zhGrid()).findByText('Paper');

      fireEvent.change(screen.getByPlaceholderText('搜尋範例...'), { target: { value: '表格資料' } });
      expect(within(zhGrid()).getByText('Paper')).toBeInTheDocument();
      expect(within(zhGrid()).queryByText('Chat')).toBeNull();
      // The tag id matches too, in either UI language.
      fireEvent.change(screen.getByPlaceholderText('搜尋範例...'), { target: { value: 'nlp' } });
      expect(within(zhGrid()).getByText('Chat')).toBeInTheDocument();
      expect(within(zhGrid()).queryByText('Paper')).toBeNull();
      unmount();
    });
  });

  it('says so when nothing is selectable', async () => {
    render(<TemplateGalleryModal />);
    await screen.findByText('No examples available');
    expect(screen.getByText('Select a template')).toBeInTheDocument();
  });
});
