import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { WelcomeScreen } from './WelcomeScreen';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { _resetPluginStoreForTesting } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import { importFile } from '../../utils/importGraphFile';
import { canFillTab } from '../../utils/tabFill';
import * as rest from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, listExamples: vi.fn(), loadExample: vi.fn() };
});

// What a picked file does is the import's own suite's business; here the call
// is what matters.
vi.mock('../../utils/importGraphFile', () => ({ importFile: vi.fn() }));

// A resolved node always has a position (the resolver defaults it), and an
// opened example is framed from it.
vi.mock('../../utils', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../utils')>();
  return {
    ...actual,
    resolveSerializedNodes: vi.fn(() => [{ id: 'n1', position: { x: 0, y: 0 } } as never]),
    resolveSerializedEdges: vi.fn(() => [{ id: 'e1' } as never]),
  };
});

const mockedRest = vi.mocked(rest);
const mockedImport = vi.mocked(importFile);

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

/** The store as the welcome screen only ever sees it: no tabs at all. */
function emptyWorkspace() {
  useTabStore.setState({ tabs: [], activeTabId: '', clipboard: null });
}

/** Let the example list's fetch land, so its update happens inside act(). */
async function examplesSettled() {
  await waitFor(() => expect(screen.queryByText('Loading examples...')).toBeNull());
}

describe('WelcomeScreen', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useNodeDefStore.setState({ definitions: [], presets: [] });
    _resetPluginStoreForTesting();
    emptyWorkspace();
    mockedRest.listExamples.mockReset();
    mockedRest.loadExample.mockReset();
    mockedRest.listExamples.mockResolvedValue([]);
    mockedImport.mockReset();
    mockedImport.mockResolvedValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('greets with the product name, a tagline and the ways in', () => {
    render(<WelcomeScreen />);
    expect(screen.getByText('CodefyUI')).toBeInTheDocument();
    expect(
      screen.getByText('Build deep learning models by dragging nodes onto a canvas.'),
    ).toBeInTheDocument();
    // A blank graph, a saved one, a file, a template, in that order.
    expect(screen.getAllByRole('button').slice(0, 4).map((b) => b.textContent)).toEqual([
      'New blank graph',
      'Open a saved graph',
      'Import...',
      'Browse all templates',
    ]);
  });

  it('opens a saved graph through an empty tab with the Graphs list open beside it', async () => {
    useUIStore.setState({ sidebarTab: 'nodes', sidebarCollapsed: true });
    render(<WelcomeScreen />);
    await examplesSettled();
    fireEvent.click(screen.getByRole('button', { name: 'Open a saved graph' }));
    const { tabs, activeTabId } = useTabStore.getState();
    expect(tabs).toHaveLength(1);
    expect(activeTabId).toBe(tabs[0].id);
    // Empty and the user's own, so the row clicked in that list fills it
    // rather than opening a second tab.
    expect(canFillTab(tabs[0])).toBe(true);
    expect(useUIStore.getState().sidebarTab).toBe('graphs');
    expect(useUIStore.getState().sidebarCollapsed).toBe(false);
  });

  it("imports a picked file the way the Graphs panel's Import... does", async () => {
    const { container } = render(<WelcomeScreen />);
    await examplesSettled();
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    expect(input.accept).toBe('.json,.cduiworkspace');
    const pick = vi.spyOn(input, 'click').mockImplementation(() => {});
    const button = screen.getByRole('button', { name: 'Import...' });
    expect(button.title).toBe('A graph (.json) or a workspace (.cduiworkspace)');
    fireEvent.click(button);
    expect(pick).toHaveBeenCalled();

    const file = new File(['{"nodes":[]}'], 'starter.json', { type: 'application/json' });
    fireEvent.change(input, { target: { files: [file] } });
    expect(mockedImport).toHaveBeenCalledWith(file);
    // Cleared at once, so picking the same file again still fires `change`.
    expect(input.value).toBe('');
  });

  it('imports nothing when the picker comes back empty', async () => {
    const { container } = render(<WelcomeScreen />);
    await examplesSettled();
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [] } });
    expect(mockedImport).not.toHaveBeenCalled();
  });

  it('opens a blank graph, which is what takes the workspace out of this state', () => {
    render(<WelcomeScreen />);
    fireEvent.click(screen.getByText('New blank graph'));
    const { tabs, activeTabId } = useTabStore.getState();
    expect(tabs).toHaveLength(1);
    expect(tabs[0].name).toBe('Tab 1');
    expect(activeTabId).toBe(tabs[0].id);
  });

  it('opens the full gallery modal from the secondary button', () => {
    useUIStore.setState({ templateGalleryOpen: false });
    render(<WelcomeScreen />);
    fireEvent.click(screen.getByText('Browse all templates'));
    expect(useUIStore.getState().templateGalleryOpen).toBe(true);
  });

  it('lists the examples under their own heading', async () => {
    mockedRest.listExamples.mockResolvedValue([
      ex({ name: 'Train MLP', node_count: 5, section: 'quickstart' }),
    ]);
    render(<WelcomeScreen />);
    expect(screen.getByText('Or start from an example')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Train MLP')).toBeInTheDocument());
    expect(screen.getByText('Quick Start')).toBeInTheDocument();
    expect(screen.getByText('5 nodes')).toBeInTheDocument();
  });

  it('opens a picked example into a tab it creates first', async () => {
    // The one behavioural difference from the empty-canvas overlay, which
    // replaces the graph in the tab the user is already standing in. Here
    // there is no tab to replace, so one has to be made -- and only after the
    // fetch succeeds, so a failed load does not strand the user on a blank
    // canvas with a toast for an explanation.
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Loadable' })]);
    mockedRest.loadExample.mockResolvedValue({
      name: 'My Model',
      nodes: [{ id: 'a' }],
      edges: [{ id: 'e' }],
    });
    render(<WelcomeScreen />);
    fireEvent.click(await screen.findByText('Loadable'));

    await waitFor(() => expect(useTabStore.getState().tabs).toHaveLength(1));
    expect(mockedRest.loadExample).toHaveBeenCalledWith('/examples/foo.json');
    const tab = useTabStore.getState().tabs[0];
    expect(tab.name).toBe('My Model');
    expect(tab.nodes).toEqual([{ id: 'n1', position: { x: 0, y: 0 } }]);
    // A template is bound to no file, so the first Save has to ask where it
    // goes rather than overwrite the example.
    expect(tab.currentGraphFile).toBeNull();
  });

  it('leaves the workspace empty when the example fails to load', async () => {
    mockedRest.listExamples.mockResolvedValue([ex({ name: 'Broken' })]);
    mockedRest.loadExample.mockRejectedValue(new Error('load failed'));
    render(<WelcomeScreen />);
    fireEvent.click(await screen.findByText('Broken'));

    await waitFor(() => expect(mockedRest.loadExample).toHaveBeenCalled());
    expect(useTabStore.getState().tabs).toEqual([]);
  });
});
