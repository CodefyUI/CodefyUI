import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { WelcomeScreen } from './WelcomeScreen';
import { useTabStore } from '../../store/tabStore';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useUIStore } from '../../store/uiStore';
import { _resetPluginStoreForTesting } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import type { ExampleSummary } from '../../api/rest';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, listExamples: vi.fn(), loadExample: vi.fn() };
});

vi.mock('../../utils', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../utils')>();
  return {
    ...actual,
    resolveSerializedNodes: vi.fn(() => [{ id: 'n1' } as never]),
    resolveSerializedEdges: vi.fn(() => [{ id: 'e1' } as never]),
  };
});

const mockedRest = vi.mocked(rest);

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

describe('WelcomeScreen', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useNodeDefStore.setState({ definitions: [], presets: [] });
    _resetPluginStoreForTesting();
    emptyWorkspace();
    mockedRest.listExamples.mockReset();
    mockedRest.loadExample.mockReset();
    mockedRest.listExamples.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('greets with the product name, a tagline and the two ways in', () => {
    render(<WelcomeScreen />);
    expect(screen.getByText('CodefyUI')).toBeInTheDocument();
    expect(
      screen.getByText('Build deep learning models by dragging nodes onto a canvas.'),
    ).toBeInTheDocument();
    expect(screen.getByText('New blank graph')).toBeInTheDocument();
    expect(screen.getByText('Browse all templates')).toBeInTheDocument();
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
    expect(tab.nodes).toEqual([{ id: 'n1' }]);
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
