import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import App from './App';
import { useTabStore } from './store/tabStore';
import { useUIStore } from './store/uiStore';
import { useProjectStore } from './store/projectStore';
import { fetchHealth } from './api/rest';

// ── Mock heavy children so we test only App's composition logic ───────────────
// Each stub renders a stable testid so we can assert presence / counts.

vi.mock('./hooks/useKeyboardShortcuts', () => ({
  useKeyboardShortcuts: vi.fn(),
}));

vi.mock('./components/Toolbar/Toolbar', () => ({
  Toolbar: () => <div data-testid="toolbar" />,
}));
vi.mock('./components/TabBar/TabBar', () => ({
  TabBar: () => <div data-testid="tabbar" />,
}));
vi.mock('./components/Sidebar/NodePalette', () => ({
  NodePalette: () => <div data-testid="node-palette" />,
}));
vi.mock('./components/Canvas/FlowCanvas', () => ({
  // Echoes the tabId prop so the single-mount tests can assert WHICH tab the
  // one mounted canvas is for.
  FlowCanvas: ({ tabId }: { tabId?: string }) => (
    <div data-testid="flow-canvas" data-tab-id={tabId} />
  ),
}));
vi.mock('./components/ConfigPanel/NodeConfigPanel', () => ({
  NodeConfigPanel: () => <div data-testid="config-panel" />,
}));
vi.mock('./components/InspectorPanel/InspectorPanel', () => ({
  InspectorPanel: () => <div data-testid="inspector-panel" />,
}));
vi.mock('./components/ResultsPanel/ResultsPanel', () => ({
  ResultsPanel: () => <div data-testid="results-panel" />,
}));
vi.mock('./components/PresetModal/PresetConfigModal', () => ({
  PresetConfigModal: () => <div data-testid="preset-modal" />,
}));
vi.mock('./components/LayersEditor/LayersEditorModal', () => ({
  LayersEditorModal: () => <div data-testid="layers-editor-modal" />,
}));
vi.mock('./components/NodeDetailModal/NodeDetailModal', () => ({
  NodeDetailModal: () => <div data-testid="node-detail-modal" />,
}));
vi.mock('./components/Nodes/VizViewerModal', () => ({
  VizViewerModal: () => <div data-testid="viz-viewer-modal" />,
}));
vi.mock('./components/shared/Toast', () => ({
  ToastContainer: () => <div data-testid="toast-container" />,
}));
vi.mock('./components/shared/ShortcutsModal', () => ({
  ShortcutsModal: () => <div data-testid="shortcuts-modal" />,
}));
vi.mock('./components/shared/DialogContainer', () => ({
  DialogContainer: () => <div data-testid="dialog-container" />,
}));

// fetchHealth drives the bootstrap effect's project branch (App.tsx:87-101).
// Left unmocked, the real implementation's fetch() call rejects in jsdom (no
// server) and the effect's .catch swallows it silently, so that branch never
// ran under test. Mocking it here lets the non-project shape PIN the existing
// tests' behavior below (rather than it being accidental) and lets one
// dedicated test drive the project-mode rehydration path.
vi.mock('./api/rest', () => ({
  fetchHealth: vi.fn(),
}));
const mockedFetchHealth = vi.mocked(fetchHealth);

// #124: App's second bootstrap effect asks the run store whether anything is
// still training. Stubbed here for the same reason ResultsPanel is — this
// file tests composition, and the real store would pull the (mocked) REST
// module in for a list request nothing here is asserting.
const mockCheckInProgress = vi.fn(async () => 0);
vi.mock('./store/runStore', () => ({
  useRunStore: { getState: () => ({ checkInProgress: mockCheckInProgress }) },
}));

// ── Helpers ──────────────────────────────────────────────────────────────────

function resetToSingleTab() {
  useTabStore.setState({
    tabs: [],
    activeTabId: null as unknown as string,
    clipboard: null,
  });
  useTabStore.getState().addTab('Tab 1');
}

beforeEach(() => {
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
  resetToSingleTab();
  useUIStore.setState({ fontSize: 'default' });
  document.documentElement.style.fontSize = '';
  localStorage.clear();
  // Default: non-project health shape, so the pre-existing tests below
  // exercise (and pin) the same bootstrap branch the real server takes
  // outside project mode, instead of silently skipping it.
  mockedFetchHealth.mockReset();
  mockedFetchHealth.mockResolvedValue({
    status: 'ok',
    // version / caches are read by the settings popover, not by App (#193
    // item 2); present here because HealthInfo now carries them.
    version: '2.2.0',
    nodes_loaded: 0,
    presets_loaded: 0,
    caches: {},
    project: null,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  document.documentElement.style.fontSize = '';
  localStorage.clear();
});

describe('App', () => {
  it('renders the top-level chrome (toolbar, tab bar, modals, containers)', () => {
    render(<App />);
    expect(screen.getByTestId('toolbar')).toBeTruthy();
    expect(screen.getByTestId('tabbar')).toBeTruthy();
    expect(screen.getByTestId('preset-modal')).toBeTruthy();
    expect(screen.getByTestId('layers-editor-modal')).toBeTruthy();
    expect(screen.getByTestId('node-detail-modal')).toBeTruthy();
    expect(screen.getByTestId('viz-viewer-modal')).toBeTruthy();
    expect(screen.getByTestId('toast-container')).toBeTruthy();
    expect(screen.getByTestId('shortcuts-modal')).toBeTruthy();
    expect(screen.getByTestId('dialog-container')).toBeTruthy();
  });

  it('renders the canvas/palette/results for the single tab', () => {
    render(<App />);
    expect(screen.getByTestId('node-palette')).toBeTruthy();
    expect(screen.getByTestId('flow-canvas')).toBeTruthy();
    expect(screen.getByTestId('results-panel')).toBeTruthy();
  });

  // #125: only the ACTIVE tab's editor surface is mounted. It used to render
  // one TabContent per tab and hide the inactive ones with display:none,
  // which mounted every tab's canvas, palette and panels at once.
  it('mounts exactly one editor surface no matter how many tabs exist', () => {
    useTabStore.getState().addTab('Tab 2');
    useTabStore.getState().addTab('Tab 3'); // Tab 3 becomes active
    render(<App />);
    expect(screen.getAllByTestId('flow-canvas')).toHaveLength(1);
    expect(screen.getAllByTestId('node-palette')).toHaveLength(1);
    expect(screen.getAllByTestId('results-panel')).toHaveLength(1);
  });

  it('hands the active tab id to the mounted canvas and follows tab switches', () => {
    const first = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    const second = useTabStore.getState().activeTabId;

    render(<App />);
    expect(screen.getByTestId('flow-canvas').dataset.tabId).toBe(second);

    act(() => {
      useTabStore.getState().setActiveTab(first);
    });
    expect(screen.getByTestId('flow-canvas').dataset.tabId).toBe(first);
  });

  it('renders nothing for the editor surface when no tab is active', () => {
    // Defensive: a store mid-rehydration can briefly name a tab that is gone.
    useTabStore.setState({ activeTabId: 'missing' });
    render(<App />);
    expect(screen.queryByTestId('flow-canvas')).toBeNull();
  });

  // ── RightColumn conditional rendering ───────────────────────────────────────

  it('does not render the config panel or inspector when nothing is selected', () => {
    render(<App />);
    expect(screen.queryByTestId('config-panel')).toBeNull();
    expect(screen.queryByTestId('inspector-panel')).toBeNull();
  });

  it('renders both config panel and inspector when a node is selected', () => {
    const tabId = useTabStore.getState().activeTabId;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId ? { ...t, selectedNodeId: 'node-1' } : t,
      ),
    });
    render(<App />);
    expect(screen.getByTestId('config-panel')).toBeTruthy();
    expect(screen.getByTestId('inspector-panel')).toBeTruthy();
  });

  it('renders only the inspector (not the config panel) when a segment is active but no node selected', () => {
    const tabId = useTabStore.getState().activeTabId;
    useTabStore.setState({
      tabs: useTabStore.getState().tabs.map((t) =>
        t.id === tabId
          ? {
              ...t,
              selectedNodeId: null,
              activeSegment: { id: 's1', headNodeId: 'a', tailNodeId: 'b' },
            }
          : t,
      ),
    });
    render(<App />);
    expect(screen.queryByTestId('config-panel')).toBeNull();
    expect(screen.getByTestId('inspector-panel')).toBeTruthy();
  });

  // ── Font-size effect ────────────────────────────────────────────────────────

  it('applies the small font scale to the document element', () => {
    // Small is a deliberate density trade for people who want more on screen,
    // not an accessibility setting. It is applied as a multiplier so it
    // composes with App.css's viewport clamp rather than replacing it.
    useUIStore.setState({ fontSize: 'small' });
    render(<App />);
    expect(
      document.documentElement.style.getPropertyValue('--font-scale'),
    ).toBe('0.92');
  });

  it('applies the large font scale to the document element', () => {
    useUIStore.setState({ fontSize: 'large' });
    render(<App />);
    expect(
      document.documentElement.style.getPropertyValue('--font-scale'),
    ).toBe('1.15');
  });

  it('applies a neutral scale for the default choice', () => {
    // The control used to write an absolute root px, where "default" cleared
    // the inline style and fell through to a `clamp(13.5px, ...)` in App.css
    // whose floor put the most-used size at 10.8px on a 1366px laptop. It is
    // now a multiplier over a clamp floored at 16px, so default means "no
    // adjustment" rather than "no value". Seed a different scale first to
    // prove the effect overwrites it rather than leaving the seeded value.
    document.documentElement.style.setProperty('--font-scale', '9');
    useUIStore.setState({ fontSize: 'default' });
    render(<App />);
    expect(
      document.documentElement.style.getPropertyValue('--font-scale'),
    ).toBe('1');
  });

  it('falls back to a neutral scale for an unknown font size value', () => {
    document.documentElement.style.setProperty('--font-scale', '9');
    // Drive an out-of-range value to hit the `?? '1'` fallback branch. It must
    // land on 1, not on an empty string: an empty custom property would make
    // the calc() in App.css invalid and collapse the root size.
    useUIStore.setState({ fontSize: 'weird' as never });
    render(<App />);
    expect(
      document.documentElement.style.getPropertyValue('--font-scale'),
    ).toBe('1');
  });

  it('reacts to font-size changes after mount', () => {
    const { rerender } = render(<App />);
    // beforeEach leaves fontSize at 'default', which applies a neutral scale.
    expect(
      document.documentElement.style.getPropertyValue('--font-scale'),
    ).toBe('1');
    useUIStore.setState({ fontSize: 'large' });
    rerender(<App />);
    expect(
      document.documentElement.style.getPropertyValue('--font-scale'),
    ).toBe('1.15');
  });

  it('invokes the keyboard shortcuts hook on mount', async () => {
    const { useKeyboardShortcuts } = await import('./hooks/useKeyboardShortcuts');
    render(<App />);
    expect(useKeyboardShortcuts).toHaveBeenCalled();
  });

  // #125 replaced the display:none toggle with mounting only the active tab,
  // so visibility is no longer an inline style to assert on. What survives of
  // the old contract is that switching tabs keeps exactly one surface up and
  // does NOT tear the ReactFlowProvider down (TabContent renders without a
  // `key`, so React reuses the instance across the switch).
  it('reuses the same TabContent instance across a tab switch', () => {
    const first = useTabStore.getState().activeTabId;
    useTabStore.getState().addTab('Tab 2');
    render(<App />);
    // TabContent renders without a `key`, so React reuses the instance --
    // and with it the ReactFlowProvider, node measurements and panel state --
    // rather than tearing the whole editor down on every switch. A remount
    // would have produced a different DOM node here.
    const canvasBefore = screen.getByTestId('flow-canvas');

    act(() => {
      useTabStore.getState().setActiveTab(first);
    });
    expect(screen.getAllByTestId('flow-canvas')).toHaveLength(1);
    expect(screen.getByTestId('flow-canvas')).toBe(canvasBefore);
    expect(canvasBefore.dataset.tabId).toBe(first);
  });

  // -- Health bootstrap -> per-project rehydration (Task 13 review gap, ID10) --
  // App's bootstrap effect calls setProject(h.project) then
  // rehydrateForProject(h.project) once fetchHealth resolves (App.tsx:90-94).
  // The tests above pin the non-project shape via the beforeEach default;
  // this one drives the project branch and asserts the tab store actually
  // rehydrated from the project-scoped localStorage key.
  it('rehydrates tabs for the resolved project once fetchHealth reports one', async () => {
    localStorage.setItem(
      'codefyui-tabs::/proj',
      JSON.stringify({
        activeTabId: 'p1',
        tabs: [{ id: 'p1', name: 'project-tab', nodes: [], edges: [] }],
      }),
    );
    mockedFetchHealth.mockResolvedValueOnce({
      status: 'ok',
      version: '2.2.0',
      nodes_loaded: 0,
      presets_loaded: 0,
      caches: {},
      project: '/proj',
    });
    render(<App />);
    await waitFor(() => {
      expect(useTabStore.getState().tabs.some((t) => t.name === 'project-tab')).toBe(true);
    });
    expect(useProjectStore.getState().projectDir).toBe('/proj');
  });

  // -- Runs still in progress (#124) --
  it('asks the run store for in-flight runs on mount', () => {
    mockCheckInProgress.mockClear();
    render(<App />);
    expect(mockCheckInProgress).toHaveBeenCalledTimes(1);
  });
});
