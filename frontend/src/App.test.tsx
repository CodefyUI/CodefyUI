import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';
import { useTabStore } from './store/tabStore';
import { useUIStore } from './store/uiStore';

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
  FlowCanvas: () => <div data-testid="flow-canvas" />,
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
vi.mock('./components/SubgraphEditor/SubgraphEditorModal', () => ({
  SubgraphEditorModal: () => <div data-testid="subgraph-modal" />,
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
  resetToSingleTab();
  useUIStore.setState({ fontSize: 'default' });
  document.documentElement.style.fontSize = '';
});

afterEach(() => {
  vi.restoreAllMocks();
  document.documentElement.style.fontSize = '';
});

describe('App', () => {
  it('renders the top-level chrome (toolbar, tab bar, modals, containers)', () => {
    render(<App />);
    expect(screen.getByTestId('toolbar')).toBeTruthy();
    expect(screen.getByTestId('tabbar')).toBeTruthy();
    expect(screen.getByTestId('preset-modal')).toBeTruthy();
    expect(screen.getByTestId('subgraph-modal')).toBeTruthy();
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

  it('renders one TabContent per tab and shows only the active one', () => {
    useTabStore.getState().addTab('Tab 2'); // Tab 2 becomes active
    const { container } = render(<App />);
    // Two canvases (one per tab).
    expect(screen.getAllByTestId('flow-canvas')).toHaveLength(2);

    // The tabContent wrappers toggle display; exactly one is flex, one is none.
    const displays = Array.from(container.querySelectorAll('div'))
      .filter((d) => d.style.display === 'flex' || d.style.display === 'none')
      .map((d) => d.style.display);
    expect(displays.filter((d) => d === 'flex')).toHaveLength(1);
    expect(displays.filter((d) => d === 'none')).toHaveLength(1);
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

  it('applies the small font size to the document element', () => {
    useUIStore.setState({ fontSize: 'small' });
    render(<App />);
    expect(document.documentElement.style.fontSize).toBe('12px');
  });

  it('applies the large font size to the document element', () => {
    useUIStore.setState({ fontSize: 'large' });
    render(<App />);
    expect(document.documentElement.style.fontSize).toBe('20px');
  });

  it('clears the inline font size for the default choice', () => {
    // Seed a non-empty inline size first to prove the effect clears it.
    document.documentElement.style.fontSize = '99px';
    useUIStore.setState({ fontSize: 'default' });
    render(<App />);
    expect(document.documentElement.style.fontSize).toBe('');
  });

  it('falls back to clearing the inline size for an unknown font size value', () => {
    document.documentElement.style.fontSize = '99px';
    // Drive an out-of-range value to hit the `?? ''` fallback branch.
    useUIStore.setState({ fontSize: 'weird' as never });
    render(<App />);
    expect(document.documentElement.style.fontSize).toBe('');
  });

  it('reacts to font-size changes after mount', () => {
    const { rerender } = render(<App />);
    expect(document.documentElement.style.fontSize).toBe('');
    useUIStore.setState({ fontSize: 'large' });
    rerender(<App />);
    expect(document.documentElement.style.fontSize).toBe('20px');
  });

  it('invokes the keyboard shortcuts hook on mount', async () => {
    const { useKeyboardShortcuts } = await import('./hooks/useKeyboardShortcuts');
    render(<App />);
    expect(useKeyboardShortcuts).toHaveBeenCalled();
  });

  it('keeps the active TabContent visible (display:flex) for the active tab only', () => {
    const tabId = useTabStore.getState().activeTabId;
    const { container } = render(<App />);
    // With a single tab it is active → its content is flex.
    const flexContents = Array.from(container.querySelectorAll('div')).filter(
      (d) => d.style.display === 'flex',
    );
    expect(flexContents.length).toBeGreaterThanOrEqual(1);
    // Switching active away (no other tab) keeps it active; sanity: id unchanged.
    expect(useTabStore.getState().activeTabId).toBe(tabId);
  });
});
