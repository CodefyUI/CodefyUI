import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import App from './App';
import { fetchHealth } from './api/rest';
import { useStudyStore } from './store/studyStore';
import { useTabStore } from './store/tabStore';
import { useToastStore } from './store/toastStore';
import { useWorkerStore } from './store/workerStore';
import { settle } from './test/utils';

// The Workers (#776) and Studies (#343) panels are lazy chunks. When a chunk
// cannot load (a deploy replaced it, the network dropped), App closes the
// panel it was opening instead of leaving an open flag with nothing on
// screen. Both chunk modules throw here; the rest of App is stubbed as in
// App.test.tsx, which this file cannot share (its mocks load those chunks).

vi.mock('./components/Workers/WorkersPanel', () => {
  throw new Error('chunk failed to load');
});
vi.mock('./components/Studies/StudiesPanel', () => {
  throw new Error('chunk failed to load');
});

vi.mock('./hooks/useKeyboardShortcuts', () => ({ useKeyboardShortcuts: vi.fn() }));
vi.mock('./components/Toolbar/Toolbar', () => ({ Toolbar: () => null }));
vi.mock('./components/Toolbar/WelcomeToolbar', () => ({ WelcomeToolbar: () => null }));
vi.mock('./components/Welcome/WelcomeScreen', () => ({ WelcomeScreen: () => null }));
vi.mock('./components/TabBar/TabBar', () => ({ TabBar: () => null }));
vi.mock('./components/Sidebar/NodePalette', () => ({ NodePalette: () => null }));
vi.mock('./components/Canvas/FlowCanvas', () => ({ FlowCanvas: () => null }));
vi.mock('./components/ConfigPanel/NodeConfigPanel', () => ({ NodeConfigPanel: () => null }));
vi.mock('./components/InspectorPanel/InspectorPanel', () => ({ InspectorPanel: () => null }));
vi.mock('./components/ResultsPanel/ResultsPanel', () => ({ ResultsPanel: () => null }));
vi.mock('./components/PresetModal/PresetConfigModal', () => ({ PresetConfigModal: () => null }));
vi.mock('./components/LayersEditor/LayersEditorModal', () => ({ LayersEditorModal: () => null }));
vi.mock('./components/NodeDetailModal/NodeDetailModal', () => ({ NodeDetailModal: () => null }));
vi.mock('./components/Nodes/VizViewerModal', () => ({ VizViewerModal: () => null }));
vi.mock('./components/SourceControl/GitDiffModal', () => ({ GitDiffModal: () => null }));
vi.mock('./components/CustomNodeManager/CustomNodeManager', () => ({ CustomNodeManagerModal: () => null }));
vi.mock('./components/shared/ShortcutsModal', () => ({ ShortcutsModal: () => null }));
vi.mock('./components/shared/DialogContainer', () => ({ DialogContainer: () => null }));
vi.mock('./api/rest', () => ({ fetchHealth: vi.fn() }));
vi.mock('./store/runStore', () => ({
  useRunStore: { getState: () => ({ checkInProgress: vi.fn(async () => 0) }) },
}));

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('Tab 1');
  vi.mocked(fetchHealth).mockResolvedValue({
    status: 'ok', version: '3.1.1', nodes_loaded: 0, presets_loaded: 0, caches: {}, project: null,
  } as never);
  useToastStore.setState({ toasts: [] });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

describe('App when a panel chunk cannot load', () => {
  it.each([
    ['Workers', () => useWorkerStore.setState({ panelOpen: true }), () => useWorkerStore.getState().panelOpen],
    ['Studies', () => useStudyStore.setState({ panelOpen: true }), () => useStudyStore.getState().panelOpen],
  ])('closes the %s panel and says why', async (_name, open, isOpen) => {
    render(<App />);
    act(open);
    await settle();
    await settle();
    expect(isOpen()).toBe(false);
    expect(useToastStore.getState().toasts.map((toast) => toast.type)).toContain('error');
  });
});
