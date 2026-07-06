import { useEffect } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { Toolbar } from './components/Toolbar/Toolbar';
import { TabBar } from './components/TabBar/TabBar';
import { NodePalette } from './components/Sidebar/NodePalette';
import { FlowCanvas } from './components/Canvas/FlowCanvas';
import { NodeConfigPanel } from './components/ConfigPanel/NodeConfigPanel';
import { InspectorPanel } from './components/InspectorPanel/InspectorPanel';
import { ResultsPanel } from './components/ResultsPanel/ResultsPanel';
import { PresetConfigModal } from './components/PresetModal/PresetConfigModal';
import { SubgraphEditorModal } from './components/SubgraphEditor/SubgraphEditorModal';
import { ToastContainer } from './components/shared/Toast';
import { ShortcutsModal } from './components/shared/ShortcutsModal';
import { DialogContainer } from './components/shared/DialogContainer';
import { PluginHost } from './plugins/PluginHost';
import { useTabStore } from './store/tabStore';
import { useUIStore } from './store/uiStore';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { fetchHealth } from './api/rest';
import { useProjectStore } from './store/projectStore';
import styles from './App.module.css';

// Map user font-size choice to the documentElement. Setting an empty string
// removes the inline style, letting App.css's responsive `clamp(...)` take over
// (capped at 18px). Large jumps clearly above the clamp ceiling so the change
// is actually visible.
const FONT_SIZE_PX: Record<string, string> = {
  small: '12px',
  default: '',
  large: '20px',
};

function RightColumn() {
  const selectedNodeId = useTabStore(
    (s) => s.tabs.find((t) => t.id === s.activeTabId)?.selectedNodeId ?? null,
  );
  const activeSegment = useTabStore(
    (s) => s.tabs.find((t) => t.id === s.activeTabId)?.activeSegment ?? null,
  );
  const hasSelection = selectedNodeId !== null;
  const hasSegment = activeSegment !== null;

  if (!hasSelection && !hasSegment) return null;

  return (
    <div className={styles.rightColumn}>
      {hasSelection && <NodeConfigPanel />}
      <InspectorPanel />
    </div>
  );
}

function TabContent({ tabId }: { tabId: string }) {
  const activeTabId = useTabStore((s) => s.activeTabId);
  const isActive = tabId === activeTabId;

  return (
    <div
      className={styles.tabContent}
      style={{ display: isActive ? 'flex' : 'none' }}
    >
      <div className={styles.tabInner}>
        <ReactFlowProvider>
          <NodePalette />
          <div className={styles.canvasHost}>
            <div className={styles.canvasFill}>
              <FlowCanvas />
            </div>
          </div>
          <RightColumn />
        </ReactFlowProvider>
      </div>
      <ResultsPanel />
    </div>
  );
}

function App() {
  useKeyboardShortcuts();
  const tabs = useTabStore((s) => s.tabs);
  const fontSize = useUIStore((s) => s.fontSize);

  useEffect(() => {
    document.documentElement.style.fontSize = FONT_SIZE_PX[fontSize] ?? '';
  }, [fontSize]);

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then((h) => {
        if (!cancelled) useProjectStore.getState().setProject(h.project);
      })
      .catch(() => {
        /* health unreachable -- stay in non-project defaults */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className={styles.root}>
      <Toolbar />
      <TabBar />
      {tabs.map((tab) => (
        <TabContent key={tab.id} tabId={tab.id} />
      ))}
      <PresetConfigModal />
      <SubgraphEditorModal />
      <ToastContainer />
      <ShortcutsModal />
      <DialogContainer />
      <PluginHost />
    </div>
  );
}

export default App;
