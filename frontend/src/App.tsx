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
import { NodeDetailModal } from './components/NodeDetailModal/NodeDetailModal';
import { TemplateGalleryModal } from './components/TemplateGallery/TemplateGalleryModal';
import { ToastContainer } from './components/shared/Toast';
import { ShortcutsModal } from './components/shared/ShortcutsModal';
import { DialogContainer } from './components/shared/DialogContainer';
import { PluginHost } from './plugins/PluginHost';
import { PluginRightPanels, usePluginPanels } from './components/PluginPanels/PluginPanels';
import { useTabStore } from './store/tabStore';
import { useUIStore } from './store/uiStore';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { fetchHealth } from './api/rest';
import { useProjectStore } from './store/projectStore';
import { useRunStore } from './store/runStore';
import styles from './App.module.css';

// Map the user's font-size choice onto the root element. Every size token in
// `styles/tokens.css` is a rem, so this scales the whole app.
//
// These used to be 12px / (unset) / 20px, where "unset" fell through to a
// viewport-responsive `clamp(13.5px, 0.35vw + 11px, 18px)` in App.css. Two
// problems with that: the root moved with the window, so the same UI rendered
// at 15.8px root on a 1366px laptop and 18px on a 2560px monitor and no size
// in the app was predictable; and at the Small setting a 0.625rem rule — of
// which there were 38 — came out at 7.5px.
//
// Fixed roots instead, with Default at the 16px browser standard. Small is a
// deliberate density trade for people who want more on screen, not an
// accessibility setting, and it still keeps body text at 13.1px.
const FONT_SIZE_PX: Record<string, string> = {
  small: '15px',
  default: '16px',
  large: '18px',
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
  // A right-docked plugin panel (#132) is a standing section, not a reaction
  // to a selection, so it keeps the column alive on its own. Without this the
  // column would vanish the moment the user clicked empty canvas, taking a
  // plugin's panel with it.
  const hasPluginPanels = usePluginPanels('right').length > 0;

  if (!hasSelection && !hasSegment && !hasPluginPanels) return null;

  return (
    <div className={styles.rightColumn}>
      {hasSelection && <NodeConfigPanel />}
      {(hasSelection || hasSegment) && <InspectorPanel />}
      <PluginRightPanels />
    </div>
  );
}

/**
 * The editor surface for whichever tab is active.
 *
 * Exactly ONE of these is ever mounted (#125). Before, App rendered a
 * TabContent per tab and hid the inactive ones with `display: none` — five
 * tabs of a 200-node graph meant a thousand mounted node components, five
 * React Flow stores each re-diffing on every store commit, and five copies
 * of the palette and panels re-rendering in lockstep. Worse, every one of
 * those hidden canvases rendered the ACTIVE tab's nodes anyway (each child
 * selects by `activeTabId`, not by its own tab), so the duplicates were not
 * even showing different graphs.
 *
 * It is rendered WITHOUT a `key` on purpose: React then reuses this instance
 * across tab switches, so the `<ReactFlowProvider>` (and the node
 * measurements, panel sizes and palette state inside it) survives the
 * switch instead of being torn down and rebuilt. The one thing that model
 * does not carry across for free is the viewport — a single provider has a
 * single pan/zoom — so FlowCanvas hands it over explicitly per tab.
 */
function TabContent({ tabId }: { tabId: string }) {
  return (
    <div className={styles.tabContent}>
      <div className={styles.tabInner}>
        <ReactFlowProvider>
          <NodePalette />
          <div className={styles.canvasHost}>
            <div className={styles.canvasFill}>
              <FlowCanvas tabId={tabId} />
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
  const activeTabId = useTabStore((s) => s.activeTabId);
  const hasActiveTab = useTabStore((s) =>
    s.tabs.some((t) => t.id === s.activeTabId),
  );
  const fontSize = useUIStore((s) => s.fontSize);

  useEffect(() => {
    document.documentElement.style.fontSize = FONT_SIZE_PX[fontSize] ?? '';
  }, [fontSize]);

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then((h) => {
        if (cancelled) return;
        useProjectStore.getState().setProject(h.project);
        useTabStore.getState().rehydrateForProject(h.project);
      })
      .catch(() => {
        /* health unreachable -- stay in non-project defaults */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // #124: a run outlives the page that started it, so a fresh load may open
  // onto training already in flight with nothing on screen saying so. One
  // non-blocking toast points at the Runs panel; `checkInProgress` is itself
  // idempotent per page load, which is what makes StrictMode's double mount
  // produce one toast rather than two.
  useEffect(() => {
    void useRunStore.getState().checkInProgress();
  }, []);

  return (
    <div className={styles.root}>
      <Toolbar />
      <TabBar />
      {hasActiveTab && <TabContent tabId={activeTabId} />}
      <PresetConfigModal />
      <SubgraphEditorModal />
      <NodeDetailModal />
      <TemplateGalleryModal />
      <ToastContainer />
      <ShortcutsModal />
      <DialogContainer />
      <PluginHost />
    </div>
  );
}

export default App;
