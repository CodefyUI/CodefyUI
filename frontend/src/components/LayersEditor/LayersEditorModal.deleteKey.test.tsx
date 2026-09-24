import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { ReactFlowProvider } from '@xyflow/react';

import { LayersEditorModal } from './LayersEditorModal';
import { FlowCanvas } from '../Canvas/FlowCanvas';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useDialogStore } from '../../store/dialogStore';
import { useI18n } from '../../i18n';

// The Delete key in the Model Architecture editor, through the REAL
// @xyflow/react (LayersEditorModal.test.tsx stubs <ReactFlow>): React Flow's
// own key handling has to be in the loop to show #501, where a shifted key
// whose Shift came up first left the first Delete after it doing nothing.

// Only drawn on an empty tab, and it fetches the example list when it is.
vi.mock('../Canvas/EmptyCanvasOverlay', () => ({
  EmptyCanvasOverlay: () => null,
}));

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;
const TAB_ID = 'tab-layers-delete';
const MODEL_ID = 'model';

/** Input -> Linear -> Output. */
const LAYERS = JSON.stringify({
  version: 2,
  nodes: [
    { id: 'in1', type: 'Input', ports: [{ id: 'ip1', name: 'x' }], position: { x: 0, y: 0 } },
    {
      id: 'lin1',
      type: 'Linear',
      params: { in_features: 4, out_features: 2 },
      position: { x: 0, y: 100 },
    },
    { id: 'out1', type: 'Output', ports: [{ id: 'op1', name: 'y' }], position: { x: 0, y: 200 } },
  ],
  edges: [
    { id: 'e1', source: 'in1', sourceHandle: 'ip1', target: 'lin1', targetHandle: null },
    { id: 'e2', source: 'lin1', sourceHandle: null, target: 'out1', targetHandle: 'op1' },
  ],
});

/** One tab holding a SequentialModel node, selected, with its editor open. */
function openEditor() {
  useTabStore.setState({
    tabs: [
      {
        ...ORIGINAL_TABS[0],
        id: TAB_ID,
        nodes: [
          {
            id: MODEL_ID,
            type: 'baseNode',
            position: { x: 0, y: 0 },
            selected: true,
            data: { label: 'SequentialModel', type: 'SequentialModel', params: { layers: LAYERS } },
          },
        ],
        edges: [],
        layersModalNodeId: MODEL_ID,
      } as any,
    ],
    activeTabId: TAB_ID,
  });
}

/** Select a layer the way a click on it does. */
function selectLayer(id: string) {
  fireEvent.click(document.querySelector(`.react-flow__node[data-id="${id}"]`)!);
}

/** One Delete, down and up, with the deletion run out. */
async function pressDelete() {
  fireEvent.keyDown(document.body, { key: 'Delete', code: 'Delete' });
  // The deletion awaits `onBeforeDelete` before it lands.
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  fireEvent.keyUp(document.body, { key: 'Delete', code: 'Delete' });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  // Every modal flag the delete gate reads, so nothing leaks between cases.
  useUIStore.setState({
    shortcutsModalOpen: false,
    templateGalleryOpen: false,
    packCenterOpen: false,
    pluginCenterOpen: false,
    customNodeManagerOpen: false,
    gitDiff: null,
  });
  useDialogStore.setState({ active: null, resolve: null });
  openEditor();
});

afterEach(() => {
  // Unmount first: resetting the stores under a mounted editor re-renders it
  // outside act().
  cleanup();
  useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
});

describe('LayersEditorModal Delete key (#501)', () => {
  it('deletes the selected layer on the first Delete after Shift+L with Shift let go first', async () => {
    render(<LayersEditorModal />);
    selectLayer('lin1');
    expect(screen.getByText('3 layers')).toBeTruthy();

    // `L` goes down with Shift held and comes up as `l` once Shift is up;
    // React Flow's own binding kept the `L` as held.
    fireEvent.keyDown(document.body, { key: 'Shift', code: 'ShiftLeft', shiftKey: true });
    fireEvent.keyDown(document.body, { key: 'L', code: 'KeyL', shiftKey: true });
    fireEvent.keyUp(document.body, { key: 'Shift', code: 'ShiftLeft' });
    fireEvent.keyUp(document.body, { key: 'l', code: 'KeyL' });

    await pressDelete();
    expect(screen.getByText('2 layers')).toBeTruthy();
    expect(document.querySelector('.react-flow__node[data-id="lin1"]')).toBeNull();
  });

  it('refuses a Delete while a dialog is open on top of it, and takes the next one', async () => {
    render(<LayersEditorModal />);
    selectLayer('lin1');
    act(() => {
      useDialogStore.setState({ active: { kind: 'confirm', title: 'sure?' }, resolve: null });
    });
    await pressDelete();
    expect(screen.getByText('3 layers')).toBeTruthy();

    act(() => {
      useDialogStore.setState({ active: null, resolve: null });
    });
    await pressDelete();
    expect(screen.getByText('2 layers')).toBeTruthy();
  });

  it('deletes in the editor only, never on the canvas behind it', async () => {
    // Both canvases hear the same Delete. The one behind the editor refuses
    // it (the editor is a modal); the editor takes it for its own selection.
    render(
      <>
        <ReactFlowProvider>
          <FlowCanvas />
        </ReactFlowProvider>
        <LayersEditorModal />
      </>,
    );
    selectLayer('lin1');
    await pressDelete();
    expect(screen.getByText('2 layers')).toBeTruthy();
    const tab = useTabStore.getState().tabs.find((t) => t.id === TAB_ID)!;
    expect(tab.nodes.map((n) => n.id)).toEqual([MODEL_ID]);
  });
});
