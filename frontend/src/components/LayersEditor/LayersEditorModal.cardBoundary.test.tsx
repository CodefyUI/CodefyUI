import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { cleanup, render, within } from '@testing-library/react';
import type { NodeTypes } from '@xyflow/react';
import { nodeProps } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';

// The editor's own canvas, stubbed as in LayersEditorModal.test: what matters
// here is the `nodeTypes` map the editor hands React Flow.
let flowNodeTypes: NodeTypes | null = null;
vi.mock('@xyflow/react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@xyflow/react')>();
  return {
    ...actual,
    useReactFlow: () => ({ screenToFlowPosition: (p: unknown) => p, fitView: () => {} }),
    ReactFlow: (props: { nodeTypes: NodeTypes }) => {
      flowNodeTypes = props.nodeTypes;
      return null;
    },
  };
});

// A layer card that throws while it draws.
vi.mock('./LayerNode', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./LayerNode')>()),
  LayerNode: function LayerNode() {
    throw new Error('layer card exploded');
  },
}));

import { LayersEditorModal } from './LayersEditorModal';

let originalTabs: ReturnType<typeof useTabStore.getState>['tabs'];
let originalActive: string;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  flowNodeTypes = null;
  originalTabs = useTabStore.getState().tabs;
  originalActive = useTabStore.getState().activeTabId;
  useTabStore.setState({
    activeTabId: 't1',
    tabs: [
      {
        ...originalTabs[0],
        id: 't1',
        nodes: [
          {
            id: 'seq',
            type: 'baseNode',
            position: { x: 0, y: 0 },
            data: { label: 'SequentialModel', type: 'SequentialModel', params: {}, executionStatus: 'idle' },
          },
        ],
        edges: [],
        layersModalNodeId: 'seq',
      },
    ],
  });
  // React reports each error a boundary catches through console.error. Only
  // that report is dropped: anything else, an act() warning above all, still
  // reaches the console and the act-warnings gate.
  const consoleError = console.error;
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    const caughtReport = args.some(
      (arg) => typeof arg === 'string' && arg.startsWith('React will try to recreate'),
    );
    if (!caughtReport) consoleError(...args);
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useTabStore.setState({ tabs: originalTabs, activeTabId: originalActive });
});

describe('the layers editor canvas', () => {
  it('draws a layer card that throws as a box naming the layer', () => {
    render(<LayersEditorModal />);
    expect(flowNodeTypes).not.toBeNull();
    const LayerCard = flowNodeTypes!.layerNode;
    // Its own container: the editor's layer palette lists "Linear" too.
    const { container } = render(
      <LayerCard
        {...nodeProps({
          id: 'lin1',
          type: 'layerNode',
          data: { layerType: 'Linear', params: {}, color: '#888888' },
        })}
      />,
    );
    const card = within(container);
    expect(card.getByText(useI18n.getState().t('node.cardFailed'))).toBeTruthy();
    expect(card.getByText('Linear')).toBeTruthy();
  });
});
