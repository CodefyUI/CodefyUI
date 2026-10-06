import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { ReactFlowProvider, type Node } from '@xyflow/react';
import { FlowCanvas } from '../Canvas/FlowCanvas';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData } from '../../types';

// The tokenizer card stands in for any card that throws while it draws: this
// one throws while its node is labelled "Boom".
vi.mock('./TokenizerVizNode', async () => {
  const { createElement } = await vi.importActual<typeof import('react')>('react');
  return {
    default: function TokenizerVizNode({ data }: { data: { label: string } }) {
      if (data.label === 'Boom') throw new Error('card exploded');
      return createElement('div', null, `tokenizer ${data.label}`);
    },
  };
});

const TAB_ID = 'card-boundary-tab';

function node(id: string, type: string, label: string, x: number): Node<NodeData> {
  return {
    id,
    type,
    position: { x, y: 0 },
    data: {
      label,
      type: label,
      params: {},
      definition: {
        node_name: label,
        category: 'Utility',
        description: '',
        inputs: [],
        outputs: [],
        params: [],
      },
      executionStatus: 'idle',
    },
  };
}

let originalTabs: ReturnType<typeof useTabStore.getState>['tabs'];
let originalActive: string;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  originalTabs = useTabStore.getState().tabs;
  originalActive = useTabStore.getState().activeTabId;
  useTabStore.setState({
    tabs: [
      {
        ...originalTabs[0],
        id: TAB_ID,
        name: 'Tab',
        nodes: [node('boom', 'tokenizerNode', 'Boom', 0), node('fine', 'baseNode', 'Fine node', 400)],
        edges: [],
        undoStack: [],
        redoStack: [],
      },
    ],
    activeTabId: TAB_ID,
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
  // Unmount before the store is put back: this hook runs before the global
  // cleanup, and a store write under a mounted canvas is an update outside act().
  cleanup();
  vi.restoreAllMocks();
  useTabStore.setState({ tabs: originalTabs, activeTabId: originalActive });
});

describe('a node card that throws on the canvas', () => {
  it('costs that card, not the page: the box names it and every other card is drawn', () => {
    render(
      <ReactFlowProvider>
        <FlowCanvas tabId={TAB_ID} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText(useI18n.getState().t('node.cardFailed'))).toBeTruthy();
    expect(screen.getByText('Boom')).toBeTruthy();
    expect(screen.getByText('Fine node')).toBeTruthy();
    expect(document.querySelectorAll('.react-flow__node').length).toBe(2);
  });

  it('is drawn again once its node data changes', () => {
    render(
      <ReactFlowProvider>
        <FlowCanvas tabId={TAB_ID} />
      </ReactFlowProvider>,
    );
    act(() => {
      useTabStore.setState((s) => ({
        tabs: s.tabs.map((tab) => ({
          ...tab,
          nodes: tab.nodes.map((n) =>
            n.id === 'boom' ? { ...n, data: { ...n.data, label: 'Fixed' } } : n,
          ),
        })),
      }));
    });
    expect(screen.getByText('tokenizer Fixed')).toBeTruthy();
    expect(screen.queryByText(useI18n.getState().t('node.cardFailed'))).toBeNull();
  });
});
