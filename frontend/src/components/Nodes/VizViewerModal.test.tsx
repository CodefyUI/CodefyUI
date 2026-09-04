import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { FlowWrapper, renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, OutputSummary } from '../../types';
import { VizViewerModal } from './VizViewerModal';
import EduSelfAttentionVizNode from './EduSelfAttentionVizNode';

const flowProps = {
  zIndex: 0,
  isConnectable: true,
  positionAbsoluteX: 0,
  positionAbsoluteY: 0,
  dragging: false,
  draggable: false,
  selectable: true,
  deletable: true,
} as const;

const TAB = 'tab-viz-host';

function tabNode(id: string, type: string, data: Partial<NodeData> = {}) {
  return {
    id,
    type,
    position: { x: 0, y: 0 },
    data: { label: `Card ${id}`, type: 'X', params: {}, ...data },
  };
}

function seed(over: {
  nodes?: ReturnType<typeof tabNode>[];
  outputSummaries?: Record<string, Record<string, OutputSummary>>;
  vizModalNodeId?: string | null;
  lastRunId?: string | null;
}) {
  useTabStore.setState((s) => ({
    activeTabId: TAB,
    tabs: [
      {
        ...s.tabs[0],
        id: TAB,
        name: 'Tab',
        nodes: [],
        edges: [],
        outputSummaries: {},
        lastRunId: 'run-1',
        vizModalNodeId: null,
        ...over,
      } as never,
    ],
  }));
}

const activeTab = () => useTabStore.getState().getActiveTab();
const dialog = () => document.querySelector('[role="dialog"]');

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  seed({});
  originalFetch = g.fetch;
  // A viewer with no inline values REST-fetches; none of these tests care
  // what comes back, only that the viewer is up.
  g.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ type: 'tensor', shape: [2, 2], values: [[1, 0], [0, 1]] }),
  }) as unknown as typeof fetch;
});

afterEach(() => {
  g.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe('VizViewerModal', () => {
  it('renders nothing while no card has asked for its viewer', () => {
    render(<VizViewerModal />);
    expect(dialog()).toBeNull();
  });

  it('renders nothing for an id the tab does not hold, or a node type without a viewer', () => {
    seed({ nodes: [], vizModalNodeId: 'ghost' });
    const { rerender } = render(<VizViewerModal />);
    expect(dialog()).toBeNull();

    seed({ nodes: [tabNode('plain', 'baseNode')], vizModalNodeId: 'plain' });
    rerender(<VizViewerModal />);
    expect(dialog()).toBeNull();
  });

  it.each([
    ['attentionHeatmapNode', 'AttentionHeatmap', { weights: { type: 'tensor', values: [[1, 0], [0, 1]] } }],
    ['attentionMaskNode', 'AttentionMask', { mask: { type: 'tensor', values: [[1, 0], [0, 1]] } }],
    ['eduCrossAttentionNode', 'EduCrossAttention', { weights: { type: 'tensor', values: [[[1, 0], [0, 1]]] } }],
    ['eduMultiHeadAttentionNode', 'EduMultiHeadAttention', { weights: { type: 'tensor', values: [[[1, 0], [0, 1]]] } }],
    ['eduSelfAttentionNode', 'EduSelfAttention', { weights: { type: 'tensor', values: [[1, 0], [0, 1]] } }],
    ['embeddingScatterNode', 'EmbeddingScatter', { points_2d: { type: 'tensor', values: [[0, 0], [1, 1]] } }],
  ] as [string, string, Record<string, OutputSummary>][])(
    'shows the %s viewer titled after the card',
    (type, kind, summary) => {
    seed({
      nodes: [tabNode('v1', type)],
      outputSummaries: { v1: summary },
      vizModalNodeId: 'v1',
    });
    render(<VizViewerModal />);
    expect(dialog()).toBeTruthy();
    expect(screen.getByText(`${kind} · Card v1`)).toBeTruthy();
    },
  );

  it('Escape and a backdrop click both close through the store', () => {
    seed({
      nodes: [tabNode('v1', 'eduSelfAttentionNode')],
      outputSummaries: { v1: { weights: { type: 'tensor', values: [[1, 0], [0, 1]] } } },
      vizModalNodeId: 'v1',
    });
    render(<VizViewerModal />);
    expect(dialog()).toBeTruthy();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(activeTab().vizModalNodeId).toBeNull();
    expect(dialog()).toBeNull();

    act(() => useTabStore.getState().openVizModal('v1'));
    expect(dialog()).toBeTruthy();
    fireEvent.click(dialog() as HTMLElement);
    expect(activeTab().vizModalNodeId).toBeNull();
    expect(dialog()).toBeNull();
  });

  it('survives the card that opened it being unmounted by viewport culling (core#324)', () => {
    const data: NodeData = {
      label: 'Attention',
      type: 'Edu-SelfAttention',
      params: {},
      executionStatus: 'completed',
    };
    seed({
      nodes: [tabNode('sa', 'eduSelfAttentionNode', data)],
      outputSummaries: { sa: { weights: { type: 'tensor', values: [[1, 0], [0, 1]] } } },
    });
    const card = (
      <EduSelfAttentionVizNode
        id="sa"
        type="eduSelfAttentionNode"
        data={data}
        selected={false}
        {...flowProps}
      />
    );
    const { rerender } = renderWithFlow(
      <>
        {card}
        <VizViewerModal />
      </>,
    );
    expect(dialog()).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Expand heatmap' }));
    expect(activeTab().vizModalNodeId).toBe('sa');
    expect(dialog()).toBeTruthy();

    // `onlyRenderVisibleElements` unmounting the card is, to the card, just
    // an unmount: the viewer it asked for must not go with it.
    rerender(
      <FlowWrapper>
        <VizViewerModal />
      </FlowWrapper>,
    );
    expect(dialog()).toBeTruthy();
    expect(screen.getByText('EduSelfAttention · Attention')).toBeTruthy();

    // And it is still the same viewer when the card comes back into view.
    rerender(
      <FlowWrapper>
        {card}
        <VizViewerModal />
      </FlowWrapper>,
    );
    expect(dialog()).toBeTruthy();
    expect(activeTab().vizModalNodeId).toBe('sa');
  });

  it('closes when the node it shows is deleted', () => {
    seed({
      nodes: [tabNode('v1', 'eduSelfAttentionNode')],
      outputSummaries: { v1: { weights: { type: 'tensor', values: [[1, 0], [0, 1]] } } },
      vizModalNodeId: 'v1',
    });
    render(<VizViewerModal />);
    expect(dialog()).toBeTruthy();
    act(() => useTabStore.getState().deleteNode('v1'));
    expect(dialog()).toBeNull();
  });

  it('follows the run and the outputs it is handed, not a snapshot taken when it opened', () => {
    seed({
      nodes: [tabNode('v1', 'eduSelfAttentionNode')],
      outputSummaries: { v1: { weights: { type: 'tensor', shape: [40, 40] } } },
      vizModalNodeId: 'v1',
      lastRunId: 'run-1',
    });
    render(<VizViewerModal />);
    // No inline values: the viewer fetches the tensor for the run it was given.
    expect(g.fetch).toHaveBeenCalledTimes(1);
    expect(String((g.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0])).toContain('run-1');

    // A later run replaces the summaries in place; the viewer re-reads them
    // rather than keeping the first run's tensor.
    act(() =>
      useTabStore
        .getState()
        .setTabOutputSummary(TAB, 'v1', { weights: { type: 'tensor', values: [[1, 0], [0, 1]] } }),
    );
    expect(dialog()).toBeTruthy();
    expect(g.fetch).toHaveBeenCalledTimes(1);
  });
});
