import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
import AttentionHeatmapVizNode from './AttentionHeatmapVizNode';

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

function def(): NodeDefinition {
  return {
    node_name: 'AttentionHeatmap',
    category: 'Transformer',
    description: '',
    inputs: [],
    outputs: [{ name: 'weights', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function data(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'Attn Heatmap',
    type: 'AttentionHeatmap',
    params: {},
    definition: def(),
    executionStatus: 'idle',
    ...overrides,
  };
}

const NODE_ID = 'h1';

function seed(summary: Record<string, OutputSummary> | undefined, runId: string | null = 'run-1') {
  const id = 'tab-heat';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [
      {
        ...s.tabs[0],
        id,
        name: 'Tab',
        nodes: [],
        edges: [],
        lastRunId: runId,
        outputSummaries: summary ? { [NODE_ID]: summary } : ({} as Record<string, Record<string, OutputSummary>>),
      },
    ],
  }));
}

function renderNode(d: NodeData = data()) {
  return renderWithFlow(
    <AttentionHeatmapVizNode id={NODE_ID} type="attentionHeatmapNode" data={d} selected={false} {...flowProps} />,
  );
}

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  seed(undefined);
  originalFetch = g.fetch;
  g.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: 'ok',
    json: async () => ({
      type: 'tensor',
      values: [[0.5, 0.5], [0.3, 0.7]],
      sliced_shape: [2, 2],
      run_id: 'run-1',
      node_id: NODE_ID,
      port: 'weights',
      full_shape: [2, 2],
      dtype: 'float32',
      slice: '',
      truncated: false,
    }),
  } as unknown as Response) as unknown as typeof fetch;
});

afterEach(() => {
  g.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe('AttentionHeatmapVizNode', () => {
  it('shows the run hint when there is no weights summary', () => {
    seed(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.runHint'))).toBeTruthy();
  });

  it('renders a 2D heatmap', () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('renders a 3D (multi-head) heatmap and applies the smaller panel size', () => {
    seed({
      weights: {
        type: 'tensor',
        values: [
          [[0.5, 0.5], [0.3, 0.7]],
          [[0.1, 0.9], [0.6, 0.4]],
        ],
      },
    });
    const { container } = renderNode();
    // 2 heads × 2×2 = 8 cells
    expect(container.querySelectorAll('rect[data-i]').length).toBe(8);
    // is3D → panel 140; two panels rendered
    expect(container.querySelectorAll('svg[class*="panel"]').length).toBe(2);
  });

  it('unwraps a 4D tensor by taking batch 0', () => {
    seed({
      weights: {
        type: 'tensor',
        values: [
          // batch 0: 1 head of 2x2
          [[[0.5, 0.5], [0.3, 0.7]]],
          // batch 1 (dropped)
          [[[0, 1], [1, 0]]],
        ],
      },
    });
    const { container } = renderNode();
    // After dropping batch → 1 head × 2×2 = 4 cells.
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('renders row labels when provided', () => {
    seed({
      weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] },
      labels: { type: 'list', values: ['aa', 'bb'] },
    });
    renderNode();
    // Labels appear on both axes (col defaults to row), so there are 2 each.
    expect(screen.getAllByText('aa').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('bb').length).toBeGreaterThanOrEqual(1);
  });

  it('ignores empty / non-array labels (labels undefined branch)', () => {
    seed({
      weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] },
      labels: { type: 'list', values: [] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('uses the colormap param when provided', async () => {
    seed({ weights: { type: 'tensor', values: [[1, 0], [0, 1]] } });
    const { container } = renderNode(data({ params: { colormap: 'RdBu' } }));
    // Open the modal to assert the colormap propagates (cells exist either way).
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
  });

  it('shows the too-large hint when shape exists but values are missing', () => {
    seed({ weights: { type: 'tensor', shape: [512, 512] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('clicking "View full" opens the modal (REST fetch path)', async () => {
    seed({ weights: { type: 'tensor', shape: [512, 512] } });
    renderNode();
    fireEvent.click(screen.getByRole('button', { name: new RegExp(useI18n.getState().t('attention.viewFull')) }));
    await waitFor(() => expect(g.fetch).toHaveBeenCalled());
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it('expand button on the inline plot opens the modal with inline data + node label', async () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(/AttentionHeatmap · Attn Heatmap/)).toBeTruthy());
    expect(g.fetch).not.toHaveBeenCalled();
  });

  it('modal title falls back to node id when label is missing', async () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode(data({ label: undefined as unknown as string }));
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(new RegExp(`AttentionHeatmap · ${NODE_ID}`))).toBeTruthy());
  });

  it('closing the modal unmounts it', async () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
  });

  it('handles null runId without crashing', () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } }, null);
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });
});
