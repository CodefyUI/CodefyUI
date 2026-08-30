import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
import AttentionMaskVizNode from './AttentionMaskVizNode';

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
    node_name: 'AttentionMask',
    category: 'Transformer',
    description: '',
    inputs: [],
    outputs: [{ name: 'mask', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function data(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'Attention Mask',
    type: 'AttentionMask',
    params: {},
    definition: def(),
    executionStatus: 'idle',
    ...overrides,
  };
}

const NODE_ID = 'm1';

function seed(summary: Record<string, OutputSummary> | undefined, runId: string | null = 'run-1') {
  const id = 'tab-mask';
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
    <AttentionMaskVizNode id={NODE_ID} type="attentionMaskNode" data={d} selected={false} {...flowProps} />,
  );
}

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  seed(undefined);
  originalFetch = g.fetch;
  // Default fetch mock keeps the modal's REST path from hitting the network.
  g.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: 'ok',
    json: async () => ({
      type: 'tensor',
      values: [[1, 0], [1, 1]],
      sliced_shape: [2, 2],
      run_id: 'run-1',
      node_id: NODE_ID,
      port: 'mask',
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

describe('AttentionMaskVizNode', () => {
  it('shows the run hint when there is no mask summary', () => {
    seed(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.maskRunHint'))).toBeTruthy();
  });

  it('renders the heatmap when mask values are present (booleans coerced to 0/1)', () => {
    seed({ mask: { type: 'tensor', values: [[true, false], [1, 0]] } });
    const { container } = renderNode();
    // 2x2 → 4 cells
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('coerces non-array rows to empty arrays', () => {
    // One valid row + one non-array row → second row maps to [] (0 cells).
    seed({ mask: { type: 'tensor', values: [[1, 0], 'bad'] } });
    const { container } = renderNode();
    // Only the first row yields 2 cells.
    expect(container.querySelectorAll('rect[data-i]').length).toBe(2);
  });

  it('shows the too-large hint when mask shape exists but values are missing', () => {
    // hasShape true (mask summary object present), but values absent → matrix null.
    seed({ mask: { type: 'tensor', shape: [512, 512] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
    expect(screen.getByRole('button', { name: new RegExp(useI18n.getState().t('attention.viewFull')) })).toBeTruthy();
  });

  it('shows the too-large hint when values is an empty array', () => {
    seed({ mask: { type: 'tensor', values: [] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('clicking "View full" in the too-large path opens the modal (REST fetch)', async () => {
    seed({ mask: { type: 'tensor', shape: [512, 512] } });
    renderNode();
    fireEvent.click(screen.getByRole('button', { name: new RegExp(useI18n.getState().t('attention.viewFull')) }));
    // Modal opens and fetches the full tensor.
    await waitFor(() => expect(g.fetch).toHaveBeenCalled());
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it('clicking the HeatmapPlot expand button opens the modal with inline data', async () => {
    seed({ mask: { type: 'tensor', values: [[1, 0], [1, 1]] } });
    const { container } = renderNode();
    const expandBtn = container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement;
    expect(expandBtn).toBeTruthy();
    fireEvent.click(expandBtn);
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
    // Inline data provided → no REST fetch.
    expect(g.fetch).not.toHaveBeenCalled();
    // Modal title uses the node label.
    expect(screen.getByText(/AttentionMask · Attention Mask/)).toBeTruthy();
  });

  it('modal title falls back to the node id when label is absent', async () => {
    seed({ mask: { type: 'tensor', values: [[1, 0], [1, 1]] } });
    const { container } = renderNode(data({ label: undefined as unknown as string }));
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(new RegExp(`AttentionMask · ${NODE_ID}`))).toBeTruthy());
  });

  it('uses null runId when the tab has no last run', () => {
    seed({ mask: { type: 'tensor', values: [[1, 0], [1, 1]] } }, null);
    const { container } = renderNode();
    // Renders fine; runId null is just forwarded to the (closed) modal.
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('closing the modal (Esc) calls onClose and unmounts it', async () => {
    seed({ mask: { type: 'tensor', values: [[1, 0], [1, 1]] } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
  });
});
