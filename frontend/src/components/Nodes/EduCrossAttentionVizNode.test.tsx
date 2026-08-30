import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
import EduCrossAttentionVizNode from './EduCrossAttentionVizNode';

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
    node_name: 'Edu-CrossAttention',
    category: 'Transformer',
    description: '',
    inputs: [],
    outputs: [{ name: 'weights', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function data(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'Cross Attn',
    type: 'Edu-CrossAttention',
    params: {},
    definition: def(),
    executionStatus: 'idle',
    ...overrides,
  };
}

const NODE_ID = 'ca1';

function seed(summary: Record<string, OutputSummary> | undefined, runId: string | null = 'run-1') {
  const id = 'tab-ca';
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
    <EduCrossAttentionVizNode id={NODE_ID} type="eduCrossAttentionNode" data={d} selected={false} {...flowProps} />,
  );
}

// A single head of a 2x3 (Q=2, K=3) matrix.
const oneHead = [
  [
    [0.5, 0.3, 0.2],
    [0.1, 0.6, 0.3],
  ],
];

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
      values: oneHead,
      sliced_shape: [1, 2, 3],
      run_id: 'run-1',
      node_id: NODE_ID,
      port: 'weights',
      full_shape: [1, 2, 3],
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

describe('EduCrossAttentionVizNode', () => {
  it('shows the run hint with no weights summary', () => {
    seed(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.runHint'))).toBeTruthy();
  });

  it('renders a 3D [H, Q, K] heatmap and the heads meta row', () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    const { container } = renderNode();
    // 1 head × 2×3 = 6 cells
    expect(container.querySelectorAll('rect[data-i]').length).toBe(6);
    expect(screen.getByText(useI18n.getState().t('attention.heads', { count: 1 }))).toBeTruthy();
    expect(screen.getByText('cross-attn [Q × K]')).toBeTruthy();
  });

  it('unwraps a 4D [B, H, Q, K] tensor by taking batch 0', () => {
    seed({
      weights: {
        type: 'tensor',
        values: [oneHead, [[[0, 0, 0], [0, 0, 0]]]],
      },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(6);
  });

  it('renders q and k labels when present', () => {
    seed({
      weights: { type: 'tensor', values: oneHead },
      q_labels: { type: 'list', values: ['qa', 'qb'] },
      k_labels: { type: 'list', values: ['ka', 'kb', 'kc'] },
    });
    renderNode();
    expect(screen.getByText('qa')).toBeTruthy();
    expect(screen.getByText('kc')).toBeTruthy();
  });

  it('ignores empty q/k labels (undefined branches)', () => {
    seed({
      weights: { type: 'tensor', values: oneHead },
      q_labels: { type: 'list', values: [] },
      k_labels: { type: 'list', values: [] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(6);
  });

  it('uses the small panel size for >=4 heads', () => {
    const fourHeads = Array.from({ length: 4 }, () => [
      [0.5, 0.5],
      [0.5, 0.5],
    ]);
    seed({ weights: { type: 'tensor', values: fourHeads } });
    const { container } = renderNode();
    expect(container.querySelectorAll('svg[class*="panel"]').length).toBe(4);
    expect(screen.getByText(useI18n.getState().t('attention.heads', { count: 4 }))).toBeTruthy();
  });

  it('falls back to num_heads param when no head data (too-large path)', () => {
    seed({ weights: { type: 'tensor', shape: [8, 64, 64] } });
    renderNode(data({ params: { num_heads: 8 } }));
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('defaults num_heads to 0 when neither data nor param is present', () => {
    seed({ weights: { type: 'tensor', shape: [8, 64, 64] } });
    // No num_heads param → Number(undefined ?? 0) = 0; panelSize uses 180.
    const { container } = renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
    expect(container.querySelector('button[class*="expandLink"]')).toBeTruthy();
  });

  it('"View full" opens the modal via REST', async () => {
    seed({ weights: { type: 'tensor', shape: [8, 64, 64] } });
    renderNode();
    fireEvent.click(screen.getByRole('button', { name: new RegExp(useI18n.getState().t('attention.viewFull')) }));
    await waitFor(() => expect(g.fetch).toHaveBeenCalled());
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it('inline expand opens the modal with the node label and no fetch', async () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(/EduCrossAttention · Cross Attn/)).toBeTruthy());
    expect(g.fetch).not.toHaveBeenCalled();
  });

  it('modal title falls back to node id when label is missing', async () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    const { container } = renderNode(data({ label: undefined as unknown as string }));
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(new RegExp(`EduCrossAttention · ${NODE_ID}`))).toBeTruthy());
  });

  it('closing the modal unmounts it', async () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
  });

  it('falls back to null runId when the tab has no last run', () => {
    seed({ weights: { type: 'tensor', values: oneHead } }, null);
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(6);
  });
});
