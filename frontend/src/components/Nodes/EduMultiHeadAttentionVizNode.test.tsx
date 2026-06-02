import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
import EduMultiHeadAttentionVizNode from './EduMultiHeadAttentionVizNode';

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
    node_name: 'Edu-MultiHeadAttention',
    category: 'Transformer',
    description: '',
    inputs: [],
    outputs: [{ name: 'weights', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function data(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'MHA',
    type: 'Edu-MultiHeadAttention',
    params: {},
    definition: def(),
    executionStatus: 'idle',
    ...overrides,
  };
}

const NODE_ID = 'mha1';

function seed(summary: Record<string, OutputSummary> | undefined, runId: string | null = 'run-1') {
  const id = 'tab-mha';
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
    <EduMultiHeadAttentionVizNode id={NODE_ID} type="eduMultiHeadAttentionNode" data={d} selected={false} {...flowProps} />,
  );
}

const oneHead = [
  [
    [0.5, 0.5],
    [0.3, 0.7],
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
      sliced_shape: [1, 2, 2],
      run_id: 'run-1',
      node_id: NODE_ID,
      port: 'weights',
      full_shape: [1, 2, 2],
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

describe('EduMultiHeadAttentionVizNode', () => {
  it('shows the run hint with no weights summary', () => {
    seed(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.runHint'))).toBeTruthy();
  });

  it('renders a 3D [H, seq, seq] heatmap and the meta row (non-causal)', () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
    expect(screen.getByText(useI18n.getState().t('attention.heads', { count: 1 }))).toBeTruthy();
    expect(screen.getByText('weights [H, seq, seq]')).toBeTruthy();
  });

  it('appends " · causal" to the heads count when causal is "true"', () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    renderNode(data({ params: { causal: 'true' } }));
    // The heads count and the causal suffix are in the same span.
    const heads = useI18n.getState().t('attention.heads', { count: 1 });
    expect(screen.getByText(`${heads} · causal`)).toBeTruthy();
  });

  it('omits the causal suffix when causal is not "true"', () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    renderNode(data({ params: { causal: 'false' } }));
    expect(screen.queryByText(/· causal/)).toBeNull();
  });

  it('unwraps a 4D tensor by taking batch 0', () => {
    seed({
      weights: { type: 'tensor', values: [oneHead, [[[0, 0], [0, 0]]]] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('renders labels when present', () => {
    seed({
      weights: { type: 'tensor', values: oneHead },
      labels: { type: 'list', values: ['t0', 't1'] },
    });
    renderNode();
    expect(screen.getAllByText('t0').length).toBeGreaterThanOrEqual(1);
  });

  it('ignores empty labels', () => {
    seed({
      weights: { type: 'tensor', values: oneHead },
      labels: { type: 'list', values: [] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('uses the small panel size for >=4 heads', () => {
    const fourHeads = Array.from({ length: 4 }, () => [
      [0.5, 0.5],
      [0.5, 0.5],
    ]);
    seed({ weights: { type: 'tensor', values: fourHeads } });
    const { container } = renderNode();
    expect(container.querySelectorAll('svg[class*="panel"]').length).toBe(4);
  });

  it('falls back to the num_heads param in the too-large path', () => {
    seed({ weights: { type: 'tensor', shape: [12, 128, 128] } });
    renderNode(data({ params: { num_heads: 12 } }));
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('defaults num_heads to 0 when neither head data nor the param exists', () => {
    seed({ weights: { type: 'tensor', shape: [12, 128, 128] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('"View full" opens the modal via REST', async () => {
    seed({ weights: { type: 'tensor', shape: [12, 128, 128] } });
    renderNode();
    fireEvent.click(screen.getByRole('button', { name: new RegExp(useI18n.getState().t('attention.viewFull')) }));
    await waitFor(() => expect(g.fetch).toHaveBeenCalled());
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it('inline expand opens the modal with the node label and no fetch', async () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(/EduMultiHeadAttention · MHA/)).toBeTruthy());
    expect(g.fetch).not.toHaveBeenCalled();
  });

  it('modal title falls back to node id when label is missing', async () => {
    seed({ weights: { type: 'tensor', values: oneHead } });
    const { container } = renderNode(data({ label: undefined as unknown as string }));
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(new RegExp(`EduMultiHeadAttention · ${NODE_ID}`))).toBeTruthy());
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
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });
});
