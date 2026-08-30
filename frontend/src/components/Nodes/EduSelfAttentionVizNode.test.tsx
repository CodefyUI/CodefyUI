import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
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

function def(): NodeDefinition {
  return {
    node_name: 'Edu-SelfAttention',
    category: 'Transformer',
    description: '',
    inputs: [],
    outputs: [{ name: 'weights', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function data(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'Self Attn',
    type: 'Edu-SelfAttention',
    params: {},
    definition: def(),
    executionStatus: 'idle',
    ...overrides,
  };
}

const NODE_ID = 'sa1';

function seed(summary: Record<string, OutputSummary> | undefined, runId: string | null = 'run-1') {
  const id = 'tab-sa';
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
    <EduSelfAttentionVizNode id={NODE_ID} type="eduSelfAttentionNode" data={d} selected={false} {...flowProps} />,
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

describe('EduSelfAttentionVizNode', () => {
  it('shows the run hint with no weights summary', () => {
    seed(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.runHint'))).toBeTruthy();
  });

  it('renders a 2D matrix', () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('unwraps a 3D tensor by taking the first slice', () => {
    seed({
      weights: {
        type: 'tensor',
        values: [
          [[0.5, 0.5], [0.3, 0.7]], // slice 0 → used
          [[0, 1], [1, 0]], // dropped
        ],
      },
    });
    const { container } = renderNode();
    // Single 2x2 panel = 4 cells.
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('renders row labels when present', () => {
    seed({
      weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] },
      labels: { type: 'list', values: ['q0', 'q1'] },
    });
    renderNode();
    expect(screen.getAllByText('q0').length).toBeGreaterThanOrEqual(1);
  });

  it('ignores empty labels', () => {
    seed({
      weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] },
      labels: { type: 'list', values: [] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });

  it('shows the causal meta row when params.causal is "true"', () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    renderNode(data({ params: { causal: 'true' } }));
    expect(screen.getByText(useI18n.getState().t('attention.causalMasked'))).toBeTruthy();
    expect(screen.getByText('causal=true')).toBeTruthy();
  });

  it('does not show the causal meta row when causal is not "true"', () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    renderNode(data({ params: { causal: 'false' } }));
    expect(screen.queryByText(useI18n.getState().t('attention.causalMasked'))).toBeNull();
  });

  it('shows the too-large hint when shape exists but values are missing', () => {
    seed({ weights: { type: 'tensor', shape: [512, 512] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('"View full" opens the modal via REST', async () => {
    seed({ weights: { type: 'tensor', shape: [512, 512] } });
    renderNode();
    fireEvent.click(screen.getByRole('button', { name: new RegExp(useI18n.getState().t('attention.viewFull')) }));
    await waitFor(() => expect(g.fetch).toHaveBeenCalled());
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it('inline expand opens the modal with the node label and no fetch', async () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(/EduSelfAttention · Self Attn/)).toBeTruthy());
    expect(g.fetch).not.toHaveBeenCalled();
  });

  it('modal title falls back to node id when label is missing', async () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode(data({ label: undefined as unknown as string }));
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText(new RegExp(`EduSelfAttention · ${NODE_ID}`))).toBeTruthy());
  });

  it('closing the modal unmounts it', async () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } });
    const { container } = renderNode();
    fireEvent.click(container.querySelector('button[aria-label="Expand heatmap"]') as HTMLButtonElement);
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeTruthy());
    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
  });

  it('falls back to null runId when the tab has no last run', () => {
    seed({ weights: { type: 'tensor', values: [[0.5, 0.5], [0.3, 0.7]] } }, null);
    const { container } = renderNode();
    expect(container.querySelectorAll('rect[data-i]').length).toBe(4);
  });
});
