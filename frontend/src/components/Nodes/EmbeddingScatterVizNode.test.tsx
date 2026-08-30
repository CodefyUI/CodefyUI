import { describe, it, expect, beforeEach } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
import EmbeddingScatterVizNode from './EmbeddingScatterVizNode';

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
    node_name: 'EmbeddingScatter',
    category: 'Data',
    description: '',
    inputs: [],
    outputs: [{ name: 'points_2d', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function data(): NodeData {
  return { label: 'Embedding', type: 'EmbeddingScatter', params: {}, definition: def(), executionStatus: 'idle' };
}

const NODE_ID = 'emb1';

function seed(summary: Record<string, OutputSummary> | undefined) {
  const id = 'tab-emb';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [
      {
        ...s.tabs[0],
        id,
        name: 'Tab',
        nodes: [],
        edges: [],
        outputSummaries: summary ? { [NODE_ID]: summary } : ({} as Record<string, Record<string, OutputSummary>>),
      },
    ],
  }));
}

function renderNode() {
  return renderWithFlow(
    <EmbeddingScatterVizNode id={NODE_ID} type="embeddingScatterNode" data={data()} selected={false} {...flowProps} />,
  );
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  seed(undefined);
});

describe('EmbeddingScatterVizNode', () => {
  it('shows the run hint with no summary', () => {
    seed(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('scatter.runHint'))).toBeTruthy();
  });

  it('shows the run hint when points_2d.values is empty', () => {
    seed({ points_2d: { type: 'tensor', values: [] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('scatter.runHint'))).toBeTruthy();
  });

  it('shows the run hint when points_2d.values is not an array', () => {
    seed({ points_2d: { type: 'tensor' } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('scatter.runHint'))).toBeTruthy();
  });

  it('renders a scatter point per row with string labels', () => {
    seed({
      points_2d: { type: 'tensor', values: [[0, 0], [1, 1], [2, 2]] },
      labels: { type: 'list', values: ['cat', 'dog', 'fish'] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('circle').length).toBe(3);
    expect(screen.getByText('cat')).toBeTruthy();
    expect(screen.getByText('fish')).toBeTruthy();
  });

  it('coerces non-number coords to 0 and treats non-array rows as empty', () => {
    seed({
      points_2d: { type: 'tensor', values: [[0, 0], ['x', 'y'], 'bad'] },
    });
    const { container } = renderNode();
    // All three rows produce a point (non-array → r=[] → x=y=0; non-number → 0).
    expect(container.querySelectorAll('circle').length).toBe(3);
  });

  it('omits a label when labels is absent or the entry is not a string', () => {
    seed({
      points_2d: { type: 'tensor', values: [[0, 0], [1, 1]] },
      // labels present but one entry is a number → label undefined for that point
      labels: { type: 'list', values: ['named', 42] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('circle').length).toBe(2);
    expect(screen.getByText('named')).toBeTruthy();
  });

  it('omits labels entirely when the labels summary is missing', () => {
    seed({ points_2d: { type: 'tensor', values: [[0, 0], [1, 1]] } });
    const { container } = renderNode();
    expect(container.querySelectorAll('circle').length).toBe(2);
    // No text labels rendered.
    expect(container.querySelectorAll('text').length).toBe(0);
  });

  it('treats a non-array labels value as no labels', () => {
    seed({
      points_2d: { type: 'tensor', values: [[0, 0]] },
      labels: { type: 'list' }, // values undefined → not an array
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('circle').length).toBe(1);
  });

  it('shows the too-large hint (not the run hint) when values are absent but a shape says N>0', () => {
    // Large N: backend omits inline values (numel > 256) but keeps the shape.
    seed({ points_2d: { type: 'tensor', shape: [200, 2] } });
    renderNode();
    const t = useI18n.getState().t;
    expect(screen.getByText(t('scatter.tooLargeInline'))).toBeTruthy();
    expect(screen.queryByText(t('scatter.runHint'))).toBeNull();
  });

  it('opens the modal from the too-large "open detailed view" button', () => {
    seed({ points_2d: { type: 'tensor', shape: [200, 2] } });
    renderNode();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /open detailed view/i }));
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
    expect(screen.getByText(/EmbeddingScatter · Embedding/)).toBeTruthy();
  });

  it('opens the modal from the inline plot expand affordance', () => {
    seed({
      points_2d: { type: 'tensor', values: [[0, 0], [1, 1]] },
      labels: { type: 'list', values: ['cat', 'dog'] },
    });
    const { container } = renderNode();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    // The expand button rendered by ScatterPlot when onExpand is wired.
    const expandBtn = container.querySelector('button') as HTMLButtonElement;
    expect(expandBtn).toBeTruthy();
    fireEvent.click(expandBtn);
    expect(document.querySelector('[role="dialog"]')).toBeTruthy();
  });

  it('titles the modal with the node id when the label is absent, and Esc closes it', () => {
    seed({ points_2d: { type: 'tensor', values: [[0, 0], [1, 1]] } });
    const noLabel = { ...data(), label: undefined as unknown as string };
    const { container } = renderWithFlow(
      <EmbeddingScatterVizNode
        id={NODE_ID}
        type="embeddingScatterNode"
        data={noLabel}
        selected={false}
        {...flowProps}
      />,
    );
    fireEvent.click(container.querySelector('button') as HTMLButtonElement);
    expect(screen.getByText(new RegExp(`EmbeddingScatter · ${NODE_ID}`))).toBeTruthy();
    // Closing the modal flips the node's expanded state back off.
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });
});
