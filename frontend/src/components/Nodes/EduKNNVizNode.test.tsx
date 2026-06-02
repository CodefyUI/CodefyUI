import { describe, it, expect, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
import EduKNNVizNode from './EduKNNVizNode';

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
    node_name: 'Edu-KNN',
    category: 'Classical',
    description: '',
    inputs: [],
    outputs: [{ name: 'train_coords', data_type: 'TENSOR', description: '', optional: false }],
    params: [],
  };
}

function data(): NodeData {
  return { label: 'KNN', type: 'Edu-KNN', params: {}, definition: def(), executionStatus: 'idle' };
}

const NODE_ID = 'knn1';

function seed(summary: Record<string, OutputSummary> | undefined) {
  const id = 'tab-knn';
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
    <EduKNNVizNode id={NODE_ID} type="eduKNNNode" data={data()} selected={false} {...flowProps} />,
  );
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  seed(undefined);
});

describe('EduKNNVizNode', () => {
  it('shows the run hint with no summary', () => {
    seed(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.runHint'))).toBeTruthy();
  });

  it('shows the run hint when train_coords is missing but query_coords is present', () => {
    // No train_coords summary at all → hasShape false → run hint.
    seed({ query_coords: { type: 'tensor', values: [[1, 2]] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.runHint'))).toBeTruthy();
  });

  it('renders the scatter with class count and the query legend', () => {
    seed({
      train_coords: { type: 'tensor', values: [[0, 0], [1, 1], [2, 2]] },
      train_labels: { type: 'list', values: ['a', 'b', 'a'] },
      query_coords: { type: 'tensor', values: [[0.5, 0.5]] },
    });
    const { container } = renderNode();
    // 3 train points + 1 query point = 4 circles.
    expect(container.querySelectorAll('circle').length).toBe(4);
    // 2 distinct classes (a, b)
    expect(screen.getByText('2 classes')).toBeTruthy();
    expect(screen.getByText('? = query')).toBeTruthy();
    // query point labelled ?0
    expect(screen.getByText('?0')).toBeTruthy();
  });

  it('defaults missing labels to empty string (single class)', () => {
    seed({
      train_coords: { type: 'tensor', values: [[0, 0], [1, 1]] },
      // no train_labels → labelStrings empty → both points share '' → 1 class
      query_coords: { type: 'tensor', values: [[0.5, 0.5]] },
    });
    renderNode();
    expect(screen.getByText('1 classes')).toBeTruthy();
  });

  it('coerces non-number coords to 0 and skips malformed rows (< 2 dims / non-array)', () => {
    seed({
      train_coords: {
        type: 'tensor',
        // valid, non-number x→0 / non-number y→0, too-short (skipped), non-array (skipped)
        values: [[1, 2], ['x', 'y'], [9], 'bad'],
      },
      train_labels: { type: 'list', values: ['a', 'b', 'c', 'd'] },
      query_coords: {
        type: 'tensor',
        // valid + non-number coords (x,y → 0) + too-short (skipped) + non-array (skipped)
        values: [[0.5, 0.5], ['a', 'b'], [1], 7],
      },
    });
    const { container } = renderNode();
    // 2 valid train rows + 2 plotted query rows (valid + coerced-to-0) = 4 circles.
    expect(container.querySelectorAll('circle').length).toBe(4);
  });

  it('shows the too-large hint when train_coords shape exists but values are missing', () => {
    seed({ train_coords: { type: 'tensor', shape: [500, 4] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
    expect(screen.getByText('view in inspector')).toBeTruthy();
  });

  it('shows the too-large hint when query_coords is missing (data null, hasShape true)', () => {
    // train_coords has values, but query_coords missing → data null; train_coords
    // present → hasShape true → too-large hint.
    seed({ train_coords: { type: 'tensor', values: [[0, 0], [1, 1]] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('shows the too-large hint when train_coords values is empty', () => {
    seed({
      train_coords: { type: 'tensor', values: [] },
      query_coords: { type: 'tensor', values: [[0.5, 0.5]] },
    });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });

  it('shows the too-large hint when query_coords values is empty', () => {
    seed({
      train_coords: { type: 'tensor', values: [[0, 0]] },
      query_coords: { type: 'tensor', values: [] },
    });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('attention.tooLargeInline'))).toBeTruthy();
  });
});
