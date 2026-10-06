import { describe, it, expect, beforeEach } from 'vitest';
import type { ComponentType } from 'react';
import type { NodeProps } from '@xyflow/react';
import { render, screen, fireEvent } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore, type LogChartPayload } from '../../store/tabStore';
import type { AppNode, NodeData, OutputSummary } from '../../types';
import { ChartView } from '../shared/ChartView';
import AttentionHeatmapVizNode from './AttentionHeatmapVizNode';
import AttentionMaskVizNode from './AttentionMaskVizNode';
import EduCrossAttentionVizNode from './EduCrossAttentionVizNode';
import EduMultiHeadAttentionVizNode from './EduMultiHeadAttentionVizNode';
import EduSelfAttentionVizNode from './EduSelfAttentionVizNode';
import { VizViewerModal } from './VizViewerModal';

/**
 * Every card that draws a matrix, its full-size viewer, and a heatmap chart,
 * fed every tensor a run can leave on the port. All of them end in
 * HeatmapPlot, which trusted its input to be [seq, seq] or [H, seq, seq]: a
 * 1-D tensor threw during render, and a NaN / inf cell (null after the
 * backend's json_safe) or a boolean one threw on hover. Either unmounted the
 * whole page.
 */

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

const NODE_ID = 'viz1';
const LABEL = 'Viz card';

// [node type, React Flow node type, card, the output port the card draws]
const CARDS: [string, string, ComponentType<NodeProps<AppNode>>, string][] = [
  ['AttentionHeatmap', 'attentionHeatmapNode', AttentionHeatmapVizNode, 'weights'],
  ['AttentionMask', 'attentionMaskNode', AttentionMaskVizNode, 'mask'],
  ['Edu-SelfAttention', 'eduSelfAttentionNode', EduSelfAttentionVizNode, 'weights'],
  ['Edu-MultiHeadAttention', 'eduMultiHeadAttentionNode', EduMultiHeadAttentionVizNode, 'weights'],
  ['Edu-CrossAttention', 'eduCrossAttentionNode', EduCrossAttentionVizNode, 'weights'],
];

// A tensor summary's `values`: the tensor's tolist(), after json_safe.
const TENSORS: [string, unknown][] = [
  ['0-D', 0.5],
  ['0-D NaN', null],
  ['1-D', [0.1, 0.4, 0.2, 0.3]],
  ['empty', []],
  ['[2, 0]', [[], []]],
  ['2-D NaN / inf', [[0.5, null], [null, 0.5]]],
  ['boolean', [[true, false], [false, true]]],
  ['3-D NaN', [[[0.5, null], [0.2, 0.8]], [[0.1, 0.9], [null, null]]]],
  ['5-D', [[[[[0.5, 0.5]]]]]],
  ['list of strings', ['a', 'b']],
  ['ragged', [[0.1, 0.2], 0.3]],
];

function showCard(
  type: string,
  rfType: string,
  Card: ComponentType<NodeProps<AppNode>>,
  port: string,
  values: unknown,
) {
  const data: NodeData = {
    label: LABEL,
    type,
    params: {},
    definition: {
      node_name: type,
      category: 'LLM',
      description: '',
      inputs: [],
      outputs: [{ name: port, data_type: 'TENSOR', description: '', optional: false }],
      params: [],
    },
    executionStatus: 'idle',
  };
  const summary = { type: 'tensor', values } as OutputSummary;
  useTabStore.setState((s) => ({
    activeTabId: 'tab-viz',
    tabs: [
      {
        ...s.tabs[0],
        id: 'tab-viz',
        name: 'Tab',
        // The viewer host reads the node off the tab.
        nodes: [{ id: NODE_ID, type: rfType, position: { x: 0, y: 0 }, data }],
        edges: [],
        vizModalNodeId: null,
        lastRunId: 'run-1',
        outputSummaries: { [NODE_ID]: { [port]: summary } },
      },
    ],
  }));
  return renderWithFlow(
    <>
      <Card id={NODE_ID} type={rfType} data={data} selected={false} {...flowProps} />
      <VizViewerModal />
    </>,
  );
}

function hoverEveryCell(root: ParentNode) {
  for (const cell of Array.from(root.querySelectorAll('rect[data-i]'))) {
    fireEvent.mouseEnter(cell, { clientX: 1, clientY: 1 });
    fireEvent.mouseLeave(cell);
  }
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe.each(CARDS)('the %s card', (type, rfType, Card, port) => {
  it.each(TENSORS)('and its viewer survive a %s tensor, hovered cell by cell', (_shape, values) => {
    const { container } = showCard(type, rfType, Card, port, values);
    hoverEveryCell(container);
    const expand = container.querySelector('button[aria-label="Open detailed view"]');
    if (expand) {
      fireEvent.click(expand);
      hoverEveryCell(screen.getByRole('dialog'));
    }
    // A throw anywhere above unmounts the whole tree, card included.
    expect(screen.getByText(LABEL)).toBeTruthy();
  });
});

describe('a heatmap chart', () => {
  it.each(TENSORS)('survives a %s matrix, hovered cell by cell', (_shape, matrix) => {
    const chart = { kind: 'heatmap', matrix } as unknown as LogChartPayload;
    const { container } = render(<ChartView chart={chart} />);
    hoverEveryCell(container);
    expect(container.querySelector('[data-chart-kind="heatmap"]')).toBeTruthy();
  });
});
