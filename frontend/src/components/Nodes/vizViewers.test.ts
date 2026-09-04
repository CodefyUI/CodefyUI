import { describe, it, expect } from 'vitest';
import type { Node } from '@xyflow/react';
import type { NodeData } from '../../types';
import {
  VIZ_VIEWERS,
  attentionHeads,
  attentionWeights,
  labelList,
  maskMatrix,
  scatterPoints,
  selfAttentionWeights,
} from './vizViewers';

const m2 = [[1, 0], [0, 1]];
const m3 = [m2, m2];
const m4 = [m3];

describe('tensor readers shared by the viz cards and their viewer', () => {
  it('attentionWeights keeps 2-D and 3-D tensors and collapses a batched 4-D one to batch 0', () => {
    expect(attentionWeights(m2)).toBe(m2);
    expect(attentionWeights(m3)).toBe(m3);
    expect(attentionWeights(m4)).toBe(m3);
    expect(attentionWeights([])).toBeNull();
    expect(attentionWeights(undefined)).toBeNull();
  });

  it('attentionHeads keeps [H, seq, seq] and collapses a batched tensor to batch 0', () => {
    expect(attentionHeads(m3)).toBe(m3);
    expect(attentionHeads(m4)).toBe(m3);
    expect(attentionHeads('no')).toBeNull();
  });

  it('selfAttentionWeights keeps [seq, seq] and collapses a [1, seq, seq] tensor to its head', () => {
    expect(selfAttentionWeights(m2)).toBe(m2);
    expect(selfAttentionWeights([m2])).toBe(m2);
    expect(selfAttentionWeights(null)).toBeNull();
  });

  it('maskMatrix coerces truthy cells to 1 and a malformed row to an empty one', () => {
    expect(maskMatrix([[true, false], [0.5, 0], 'bad'])).toEqual([[1, 0], [1, 0], []]);
    expect(maskMatrix([])).toBeNull();
  });

  it('labelList stringifies a list and is undefined for an empty or missing one', () => {
    expect(labelList(['a', 1])).toEqual(['a', '1']);
    expect(labelList([])).toBeUndefined();
    expect(labelList(undefined)).toBeUndefined();
  });

  it('scatterPoints pairs coordinates with labels and tolerates malformed rows', () => {
    expect(scatterPoints([[1, 2], [3], 'x'], ['a', 7])).toEqual([
      { x: 1, y: 2, label: 'a', cluster: 0 },
      { x: 3, y: 0, label: undefined, cluster: 1 },
      { x: 0, y: 0, label: undefined, cluster: 2 },
    ]);
    expect(scatterPoints([[1, 2]], undefined)![0].label).toBeUndefined();
    expect(scatterPoints([], [])).toBeNull();
    expect(scatterPoints(undefined, [])).toBeNull();
  });
});

describe('VIZ_VIEWERS', () => {
  const node = (type: string, data: Partial<NodeData> = {}): Node<NodeData> => ({
    id: 'n1',
    type,
    position: { x: 0, y: 0 },
    data: { label: 'Card', type: 'X', params: {}, ...data },
  });

  it('covers exactly the six viz cards registered in FlowCanvas', () => {
    expect(Object.keys(VIZ_VIEWERS).sort()).toEqual([
      'attentionHeatmapNode',
      'attentionMaskNode',
      'eduCrossAttentionNode',
      'eduMultiHeadAttentionNode',
      'eduSelfAttentionNode',
      'embeddingScatterNode',
    ]);
  });

  it('titles the viewer with the node label, and with the id when the label is absent', () => {
    const spec = VIZ_VIEWERS.eduSelfAttentionNode(node('eduSelfAttentionNode'), undefined, null);
    expect(spec.kind).toBe('heatmap');
    expect(spec.props.title).toBe('EduSelfAttention · Card');
    const noLabel = node('eduSelfAttentionNode', { label: undefined as unknown as string });
    expect(VIZ_VIEWERS.eduSelfAttentionNode(noLabel, undefined, null).props.title).toBe(
      'EduSelfAttention · n1',
    );
  });

  it('hands the heatmap viewer the port to fetch and the run to fetch it from', () => {
    const spec = VIZ_VIEWERS.attentionHeatmapNode(
      node('attentionHeatmapNode', { params: { colormap: 'magma' } }),
      { weights: { type: 'tensor', shape: [40, 40] }, labels: { type: 'list', values: ['a'] } },
      'run-9',
    );
    expect(spec.kind).toBe('heatmap');
    expect(spec.props).toMatchObject({
      inlineData: null,
      rowLabels: ['a'],
      colormap: 'magma',
      runId: 'run-9',
      nodeId: 'n1',
      port: 'weights',
    });
    expect(
      VIZ_VIEWERS.attentionHeatmapNode(node('attentionHeatmapNode'), undefined, null).props,
    ).toMatchObject({ colormap: 'viridis', runId: null });
  });

  it('reads the mask, the cross-attention axes and the multi-head weights off their own ports', () => {
    const mask = VIZ_VIEWERS.attentionMaskNode(
      node('attentionMaskNode'),
      { mask: { type: 'tensor', values: [[true, false]] } },
      null,
    );
    expect(mask.props).toMatchObject({ inlineData: [[1, 0]], port: 'mask', variant: 'mask' });

    const cross = VIZ_VIEWERS.eduCrossAttentionNode(
      node('eduCrossAttentionNode'),
      {
        weights: { type: 'tensor', values: m3 },
        q_labels: { type: 'list', values: ['q'] },
        k_labels: { type: 'list', values: ['k1', 'k2'] },
      },
      null,
    );
    expect(cross.props).toMatchObject({
      inlineData: m3,
      rowLabels: ['q'],
      colLabels: ['k1', 'k2'],
      detectCausalMask: false,
    });

    const heads = VIZ_VIEWERS.eduMultiHeadAttentionNode(
      node('eduMultiHeadAttentionNode'),
      { weights: { type: 'tensor', values: m4 }, labels: { type: 'list', values: ['t'] } },
      null,
    );
    expect(heads.props).toMatchObject({ inlineData: m3, rowLabels: ['t'], detectCausalMask: true });
  });

  it('builds the scatter viewer from the points and labels ports', () => {
    const spec = VIZ_VIEWERS.embeddingScatterNode(
      node('embeddingScatterNode'),
      {
        points_2d: { type: 'tensor', values: [[0, 1]] },
        labels: { type: 'list', values: ['cat'] },
      },
      'run-1',
    );
    expect(spec.kind).toBe('scatter');
    expect(spec.props).toMatchObject({
      title: 'EmbeddingScatter · Card',
      inlinePoints: [{ x: 0, y: 1, label: 'cat', cluster: 0 }],
      runId: 'run-1',
      nodeId: 'n1',
      pointsPort: 'points_2d',
      labelsPort: 'labels',
    });
  });
});
