import type { Node } from '@xyflow/react';
import type { NodeData, OutputSummary } from '../../types';
import type { HeatmapColormap } from '../shared/HeatmapPlot';
import type { HeatmapModalProps } from '../shared/HeatmapModal';
import type { ScatterModalProps } from '../shared/ScatterModal';
import type { ScatterPoint } from '../shared/ScatterPlot';

/**
 * What each visualization card's "View full" viewer shows, and how to read it
 * off the store (core#324).
 *
 * The six viz cards used to hold their viewer as a `useState` flag and render
 * the modal themselves. `onlyRenderVisibleElements` (#162) unmounts a card the
 * moment it leaves the viewport -- a window resize, a browser zoom, Shift+L or
 * Ctrl+Z can all do that while the viewer is up -- and the flag went with it,
 * so the viewer closed under the user's cursor. The viewer is now rendered
 * once at the app root by `VizViewerModal`, which nothing culls, and driven by
 * `tab.vizModalNodeId`. This module is what lets that one host stand in for
 * six cards: keyed by the React Flow node type each card is registered under
 * in `FlowCanvas`'s `nodeTypes`, it builds the modal's props from the node and
 * its streamed output summaries. The cards read their inline plot through the
 * same helpers, so the small plot and the full one cannot disagree about what
 * a tensor means.
 */

/** The per-port output summaries the backend streamed for one node, if it has run. */
export type NodeSummaries = Record<string, OutputSummary> | undefined;

export type VizViewerSpec =
  | { kind: 'heatmap'; props: Omit<HeatmapModalProps, 'isOpen' | 'onClose'> }
  | { kind: 'scatter'; props: Omit<ScatterModalProps, 'isOpen' | 'onClose'> };

export type VizViewerBuilder = (
  node: Node<NodeData>,
  summaries: NodeSummaries,
  runId: string | null,
) => VizViewerSpec;

/** True when `values[0][0]...` is an array `depth` levels down (a tensor of that rank). */
function rankAtLeast(values: unknown, depth: number): boolean {
  let cur: unknown = values;
  for (let d = 0; d < depth; d++) {
    if (!Array.isArray(cur)) return false;
    cur = cur[0];
  }
  return true;
}

/**
 * Attention weights as the heatmap draws them: `[seq, seq]` or `[H, seq, seq]`.
 * A batched `[B, H, seq, seq]` collapses to batch 0.
 */
export function attentionWeights(values: unknown): number[][] | number[][][] | null {
  if (!Array.isArray(values) || values.length === 0) return null;
  if (rankAtLeast(values, 4)) return (values as number[][][][])[0];
  return values as number[][] | number[][][];
}

/** Per-head weights as `[H, seq, seq]`; a batched 4-D tensor collapses to batch 0. */
export function attentionHeads(values: unknown): number[][][] | null {
  if (!Array.isArray(values) || values.length === 0) return null;
  if (rankAtLeast(values, 4)) return (values as number[][][][])[0];
  return values as number[][][];
}

/** Single-head weights as `[seq, seq]`; a `[1, seq, seq]` tensor collapses to its one head. */
export function selfAttentionWeights(values: unknown): number[][] | null {
  if (!Array.isArray(values) || values.length === 0) return null;
  if (rankAtLeast(values, 3)) return (values as number[][][])[0];
  return values as number[][];
}

/** A boolean / 0-1 mask as 0/1 numbers, row by row. */
export function maskMatrix(values: unknown): number[][] | null {
  if (!Array.isArray(values) || values.length === 0) return null;
  return (values as unknown[]).map((row) =>
    Array.isArray(row) ? row.map((x) => (x ? 1 : 0)) : [],
  );
}

/** A `LIST[str]` port's values as strings, or undefined when the port is empty. */
export function labelList(values: unknown): string[] | undefined {
  if (!Array.isArray(values) || values.length === 0) return undefined;
  return values.map((s) => String(s));
}

/** `[N, 2]` coordinates plus an optional label list, as scatter points. */
export function scatterPoints(coords: unknown, labels: unknown): ScatterPoint[] | null {
  if (!Array.isArray(coords) || coords.length === 0) return null;
  return coords.map((row, i) => {
    const r = Array.isArray(row) ? row : [];
    const x = typeof r[0] === 'number' ? r[0] : 0;
    const y = typeof r[1] === 'number' ? r[1] : 0;
    const label =
      Array.isArray(labels) && typeof labels[i] === 'string' ? (labels[i] as string) : undefined;
    return { x, y, label, cluster: i };
  });
}

const titleOf = (kind: string, node: Node<NodeData>) => `${kind} · ${node.data.label ?? node.id}`;

/**
 * React Flow node type (the `nodeTypes` key in `FlowCanvas`) -> its viewer.
 * A node type absent from this table has no full-size viewer, and
 * `VizViewerModal` renders nothing for it.
 */
export const VIZ_VIEWERS: Record<string, VizViewerBuilder> = {
  attentionHeatmapNode: (node, s, runId) => ({
    kind: 'heatmap',
    props: {
      title: titleOf('AttentionHeatmap', node),
      inlineData: attentionWeights(s?.weights?.values),
      rowLabels: labelList(s?.labels?.values),
      colormap: (node.data.params?.colormap as HeatmapColormap | undefined) ?? 'viridis',
      runId,
      nodeId: node.id,
      port: 'weights',
      detectCausalMask: true,
      normalizePerRow: true,
    },
  }),
  attentionMaskNode: (node, s, runId) => ({
    kind: 'heatmap',
    props: {
      title: titleOf('AttentionMask', node),
      inlineData: maskMatrix(s?.mask?.values),
      colormap: 'RdBu',
      detectCausalMask: false,
      runId,
      nodeId: node.id,
      port: 'mask',
      variant: 'mask',
    },
  }),
  eduCrossAttentionNode: (node, s, runId) => ({
    kind: 'heatmap',
    props: {
      title: titleOf('EduCrossAttention', node),
      inlineData: attentionHeads(s?.weights?.values),
      rowLabels: labelList(s?.q_labels?.values),
      colLabels: labelList(s?.k_labels?.values),
      runId,
      nodeId: node.id,
      port: 'weights',
      // Cross-attention is always rectangular and lives outside the causal
      // regime -- don't try to detect a causal pattern.
      detectCausalMask: false,
      normalizePerRow: true,
    },
  }),
  eduMultiHeadAttentionNode: (node, s, runId) => ({
    kind: 'heatmap',
    props: {
      title: titleOf('EduMultiHeadAttention', node),
      inlineData: attentionHeads(s?.weights?.values),
      rowLabels: labelList(s?.labels?.values),
      runId,
      nodeId: node.id,
      port: 'weights',
      detectCausalMask: true,
      normalizePerRow: true,
    },
  }),
  eduSelfAttentionNode: (node, s, runId) => ({
    kind: 'heatmap',
    props: {
      title: titleOf('EduSelfAttention', node),
      inlineData: selfAttentionWeights(s?.weights?.values),
      rowLabels: labelList(s?.labels?.values),
      runId,
      nodeId: node.id,
      port: 'weights',
      detectCausalMask: true,
      normalizePerRow: true,
    },
  }),
  embeddingScatterNode: (node, s, runId) => ({
    kind: 'scatter',
    props: {
      title: titleOf('EmbeddingScatter', node),
      inlinePoints: scatterPoints(s?.points_2d?.values, s?.labels?.values),
      runId,
      nodeId: node.id,
      pointsPort: 'points_2d',
      labelsPort: 'labels',
    },
  }),
};
