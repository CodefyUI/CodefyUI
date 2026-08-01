import type { ReactNode } from 'react';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, OutputSummary } from '../../types';
import type { TranslationKey } from '../../i18n';
import { StepTraceView } from '../InspectorPanel/StepTraceView';
import { BackwardView } from '../InspectorPanel/BackwardView';
import { CapturesTab } from './CapturesTab';
import { StatsTab } from './StatsTab';
import { DocsTab } from './DocsTab';

/**
 * Everything a Node Detail Modal tab is handed. Treat it as an additive
 * contract: fields may be appended for later tabs, but an existing field never
 * changes meaning, so a tab written today keeps working.
 */
export interface NodeDetailTabContext {
  /** Canvas id of the node the modal is showing. */
  nodeId: string;
  /** The live node, re-read from the store on every render. */
  node: Node<NodeData>;
  /** Run whose captures this tab should read, or null when nothing has run. */
  runId: string | null;
  /** Every node on the tab — needed to name the upstream side of an input. */
  nodes: Node<NodeData>[];
  /** Every edge on the tab — needed to resolve which inputs are connected. */
  edges: Edge[];
  /** Whether the tab's "Record outputs" switch is currently on. */
  recordOutputs: boolean;
  /** Per-node, per-port summaries streamed during the last run. */
  outputSummaries: Record<string, Record<string, OutputSummary>>;
}

export interface NodeDetailTabSpec {
  /** Stable identifier; also the value stored as the modal's active tab. */
  id: string;
  /** i18n key for the tab button label. */
  labelKey: TranslationKey;
  /**
   * Sort key. Built-ins are spaced by ten so a later tab can be slotted
   * between two of them without renumbering anything.
   */
  order: number;
  /**
   * Hide the tab entirely when this returns false — used by Steps / Backward,
   * which have nothing to show until a run exists. Omit for always-on tabs.
   */
  isEnabled?: (ctx: NodeDetailTabContext) => boolean;
  /** The tab's body. Return an element; hooks belong inside that component. */
  render: (ctx: NodeDetailTabContext) => ReactNode;
}

/**
 * The tabs this issue ships. Later issues extend the modal through
 * {@link registerNodeDetailTab} rather than editing this array — see the
 * module docs on that function.
 */
export const BUILTIN_NODE_DETAIL_TABS: readonly NodeDetailTabSpec[] = [
  {
    id: 'inputs',
    labelKey: 'nodeDetail.tabs.inputs',
    order: 10,
    render: (ctx) => <CapturesTab kind="input" ctx={ctx} />,
  },
  {
    id: 'outputs',
    labelKey: 'nodeDetail.tabs.outputs',
    order: 20,
    render: (ctx) => <CapturesTab kind="output" ctx={ctx} />,
  },
  {
    id: 'steps',
    labelKey: 'nodeDetail.tabs.steps',
    order: 30,
    isEnabled: (ctx) => ctx.runId !== null,
    render: (ctx) => (
      <StepTraceView
        key={`${ctx.runId}:${ctx.nodeId}`}
        runId={ctx.runId as string}
        nodeId={ctx.nodeId}
      />
    ),
  },
  {
    id: 'backward',
    labelKey: 'nodeDetail.tabs.backward',
    order: 40,
    isEnabled: (ctx) => ctx.runId !== null,
    render: (ctx) => (
      <BackwardView
        key={`${ctx.runId}:${ctx.nodeId}`}
        runId={ctx.runId as string}
        nodeId={ctx.nodeId}
      />
    ),
  },
  {
    id: 'stats',
    labelKey: 'nodeDetail.tabs.stats',
    order: 50,
    render: () => <StatsTab />,
  },
  {
    id: 'docs',
    labelKey: 'nodeDetail.tabs.docs',
    order: 60,
    render: (ctx) => <DocsTab node={ctx.node} />,
  },
];

/** Tabs contributed after module load, keyed by id. */
const extraTabs = new Map<string, NodeDetailTabSpec>();

/**
 * Add a tab to the Node Detail Modal, or replace a built-in one.
 *
 * Registering an id that already exists as a built-in REPLACES it while
 * keeping its slot — which is how P3-14 turns the placeholder `stats` tab into
 * a real one without touching this file, and how a code-editor tab lands next
 * to the others. Registering a new id inserts it by `order`.
 *
 * Call at module scope from the feature's own module; the modal reads the
 * registry on every render, so a tab registered before the modal opens is
 * always visible.
 */
export function registerNodeDetailTab(spec: NodeDetailTabSpec): void {
  extraTabs.set(spec.id, spec);
}

/** Remove a previously registered tab. A built-in id reverts to the built-in. */
export function unregisterNodeDetailTab(id: string): void {
  extraTabs.delete(id);
}

/**
 * The tabs to show for one node, in display order, with anything disabled for
 * this context already filtered out.
 */
export function getNodeDetailTabs(ctx: NodeDetailTabContext): NodeDetailTabSpec[] {
  const merged = new Map<string, NodeDetailTabSpec>();
  for (const spec of BUILTIN_NODE_DETAIL_TABS) merged.set(spec.id, spec);
  for (const [id, spec] of extraTabs) merged.set(id, spec);
  return [...merged.values()]
    .filter((spec) => spec.isEnabled === undefined || spec.isEnabled(ctx))
    .sort((a, b) => a.order - b.order);
}
