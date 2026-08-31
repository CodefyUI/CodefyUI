/**
 * Compile-time guard: the published plugin contract (`./contract`) must stay in
 * sync with the host's real implementation types. If the host changes a
 * contract-relevant type without updating `contract.ts`, one of the assertions
 * below resolves to something other than `true` and `tsc -b` fails — caught by
 * the Frontend Build Check before any plugin sees stale autocomplete.
 *
 * This file emits no runtime code; it exists purely so the type checker visits
 * these assertions.
 */
import type { GraphOp as HGraphOp, OpResult as HOpResult } from './ops';
import type {
  ApplyResult as HApplyResult,
  CodefyUIPluginAPI as HApi,
  GraphView as HGraphView,
  GraphViewLevel as HGraphViewLevel,
  RunListOptions as HRunListOptions,
  WorkspaceApplyRequest as HWorkspaceApplyRequest,
  WorkspaceApplyResult as HWorkspaceApplyResult,
  WorkspaceConflict as HWorkspaceConflict,
  WorkspaceEvent as HWorkspaceEvent,
  WorkspaceGraphInput as HWorkspaceGraphInput,
  WorkspaceOpenEntry as HWorkspaceOpenEntry,
  WorkspaceOpenResult as HWorkspaceOpenResult,
  WorkspaceSnapshot as HWorkspaceSnapshot,
  WorkspaceSource as HWorkspaceSource,
  WorkspaceTabInfo as HWorkspaceTabInfo,
} from './api';
import type {
  NodeDefinition as HNodeDef,
  ParamDefinition as HParamDef,
  PortDefinition as HPortDef,
} from '../types';
import type { PluginNodeRenderer as HRenderer } from './nodeRenderers';
import type {
  PluginPanelDock as HPanelDock,
  PluginPanelOptions as HPanelOptions,
} from './panels';
import type { PluginToolbarButtonOptions as HToolbarOptions } from './toolbarButtons';
import type {
  ExecutionEvent as HExecutionEvent,
  ExecutionFinishStatus as HFinishStatus,
} from './executionEvents';
import type {
  RunInfo as HRunInfo,
  RunListPage as HRunListPage,
  RunMetricPoint as HRunMetricPoint,
  RunMetrics as HRunMetrics,
  RunStatus as HRunStatus,
  RunSummary as HRunSummary,
} from '../api/rest';
import type {
  ApplyResult as CApplyResult,
  CodefyUIPluginAPI as CApi,
  ExecutionEvent as CExecutionEvent,
  ExecutionFinishStatus as CFinishStatus,
  GraphOp as CGraphOp,
  GraphView as CGraphView,
  GraphViewLevel as CGraphViewLevel,
  NodeDefinition as CNodeDef,
  OpResult as COpResult,
  ParamDefinition as CParamDef,
  PluginNodeRenderer as CRenderer,
  PluginPanelDock as CPanelDock,
  PluginPanelOptions as CPanelOptions,
  PluginToolbarButtonOptions as CToolbarOptions,
  PortDefinition as CPortDef,
  RunInfo as CRunInfo,
  RunListOptions as CRunListOptions,
  RunListPage as CRunListPage,
  RunMetricPoint as CRunMetricPoint,
  RunMetrics as CRunMetrics,
  RunStatus as CRunStatus,
  RunSummary as CRunSummary,
  WorkspaceApplyRequest as CWorkspaceApplyRequest,
  WorkspaceApplyResult as CWorkspaceApplyResult,
  WorkspaceConflict as CWorkspaceConflict,
  WorkspaceEvent as CWorkspaceEvent,
  WorkspaceGraphInput as CWorkspaceGraphInput,
  WorkspaceOpenEntry as CWorkspaceOpenEntry,
  WorkspaceOpenResult as CWorkspaceOpenResult,
  WorkspaceSnapshot as CWorkspaceSnapshot,
  WorkspaceSource as CWorkspaceSource,
  WorkspaceTabInfo as CWorkspaceTabInfo,
} from './contract';

type Extends<A, B> = A extends B ? true : false;
/** True only when A and B are mutually assignable (structurally equal). */
type Mutual<A, B> = Extends<A, B> extends true ? Extends<B, A> : false;
/** Compile error unless the argument resolves to exactly `true`. */
type Expect<T extends true> = T;
/**
 * `Omit`, applied to each member of a union SEPARATELY.
 *
 * A conditional type over a naked type parameter distributes, so every branch
 * is omitted from on its own and the union is put back together with all of
 * them still in it. Plain `Omit` cannot do that on a union: `keyof (A | B)` is
 * the INTERSECTION of the two key sets, so `Omit<A | B, K>` keeps only the
 * keys A and B share — `never` for a union whose branches have nothing in
 * common, which collapses the whole thing to `{}` and compares nothing at all.
 */
type DistributiveOmit<T, K extends PropertyKey> = T extends unknown
  ? Omit<T, K>
  : never;

// ── data contracts: must match the host structurally, both directions ──
// (ParamDefinition.default is `any` on the host and `unknown` in the contract;
// the two are mutually assignable, so this still catches every real drift —
// an added/removed/renamed field or a changed param_type/op union member.)
export type _GraphOp = Expect<Mutual<HGraphOp, CGraphOp>>;
export type _OpResult = Expect<Mutual<HOpResult, COpResult>>;
export type _ApplyResult = Expect<Mutual<HApplyResult, CApplyResult>>;
export type _NodeDef = Expect<Mutual<HNodeDef, CNodeDef>>;
export type _PortDef = Expect<Mutual<HPortDef, CPortDef>>;
export type _ParamDef = Expect<Mutual<HParamDef, CParamDef>>;
export type _Renderer = Expect<Mutual<HRenderer, CRenderer>>;

// ── apiVersion 3 additions ────────────────────────────────────────────────
export type _PanelDock = Expect<Mutual<HPanelDock, CPanelDock>>;
export type _PanelOptions = Expect<Mutual<HPanelOptions, CPanelOptions>>;
export type _ToolbarOptions = Expect<Mutual<HToolbarOptions, CToolbarOptions>>;
export type _FinishStatus = Expect<Mutual<HFinishStatus, CFinishStatus>>;
export type _ExecutionEvent = Expect<Mutual<HExecutionEvent, CExecutionEvent>>;
export type _RunStatus = Expect<Mutual<HRunStatus, CRunStatus>>;
export type _RunSummary = Expect<Mutual<HRunSummary, CRunSummary>>;
export type _RunInfo = Expect<Mutual<HRunInfo, CRunInfo>>;
export type _RunListPage = Expect<Mutual<HRunListPage, CRunListPage>>;
export type _RunListOptions = Expect<Mutual<HRunListOptions, CRunListOptions>>;
export type _RunMetricPoint = Expect<Mutual<HRunMetricPoint, CRunMetricPoint>>;
export type _RunMetrics = Expect<Mutual<HRunMetrics, CRunMetrics>>;

// ── apiVersion 4 additions (#200 item 7) ──────────────────────────────────
export type _GraphViewLevel = Expect<Mutual<HGraphViewLevel, CGraphViewLevel>>;
export type _GraphView = Expect<Mutual<HGraphView, CGraphView>>;

// ── apiVersion 5 additions (#341, #342) ───────────────────────────────────
export type _WorkspaceSource = Expect<Mutual<HWorkspaceSource, CWorkspaceSource>>;
export type _WorkspaceGraphInput = Expect<Mutual<HWorkspaceGraphInput, CWorkspaceGraphInput>>;
export type _WorkspaceOpenResult = Expect<Mutual<HWorkspaceOpenResult, CWorkspaceOpenResult>>;
export type _WorkspaceTabInfo = Expect<Mutual<HWorkspaceTabInfo, CWorkspaceTabInfo>>;
export type _WorkspaceApplyRequest = Expect<Mutual<HWorkspaceApplyRequest, CWorkspaceApplyRequest>>;
export type _WorkspaceApplyResult = Expect<Mutual<HWorkspaceApplyResult, CWorkspaceApplyResult>>;
export type _WorkspaceConflict = Expect<Mutual<HWorkspaceConflict, CWorkspaceConflict>>;
export type _WorkspaceEvent = Expect<Mutual<HWorkspaceEvent, CWorkspaceEvent>>;
// `WorkspaceOpenEntry.graph` is the ONE field deliberately not compared: the
// host takes the loose `WorkspaceGraphInput` a plugin can actually build,
// while the contract shows a plugin the `SerializedGraph` it read. `Omit`
// keeps the rest of the entry honest — which is where a renamed or dropped
// field would otherwise hide. Same carve-out `SerializedGraph` itself has had
// since v1, and for the same reason.
export type _WorkspaceOpenEntryMeta = Expect<
  Mutual<Omit<HWorkspaceOpenEntry, 'graph'>, Omit<CWorkspaceOpenEntry, 'graph'>>
>;
// The same carve-out, one level in — and applied to BOTH branches of the
// union, which is the part `Exclude` could not do (#398). Nothing else
// reaches either branch: `_WorkspaceKeys` compares the key NAMES of
// `workspace`, never the return type of `snapshot`. `Exclude<..., { graph:
// unknown }>` closed the `{ error: 'unknown_tab' }` half by isolating it, but
// it isolates by DROPPING the graph-bearing half, so a host-only field added
// next to `graph` was compared against nothing at all. Omitting `graph` from
// each branch separately leaves both of them in the comparison, member for
// member. (`graph` itself stays out for the reason above: the host's
// `SerializedGraph` and the contract's are the one deliberate divergence.)
export type _WorkspaceSnapshotMeta = Expect<
  Mutual<
    DistributiveOmit<HWorkspaceSnapshot, 'graph'>,
    DistributiveOmit<CWorkspaceSnapshot, 'graph'>
  >
>;
export type _WorkspaceKeys = Expect<
  Mutual<keyof HApi['workspace'], keyof CApi['workspace']>
>;

// ── API surface: same top-level sections; apiVersion is intentionally widened
// from the host's literal `5` to `number` so plugins can defensively check it. ──
export type _ApiKeys = Expect<Mutual<keyof HApi, keyof CApi>>;
export type _ApiVersion = Expect<Extends<HApi['apiVersion'], CApi['apiVersion']>>;
export type _UiKeys = Expect<Mutual<keyof HApi['ui'], keyof CApi['ui']>>;
// `graph` was the one section with no key check of its own, which is how a
// member could have been published on the host and never reach the contract
// plugins read (#200 item 7 added the first new member since).
export type _GraphKeys = Expect<Mutual<keyof HApi['graph'], keyof CApi['graph']>>;
export type _EventsKeys = Expect<Mutual<keyof HApi['events'], keyof CApi['events']>>;
export type _RunsKeys = Expect<Mutual<keyof HApi['runs'], keyof CApi['runs']>>;
