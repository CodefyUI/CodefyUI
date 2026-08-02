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
  RunListOptions as HRunListOptions,
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
} from './contract';

type Extends<A, B> = A extends B ? true : false;
/** True only when A and B are mutually assignable (structurally equal). */
type Mutual<A, B> = Extends<A, B> extends true ? Extends<B, A> : false;
/** Compile error unless the argument resolves to exactly `true`. */
type Expect<T extends true> = T;

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

// ── API surface: same top-level sections; apiVersion is intentionally widened
// from the host's literal `3` to `number` so plugins can defensively check it. ──
export type _ApiKeys = Expect<Mutual<keyof HApi, keyof CApi>>;
export type _ApiVersion = Expect<Extends<HApi['apiVersion'], CApi['apiVersion']>>;
export type _UiKeys = Expect<Mutual<keyof HApi['ui'], keyof CApi['ui']>>;
export type _EventsKeys = Expect<Mutual<keyof HApi['events'], keyof CApi['events']>>;
export type _RunsKeys = Expect<Mutual<keyof HApi['runs'], keyof CApi['runs']>>;
