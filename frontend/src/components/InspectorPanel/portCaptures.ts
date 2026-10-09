import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchOutput,
  listRunOutputs,
  NoValueError,
  RunDataExpiredError,
  PayloadTooLargeError,
} from '../../api/executionOutputs';
import { ACTIVE_RUN_STATUSES, getRun } from '../../api/rest';
import type { ExecutionStatus, NodeData, OutputData } from '../../types';
import type { Edge, Node } from '@xyflow/react';
import {
  useTabStore,
  type LogEntry,
  type LogImagePayload,
  type LogVideoPayload,
} from '../../store/tabStore';
import { runNodePrefix, subgraphIdOf } from '../../utils/subgraph';
import { useI18n, type TranslationKey } from '../../i18n';
import { keyOf, type FetchMap, type PortTarget } from './PortGroup';

/**
 * The capture-reading half of the Inspector, factored out so the side panel
 * and the Node Detail Modal (#127) read the *same* run data through the *same*
 * code path. Anything that resolves which ports belong to a node, or turns
 * those ports into `/api/execution/outputs` results, belongs here — never
 * copied into a second component.
 */

/** The most values a preview of a too-large tensor carries. */
export const PREVIEW_MAX_ELEMENTS = 65536;

/**
 * Fetch one port, falling back to a bounded preview if the server refuses the
 * full payload. The server chooses the preview's slice from the captured
 * shape (#640), so an image batch, a long vector and a wide matrix all come
 * back as their leading part, with `slice` and `truncated` saying so. Without
 * the retry a large activation shows an error instead.
 */
export async function fetchPortWithSliceFallback(
  runId: string,
  nodeId: string,
  port: string,
): Promise<OutputData> {
  try {
    return await fetchOutput(runId, nodeId, port);
  } catch (e) {
    if (e instanceof PayloadTooLargeError) {
      return await fetchOutput(runId, nodeId, port, {
        preview: true,
        maxElements: PREVIEW_MAX_ELEMENTS,
      });
    }
    throw e;
  }
}

/* ── What a finished run recorded, node by node ─────────────────────────────
 *
 * A node's captures are written when it completes, so a finished run's port
 * list names every node the run has anything for. Reads use it to skip
 * requests that could only be answered 404 -- which every view reads as "Run
 * data expired" -- and say what is true instead, but only what is certain:
 *
 * - "Not in the last run" only from a port list of a finished run that
 *   recorded outputs, for a node that declares outputs and is not in it.
 * - "Failed in the last run" from the card's own status, at the top level
 *   only: inside an open block the cards never get a run status.
 * - The Record node outputs hint when the run's record says it was off.
 *
 * Anything else is asked for as before, and a 404 still reads as expired.
 *
 * The run record is read first. It says whether the run is over -- the run
 * service marks it finished after its last capture is written, so a list read
 * after that is whole -- and how it was run. A list read while the run still
 * goes is partial (a tab can name a live run without running itself: Watch in
 * the Runs panel, a reload mid-run): a node in it has finished and is read,
 * and any other waits for the run to end, as in the tab's own run. The
 * service also sends the run's last event before it marks the run finished,
 * so a record that still says "running" when the tab's run has just ended is
 * asked again a few times before it is taken at its word.
 */

/** Node ids, as the run names them, that a finished run recorded anything for. */
export type RunIndex = ReadonlySet<string>;

/** What the server says about a run, as far as the reads below need to know. */
export type RunCaptures =
  /**
   * Still going, though the tab is not running it: Watch in the Runs panel,
   * or a reload mid-run. `nodes` is what its list holds SO FAR (null while
   * the run is queued, or before it stores anything) -- each node in it has
   * finished -- and nothing about the rest is final.
   */
  | { state: 'live'; recorded: boolean; gradients: boolean; nodes: RunIndex | null }
  /** No record, or it could not be read. */
  | { state: 'unknown' }
  | {
      state: 'finished';
      /** Record node outputs was on for the run. */
      recorded: boolean;
      /** Capture gradients was on for the run. */
      gradients: boolean;
      /** Nodes with anything stored; null when there is no list (404, unreadable, or never read). */
      nodes: RunIndex | null;
      /** Whether the list holds a forward value, not only gradients. */
      forward: boolean;
    };

/** Answers kept; the server itself keeps only the last few runs' captures. */
const RUN_CAPTURES_KEPT = 8;
const runCaptures = new Map<string, Promise<RunCaptures>>();

/** How often, and how far apart, a record still saying "running" is asked again. */
let recordRetries = 3;
let recordRetryMs = 200;

/** A port a node put out -- not a gradient, a step, or other bookkeeping. */
function isForwardPort(port: string): boolean {
  return !port.startsWith('__') && !port.endsWith('__grad') && !port.endsWith('__grad__meta');
}

/** One read; `settled` when the answer cannot change any more. */
async function readRunCaptures(runId: string): Promise<{ captures: RunCaptures; settled: boolean }> {
  let run;
  let live = false;
  for (let attempt = 0; ; attempt += 1) {
    try {
      run = await getRun(runId);
    } catch {
      return { captures: { state: 'unknown' }, settled: false };
    }
    if (!run) return { captures: { state: 'unknown' }, settled: true };
    if (!ACTIVE_RUN_STATUSES.includes(run.status)) break;
    // The run's last event can end the tab's run a moment before the row
    // says so; past that, the run really is still going.
    if (attempt >= recordRetries) {
      live = true;
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, recordRetryMs));
  }
  const recorded = run.options?.record_outputs !== false;
  const gradients = run.options?.backward_mode !== false;
  // The list is not asked for when it could not change a note: recording was
  // off (it holds gradients at most), or the run is still queued (nothing in
  // it has run, so the answer could only be a 404).
  let nodes: RunIndex | null = null;
  let forward = false;
  let listSettled = true;
  if (recorded && run.status !== 'queued') {
    try {
      const refs = await listRunOutputs(runId);
      nodes = new Set(refs.map((ref) => ref.node_id));
      forward = refs.some((ref) => isForwardPort(ref.port));
    } catch (e) {
      // 404: the server holds nothing for the run. That is final too, but
      // says nothing about any one node: the run stored nothing, or its
      // captures expired.
      listSettled = e instanceof RunDataExpiredError;
    }
  }
  if (live) return { captures: { state: 'live', recorded, gradients, nodes }, settled: false };
  return {
    captures: { state: 'finished', recorded, gradients, nodes, forward },
    settled: listSettled,
  };
}

/* ── The end of a run the tab is not running ────────────────────────────────
 *
 * A tab that watches a run (Watch in the Runs panel), or was reloaded while
 * one went on, may never hear that run end: the tab's own run status is what
 * the views follow, and it can stay idle. While a view waits on such a run,
 * its record is polled, and the view reads again once the run is over.
 */

let runEndPollMs = 2000;

interface RunEndWatch {
  listeners: Set<() => void>;
  timer: ReturnType<typeof setTimeout> | null;
}

const runEndWatches = new Map<string, RunEndWatch>();

function pollRunEnd(runId: string, watch: RunEndWatch): void {
  watch.timer = setTimeout(() => {
    void (async () => {
      let over = false;
      try {
        const run = await getRun(runId);
        over = !run || !ACTIVE_RUN_STATUSES.includes(run.status);
      } catch {
        // Server unreachable: keep waiting while anyone is.
      }
      if (runEndWatches.get(runId) !== watch) return;
      if (!over) {
        pollRunEnd(runId, watch);
        return;
      }
      runEndWatches.delete(runId);
      runCaptures.delete(runId);
      for (const listener of [...watch.listeners]) listener();
    })();
  }, runEndPollMs);
}

/**
 * Call `onEnd` once run `runId` is no longer going, polling its record while
 * anyone listens. Returns the way to stop listening.
 */
export function onRunEnd(runId: string, onEnd: () => void): () => void {
  let watch = runEndWatches.get(runId);
  if (!watch) {
    watch = { listeners: new Set(), timer: null };
    runEndWatches.set(runId, watch);
    pollRunEnd(runId, watch);
  }
  const current = watch;
  current.listeners.add(onEnd);
  return () => {
    current.listeners.delete(onEnd);
    if (current.listeners.size === 0 && runEndWatches.get(runId) === current) {
      if (current.timer) clearTimeout(current.timer);
      runEndWatches.delete(runId);
    }
  };
}

/** Whether a note means "the run is still going": read again once it ends. */
export function isRunStillGoingNote(note: TranslationKey | 'none' | null): boolean {
  return note === 'inspector.nodePending' || note === 'inspector.runRunning';
}

/** Tests only: how often a run the tab is not running is polled for its end. */
export function _setRunEndPollForTests(ms: number): void {
  runEndPollMs = ms;
}

/**
 * What the server says about run `runId`, read once it is final and kept. A
 * read that is not settled is forgotten, so the next one asks again.
 */
export function loadRunCaptures(runId: string): Promise<RunCaptures> {
  let captures = runCaptures.get(runId);
  if (!captures) {
    const read = readRunCaptures(runId);
    const kept = read.then((r) => r.captures);
    captures = kept;
    runCaptures.set(runId, kept);
    void read.then((r) => {
      if (!r.settled && runCaptures.get(runId) === kept) runCaptures.delete(runId);
    });
    if (runCaptures.size > RUN_CAPTURES_KEPT) {
      runCaptures.delete(runCaptures.keys().next().value!);
    }
  }
  return captures;
}

/** Tests only: forget every answer read so far, and stop every poll. */
export function _resetRunIndexesForTests(): void {
  runCaptures.clear();
  for (const watch of runEndWatches.values()) {
    if (watch.timer) clearTimeout(watch.timer);
  }
  runEndWatches.clear();
}

/** Tests only: how a record still saying "running" is asked again. */
export function _setRunRecordRetryForTests(retries: number, ms: number): void {
  recordRetries = retries;
  recordRetryMs = ms;
}

/**
 * Who a read is for. `status` is the card's last run status, given at the TOP
 * level only -- inside an open block the cards never get one. `hasOutputs` is
 * false for a node that declares no data output (Start): the run's list has
 * nothing for it even when it ran.
 */
export interface ReadSubject {
  runNodeId: string;
  status?: ExecutionStatus;
  hasOutputs?: boolean;
}

/** A positive list of a finished, recorded run, without this node: certain. */
function absentFromRecordedList(c: RunCaptures, subject: ReadSubject): boolean {
  return (
    c.state === 'finished' &&
    c.recorded &&
    c.nodes !== null &&
    c.forward &&
    (subject.hasOutputs ?? true) &&
    !c.nodes.has(subject.runNodeId)
  );
}

/**
 * What a read of a node's VALUES (ports, statistics, steps) says instead of
 * asking the server, or null to ask as before -- and let a 404 read as
 * expired.
 */
export async function missingFromRunNote(
  runId: string,
  subject: ReadSubject,
): Promise<TranslationKey | null> {
  const c = await loadRunCaptures(runId);
  // Recording was off: no value exists, though the list may name the node
  // for its gradients. The views redraw this line as the setting changes
  // (see `followRecordingSetting`).
  if (c.state !== 'unknown' && !c.recorded) {
    if (c.state === 'finished' && subject.status === 'error') return 'inspector.capture.failedInRun';
    return recordingOffNote(recordOutputsNow());
  }
  if (c.state === 'live') {
    // A node in the list so far has finished; the rest wait for the run, as
    // they do in the tab's own run -- a request now could only 404. (No card
    // status here: the tab has heard nothing of this run.)
    if (c.nodes?.has(subject.runNodeId) || !(subject.hasOutputs ?? true)) return null;
    return 'inspector.nodePending';
  }
  if (subject.status === 'error') return 'inspector.capture.failedInRun';
  if (c.state === 'finished' && c.nodes?.has(subject.runNodeId)) return null;
  if (!absentFromRecordedList(c, subject)) return null;
  // A bypassed node never runs; it is in the list only for a port something
  // passed through (#559). Nothing did, which is not the same as expiry.
  if (subject.status === 'bypassed') return 'inspector.capture.bypassedInRun';
  // A node a Switch's param did not pick never ran (#656).
  if (subject.status === 'unselected') return 'inspector.capture.unselectedInRun';
  return 'inspector.capture.notInRun';
}

/**
 * The same for a node's GRADIENTS, which Record node outputs does not govern.
 * `'none'` when the run's record says Capture gradients was off: there are
 * none, and nothing to ask for.
 */
export async function missingGradientsNote(
  runId: string,
  subject: ReadSubject,
): Promise<TranslationKey | 'none' | null> {
  const c = await loadRunCaptures(runId);
  // Gradients are written after the whole forward pass: a run still going
  // has none yet, as the tab's own run says.
  if (c.state === 'live') return c.gradients ? 'inspector.runRunning' : 'none';
  if (subject.status === 'error') return 'inspector.capture.failedInRun';
  if (c.state === 'finished' && !c.gradients) return 'none';
  if (c.state === 'finished' && c.nodes?.has(subject.runNodeId)) return null;
  return absentFromRecordedList(c, subject) ? 'inspector.capture.notInRun' : null;
}

/** The last run status of canvas node `nodeId` on the active tab's open level. */
export function canvasNodeStatus(nodeId: string): ExecutionStatus | undefined {
  const { tabs, activeTabId } = useTabStore.getState();
  return tabs
    .find((t) => t.id === activeTabId)
    ?.nodes.find((n) => n.id === nodeId)?.data.executionStatus;
}

/**
 * Whether canvas node `nodeId` is a block or preset card (#559). A card never
 * runs: what it recorded -- steps, gradients -- is its inner nodes'.
 */
export function canvasNodeIsContainer(nodeId: string): boolean {
  const { tabs, activeTabId } = useTabStore.getState();
  const type = tabs.find((t) => t.id === activeTabId)?.nodes.find((n) => n.id === nodeId)?.data.type;
  return subgraphIdOf(type) !== null || (typeof type === 'string' && type.startsWith('preset:'));
}

/**
 * How a card's Steps and Backward name the node inside it that recorded an
 * entry (#559): its run id after the card's, as validation and the
 * Inspector name inner nodes (`nest/mul` in block `blk`, `mul` in preset
 * card `card`).
 */
export function innerNodeLabel(cardRunId: string, producer: string): string {
  for (const separator of ['/', '__']) {
    if (producer.startsWith(cardRunId + separator)) {
      return producer.slice(cardRunId.length + separator.length);
    }
  }
  return producer;
}

/** Whether canvas node `nodeId` declares a data output (Start does not). */
export function canvasNodeHasOutputs(nodeId: string): boolean {
  const { tabs, activeTabId } = useTabStore.getState();
  const node = tabs.find((t) => t.id === activeTabId)?.nodes.find((n) => n.id === nodeId);
  return (node?.data.definition?.outputs ?? []).some((o) => o.data_type !== 'TRIGGER');
}

/**
 * Whether the active tab's run is still going, read at the moment a request
 * is decided rather than subscribed to: nothing re-renders for it.
 */
export function runInProgressNow(): boolean {
  const { tabs, activeTabId } = useTabStore.getState();
  return tabs.find((t) => t.id === activeTabId)?.status === 'running';
}

/** The active tab's Record node outputs setting, read the same way. */
function recordOutputsNow(): boolean {
  const { tabs, activeTabId } = useTabStore.getState();
  return tabs.find((t) => t.id === activeTabId)?.recordOutputs ?? true;
}

/** The active tab's Record node outputs setting, subscribed: the views redraw when it is switched. */
export function useRecordOutputs(): boolean {
  return useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)?.recordOutputs ?? true);
}

/**
 * What a run made with Record node outputs off says about a node's values:
 * turn the setting on while it is off, or run the graph once it is on.
 */
function recordingOffNote(recordOutputs: boolean): TranslationKey {
  return recordOutputs ? 'inspector.capture.runHint' : 'inspector.empty.notRunHint';
}

/**
 * `key`, with that line swapped for the one the setting calls for NOW. Applied
 * as a view draws, like the phase notes, so the line follows the Settings
 * switch with the node still selected, without asking anything again.
 */
export function followRecordingSetting<K extends TranslationKey | null | undefined>(
  key: K,
  recordOutputs: boolean,
): K | TranslationKey {
  return key === 'inspector.capture.runHint' || key === 'inspector.empty.notRunHint'
    ? recordingOffNote(recordOutputs)
    : key;
}

/** {@link followRecordingSetting} over a map of results; the same map when no line changes. */
export function withRecordingSetting<T extends { noteKey?: TranslationKey | null }>(
  results: Record<string, T>,
  recordOutputs: boolean,
): Record<string, T> {
  let out = results;
  for (const [key, entry] of Object.entries(results)) {
    const noteKey = followRecordingSetting(entry.noteKey, recordOutputs);
    if (noteKey === entry.noteKey) continue;
    if (out === results) out = { ...results };
    out[key] = { ...entry, noteKey };
  }
  return out;
}

/* ── The id a run gave a canvas node (#621) ─────────────────────────────────
 *
 * The engine flattens a block before it runs: a node inside one runs, and is
 * captured, as `<instance>/<inner>` -- `<outer>/<inner instance>/<node>` one
 * level further in. An open block shows the definition's nodes under their
 * own ids, so every read from inside one puts the entered instances in front
 * of the canvas id. Two copies of one block hold the same inner ids, which is
 * why a read's result is kept under the run's id and never under the canvas
 * id alone.
 */

// What goes in front of a canvas id on the open level. Kept beside the block
// helpers, where the store paints a run's inner statuses with it too (#559).
export { runNodePrefix };

/**
 * {@link runNodePrefix} for the active tab. The selector returns the string,
 * not the stack, so the caller re-renders only when the prefix changes.
 */
export function useRunNodePrefix(): string {
  return useTabStore((s) =>
    runNodePrefix(s.tabs.find((t) => t.id === s.activeTabId)?.subgraphStack),
  );
}

/** The id the last run gave canvas node `canvasId`, on the level on screen. */
export function useRunNodeId(canvasId: string): string {
  return useRunNodePrefix() + canvasId;
}

/** `ports` with every node id as the run knows it; the same array at the top level. */
export function toRunPorts(ports: readonly PortTarget[], prefix: string): readonly PortTarget[] {
  return prefix ? ports.map((p) => ({ ...p, nodeId: prefix + p.nodeId })) : ports;
}

/**
 * A map kept under run ids (see {@link toRunPorts}), keyed by canvas id again
 * for the ports on screen -- what every view reads by. The map itself at the
 * top level, where the two ids agree.
 */
export function fromRunKeys<T>(
  byRunKey: Record<string, T>,
  ports: readonly PortTarget[],
  prefix: string,
): Record<string, T> {
  if (!prefix) return byRunKey;
  const out: Record<string, T> = {};
  for (const p of ports) {
    const entry = byRunKey[keyOf(prefix + p.nodeId, p.port)];
    if (entry !== undefined) out[keyOf(p.nodeId, p.port)] = entry;
  }
  return out;
}

/** Look up a source port's declared data type from its node definition. */
export function portDataType(
  nodes: readonly {
    id: string;
    data: { definition?: { outputs?: { name: string; data_type: string }[] } };
  }[],
  nodeId: string,
  port: string,
): string | undefined {
  const n = nodes.find((x) => x.id === nodeId);
  return n?.data.definition?.outputs?.find((o) => o.name === port)?.data_type;
}

/**
 * The upstream (source node, source port) pairs feeding `nodeId`. Trigger
 * edges are control-flow markers with no value behind them, and an edge
 * without a `sourceHandle` names no port, so both are skipped.
 */
export function resolveInputSources(
  nodeId: string,
  edges: readonly Pick<Edge, 'source' | 'target' | 'sourceHandle' | 'type' | 'data'>[],
): PortTarget[] {
  const result: PortTarget[] = [];
  for (const e of edges) {
    if (e.target !== nodeId) continue;
    const isTrigger =
      e.type === 'triggerEdge' || (e.data as { type?: string } | undefined)?.type === 'trigger';
    if (isTrigger) continue;
    if (!e.sourceHandle) continue;
    result.push({ nodeId: e.source, port: e.sourceHandle });
  }
  return result;
}

/** Stable empty reference — a fresh `[]` from a zustand selector re-renders forever. */
const NO_LOGS: LogEntry[] = [];

/** What a media port produced, ready to render. */
export type PortMedia =
  | { kind: 'image'; image: LogImagePayload }
  | { kind: 'video'; video: LogVideoPayload };

export type PortMediaMap = Record<string, PortMedia>;

/**
 * What each media port produced last run, keyed by {@link keyOf}.
 *
 * The capture fetch cannot be the source for either kind. A port declaring
 * `media=MEDIA_IMAGE` carries a base64 PNG and `/api/execution/outputs`
 * truncates every string it serves at 4000 chars — two orders of magnitude
 * under a plot — so what it returns for such a port is a fragment that decodes
 * to nothing. A port declaring `media=MEDIA_VIDEO` carries a reference dict,
 * which the capture path can only describe as a `repr`. Both ride the
 * `node_status` stream instead, which is how the results panel has always
 * drawn them, and the entry names the port it came from (see
 * `build_node_output_entries`). Reading the log rather than the capture also
 * means they still show with "Record outputs" off, the same property the chart
 * block relies on.
 *
 * Later entries win, so a re-run replaces the previous run's media rather than
 * stacking behind it.
 */
export function usePortMedia(): PortMediaMap {
  const logs = useTabStore(
    (s) => s.tabs.find((t) => t.id === s.activeTabId)?.logs ?? NO_LOGS,
  );
  return useMemo(() => {
    const out: PortMediaMap = {};
    for (const entry of logs) {
      if (!entry.nodeId) continue;
      // An entry with no `port` predates the named-port contract (or came
      // from the legacy flat image field): it cannot be attributed to a row,
      // and guessing would put one node's media on another node's port.
      if (entry.kind === 'image' && entry.image?.data && entry.image.port) {
        out[keyOf(entry.nodeId, entry.image.port)] = { kind: 'image', image: entry.image };
      } else if (entry.kind === 'video' && entry.video?.url && entry.video.port) {
        out[keyOf(entry.nodeId, entry.video.port)] = { kind: 'video', video: entry.video };
      }
    }
    return out;
  }, [logs]);
}

export interface NodePorts {
  inputs: PortTarget[];
  outputs: PortTarget[];
}

/**
 * Every port worth showing for one node: connected upstream sources on the
 * input side (labelled with their provenance, since the values are foreign)
 * and the node's own declared outputs on the other.
 *
 * Returns empty lists for a node that is absent or has no definition, so
 * callers never have to special-case a half-loaded graph.
 */
export function resolveSingleNodePorts(
  nodeId: string,
  nodes: readonly Node<NodeData>[],
  edges: readonly Edge[],
): NodePorts {
  const node = nodes.find((n) => n.id === nodeId);
  if (!node) return { inputs: [], outputs: [] };

  const inputs = resolveInputSources(node.id, edges).map((p) => {
    const srcNode = nodes.find((n) => n.id === p.nodeId);
    const srcLabel = srcNode?.data.label || p.nodeId.slice(0, 6);
    return {
      ...p,
      displayName: `${srcLabel}.${p.port}`,
      dataType: portDataType(nodes, p.nodeId, p.port),
    };
  });

  // A trigger output (Start's) is left out for the reason `resolveInputSources`
  // skips trigger edges: no value is ever captured behind it, so a request for
  // it could only 404.
  const outputs: PortTarget[] = (node.data.definition?.outputs ?? [])
    .filter((o) => o.data_type !== 'TRIGGER')
    .map((o) => ({
      nodeId: node.id,
      port: o.name,
      dataType: o.data_type,
    }));

  return { inputs, outputs };
}

/**
 * What an Inputs group with no wire on this level says -- one sentence for the
 * Inspector and Node details. An open block draws no wire from its own inputs,
 * so a node fed through one would otherwise read as having nothing connected.
 */
export function useInputsEmptyText(nodeId: string): string {
  const { t } = useI18n();
  const blockInputs = useTabStore((s) => {
    const tab = s.tabs.find((x) => x.id === s.activeTabId);
    const stack = tab?.subgraphStack ?? [];
    const frame = stack[stack.length - 1];
    if (!tab || !frame) return '';
    const open = tab.subgraphs.find((d) => d.id === frame.subgraphId);
    return (open?.interface?.inputs ?? [])
      .filter((p) => p.innerNode === nodeId)
      .map((p) => p.port)
      .join(', ');
  });
  return blockInputs
    ? t('inspector.capture.fromBlock', { ports: blockInputs })
    : t('nodeDetail.inputs.empty');
}

/* ── When a port can be read at all ─────────────────────────────────────────
 *
 * The engine writes a node's captures only after the node returns, and
 * `/api/execution/outputs` answers 404 for anything not written yet. So
 * mid-run a port has THREE states, not two: "nothing recorded" and "nothing
 * recorded YET" look identical to the fetch and mean opposite things to the
 * reader. Every view decides between them here — the rule is not copied.
 *
 * Which node decides is the node that OWNS the port: for an input row that is
 * the upstream source, whose value the row shows, not the node on screen.
 */

export type CapturePhase = 'pending' | 'running' | 'bypassed' | 'unselected' | 'settled';

/**
 * Whether the owner of a port has written its captures.
 *
 * `settled` means readable: the node reported a terminal status, or no run is
 * in progress at all and whatever is on the server is all there will be. A
 * node the run has not reached is `pending` rather than `running` — it is
 * queued, and saying it is running would be a claim we cannot make. A node
 * the run bypassed is neither (#559): it will never run, and what passes
 * through it is the upstream node's value, readable once the run is over.
 */
export function capturePhase(
  status: ExecutionStatus | undefined,
  runInProgress: boolean,
): CapturePhase {
  if (!runInProgress) return 'settled';
  if (status === 'running') return 'running';
  if (status === 'bypassed') return 'bypassed';
  // #656: left out because a Switch's param did not pick its branch. It will
  // never run, and nothing passes through it.
  if (status === 'unselected') return 'unselected';
  if (status === undefined || status === 'idle') return 'pending';
  return 'settled';
}

/**
 * The line to show for a phase, as a KEY — translated by whoever renders it,
 * so it follows a locale switch that no refetch would reach.
 */
export function capturePhaseNoteKey(phase: CapturePhase): TranslationKey | null {
  if (phase === 'running') return 'inspector.nodeRunning';
  if (phase === 'pending') return 'inspector.nodePending';
  if (phase === 'bypassed') return 'inspector.nodeBypassed';
  if (phase === 'unselected') return 'inspector.nodeUnselected';
  return null;
}

/**
 * Which ports to request now, given what has already been requested.
 *
 * `asked` is the per-run record of ports a request has gone out for, and this
 * mutates it. Three rules, all of them things a user would notice:
 *
 *  - a settled port is handed out ONCE, so a 100 MB upstream tensor is not
 *    downloaded again every time some other node finishes;
 *  - a port whose owner is running or queued is withheld AND forgotten, so
 *    the value that owner is about to write is read once it lands — which is
 *    also what makes a loop's second pass readable;
 *  - a port that left the view is forgotten, so coming back to it reads it
 *    afresh rather than showing another node's leftovers.
 */
export function takeDuePorts(
  ports: readonly PortTarget[],
  phases: readonly CapturePhase[],
  asked: Set<string>,
): PortTarget[] {
  const due: PortTarget[] = [];
  const present = new Set<string>();
  for (let i = 0; i < ports.length; i++) {
    const key = keyOf(ports[i].nodeId, ports[i].port);
    if (present.has(key)) continue;
    present.add(key);
    if (phases[i] !== 'settled') {
      asked.delete(key);
      continue;
    }
    if (!asked.has(key)) due.push(ports[i]);
  }
  for (const key of asked) if (!present.has(key)) asked.delete(key);
  for (const p of due) asked.add(keyOf(p.nodeId, p.port));
  return due;
}

// A phase per character, so a whole list of them travels as a string. A
// zustand selector that returned the array itself would hand React a new
// object on every store event and re-render forever (see NO_LOGS above).
const PHASE_CHAR: Record<CapturePhase, string> = {
  pending: 'p',
  running: 'r',
  bypassed: 'b',
  unselected: 'u',
  settled: 's',
};
const CHAR_PHASE: Record<string, CapturePhase> = {
  p: 'pending',
  r: 'running',
  b: 'bypassed',
  u: 'unselected',
  s: 'settled',
};

/** Whether the active tab has a run in flight right now. */
export function useRunInProgress(): boolean {
  return useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)?.status === 'running');
}

/** {@link capturePhase} for one node, from the live status on the active tab. */
export function useCapturePhase(nodeId: string): CapturePhase {
  const char = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    const node = tab?.nodes.find((n) => n.id === nodeId);
    return PHASE_CHAR[capturePhase(node?.data.executionStatus, tab?.status === 'running')];
  });
  return CHAR_PHASE[char];
}

/** {@link capturePhase} for each port's OWNER, in port order. */
export function usePortPhases(ports: readonly PortTarget[]): CapturePhase[] {
  const key = useTabStore((s) => {
    const tab = s.tabs.find((t) => t.id === s.activeTabId);
    const inProgress = tab?.status === 'running';
    let out = '';
    for (const p of ports) {
      const node = tab?.nodes.find((n) => n.id === p.nodeId);
      out += PHASE_CHAR[capturePhase(node?.data.executionStatus, inProgress)];
    }
    return out;
  });
  return useMemo(() => Array.from(key, (c) => CHAR_PHASE[c]), [key]);
}

/**
 * Replace the stored result of every port that cannot be read yet with its
 * note, at render time.
 *
 * Derived rather than stored on purpose: a node that started running again
 * must not show last pass's tensor, nor an expired line from a read that was
 * merely too early, for even one frame — and no write to state can be that
 * prompt.
 */
function withPhaseNotes(
  fetches: FetchMap,
  ports: readonly PortTarget[],
  phases: readonly CapturePhase[],
): FetchMap {
  let out = fetches;
  for (let i = 0; i < ports.length; i++) {
    const noteKey = capturePhaseNoteKey(phases[i]);
    if (!noteKey) continue;
    if (out === fetches) out = { ...fetches };
    out[keyOf(ports[i].nodeId, ports[i].port)] = {
      loading: false,
      error: null,
      errorKey: null,
      noteKey,
      data: null,
    };
  }
  return out;
}

/**
 * Load every port in `ports` from `runId` and keep the results in a map keyed
 * by {@link keyOf}.
 *
 * Refetching is keyed on the (run, port-set, per-port phase) identity rather
 * than the array reference: a caller that rebuilds an equivalent list on every
 * render — which any `useMemo` over the live nodes array eventually does —
 * must not restart the whole fetch (#166), and a node finishing must restart
 * exactly the ports it owns. Results for ports that disappear are left in the
 * map; they are unreachable by key and cost one entry.
 *
 * A request is never issued for a port whose owner has not returned: it could
 * only be answered 404, and that 404 is indistinguishable from expiry.
 *
 * Inside an open block every request names the node as the run does, and the
 * results are kept under that id (#621), so the same canvas id in two copies
 * of one block never shares a result. The map handed back is still keyed by
 * canvas id.
 *
 * Once the run is over, a port whose owner it recorded nothing for is not
 * asked for at all: its row says the node was not in the run, or failed in
 * it (see {@link missingFromRunNote}). While a run the tab is not running
 * still goes, a port whose owner it has not finished waits, and is read once
 * the run's record says it is over (see {@link onRunEnd}). In a run made with
 * Record node outputs off, a row's line follows the setting as it is switched
 * (see {@link followRecordingSetting}).
 */
export function usePortFetches(
  runId: string | null,
  ports: readonly PortTarget[],
): FetchMap {
  const [fetches, setFetches] = useState<FetchMap>({});
  const phases = usePortPhases(ports);
  const prefix = useRunNodePrefix();
  const recordOutputs = useRecordOutputs();

  // The effect depends on the port SET, not the array object, so keep the
  // latest arrays in refs for the effect body to read.
  const portsRef = useRef(ports);
  portsRef.current = ports;
  const phasesRef = useRef(phases);
  phasesRef.current = phases;
  const portsKey = ports.map((p) => keyOf(p.nodeId, p.port)).join('|');
  const phasesKey = phases.join('');

  // An answer is dropped only when nobody is left to want it: the run moved
  // on, the hook went away, or a later request for the same port superseded
  // this one (a loop's second pass must not be overwritten by its first).
  // Anything looser drops results that are still wanted, because the effect
  // now re-runs whenever ANY node's status changes.
  const runIdRef = useRef(runId);
  runIdRef.current = runId;
  const aliveRef = useRef(true);
  const askedRef = useRef<{ runId: string | null; keys: Set<string> }>({
    runId: null,
    keys: new Set(),
  });
  const seqRef = useRef<Map<string, number>>(new Map());
  // A run the tab is not running can end without the tab hearing of it: a
  // row waiting on one is read again once its record says it is over.
  const [runEnded, setRunEnded] = useState(0);
  const runEndRef = useRef<{ runId: string; stop: () => void } | null>(null);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      runEndRef.current?.stop();
      runEndRef.current = null;
    };
  }, []);

  useEffect(() => {
    // Stop waiting on a run the view has left, or one the tab now runs itself
    // (an attach was acknowledged): its own frames end it from here on, and
    // the phases re-run this effect when they do.
    if (runEndRef.current && (runEndRef.current.runId !== runId || runInProgressNow())) {
      runEndRef.current.stop();
      runEndRef.current = null;
    }
    if (!runId) return;
    if (askedRef.current.runId !== runId) {
      askedRef.current = { runId, keys: new Set() };
    }
    const due = takeDuePorts(
      toRunPorts(portsRef.current, prefix),
      phasesRef.current,
      askedRef.current.keys,
    );
    if (due.length === 0) return;

    const updates: FetchMap = {};
    const issued = new Map<string, number>();
    for (const t of due) {
      const key = keyOf(t.nodeId, t.port);
      const seq = (seqRef.current.get(key) ?? 0) + 1;
      seqRef.current.set(key, seq);
      issued.set(key, seq);
      updates[key] = { loading: true, error: null, errorKey: null, data: null };
    }
    setFetches((prev) => ({ ...prev, ...updates }));

    const stale = (key: string, seq: number) =>
      !aliveRef.current || runIdRef.current !== runId || seqRef.current.get(key) !== seq;
    // Read now, not when the answers land: the index is only whole once the
    // run is over, and the status is the one this pass was decided on. Inside
    // an open block there is no status to read: the cards never get one.
    const runOver = !runInProgressNow();
    const statuses = due.map((t) => (prefix ? undefined : canvasNodeStatus(t.nodeId)));

    void Promise.all(
      due.map(async (t, i) => {
        const key = keyOf(t.nodeId, t.port);
        const seq = issued.get(key)!;
        try {
          const missing = runOver
            ? await missingFromRunNote(runId, { runNodeId: t.nodeId, status: statuses[i] })
            : null;
          if (missing) {
            if (stale(key, seq)) return;
            if (isRunStillGoingNote(missing)) {
              // Not asked for yet: read once the run is over.
              askedRef.current.keys.delete(key);
              if (!runEndRef.current) {
                const stop = onRunEnd(runId, () => {
                  runEndRef.current = null;
                  setRunEnded((n) => n + 1);
                });
                runEndRef.current = { runId, stop };
              }
            }
            setFetches((prev) => ({
              ...prev,
              [key]: { loading: false, error: null, errorKey: null, noteKey: missing, data: null },
            }));
            return;
          }
          const data = await fetchPortWithSliceFallback(runId, t.nodeId, t.port);
          if (stale(key, seq)) return;
          setFetches((prev) => ({
            ...prev,
            [key]: { loading: false, error: null, errorKey: null, data },
          }));
        } catch (e) {
          if (stale(key, seq)) return;
          // The node ran and left this port empty: a neutral note, not an
          // error, and certainly not expiry.
          if (e instanceof NoValueError) {
            setFetches((prev) => ({
              ...prev,
              [key]: {
                loading: false,
                error: null,
                errorKey: null,
                noteKey: 'inspector.noValue',
                data: null,
              },
            }));
            return;
          }
          // Expiry is the one failure we recognise, so it travels as a key
          // that PortGroup translates on every render. Translating it here
          // would freeze the wording into state, where a later locale
          // switch cannot reach it.
          const expired = e instanceof RunDataExpiredError;
          setFetches((prev) => ({
            ...prev,
            [key]: {
              loading: false,
              error: expired ? null : (e as Error).message,
              errorKey: expired ? 'inspector.dataExpired' : null,
              data: null,
            },
          }));
        }
      }),
    );
  }, [runId, prefix, portsKey, phasesKey, runEnded]);

  return withPhaseNotes(
    withRecordingSetting(fromRunKeys(fetches, ports, prefix), recordOutputs),
    ports,
    phases,
  );
}
