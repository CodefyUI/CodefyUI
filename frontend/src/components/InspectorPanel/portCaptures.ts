import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchOutput,
  RunDataExpiredError,
  PayloadTooLargeError,
} from '../../api/executionOutputs';
import type { ExecutionStatus, NodeData, OutputData } from '../../types';
import type { Edge, Node } from '@xyflow/react';
import {
  useTabStore,
  type LogEntry,
  type LogImagePayload,
  type LogVideoPayload,
} from '../../store/tabStore';
import type { TranslationKey } from '../../i18n';
import { keyOf, type FetchMap, type PortTarget } from './PortGroup';

/**
 * The capture-reading half of the Inspector, factored out so the side panel
 * and the Node Detail Modal (#127) read the *same* run data through the *same*
 * code path. Anything that resolves which ports belong to a node, or turns
 * those ports into `/api/execution/outputs` results, belongs here — never
 * copied into a second component.
 */

/**
 * Fetch one port, narrowing to the first index along every leading dim if the
 * server refuses the full payload. Without the retry a large activation shows
 * an error instead of its leading slice.
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
      return await fetchOutput(runId, nodeId, port, { slice: '0,:,:', maxElements: 65536 });
    }
    throw e;
  }
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

  const outputs: PortTarget[] = (node.data.definition?.outputs ?? []).map((o) => ({
    nodeId: node.id,
    port: o.name,
    dataType: o.data_type,
  }));

  return { inputs, outputs };
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

export type CapturePhase = 'pending' | 'running' | 'settled';

/**
 * Whether the owner of a port has written its captures.
 *
 * `settled` means readable: the node reported a terminal status, or no run is
 * in progress at all and whatever is on the server is all there will be. A
 * node the run has not reached is `pending` rather than `running` — it is
 * queued, and saying it is running would be a claim we cannot make.
 */
export function capturePhase(
  status: ExecutionStatus | undefined,
  runInProgress: boolean,
): CapturePhase {
  if (!runInProgress) return 'settled';
  if (status === 'running') return 'running';
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
  settled: 's',
};
const CHAR_PHASE: Record<string, CapturePhase> = {
  p: 'pending',
  r: 'running',
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
 */
export function usePortFetches(
  runId: string | null,
  ports: readonly PortTarget[],
): FetchMap {
  const [fetches, setFetches] = useState<FetchMap>({});
  const phases = usePortPhases(ports);

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

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!runId) return;
    if (askedRef.current.runId !== runId) {
      askedRef.current = { runId, keys: new Set() };
    }
    const due = takeDuePorts(portsRef.current, phasesRef.current, askedRef.current.keys);
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

    void Promise.all(
      due.map(async (t) => {
        const key = keyOf(t.nodeId, t.port);
        const seq = issued.get(key)!;
        try {
          const data = await fetchPortWithSliceFallback(runId, t.nodeId, t.port);
          if (stale(key, seq)) return;
          setFetches((prev) => ({
            ...prev,
            [key]: { loading: false, error: null, errorKey: null, data },
          }));
        } catch (e) {
          if (stale(key, seq)) return;
          // Expiry is the one cause we recognise, so it travels as a key
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
  }, [runId, portsKey, phasesKey]);

  return withPhaseNotes(fetches, ports, phases);
}
