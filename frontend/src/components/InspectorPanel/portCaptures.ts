import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchOutput,
  RunDataExpiredError,
  PayloadTooLargeError,
} from '../../api/executionOutputs';
import type { NodeData, OutputData } from '../../types';
import type { Edge, Node } from '@xyflow/react';
import {
  useTabStore,
  type LogEntry,
  type LogImagePayload,
  type LogVideoPayload,
} from '../../store/tabStore';
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

/**
 * Load every port in `ports` from `runId` and keep the results in a map keyed
 * by {@link keyOf}.
 *
 * Refetching is keyed on the (run, port-set) identity rather than the array
 * reference: a caller that rebuilds an equivalent list on every render — which
 * any `useMemo` over the live nodes array eventually does — must not restart
 * the whole fetch. Results for ports that disappear are left in the map; they
 * are unreachable by key and cost one entry.
 */
export function usePortFetches(
  runId: string | null,
  ports: readonly PortTarget[],
): FetchMap {
  const [fetches, setFetches] = useState<FetchMap>({});

  // The effect depends on the port SET, not the array object, so keep the
  // latest array in a ref for the effect body to read.
  const portsRef = useRef(ports);
  portsRef.current = ports;
  const portsKey = ports.map((p) => keyOf(p.nodeId, p.port)).join('|');

  useEffect(() => {
    if (!runId) return;
    const all = portsRef.current;
    if (all.length === 0) return;
    let cancelled = false;

    const run = async () => {
      const updates: FetchMap = {};
      for (const t of all) {
        updates[keyOf(t.nodeId, t.port)] = { loading: true, error: null, data: null };
      }
      setFetches((prev) => ({ ...prev, ...updates }));

      await Promise.all(
        all.map(async (t) => {
          try {
            const data = await fetchPortWithSliceFallback(runId, t.nodeId, t.port);
            if (cancelled) return;
            setFetches((prev) => ({
              ...prev,
              [keyOf(t.nodeId, t.port)]: { loading: false, error: null, data },
            }));
          } catch (e) {
            if (cancelled) return;
            const msg =
              e instanceof RunDataExpiredError
                ? 'run data expired — re-run to capture'
                : (e as Error).message;
            setFetches((prev) => ({
              ...prev,
              [keyOf(t.nodeId, t.port)]: { loading: false, error: msg, data: null },
            }));
          }
        }),
      );
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [runId, portsKey]);

  return fetches;
}
