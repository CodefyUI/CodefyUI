import { useEffect, useMemo, useState } from 'react';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import {
  fetchOutput,
  RunDataExpiredError,
  PayloadTooLargeError,
} from '../../api/executionOutputs';
import type { OutputData, TensorOutput } from '../../types';
import { TensorGridView } from './TensorGridView';
import { ValueDiff } from './ValueDiff';
import { computeSegmentNodes } from '../../utils/segmentPath';
import styles from './InspectorPanel.module.css';

interface PortFetchState {
  loading: boolean;
  error: string | null;
  data: OutputData | null;
}

type FetchMap = Record<string, PortFetchState>;

interface PortTarget {
  nodeId: string;
  port: string;
  /** Optional extra label shown above the tensor view (e.g. `→ NodeX.tensor`) */
  displayName?: string;
}

function keyOf(nodeId: string, port: string): string {
  return `${nodeId}::${port}`;
}

async function fetchPortWithSliceFallback(
  runId: string,
  nodeId: string,
  port: string,
): Promise<OutputData> {
  try {
    return await fetchOutput(runId, nodeId, port);
  } catch (e) {
    if (e instanceof PayloadTooLargeError) {
      // Narrow to the first index along every leading dim
      return await fetchOutput(runId, nodeId, port, { slice: '0,:,:', maxElements: 65536 });
    }
    throw e;
  }
}

export function InspectorPanel() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const selectedNodeId = activeTab.selectedNodeId;
  const activeSegment = activeTab.activeSegment;
  const lastRunId = activeTab.lastRunId;
  const nodes = activeTab.nodes;
  const edges = activeTab.edges;
  const { t } = useI18n();

  const [fetches, setFetches] = useState<FetchMap>({});
  const [collapsed, setCollapsed] = useState(false);

  // Determine what to fetch based on mode
  const targets = useMemo(() => {
    if (!lastRunId) return { mode: 'none' as const };
    if (activeSegment) {
      const head = nodes.find((n) => n.id === activeSegment.headNodeId);
      const tail = nodes.find((n) => n.id === activeSegment.tailNodeId);
      if (!head || !tail) return { mode: 'none' as const };

      // Gather every node that lies on a head→tail path. Treat the segment
      // as a logical block: any data edge whose source is OUTSIDE this set
      // and whose target is INSIDE is a segment-level input, regardless of
      // which internal node it feeds. This surfaces all entry lines, not
      // just the head's direct inputs.
      const segmentSet = computeSegmentNodes(head.id, tail.id, nodes, edges);

      const inputs: PortTarget[] = [];
      const seenInputKey = new Set<string>();
      for (const e of edges) {
        const isTrigger =
          (e as { type?: string }).type === 'triggerEdge' ||
          (e.data as { type?: string } | undefined)?.type === 'trigger';
        if (isTrigger) continue;
        if (!segmentSet.has(e.target) || segmentSet.has(e.source)) continue;
        if (!e.sourceHandle) continue;
        const key = `${e.source}::${e.sourceHandle}->${e.target}::${e.targetHandle ?? ''}`;
        if (seenInputKey.has(key)) continue;
        seenInputKey.add(key);
        const targetNode = nodes.find((n) => n.id === e.target);
        const targetLabel = targetNode?.data.label ?? e.target.slice(0, 6);
        inputs.push({
          nodeId: e.source,
          port: e.sourceHandle,
          displayName: `→ ${targetLabel}.${e.targetHandle ?? ''}`,
        });
      }

      const outputs: PortTarget[] = (tail.data.definition?.outputs ?? []).map((o) => ({
        nodeId: tail.id,
        port: o.name,
        displayName: `${o.name}`,
      }));

      return {
        mode: 'segment' as const,
        headName: head.data.label,
        tailName: tail.data.label,
        inputs,
        outputs,
      };
    }
    if (selectedNodeId) {
      const node = nodes.find((n) => n.id === selectedNodeId);
      if (!node) return { mode: 'none' as const };
      const inputs = resolveInputSources(node.id, edges);
      const outputs: PortTarget[] = (node.data.definition?.outputs ?? []).map((o) => ({
        nodeId: node.id,
        port: o.name,
      }));
      return {
        mode: 'single' as const,
        nodeName: node.data.label,
        inputs,
        outputs,
      };
    }
    return { mode: 'none' as const };
  }, [selectedNodeId, activeSegment, lastRunId, nodes, edges]);

  // Fetch effect
  useEffect(() => {
    if (targets.mode === 'none' || !lastRunId) return;
    const all = [...targets.inputs, ...targets.outputs];
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
            const data = await fetchPortWithSliceFallback(lastRunId, t.nodeId, t.port);
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
  }, [targets, lastRunId]);

  const collapseButton = (
    <button
      className={styles.collapseBtn}
      onClick={() => setCollapsed((v) => !v)}
      title={collapsed ? t('inspector.expand') : t('inspector.collapse')}
      aria-label={collapsed ? 'Expand inspector' : 'Collapse inspector'}
    >
      {collapsed ? '‹' : '›'}
    </button>
  );

  if (collapsed) {
    return (
      <div className={`${styles.panel} ${styles.collapsed}`}>
        {collapseButton}
        <div className={styles.collapsedStub}>INSPECTOR</div>
      </div>
    );
  }

  if (targets.mode === 'none') {
    if (!lastRunId) {
      return (
        <div className={styles.panel}>
          {collapseButton}
          <div className={styles.panelHeader}>
            <span className={styles.panelTitle}>{t('inspector.title')}</span>
          </div>
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>▶</div>
            <div>{t('inspector.empty.notRun')}</div>
            <div className={styles.emptyHint}>{t('inspector.empty.notRunHint')}</div>
          </div>
        </div>
      );
    }
    return (
      <div className={styles.panel}>
        {collapseButton}
        <div className={styles.panelHeader}>
          <span className={styles.panelTitle}>{t('inspector.title')}</span>
        </div>
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>⌖</div>
          <div>{t('inspector.empty.noSelection')}</div>
          <div className={styles.emptyHint}>{t('inspector.empty.noSelectionHint')}</div>
        </div>
      </div>
    );
  }

  if (targets.mode === 'segment') {
    return (
      <div className={styles.panel}>
        {collapseButton}
        <div className={styles.panelHeader}>
          <span className={styles.segmentBadge}>SEGMENT</span>
          <span className={styles.segmentNames}>
            {targets.headName} → {targets.tailName}
          </span>
        </div>
        <div className={styles.panelContent}>
          <SegmentSide
            kind="input"
            title={t('inspector.segment.inputs', { count: targets.inputs.length })}
            ports={targets.inputs}
            fetches={fetches}
          />
          <div className={styles.segmentDivider}>↓</div>
          <SegmentSide
            kind="output"
            title={t('inspector.segment.outputs', { count: targets.outputs.length })}
            ports={targets.outputs}
            fetches={fetches}
          />
        </div>
      </div>
    );
  }

  // Single node mode
  const inputs = targets.inputs;
  const outputs = targets.outputs;
  return (
    <div className={styles.panel}>
      {collapseButton}
      <div className={styles.panelHeader}>
        <span className={styles.panelTitle}>{t('inspector.title')}</span>
        <span className={styles.panelSubtitle}>{targets.nodeName}</span>
      </div>
      <div className={styles.panelContent}>
        {inputs.length === 0 && outputs.length === 0 && (
          <div className={styles.emptyState}>This node has no ports.</div>
        )}
        {/* Pair input[i] with output[i] when possible */}
        {pairLists(inputs, outputs).map((row, i) => {
          const inKey = row.input ? keyOf(row.input.nodeId, row.input.port) : null;
          const outKey = row.output ? keyOf(row.output.nodeId, row.output.port) : null;
          const inState = inKey ? fetches[inKey] : null;
          const outState = outKey ? fetches[outKey] : null;
          return (
            <div key={i} className={styles.portBlock}>
              <div className={styles.portHeader}>
                {row.input ? `in: ${row.input.port}` : 'in: —'} &nbsp;&nbsp;·&nbsp;&nbsp;{' '}
                {row.output ? `out: ${row.output.port}` : 'out: —'}
              </div>
              {renderErrors(inState, outState)}
              <ValueDiff input={inState?.data ?? null} output={outState?.data ?? null} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SegmentSide({
  kind,
  title,
  ports,
  fetches,
}: {
  kind: 'input' | 'output';
  title: string;
  ports: PortTarget[];
  fetches: FetchMap;
}) {
  if (ports.length === 0) {
    return (
      <div className={styles.segmentSide}>
        <div className={styles.segmentSideTitle}>{title}</div>
        <div className={styles.diffMissing}>—</div>
      </div>
    );
  }
  return (
    <div className={styles.segmentSide}>
      <div className={styles.segmentSideTitle}>{title}</div>
      {ports.map((p) => {
        const key = keyOf(p.nodeId, p.port);
        const state = fetches[key];
        return (
          <div key={key} className={styles.portBlock}>
            <div className={styles.portHeader}>
              {kind === 'input' ? '⟵ ' : '⟶ '}
              <span className={styles.portName}>
                {p.displayName ?? p.port}
              </span>
            </div>
            {state?.error && <div className={styles.portError}>{state.error}</div>}
            {state?.data && state.data.type === 'tensor' && (
              <TensorGridView tensor={state.data as TensorOutput} />
            )}
            {state?.data && state.data.type !== 'tensor' && (
              <div className={styles.tensorScalar}>
                {state.data.type === 'scalar' && String((state.data as { value?: unknown }).value)}
                {state.data.type === 'string' && (state.data as { value?: string }).value}
                {state.data.type === 'model' && (
                  `${(state.data as { class?: string }).class ?? 'Module'} · params ${(state.data as { params?: number }).params ?? '?'}`
                )}
              </div>
            )}
            {!state?.data && !state?.error && (
              <div className={styles.diffMissing}>…</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function pairLists<T>(a: T[], b: T[]): { input: T | null; output: T | null }[] {
  const n = Math.max(a.length, b.length);
  return Array.from({ length: n }, (_, i) => ({
    input: a[i] ?? null,
    output: b[i] ?? null,
  }));
}

function renderErrors(
  inState: PortFetchState | null | undefined,
  outState: PortFetchState | null | undefined,
) {
  const msgs: string[] = [];
  if (inState?.error) msgs.push(`input: ${inState.error}`);
  if (outState?.error) msgs.push(`output: ${outState.error}`);
  if (msgs.length === 0) return null;
  return <div className={styles.portError}>{msgs.join(' · ')}</div>;
}

function resolveInputSources(
  nodeId: string,
  edges: {
    source: string;
    target: string;
    sourceHandle?: string | null;
    targetHandle?: string | null;
    type?: string;
    data?: unknown;
  }[],
): PortTarget[] {
  const result: PortTarget[] = [];
  for (const e of edges) {
    if (e.target !== nodeId) continue;
    // Skip trigger edges
    const isTrigger =
      e.type === 'triggerEdge' || (e.data as { type?: string } | undefined)?.type === 'trigger';
    if (isTrigger) continue;
    if (!e.sourceHandle) continue;
    result.push({ nodeId: e.source, port: e.sourceHandle });
  }
  return result;
}
