import { useEffect, useMemo, useState } from 'react';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import {
  fetchOutput,
  RunDataExpiredError,
  PayloadTooLargeError,
} from '../../api/executionOutputs';
import type { OutputData } from '../../types';
import { StepTraceView } from './StepTraceView';
import { BackwardView } from './BackwardView';
import { TokenChipsView } from './TokenChipsView';
import { PortGroup, FlowDivider, keyOf } from './PortGroup';
import type { PortTarget, FetchMap } from './PortGroup';
import { isTensor, shapesEqual, makeHighlight } from './diff';
import { computeSegmentNodes } from '../../utils/segmentPath';
import styles from './InspectorPanel.module.css';

type InspectorTab = 'forward' | 'steps' | 'backward';

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
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>('forward');

  // Reset tab when selection changes so Steps don't linger between nodes.
  useEffect(() => {
    setInspectorTab('forward');
  }, [selectedNodeId, lastRunId]);

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
          dataType: portDataType(nodes, e.source, e.sourceHandle),
        });
      }

      const outputs: PortTarget[] = (tail.data.definition?.outputs ?? []).map((o) => ({
        nodeId: tail.id,
        port: o.name,
        displayName: `${o.name}`,
        dataType: o.data_type,
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
      // Inputs are the connected upstream ports; label them with their
      // provenance (source node + port) since they are foreign values.
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
      return {
        mode: 'single' as const,
        nodeName: node.data.label,
        nodeType: node.data.type,
        nodeId: node.id,
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
    <button type="button"
      className={styles.collapseBtn}
      onClick={() => setCollapsed((v) => !v)}
      title={collapsed ? t('inspector.expand') : t('inspector.collapse')}
      aria-label={collapsed ? t('inspector.expand') : t('inspector.collapse')}
    >
      {collapsed ? '‹' : '›'}
    </button>
  );

  if (collapsed) {
    return (
      <div className={`${styles.panel} ${styles.collapsed}`}>
        {collapseButton}
        <div className={styles.collapsedStub}>{t('inspector.collapsedStub')}</div>
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
          <span className={styles.segmentBadge}>{t('inspector.segmentBadge')}</span>
          <span className={styles.segmentNames}>
            {targets.headName} → {targets.tailName}
          </span>
        </div>
        <div className={styles.panelContent}>
          <PortGroup
            kind="input"
            title={t('inspector.segment.inputs', { count: targets.inputs.length })}
            ports={targets.inputs}
            fetches={fetches}
          />
          <FlowDivider />
          <PortGroup
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

  // The in→out transform summary (shape chip / cell heat) is only
  // well-defined when the node has exactly one input and one output.
  let shapeChip: string | null = null;
  let outHighlight: { portKey: string; fn: (i: number, j: number) => number } | undefined;
  if (inputs.length === 1 && outputs.length === 1) {
    const inData = fetches[keyOf(inputs[0].nodeId, inputs[0].port)]?.data ?? null;
    const outData = fetches[keyOf(outputs[0].nodeId, outputs[0].port)]?.data ?? null;
    if (isTensor(inData) && isTensor(outData)) {
      if (shapesEqual(inData.full_shape, outData.full_shape)) {
        const fn = makeHighlight(inData, outData);
        if (fn) outHighlight = { portKey: keyOf(outputs[0].nodeId, outputs[0].port), fn };
      } else {
        shapeChip = `[${inData.full_shape.join(', ')}] → [${outData.full_shape.join(', ')}]`;
      }
    }
  }

  return (
    <div className={styles.panel}>
      {collapseButton}
      <div className={styles.panelHeader}>
        <span className={styles.panelTitle}>{t('inspector.title')}</span>
        <span className={styles.panelSubtitle}>{targets.nodeName}</span>
      </div>
      <div className={styles.tabStrip} role="tablist">
        <button type="button"
          role="tab"
          aria-selected={inspectorTab === 'forward'}
          className={`${styles.tabBtn} ${inspectorTab === 'forward' ? styles.tabActive : ''}`}
          onClick={() => setInspectorTab('forward')}
        >
          {t('inspector.tabs.forward')}
        </button>
        <button type="button"
          role="tab"
          aria-selected={inspectorTab === 'steps'}
          className={`${styles.tabBtn} ${inspectorTab === 'steps' ? styles.tabActive : ''}`}
          onClick={() => setInspectorTab('steps')}
        >
          {t('inspector.tabs.steps')}
        </button>
        <button type="button"
          role="tab"
          aria-selected={inspectorTab === 'backward'}
          className={`${styles.tabBtn} ${inspectorTab === 'backward' ? styles.tabActive : ''}`}
          onClick={() => setInspectorTab('backward')}
        >
          {t('inspector.tabs.backward')}
        </button>
      </div>
      <div className={styles.panelContent}>
        {inspectorTab === 'forward' && (
          <>
            {targets.nodeType === 'Tokenizer' && (
              <TokenChipsView
                tokens={fetches[keyOf(targets.nodeId, 'tokens')]?.data ?? null}
                tokenIds={fetches[keyOf(targets.nodeId, 'token_ids')]?.data ?? null}
                offsets={fetches[keyOf(targets.nodeId, 'offsets')]?.data ?? null}
              />
            )}
            {inputs.length === 0 && outputs.length === 0 ? (
              <div className={styles.emptyState}>{t('inspector.emptyPorts')}</div>
            ) : (
              <>
                <PortGroup
                  kind="input"
                  title={t('inspector.node.inputs', { count: inputs.length })}
                  ports={inputs}
                  fetches={fetches}
                  emptyText={t('inspector.node.inputsEmpty')}
                />
                <FlowDivider chip={shapeChip} />
                <PortGroup
                  kind="output"
                  title={t('inspector.node.outputs', { count: outputs.length })}
                  ports={outputs}
                  fetches={fetches}
                  emptyText={t('inspector.node.outputsEmpty')}
                  highlight={outHighlight}
                />
              </>
            )}
          </>
        )}
        {inspectorTab === 'steps' && lastRunId && selectedNodeId && (
          <StepTraceView
            key={`${lastRunId}:${selectedNodeId}`}
            runId={lastRunId}
            nodeId={selectedNodeId}
          />
        )}
        {inspectorTab === 'backward' && lastRunId && selectedNodeId && (
          <BackwardView
            key={`${lastRunId}:${selectedNodeId}`}
            runId={lastRunId}
            nodeId={selectedNodeId}
          />
        )}
      </div>
    </div>
  );
}

/** Look up a source port's declared data type from its node definition. */
function portDataType(
  nodes: { id: string; data: { definition?: { outputs?: { name: string; data_type: string }[] } } }[],
  nodeId: string,
  port: string,
): string | undefined {
  const n = nodes.find((x) => x.id === nodeId);
  return n?.data.definition?.outputs?.find((o) => o.name === port)?.data_type;
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
