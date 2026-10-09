import { useEffect, useState } from 'react';
import {
  fetchGradIndex,
  RunDataExpiredError,
  type GradIndexEntry,
} from '../../api/executionOutputs';
import type { OutputData, TensorOutput } from '../../types';
import { TensorGridView } from './TensorGridView';
import {
  canvasNodeHasOutputs,
  canvasNodeIsContainer,
  canvasNodeStatus,
  fetchPortWithSliceFallback,
  innerNodeLabel,
  isRunStillGoingNote,
  missingGradientsNote,
  onRunEnd,
  useRunInProgress,
  useRunNodeId,
} from './portCaptures';
import { useI18n, type TranslationKey } from '../../i18n';
import styles from './InspectorPanel.module.css';

interface Props {
  runId: string;
  nodeId: string;
}

interface TensorState {
  loading: boolean;
  error: string | null;
  data: OutputData | null;
}

type TensorMap = Record<string, TensorState>;

function entryKey(entry: GradIndexEntry): string {
  // A card's entries come from several nodes inside it (#559).
  return `${entry.node_id ?? ''}::${entry.kind}::${entry.port}`;
}

function entryStorePort(entry: GradIndexEntry): string {
  // The port name to query /api/execution/outputs/{run}/{node}/{port}
  return entry.kind === 'weight'
    ? `__weight_grad__${entry.port}`
    : `${entry.port}__grad`;
}

function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(2);
  return v.toFixed(4);
}

function HealthChip({ status, norm }: { status: string; norm: number }) {
  const { t } = useI18n();
  const colors: Record<string, { bg: string; fg: string; border: string }> = {
    healthy: { bg: 'rgba(124, 207, 124, 0.12)', fg: '#7ccf7c', border: 'rgba(124, 207, 124, 0.4)' },
    vanishing: { bg: 'rgba(245, 166, 35, 0.12)', fg: '#f5a623', border: 'rgba(245, 166, 35, 0.4)' },
    exploding: { bg: 'rgba(230, 57, 70, 0.12)', fg: '#e63946', border: 'rgba(230, 57, 70, 0.4)' },
  };
  const c = colors[status] ?? colors.healthy;
  const label =
    status === 'vanishing'
      ? t('inspector.backward.health.vanishing')
      : status === 'exploding'
        ? t('inspector.backward.health.exploding')
        : t('inspector.backward.health.healthy');
  return (
    <span
      style={{
        background: c.bg,
        color: c.fg,
        border: `1px solid ${c.border}`,
        padding: '1px 6px',
        borderRadius: 8,
        fontSize: 11,
        fontFamily: 'ui-monospace, SFMono-Regular, monospace',
      }}
    >
      {label} · ‖g‖ {formatNumber(norm)}
    </span>
  );
}

/**
 * Render the captured gradients for one node in one run.
 * Shows two sections: per-output-port gradients ("port") and per-parameter
 * weight gradients ("weight"). Each entry gets a TensorGridView with
 * heat coloring based on |grad| / max|grad|.
 */
export function BackwardView({ runId, nodeId }: Props) {
  const { t } = useI18n();
  const [entries, setEntries] = useState<GradIndexEntry[] | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [tensors, setTensors] = useState<TensorMap>({});
  // Set when the finished run recorded nothing for this node: it was not in
  // that run, or failed in it, and "turn on Capture gradients" would be the
  // wrong hint.
  const [missingKey, setMissingKey] = useState<TranslationKey | null>(null);
  // Bumped when a run the tab is not running ends, so its gradients are read.
  const [runEnded, setRunEnded] = useState(0);
  // Gradients are written by the backward pass, which runs after the WHOLE
  // forward pass — so unlike the Forward tab, what this view waits for is the
  // run, not the selected node. Read mid-run the index comes back empty,
  // which reads as "no gradients captured": an instruction to turn on a
  // setting that may well already be on.
  const runInProgress = useRunInProgress();
  // Inside an open block the run recorded this node as `<instance>/<inner>`
  // (#621).
  const runNodeId = useRunNodeId(nodeId);

  // The parent remounts this component (via a `key` on runId:nodeId), so each
  // mount starts from fresh state — no manual reset needed, and the prior
  // node's gradients never flash before the new fetch resolves.
  useEffect(() => {
    if (runInProgress) return;
    let cancelled = false;
    let stopWaiting: (() => void) | null = null;
    // Inside an open block the card has no run status to go by. A block or
    // preset card's gradients are its inner nodes' (#559): none is under its
    // own id, so it is never "not in the run".
    const container = canvasNodeIsContainer(nodeId);
    missingGradientsNote(runId, {
      runNodeId,
      status: runNodeId === nodeId ? canvasNodeStatus(nodeId) : undefined,
      hasOutputs: !container && canvasNodeHasOutputs(nodeId),
    })
      .then((note) => {
        if (cancelled) return;
        if (note === 'none') {
          // Capture gradients was off for the run: there are none, and
          // nothing to ask for.
          setMissingKey(null);
          setEntries([]);
          setTensors({});
          return;
        }
        setMissingKey(note);
        if (note) {
          // A run the tab is not running: read again once it is over.
          if (isRunStillGoingNote(note)) {
            stopWaiting = onRunEnd(runId, () => setRunEnded((n) => n + 1));
          }
          return;
        }
        return fetchGradIndex(runId, runNodeId, container).then((es) => {
          if (cancelled) return;
          setEntries(es);
          // Seed loading placeholders for every gradient the next effect fetches.
          const initial: TensorMap = {};
          for (const e of es) {
            initial[entryKey(e)] = { loading: true, error: null, data: null };
          }
          setTensors(initial);
        });
      })
      .catch((e) => {
        if (cancelled) return;
        setIndexError(
          e instanceof RunDataExpiredError
            ? t('inspector.dataExpiredBackward')
            : (e as Error).message,
        );
      });
    return () => {
      cancelled = true;
      stopWaiting?.();
    };
  }, [runId, runNodeId, nodeId, t, runInProgress, runEnded]);

  // The loading placeholders were already seeded alongside setEntries above.
  useEffect(() => {
    if (!entries || entries.length === 0) return;
    let cancelled = false;
    Promise.all(
      entries.map(async (e) => {
        const port = entryStorePort(e);
        try {
          const data = await fetchPortWithSliceFallback(runId, e.node_id ?? runNodeId, port);
          if (cancelled) return;
          setTensors((prev) => ({
            ...prev,
            [entryKey(e)]: { loading: false, error: null, data },
          }));
        } catch (err) {
          if (cancelled) return;
          setTensors((prev) => ({
            ...prev,
            [entryKey(e)]: {
              loading: false,
              error:
                err instanceof RunDataExpiredError
                  ? t('inspector.tensorExpired')
                  : (err as Error).message,
              data: null,
            },
          }));
        }
      }),
    );
    return () => {
      cancelled = true;
    };
  }, [entries, runId, runNodeId, t]);

  if (runInProgress) {
    return <div className={styles.diffMissing}>{t('inspector.runRunning')}</div>;
  }

  if (missingKey) {
    return <div className={styles.diffMissing}>{t(missingKey)}</div>;
  }

  if (indexError) {
    return <div className={styles.portError}>{indexError}</div>;
  }

  if (entries === null) {
    return <div className={styles.diffMissing}>…</div>;
  }

  if (entries.length === 0) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyIcon}>∂</div>
        <div>{t('inspector.backward.empty')}</div>
        <div className={styles.emptyHint}>{t('inspector.backward.disabled')}</div>
      </div>
    );
  }

  const portEntries = entries.filter((e) => e.kind === 'port');
  const weightEntries = entries.filter((e) => e.kind === 'weight');

  return (
    <div className={styles.stepList}>
      {portEntries.length > 0 && (
        <Section
          title={t('inspector.backward.portSection')}
          entries={portEntries}
          tensors={tensors}
          runNodeId={runNodeId}
        />
      )}
      {weightEntries.length > 0 && (
        <Section
          title={t('inspector.backward.weightSection')}
          entries={weightEntries}
          tensors={tensors}
          runNodeId={runNodeId}
        />
      )}
    </div>
  );
}

function Section({
  title,
  entries,
  tensors,
  runNodeId,
}: {
  title: string;
  entries: GradIndexEntry[];
  tensors: TensorMap;
  runNodeId: string;
}) {
  return (
    <div>
      <div className={styles.portGroupTitle}>{title}</div>
      {entries.map((e) => {
        const state = tensors[entryKey(e)];
        const tensorData =
          state?.data && state.data.type === 'tensor'
            ? (state.data as TensorOutput)
            : null;
        const maxAbs =
          tensorData && tensorData.max !== undefined
            ? Math.max(Math.abs(tensorData.max), Math.abs(tensorData.min ?? 0))
            : null;

        const highlight = (i: number, j: number): number => {
          if (!tensorData || !maxAbs || maxAbs === 0) return 0;
          // Pull cell value via leading dims = []? TensorGridView already
          // applies leading dim selection; this fn only sees flat (i,j).
          const grid = tensorData.values as unknown;
          let cell = 0;
          // highlight() runs only for tensor data, whose values is always an array
          /* v8 ignore start */
          if (Array.isArray(grid)) {
            const row = (grid as unknown[])[i];
            if (Array.isArray(row)) {
              const v = (row as unknown[])[j];
              if (typeof v === 'number') cell = v;
            } else if (typeof row === 'number') {
              cell = row;
            }
          }
          /* v8 ignore stop */
          return Math.min(1, Math.abs(cell) / maxAbs);
        };

        return (
          <div key={entryKey(e)} className={styles.stepTensor} style={{ marginTop: 8 }}>
            <div
              className={styles.stepTensorLabel}
              style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}
            >
              {/* A card's entry names the node inside it that it belongs to (#559). */}
              <span>{e.node_id === undefined ? e.port : `${innerNodeLabel(runNodeId, e.node_id)} · ${e.port}`}</span>
              {e.health && (
                <HealthChip status={e.health.status} norm={e.health.norm} />
              )}
            </div>
            {state?.error && <div className={styles.portError}>{state.error}</div>}
            {state?.loading && <div className={styles.diffMissing}>…</div>}
            {tensorData && <TensorGridView tensor={tensorData} highlight={highlight} />}
          </div>
        );
      })}
    </div>
  );
}
