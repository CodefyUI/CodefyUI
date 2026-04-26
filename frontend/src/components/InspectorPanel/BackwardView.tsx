import { useEffect, useState } from 'react';
import {
  fetchGradIndex,
  fetchOutput,
  PayloadTooLargeError,
  RunDataExpiredError,
  type GradIndexEntry,
} from '../../api/executionOutputs';
import type { OutputData, TensorOutput } from '../../types';
import { TensorGridView } from './TensorGridView';
import { useI18n } from '../../i18n';
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
  return `${entry.kind}::${entry.port}`;
}

function entryStorePort(entry: GradIndexEntry): string {
  // The port name to query /api/execution/outputs/{run}/{node}/{port}
  return entry.kind === 'weight'
    ? `__weight_grad__${entry.port}`
    : `${entry.port}__grad`;
}

async function fetchTensorWithFallback(
  runId: string,
  nodeId: string,
  port: string,
): Promise<OutputData> {
  try {
    return await fetchOutput(runId, nodeId, port);
  } catch (e) {
    if (e instanceof PayloadTooLargeError) {
      return await fetchOutput(runId, nodeId, port, {
        slice: '0,:,:',
        maxElements: 65536,
      });
    }
    throw e;
  }
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

  useEffect(() => {
    let cancelled = false;
    setEntries(null);
    setIndexError(null);
    setTensors({});
    fetchGradIndex(runId, nodeId)
      .then((es) => {
        if (cancelled) return;
        setEntries(es);
      })
      .catch((e) => {
        if (cancelled) return;
        setIndexError(
          e instanceof RunDataExpiredError
            ? 'run data expired — re-run with Backward to capture'
            : (e as Error).message,
        );
      });
    return () => {
      cancelled = true;
    };
  }, [runId, nodeId]);

  useEffect(() => {
    if (!entries || entries.length === 0) return;
    let cancelled = false;
    const initial: TensorMap = {};
    for (const e of entries) {
      initial[entryKey(e)] = { loading: true, error: null, data: null };
    }
    setTensors(initial);

    Promise.all(
      entries.map(async (e) => {
        const port = entryStorePort(e);
        try {
          const data = await fetchTensorWithFallback(runId, nodeId, port);
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
                  ? 'expired'
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
  }, [entries, runId, nodeId]);

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
        />
      )}
      {weightEntries.length > 0 && (
        <Section
          title={t('inspector.backward.weightSection')}
          entries={weightEntries}
          tensors={tensors}
        />
      )}
    </div>
  );
}

function Section({
  title,
  entries,
  tensors,
}: {
  title: string;
  entries: GradIndexEntry[];
  tensors: TensorMap;
}) {
  return (
    <div>
      <div className={styles.segmentSideTitle}>{title}</div>
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
          if (Array.isArray(grid)) {
            const row = (grid as unknown[])[i];
            if (Array.isArray(row)) {
              const v = (row as unknown[])[j];
              if (typeof v === 'number') cell = v;
            } else if (typeof row === 'number') {
              cell = row;
            }
          }
          return Math.min(1, Math.abs(cell) / maxAbs);
        };

        return (
          <div key={entryKey(e)} className={styles.stepTensor} style={{ marginTop: 8 }}>
            <div
              className={styles.stepTensorLabel}
              style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}
            >
              <span>{e.port}</span>
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
