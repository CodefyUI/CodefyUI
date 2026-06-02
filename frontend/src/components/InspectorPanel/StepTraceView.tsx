import { useEffect, useState } from 'react';
import {
  fetchOutput,
  fetchStepIndex,
  PayloadTooLargeError,
  RunDataExpiredError,
  type StepIndexEntry,
} from '../../api/executionOutputs';
import type { OutputData, TensorOutput } from '../../types';
import { TensorGridView } from './TensorGridView';
import { MathText } from '../shared/MathText';
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

type TensorMap = Record<string, TensorState>; // key: `${stepIdx}::${tensorName}`

function tkey(stepIdx: number, tensorName: string): string {
  return `${stepIdx}::${tensorName}`;
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

/**
 * Render the algorithmic step trace for a single node in a single run.
 * Each step card shows the step name, its description, optional scalar
 * parameters, and a TensorGridView for every tensor recorded.
 */
export function StepTraceView({ runId, nodeId }: Props) {
  const { t } = useI18n();
  const [steps, setSteps] = useState<StepIndexEntry[] | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [tensors, setTensors] = useState<TensorMap>({});
  const [collapsed, setCollapsed] = useState<Record<number, boolean>>({});

  // Fetch the step index for this (run, node) pair. The parent remounts this
  // component (via a `key` on runId:nodeId), so each mount starts from fresh
  // state — no manual reset needed, and the user never sees the prior node's
  // trace flash before the new fetch resolves.
  useEffect(() => {
    let cancelled = false;
    fetchStepIndex(runId, nodeId)
      .then((entries) => {
        if (cancelled) return;
        setSteps(entries);
        // Seed loading placeholders for every tensor the next effect fetches.
        const initial: TensorMap = {};
        for (const step of entries) {
          for (const name of step.tensor_keys) {
            initial[tkey(step.index, name)] = {
              loading: true,
              error: null,
              data: null,
            };
          }
        }
        setTensors(initial);
      })
      .catch((e) => {
        if (cancelled) return;
        setIndexError(
          e instanceof RunDataExpiredError
            ? 'run data expired — re-run to capture'
            : (e as Error).message,
        );
      });
    return () => {
      cancelled = true;
    };
  }, [runId, nodeId]);

  // After we have the step index, fetch each tensor in parallel. The loading
  // placeholders were already seeded alongside setSteps above.
  useEffect(() => {
    if (!steps || steps.length === 0) return;
    let cancelled = false;
    const tasks: Promise<void>[] = [];
    for (const step of steps) {
      for (const name of step.tensor_keys) {
        const port = `__step__${step.index}__${name}`;
        tasks.push(
          (async () => {
            try {
              const data = await fetchTensorWithFallback(runId, nodeId, port);
              if (cancelled) return;
              setTensors((prev) => ({
                ...prev,
                [tkey(step.index, name)]: {
                  loading: false,
                  error: null,
                  data,
                },
              }));
            } catch (e) {
              if (cancelled) return;
              setTensors((prev) => ({
                ...prev,
                [tkey(step.index, name)]: {
                  loading: false,
                  error:
                    e instanceof RunDataExpiredError
                      ? 'expired'
                      : (e as Error).message,
                  data: null,
                },
              }));
            }
          })(),
        );
      }
    }
    Promise.all(tasks);
    return () => {
      cancelled = true;
    };
  }, [steps, runId, nodeId]);

  if (indexError) {
    return <div className={styles.portError}>{indexError}</div>;
  }

  if (steps === null) {
    return <div className={styles.diffMissing}>…</div>;
  }

  if (steps.length === 0) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyIcon}>{'{ƒ}'}</div>
        <div>{t('inspector.steps.empty')}</div>
        <div className={styles.emptyHint}>{t('inspector.steps.requireVerbose')}</div>
      </div>
    );
  }

  return (
    <div className={styles.stepList}>
      {steps.map((step) => {
        const isCollapsed = collapsed[step.index] ?? false;
        return (
          <div key={step.index} className={styles.stepCard}>
            <button
              type="button"
              className={styles.stepHeader}
              onClick={() =>
                setCollapsed((prev) => ({
                  ...prev,
                  [step.index]: !isCollapsed,
                }))
              }
            >
              <span className={styles.stepIndex}>{step.index + 1}.</span>
              <span className={styles.stepName}>{step.name}</span>
              <span className={styles.stepCaret}>{isCollapsed ? '▸' : '▾'}</span>
            </button>
            {!isCollapsed && (
              <div className={styles.stepBody}>
                {step.description && (
                  <MathText
                    as="div"
                    className={styles.stepDescription}
                    text={step.description}
                  />
                )}
                {Object.keys(step.scalars).length > 0 && (
                  <div className={styles.stepScalars}>
                    {Object.entries(step.scalars).map(([k, v]) => (
                      <span key={k} className={styles.stepScalarChip}>
                        {k} = {formatScalar(v)}
                      </span>
                    ))}
                  </div>
                )}
                {step.tensor_keys.length === 0 && (
                  <div className={styles.diffMissing}>(no tensors)</div>
                )}
                {step.tensor_keys.map((name) => {
                  const state = tensors[tkey(step.index, name)];
                  return (
                    <div key={name} className={styles.stepTensor}>
                      <div className={styles.stepTensorLabel}>{name}</div>
                      {state?.error && (
                        <div className={styles.portError}>{state.error}</div>
                      )}
                      {state?.loading && (
                        <div className={styles.diffMissing}>…</div>
                      )}
                      {state?.data && state.data.type === 'tensor' && (
                        <TensorGridView tensor={state.data as TensorOutput} />
                      )}
                      {state?.data && state.data.type !== 'tensor' && (
                        <div className={styles.tensorScalar}>
                          {state.data.type === 'scalar' &&
                            String((state.data as { value?: unknown }).value)}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function formatScalar(v: number): string {
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(2);
  return v.toFixed(4);
}
