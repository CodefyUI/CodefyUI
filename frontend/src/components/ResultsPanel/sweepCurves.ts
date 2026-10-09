import type { RunMetrics, SweepDetail } from '../../api/rest';

export const MAX_SWEEP_CURVES = 32;
export const SWEEP_CURVE_CONCURRENCY = 4;

/** One child's objective series. Its legend label is the view's to word. */
export interface SweepCurve {
  runId: string;
  variantIndex: number;
  points: { x: number; y: number }[];
}

/**
 * Series of finished children, by run id, for one sweep. A run writes its
 * last metric before it files a terminal status, so a child the sweep read
 * reports as finished has a series that can no longer change.
 */
export type SweepCurveCache = Map<string, SweepCurve | null>;

const FINISHED = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);

/**
 * Turn one run's chosen objective series into the chart's common shape.
 *
 * Only the objective's producer is drawn (#641): the named node's points, or,
 * for a name-only objective, the one node that logged the name. A name
 * several nodes logged is no curve at all, as it is no rank: joining their
 * points would draw one line out of different measurements.
 */
export function objectiveCurve(
  variantIndex: number,
  metrics: RunMetrics,
  objectiveMetric: string,
  objectiveNode: string | null = null,
): SweepCurve | null {
  const series = metrics.metrics.filter((point) => point.name === objectiveMetric);
  const producers = new Set(series.map((point) => point.node_id ?? null));
  if (objectiveNode === null && producers.size > 1) return null;
  const points = series
    .filter((point) => (objectiveNode === null || point.node_id === objectiveNode)
      && point.value !== null
      && Number.isFinite(point.value))
    .map((point) => ({ x: point.step, y: point.value as number }))
    .sort((left, right) => left.x - right.x);
  if (points.length === 0) return null;
  return { runId: metrics.run_id, variantIndex, points };
}

/**
 * Load only real child runs, bounded both in total and in-flight requests.
 * A pruned run or missing metric is an absent line, never a fabricated zero.
 * With a `cache`, a finished child is read once and only live children are
 * read again; a queued child has logged nothing yet and is not read at all.
 */
export async function loadObjectiveCurves(
  detail: SweepDetail,
  fetchMetrics: (runId: string) => Promise<RunMetrics>,
  signal?: AbortSignal,
  cache?: SweepCurveCache,
): Promise<SweepCurve[]> {
  const candidates = detail.variants
    .filter((variant) => variant.run_exists === true && variant.run_id !== null)
    .slice(0, MAX_SWEEP_CURVES);
  let next = 0;
  const curves: Array<SweepCurve | null> = new Array(candidates.length).fill(null);

  async function worker(): Promise<void> {
    while (next < candidates.length) {
      if (signal?.aborted) return;
      const index = next++;
      const variant = candidates[index];
      const runId = variant.run_id!;
      if (cache?.has(runId)) {
        curves[index] = cache.get(runId) ?? null;
        continue;
      }
      if (variant.status === 'queued') continue;
      try {
        const metrics = await fetchMetrics(runId);
        if (signal?.aborted) return;
        curves[index] = objectiveCurve(
          variant.index,
          metrics,
          detail.objective.metric,
          detail.objective.node_id ?? null,
        );
        if (FINISHED.has(variant.status ?? '')) cache?.set(runId, curves[index]);
      } catch {
        // A child may be pruned between the sweep read and this request.
      }
    }
  }

  await Promise.all(Array.from(
    { length: Math.min(SWEEP_CURVE_CONCURRENCY, candidates.length) },
    () => worker(),
  ));
  if (signal?.aborted) return [];
  return curves.filter((curve): curve is SweepCurve => curve !== null);
}
