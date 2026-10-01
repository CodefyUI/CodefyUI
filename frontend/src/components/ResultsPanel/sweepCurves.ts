import type { RunMetrics, SweepDetail } from '../../api/rest';
import type { ChartSeries } from './LossChart';

export const MAX_SWEEP_CURVES = 32;
export const SWEEP_CURVE_CONCURRENCY = 4;

export interface SweepCurve extends ChartSeries {
  runId: string;
  variantIndex: number;
}

/** Turn one run's chosen objective series into the chart's common shape. */
export function objectiveCurve(
  variantIndex: number,
  metrics: RunMetrics,
  objectiveMetric: string,
): SweepCurve | null {
  const points = metrics.metrics
    .filter((point) => point.name === objectiveMetric
      && point.value !== null
      && Number.isFinite(point.value))
    .map((point) => ({ x: point.step, y: point.value as number }))
    .sort((left, right) => left.x - right.x);
  if (points.length === 0) return null;
  return {
    runId: metrics.run_id,
    variantIndex,
    name: `Variant ${variantIndex + 1}`,
    points,
  };
}

/**
 * Load only real child runs, bounded both in total and in-flight requests.
 * A pruned run or missing metric is an absent line, never a fabricated zero.
 */
export async function loadObjectiveCurves(
  detail: SweepDetail,
  fetchMetrics: (runId: string) => Promise<RunMetrics>,
  signal?: AbortSignal,
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
      try {
        const metrics = await fetchMetrics(variant.run_id!);
        if (signal?.aborted) return;
        curves[index] = objectiveCurve(
          variant.index,
          metrics,
          detail.objective.metric,
        );
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
