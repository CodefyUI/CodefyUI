import { describe, expect, it, vi } from 'vitest';
import type { RunMetrics, SweepDetail } from '../../api/rest';
import {
  loadObjectiveCurves,
  objectiveCurve,
} from './sweepCurves';

function detail(count: number): SweepDetail {
  return {
    sweep_id: 's1', name: 'Search', state: 'finished', method: 'grid',
    seed: null, seed_variants: false,
    objective: { metric: 'loss', direction: 'minimize' },
    created_at: '2026-10-01T00:00:00Z', finished_at: null, error: null,
    counts: { queued: 0, running: 0, succeeded: count, failed: 0, cancelled: 0, interrupted: 0, missing: 0 },
    params: [], best: null,
    variants: Array.from({ length: count }, (_, index) => ({
      index, domain_index: index, run_id: `r${index}`, status: 'succeeded' as const,
      params: [], seed: null, objective: index, rank: index + 1,
      run_exists: true, final_metrics: { loss: index },
    })),
  };
}

function metrics(runId: string, values: Array<number | null>): RunMetrics {
  return {
    run_id: runId,
    names: ['loss', 'accuracy'],
    metrics: values.map((value, index) => ({
      node_id: 'train', name: 'loss', step: index + 1, value,
    })).concat([{ node_id: 'train', name: 'accuracy', step: 1, value: 0.9 }]),
  };
}

describe('objectiveCurve', () => {
  it('keeps only finite objective points in step order', () => {
    expect(objectiveCurve(2, metrics('r2', [0.7, null, 0.4]), 'loss')).toEqual({
      runId: 'r2',
      variantIndex: 2,
      name: 'Variant 3',
      points: [{ x: 1, y: 0.7 }, { x: 3, y: 0.4 }],
    });
  });

  it('returns null when the objective series has no points', () => {
    expect(objectiveCurve(0, metrics('r0', []), 'loss')).toBeNull();
  });
});

describe('loadObjectiveCurves', () => {
  it('skips deleted children, failed requests and empty series', async () => {
    const sweep = detail(4);
    sweep.variants[1].run_exists = false;
    const fetcher = vi.fn(async (runId: string) => {
      if (runId === 'r2') throw new Error('gone');
      return runId === 'r3' ? metrics(runId, []) : metrics(runId, [1, 0.5]);
    });

    const curves = await loadObjectiveCurves(sweep, fetcher);

    expect(fetcher.mock.calls.map(([id]) => id)).toEqual(['r0', 'r2', 'r3']);
    expect(curves.map((curve) => curve.runId)).toEqual(['r0']);
  });

  it('fetches at most 32 runs with at most four requests in flight', async () => {
    let active = 0;
    let peak = 0;
    const releases: Array<() => void> = [];
    const fetcher = vi.fn((runId: string) => new Promise<RunMetrics>((resolve) => {
      active += 1;
      peak = Math.max(peak, active);
      releases.push(() => {
        active -= 1;
        resolve(metrics(runId, [1]));
      });
    }));

    const loading = loadObjectiveCurves(detail(40), fetcher);
    await Promise.resolve();
    expect(fetcher).toHaveBeenCalledTimes(4);
    while (releases.length) {
      releases.shift()!();
      await Promise.resolve();
      await Promise.resolve();
    }
    const curves = await loading;

    expect(fetcher).toHaveBeenCalledTimes(32);
    expect(peak).toBeLessThanOrEqual(4);
    expect(curves).toHaveLength(32);
  });
});
