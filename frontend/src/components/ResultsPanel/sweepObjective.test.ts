import { describe, expect, it } from 'vitest';
import { metricProducers, producerLabel, recordedMetricNames } from './sweepObjective';

describe('metricProducers', () => {
  it('unions every run\'s producers of one metric, run level first, then by id', () => {
    const runs: Array<{ metric_producers?: Record<string, Array<string | null>> }> = [
      { metric_producers: { loss: ['b', null, 'a'] } },
      { metric_producers: { loss: [null, 'c', 'a'], acc: ['z'] } },
      { metric_producers: {} },
      {},
    ];
    expect(metricProducers(runs, 'loss')).toEqual([null, 'a', 'b', 'c']);
    expect(metricProducers(runs, 'acc')).toEqual(['z']);
    expect(metricProducers(runs, 'missing')).toEqual([]);
    expect(metricProducers([{ metric_producers: { loss: ['a', null] } }], 'loss')).toEqual([null, 'a']);
  });
});

describe('recordedMetricNames', () => {
  it('reads names from the summary and from the producers', () => {
    expect(recordedMetricNames([
      { final_metrics: { loss: 1 }, metric_producers: { acc: ['n'] } },
      {} as never,
    ]).sort()).toEqual(['acc', 'loss']);
  });
});

describe('producerLabel', () => {
  const node = (id: string, data: Record<string, unknown>) => ({ id, data: data as never });

  it('names a node the graph can vouch for and keeps the id for one it cannot', () => {
    expect(producerLabel('t', [node('t', { label: 'Trainer', type: 'Train' })])).toBe('Trainer (t)');
    expect(producerLabel('t', [node('t', { label: '', type: 'Train' })])).toBe('Train (t)');
    expect(producerLabel('t', [node('t', { label: 't', type: 'Train' })])).toBe('t');
    expect(producerLabel('t', [node('t', {})])).toBe('t');
    expect(producerLabel('block/inner', [node('block', { label: 'Block' })])).toBe('block/inner');
    expect(producerLabel('t', [node('t', { label: 'A' }), node('t', { label: 'B' })])).toBe('t');
  });
});
