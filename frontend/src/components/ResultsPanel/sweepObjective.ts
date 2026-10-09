import type { RunSummary } from '../../api/rest';
import type { AppNode } from '../../types';

/**
 * Every producer of `metric` the listed runs recorded (#641): node ids,
 * sorted, with `null` (the run-level series, no node) first. This is the
 * evidence a sweep objective is chosen from -- a run's `final_metrics`
 * collapses a name several nodes log, `metric_producers` keeps them apart.
 */
export function metricProducers(
  runs: readonly Pick<RunSummary, 'metric_producers'>[],
  metric: string,
): Array<string | null> {
  const found = new Set<string | null>();
  for (const run of runs) {
    for (const producer of run.metric_producers?.[metric] ?? []) found.add(producer);
  }
  // A Set holds no duplicates, so two entries are never equal here.
  return [...found].sort((left, right) => {
    if (left === null) return -1;
    if (right === null) return 1;
    return left.localeCompare(right);
  });
}

/** Every series name the listed runs recorded, from either field. */
export function recordedMetricNames(runs: readonly Pick<RunSummary, 'final_metrics' | 'metric_producers'>[]): string[] {
  const names = new Set<string>();
  for (const run of runs) {
    for (const name of Object.keys(run.final_metrics ?? {})) names.add(name);
    for (const name of Object.keys(run.metric_producers ?? {})) names.add(name);
  }
  return [...names];
}

/**
 * A node id as the open graph names it: its label with the id after it, or
 * the id alone for a node the graph cannot vouch for -- an inner node of a
 * block logs under its flattened id (`block/inner`), and a sweep may come
 * from another graph.
 */
export function producerLabel(
  nodeId: string,
  nodes: readonly Pick<AppNode, 'id' | 'data'>[],
): string {
  const matches = nodes.filter((node) => node.id === nodeId);
  if (matches.length !== 1) return nodeId;
  const { data } = matches[0];
  const name = data.label || data.type;
  return name && name !== nodeId ? `${name} (${nodeId})` : nodeId;
}
