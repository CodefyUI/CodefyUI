/**
 * EXAMPLE — a live run-metrics dashboard in a bottom dock tab.
 *
 * Nothing imports this file by default. Copy it, or wire it up from
 * `src/index.tsx`:
 *
 * ```tsx
 * import { mountRunMetricsPanel } from './examples/run-metrics-panel';
 *
 * export default function activate(api: CodefyUIPluginAPI) {
 *   mountRunMetricsPanel(api);
 * }
 * ```
 *
 * It shows the two halves of the apiVersion 3 run surface working together,
 * and which one to reach for:
 *
 *   - `api.events.onExecution` is the LIVE half. It streams what is happening
 *     right now — batched onto animation frames, so a training run pushing
 *     hundreds of metrics a second costs you one render per frame, not one
 *     per point. The editor re-attaching to a run cannot hand you the same
 *     entry twice, and a hole in `event.seq` — NOT in `event.cursor` — is the
 *     signal that events were dropped. (`cursor` skips durable entries this
 *     stream does not publish, an artifact for every saved checkpoint among
 *     them, so cursor gaps are ordinary and mean nothing.)
 *   - `api.runs` is the HISTORY half. Read-only: `list()`, `get(id)`,
 *     `metrics(id)`. Use it to fill in everything that happened before you
 *     were listening, and to recover anything a `seq` hole says the live
 *     stream dropped.
 *
 * The pattern below is the one most dashboards want: subscribe first, then
 * back-fill, so no event can slip through the gap between the two.
 */
import React from 'react';
import { mountPanel, useExecutionEvents, useRuns } from '../sdk';
import type { CodefyUIPluginAPI, RunMetricPoint } from '../sdk';

interface Series {
  name: string;
  /** Last finite value seen; null until one arrives. */
  last: number | null;
  /** How many points were recorded, gaps included. */
  points: number;
  /** How many of them were non-finite. */
  gaps: number;
}

/**
 * Fold recorded points into the series table.
 *
 * It takes `RunMetricPoint[]`, which is what BOTH halves of the run surface
 * hand you: `event.points` from the live tail, and `runs.metrics().metrics`
 * from the REST back-fill. One function serves both — that is the reason the
 * live `metric` event carries its batch instead of one event per point.
 *
 * `value: null` is a diverged (non-finite) number: a gap in the curve, not a
 * zero. Draw a break, do not fold it into an average.
 */
function foldPoints(
  series: Record<string, Series>,
  points: readonly RunMetricPoint[],
): Record<string, Series> {
  // One pass, one new object — not one spread per point.
  const next = { ...series };
  for (const point of points) {
    const previous = next[point.name];
    next[point.name] = {
      name: point.name,
      last: point.value ?? previous?.last ?? null,
      gaps: (previous?.gaps ?? 0) + (point.value === null ? 1 : 0),
      points: (previous?.points ?? 0) + 1,
    };
  }
  return next;
}

function RunMetricsPanel() {
  const runs = useRuns();
  const [runId, setRunId] = React.useState<string | null>(null);
  const [status, setStatus] = React.useState<string>('idle');
  const [series, setSeries] = React.useState<Record<string, Series>>({});

  // 1. The live tail. Subscribing happens on mount, before the back-fill
  //    below runs, so nothing arrives in between and gets lost.
  useExecutionEvents((event) => {
    if (event.type === 'run_started') {
      setRunId(event.run_id);
      setStatus('running');
      setSeries({});
      return;
    }
    if (event.type === 'run_finished') {
      setStatus(event.status);
      return;
    }
    if (event.type === 'metric') {
      setSeries((current) => foldPoints(current, event.points));
    }
  });

  // 2. The back-fill. On mount there may already be a run in flight that
  //    started before this panel existed; ask the REST side for it, and for
  //    the metrics recorded so far.
  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      const page = await runs.list({ status: ['running'], limit: 1 });
      const active = page.runs[0];
      if (cancelled || !active) return;
      setRunId(active.id);
      setStatus(active.status);
      const recorded = await runs.metrics(active.id);
      if (cancelled) return;
      // The same fold as the live tail, because it is the same type. Only
      // series the tail has not already covered are back-filled.
      setSeries((current) => foldPoints(
        current,
        recorded.metrics.filter((point) => !current[point.name]),
      ));
    })();
    return () => { cancelled = true; };
  }, [runs]);

  const rows = Object.values(series).sort((a, b) => a.name.localeCompare(b.name));

  return (
    <div className="cdui-metrics">
      <div className="cdui-metrics__head">
        <b>{runId ?? 'no run yet'}</b>
        <span className="cdui-metrics__status">{status}</span>
      </div>
      {rows.length === 0 ? (
        <div className="cdui-metrics__empty">
          Run a graph that records metrics and they appear here live.
        </div>
      ) : (
        <table className="cdui-metrics__table">
          <thead>
            <tr><th>Series</th><th>Last</th><th>Points</th><th>Gaps</th></tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td>{row.name}</td>
                <td>{row.last === null ? '--' : row.last.toFixed(4)}</td>
                <td>{row.points}</td>
                <td>{row.gaps}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/** Add the panel as a tab in the editor's bottom dock. */
export function mountRunMetricsPanel(api: CodefyUIPluginAPI): void {
  if (api.apiVersion < 3) {
    api.ui.toast('This panel needs CodefyUI 1.5 or newer', 'warning');
    return;
  }
  mountPanel(
    api,
    { id: 'run-metrics', title: 'Run Metrics', icon: '~' },
    RunMetricsPanel,
  );
}
