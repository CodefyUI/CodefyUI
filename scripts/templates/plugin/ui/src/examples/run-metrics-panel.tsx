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
 *     per point. It starts wherever you happened to subscribe: it is a tail,
 *     not a history.
 *   - `api.runs` is the HISTORY half. Read-only: `list()`, `get(id)`,
 *     `metrics(id)`. Use it to fill in everything that happened before you
 *     were listening, and to back-fill anything the live stream dropped.
 *
 * The pattern below is the one most dashboards want: subscribe first, then
 * back-fill, so no event can slip through the gap between the two.
 */
import React from 'react';
import { mountPanel, useExecutionEvents, useRuns } from '../sdk';
import type { CodefyUIPluginAPI, ExecutionEvent } from '../sdk';

interface Series {
  name: string;
  last: number;
  points: number;
}

/** Fold one event into the series table. Pure, so it is easy to test. */
function reduceEvent(
  series: Record<string, Series>,
  event: ExecutionEvent,
): Record<string, Series> {
  if (event.type !== 'metric') return series;
  const previous = series[event.name];
  return {
    ...series,
    [event.name]: {
      name: event.name,
      last: event.value,
      points: (previous?.points ?? 0) + 1,
    },
  };
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
    setSeries((current) => reduceEvent(current, event));
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
      setSeries((current) => {
        const merged = { ...current };
        for (const point of recorded.metrics) {
          if (point.value === null) continue;      // a diverged loss: a gap
          if (merged[point.name]) continue;        // the live tail wins
          merged[point.name] = {
            name: point.name, last: point.value, points: 1,
          };
        }
        return merged;
      });
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
            <tr><th>Series</th><th>Last</th><th>Points</th></tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td>{row.name}</td>
                <td>{row.last.toFixed(4)}</td>
                <td>{row.points}</td>
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
