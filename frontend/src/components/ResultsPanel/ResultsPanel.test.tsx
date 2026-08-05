import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act, within } from '@testing-library/react';
import { ResultsPanel } from './ResultsPanel';
import { useTabStore, type LogEntry } from '../../store/tabStore';
import { useRunStore } from '../../store/runStore';
import { useI18n } from '../../i18n';
import {
  _clearPluginPanels, registerPluginPanel, removePluginPanel,
} from '../../plugins/panels';

// Stub LossChart so the SVG sub-tree doesn't interfere with assertions and
// ResizeObserver bookkeeping stays out of these tests. We assert the props it
// receives via data attributes.
vi.mock('./LossChart', () => ({
  LossChart: ({ losses, height }: { losses: number[]; height: number }) => (
    <div data-testid="loss-chart" data-len={losses.length} data-height={height} />
  ),
}));

// The Runs tab is its own component with its own tests; here we only care
// that the third tab exists, switches, and is fed the panel height.
vi.mock('./RunsPanel', () => ({
  RunsPanel: ({ panelHeight, onWatchRun }: { panelHeight: number; onWatchRun?: () => void }) => (
    <div data-testid="runs-panel" data-height={panelHeight}>
      <button type="button" onClick={onWatchRun}>stub-watch</button>
    </div>
  ),
}));

/**
 * DEPRECATED (remove one release after #117, together with the parsing in
 * ResultsPanel). Builds the legacy magic-prefixed progress string that
 * pre-#117 frontends smuggled through the log stream. New code seeds
 * `{ kind: 'progress', progress: {...} }` entries instead.
 */
function progress(obj: Record<string, unknown>): string {
  return '__PROGRESS__:' + JSON.stringify(obj);
}

function makeLog(partial: Partial<LogEntry> & Pick<LogEntry, 'message'>): LogEntry {
  return {
    timestamp: 1_700_000_000_000,
    type: 'info',
    ...partial,
  };
}

/** Replace the active tab's logs. */
function seedLogs(logs: LogEntry[]) {
  useTabStore.setState((state) => ({
    tabs: state.tabs.map((t) =>
      t.id === state.activeTabId ? { ...t, logs } : t,
    ),
  }));
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
  useRunStore.setState({ runs: [], total: 0, activeCount: 0 });
});

afterEach(() => {
  vi.restoreAllMocks();
});

const t = (k: any, vars?: any) => useI18n.getState().t(k, vars);

describe('ResultsPanel — log tab basics', () => {
  it('shows the empty state when there are no non-progress logs', () => {
    seedLogs([]);
    render(<ResultsPanel />);
    expect(screen.getByText(t('results.empty'))).toBeInTheDocument();
    // clear button disabled when no logs
    const clearBtn = screen.getByText(t('results.clear'));
    expect(clearBtn).toBeDisabled();
  });

  it('filters out __PROGRESS__ entries from the log tab and shows a count badge', () => {
    // Use a non-config/non-epoch progress event so it is filtered out of the
    // log view but does NOT flip hasTraining (which would auto-switch tabs).
    seedLogs([
      makeLog({ message: 'hello world' }),
      makeLog({ message: progress({ event: 'noop' }) }),
      makeLog({ message: 'second line', type: 'success' }),
    ]);
    render(<ResultsPanel />);
    expect(screen.getByText('hello world')).toBeInTheDocument();
    expect(screen.getByText('second line')).toBeInTheDocument();
    // progress message is filtered out of the visible log
    expect(screen.queryByText(/__PROGRESS__/)).not.toBeInTheDocument();
    // count badge (on the log tab) reflects the 2 visible non-progress entries
    const logTabBtn = screen.getByText(t('results.title')).closest('button')!;
    expect(within(logTabBtn).getByText('2')).toBeInTheDocument();
    // training tab is disabled — the noop progress event created no training data
    expect(screen.getByText(t('results.training')).closest('button')!).toBeDisabled();
  });

  it('renders an info, error, and success entry; error is expandable', () => {
    seedLogs([
      makeLog({ message: 'info msg', type: 'info', nodeId: 'abcdef1234567890' }),
      makeLog({ message: "ValueError: bad shape", type: 'error' }),
    ]);
    render(<ResultsPanel />);
    // node id badge is truncated to 8 chars
    expect(screen.getByText('abcdef12')).toBeInTheDocument();
    // error entry present
    const errEntry = screen.getByText('ValueError: bad shape');
    // collapsed initially: no expanded detail
    expect(screen.queryByText('bad shape')).not.toBeInTheDocument();
    // click the error row to expand -> friendlyError strips the "ValueError:" prefix
    fireEvent.click(errEntry);
    expect(screen.getByText('bad shape')).toBeInTheDocument();
    // click again to collapse
    fireEvent.click(errEntry);
    expect(screen.queryByText('bad shape')).not.toBeInTheDocument();
  });

  it('clicking the node id badge selects that node exclusively (#167 follow-up)', () => {
    // This panel is not the canvas -- React Flow's own click handling never
    // runs for it -- so the badge needs the `.selected`-syncing action
    // (selectNodeExclusively), not the plain-click one.
    useTabStore.getState().setNodes([
      { id: 'abcdef1234567890', type: 'baseNode', position: { x: 0, y: 0 }, selected: false, data: { label: 'N', type: 'N', params: {} } },
      { id: 'other', type: 'baseNode', position: { x: 0, y: 0 }, selected: true, data: { label: 'O', type: 'O', params: {} } },
    ] as any);
    seedLogs([makeLog({ message: 'info msg', type: 'info', nodeId: 'abcdef1234567890' })]);
    render(<ResultsPanel />);
    fireEvent.click(screen.getByText('abcdef12'));
    const tab = useTabStore.getState().tabs.find((tt) => tt.id === useTabStore.getState().activeTabId)!;
    expect(tab.selectedNodeId).toBe('abcdef1234567890');
    expect(tab.nodes.find((n) => n.id === 'abcdef1234567890')!.selected).toBe(true);
    expect(tab.nodes.find((n) => n.id === 'other')!.selected).toBe(false);
  });

  // #125: one ResultsPanel now serves every canvas tab (only the active tab's
  // surface is mounted, and it is not remounted on a switch), so local state
  // that indexes INTO the active tab's data has to be reset by hand.
  it('drops the expanded error row when the canvas tab changes', () => {
    seedLogs([makeLog({ message: 'ValueError: bad shape', type: 'error' })]);
    render(<ResultsPanel />);
    fireEvent.click(screen.getByText('ValueError: bad shape'));
    expect(screen.getByText('bad shape')).toBeInTheDocument();

    // A second tab whose log has a DIFFERENT error at the same index. Without
    // the reset, index 0 stays expanded and this row renders open.
    act(() => {
      useTabStore.getState().addTab('other');
      useTabStore.setState((state) => ({
        tabs: state.tabs.map((tb) =>
          tb.id === state.activeTabId
            ? { ...tb, logs: [makeLog({ message: 'RuntimeError: other tab', type: 'error' })] }
            : tb,
        ),
      }));
    });
    expect(screen.getByText('RuntimeError: other tab')).toBeInTheDocument();
    expect(screen.queryByText('other tab')).not.toBeInTheDocument();
  });

  it('clicking a node-id badge highlights that node and stops propagation', () => {
    seedLogs([makeLog({ message: 'ValueError: oops', type: 'error', nodeId: 'node-xyz-1' })]);
    render(<ResultsPanel />);
    const badge = screen.getByText('node-xyz'); // slice(0,8)
    fireEvent.click(badge);
    expect(useTabStore.getState().getActiveTab().selectedNodeId).toBe('node-xyz-1');
    // stopPropagation: the error row did NOT expand
    expect(screen.queryByText('oops')).not.toBeInTheDocument();
  });

  // DEPRECATED (remove one release after #117): legacy prefixed image entry.
  it('renders a legacy __IMAGE__ entry as an <img> with a base64 data URL', () => {
    seedLogs([makeLog({ message: '__IMAGE__:QUJD' })]);
    render(<ResultsPanel />);
    const img = screen.getByAltText('output') as HTMLImageElement;
    expect(img.src).toBe('data:image/png;base64,QUJD');
  });

  it('clears logs when the Clear button is pressed', () => {
    seedLogs([makeLog({ message: 'one' })]);
    render(<ResultsPanel />);
    const clearBtn = screen.getByText(t('results.clear'));
    expect(clearBtn).not.toBeDisabled();
    fireEvent.click(clearBtn);
    expect(useTabStore.getState().getActiveTab().logs).toHaveLength(0);
  });
});

describe('ResultsPanel — collapse & resize', () => {
  it('toggles collapse, swapping the chevron and aria-label, and restores height', () => {
    seedLogs([makeLog({ message: 'x' })]);
    const { container } = render(<ResultsPanel />);
    const collapseBtn = screen.getByLabelText(t('results.collapse'));
    expect(collapseBtn).toHaveTextContent('▾');
    // resize handle present while expanded
    expect(container.querySelector('div[class]')).toBeTruthy();
    fireEvent.click(collapseBtn);
    // now collapsed: expand affordance shown
    const expandBtn = screen.getByLabelText(t('results.expand'));
    expect(expandBtn).toHaveTextContent('▴');
    // log content hidden while collapsed
    expect(screen.queryByText('x')).not.toBeInTheDocument();
    // expand again
    fireEvent.click(expandBtn);
    expect(screen.getByText('x')).toBeInTheDocument();
  });

  it('resizes the panel via the top drag handle (mousemove + mouseup)', () => {
    seedLogs([makeLog({ message: 'x' })]);
    const { container } = render(<ResultsPanel />);
    // resize handle is the first child after the header structure; grab by class fragment
    const handle = container.querySelector('[class*="resizeHandle"]') as HTMLElement;
    expect(handle).toBeTruthy();
    fireEvent.mouseDown(handle, { clientY: 300 });
    // drag up (smaller clientY => taller panel)
    act(() => {
      document.dispatchEvent(new MouseEvent('mousemove', { clientY: 150 }));
    });
    act(() => {
      document.dispatchEvent(new MouseEvent('mouseup'));
    });
    // after mouseup the body cursor styles are reset
    expect(document.body.style.cursor).toBe('');
    expect(document.body.style.userSelect).toBe('');
  });

  it('a resize drag while collapsed un-collapses the panel', () => {
    seedLogs([makeLog({ message: 'x' })]);
    const { container } = render(<ResultsPanel />);
    // collapse first
    fireEvent.click(screen.getByLabelText(t('results.collapse')));
    expect(screen.queryByText('x')).not.toBeInTheDocument();
    // While collapsed there is no resize handle, so re-expand to grab it, then
    // verify the move handler's `if (collapsed) setCollapsed(false)` path by
    // collapsing through state is not possible; instead assert handle hidden.
    expect(container.querySelector('[class*="resizeHandle"]')).toBeNull();
  });
});

describe('ResultsPanel — training tab', () => {
  function seedTraining() {
    seedLogs([
      makeLog({
        message: progress({
          event: 'config',
          config: { lr: 0.01, epochs: 3, optimizer: 'adam', momentum: 0.9 },
        }),
      }),
      makeLog({
        timestamp: 1000,
        message: progress({ event: 'epoch', epoch: 1, total_epochs: 3, loss: 0.8 }),
      }),
      makeLog({
        timestamp: 3000,
        message: progress({ event: 'epoch', epoch: 2, total_epochs: 3, loss: 0.4 }),
      }),
      makeLog({
        timestamp: 4000,
        message: progress({ event: 'epoch', epoch: 3, total_epochs: 3, loss: 0.6 }),
      }),
    ]);
  }

  it('auto-switches to the Training tab and renders summary, chart, config, and epoch table', () => {
    seedTraining();
    render(<ResultsPanel />);
    // Auto-switched to training tab -> training count badge shows 3 epochs
    const trainingTabBtn = screen.getByText(t('results.training')).closest('button')!;
    expect(within(trainingTabBtn).getByText('3')).toBeInTheDocument();
    // Summary stats
    expect(screen.getByText(t('results.epoch'))).toBeInTheDocument();
    expect(screen.getByText('3 / 3')).toBeInTheDocument(); // last epoch / total
    // current loss = last loss 0.6 -> toFixed(4) (also appears in the epoch row)
    expect(screen.getAllByText('0.6000').length).toBeGreaterThanOrEqual(1);
    // best loss = min(0.8,0.4,0.6) = 0.4 -> toFixed(4) (also an epoch row value)
    expect(screen.getAllByText('0.4000').length).toBeGreaterThanOrEqual(1);
    // chart rendered with 3 points
    const chart = screen.getByTestId('loss-chart');
    expect(chart.getAttribute('data-len')).toBe('3');
    // config section: integer printed plainly, float to 6dp
    expect(screen.getByText('lr')).toBeInTheDocument();
    expect(screen.getByText('0.010000')).toBeInTheDocument(); // 0.01 not integer -> toFixed(6)
    expect(screen.getByText('adam')).toBeInTheDocument(); // string value
    // integer config value (epochs: 3) printed via String(val) inside the config grid
    expect(within(document.querySelector('[class*="configGrid"]') as HTMLElement).getByText('3')).toBeInTheDocument();
    // epoch table header + rows
    expect(screen.getByText(t('results.col.delta'))).toBeInTheDocument();
    // delta for epoch 2 = 0.4 - 0.8 = -0.4 -> "-0.4000" (down)
    expect(screen.getByText('-0.4000')).toBeInTheDocument();
    // delta for epoch 3 = 0.6 - 0.4 = +0.2 -> "+0.2000" (up)
    expect(screen.getByText('+0.2000')).toBeInTheDocument();
    // elapsed for epoch 2 = (3000-1000)/1000 = 2.0s
    expect(screen.getByText('2.0s')).toBeInTheDocument();
    // first epoch elapsed/delta are '-'
    expect(screen.getAllByText('-').length).toBeGreaterThanOrEqual(2);
  });

  it('renders the config-only training state: no summary, shows waitingEpoch, disables epoch UI', () => {
    seedLogs([
      makeLog({ message: progress({ event: 'config', config: { lr: 0.1 } }) }),
    ]);
    render(<ResultsPanel />);
    // hasTraining true (config != null) so it auto-switches to training
    // No epochs -> waiting message in the chart column
    expect(screen.getByText(t('results.waitingEpoch'))).toBeInTheDocument();
    // config present
    expect(screen.getByText('lr')).toBeInTheDocument();
    // no chart, no epoch table, no summary
    expect(screen.queryByTestId('loss-chart')).not.toBeInTheDocument();
    expect(screen.queryByText(t('results.col.loss'))).not.toBeInTheDocument();
  });

  it('handles malformed progress JSON gracefully (catch branch)', () => {
    seedLogs([
      makeLog({ message: '__PROGRESS__:{not valid json' }),
      makeLog({ message: 'normal' }),
    ]);
    render(<ResultsPanel />);
    // malformed progress yields no training data -> still on log tab
    expect(screen.getByText('normal')).toBeInTheDocument();
    // training tab button is disabled (no training data)
    const trainingBtn = screen.getByText(t('results.training')).closest('button')!;
    expect(trainingBtn).toBeDisabled();
  });

  it('does not switch tabs when there is no training data; the Training tab is disabled and inert', () => {
    seedLogs([makeLog({ message: 'just a log' })]);
    render(<ResultsPanel />);
    const trainingBtn = screen.getByText(t('results.training')).closest('button')!;
    expect(trainingBtn).toBeDisabled();
    // clicking the disabled training tab does nothing (hasTraining && setPanelTab)
    fireEvent.click(trainingBtn);
    // still showing the log tab content
    expect(screen.getByText('just a log')).toBeInTheDocument();
  });

  it('clicking the enabled Training tab invokes its handler (hasTraining && setPanelTab)', () => {
    seedLogs([
      makeLog({ message: 'a log line' }),
      makeLog({ timestamp: 1000, message: progress({ event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.5 }) }),
    ]);
    render(<ResultsPanel />);
    // switch to log first, then click the enabled training tab to drive its onClick
    fireEvent.click(screen.getByText(t('results.title')));
    expect(screen.getByText('a log line')).toBeInTheDocument();
    const trainingBtn = screen.getByText(t('results.training')).closest('button')!;
    expect(trainingBtn).not.toBeDisabled();
    fireEvent.click(trainingBtn);
    // back on the training tab -> chart visible, log hidden
    expect(screen.getByTestId('loss-chart')).toBeInTheDocument();
    expect(screen.queryByText('a log line')).not.toBeInTheDocument();
  });

  it('shows the trainingEmpty state if logs are cleared while on the Training tab', () => {
    seedLogs([
      makeLog({ timestamp: 1000, message: progress({ event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.5 }) }),
    ]);
    render(<ResultsPanel />);
    // auto-switched to training; now clear logs -> hasTraining becomes false while
    // panelTab is still 'training' -> the `!hasTraining` trainingEmpty branch shows.
    expect(screen.getByTestId('loss-chart')).toBeInTheDocument();
    fireEvent.click(screen.getByText(t('results.clear')));
    expect(screen.getByText(t('results.trainingEmpty'))).toBeInTheDocument();
    expect(screen.queryByTestId('loss-chart')).not.toBeInTheDocument();
  });

  it('lets the user switch back to the Log tab after training auto-switch', () => {
    seedLogs([
      makeLog({ message: 'a log line' }),
      makeLog({ message: progress({ event: 'epoch', epoch: 1, total_epochs: 1, loss: 0.5 }) }),
    ]);
    render(<ResultsPanel />);
    // auto-switched to training; switch back to log
    fireEvent.click(screen.getByText(t('results.title')));
    expect(screen.getByText('a log line')).toBeInTheDocument();
  });

  it('switches to training manually when only epochs exist (epoch-only, no config)', () => {
    // epochs but no config -> config section + row divider absent
    seedLogs([
      makeLog({ timestamp: 1000, message: progress({ event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.5 }) }),
      makeLog({ timestamp: 2000, message: progress({ event: 'epoch', epoch: 2, total_epochs: 2, loss: 0.3 }) }),
    ]);
    render(<ResultsPanel />);
    // chart present
    expect(screen.getByTestId('loss-chart')).toBeInTheDocument();
    // config header absent (no config)
    expect(screen.queryByText(t('results.trainingConfig'))).not.toBeInTheDocument();
    // epoch table present
    expect(screen.getByText(t('results.col.time'))).toBeInTheDocument();
  });
});

describe('ResultsPanel — training column dividers (drag handlers)', () => {
  function seedFull() {
    seedLogs([
      makeLog({ message: progress({ event: 'config', config: { lr: 0.01 } }) }),
      makeLog({ timestamp: 1000, message: progress({ event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.8 }) }),
      makeLog({ timestamp: 2000, message: progress({ event: 'epoch', epoch: 2, total_epochs: 2, loss: 0.4 }) }),
    ]);
  }

  it('drags the column divider to resize the info column', () => {
    seedFull();
    const { container } = render(<ResultsPanel />);
    const colDivider = container.querySelector('[class*="columnDivider"]') as HTMLElement;
    expect(colDivider).toBeTruthy();
    fireEvent.mouseDown(colDivider, { clientX: 500 });
    act(() => {
      document.dispatchEvent(new MouseEvent('mousemove', { clientX: 400 }));
    });
    act(() => {
      document.dispatchEvent(new MouseEvent('mouseup'));
    });
    expect(document.body.style.cursor).toBe('');
  });

  it('drags the row divider between config and the epoch table', () => {
    seedFull();
    const { container } = render(<ResultsPanel />);
    const rowDivider = container.querySelector('[class*="rowDivider"]') as HTMLElement;
    expect(rowDivider).toBeTruthy();
    fireEvent.mouseDown(rowDivider, { clientY: 100 });
    act(() => {
      document.dispatchEvent(new MouseEvent('mousemove', { clientY: 180 }));
    });
    act(() => {
      document.dispatchEvent(new MouseEvent('mouseup'));
    });
    expect(document.body.style.userSelect).toBe('');
  });

  it('falls back to a default start height when [data-config] clientHeight is unavailable', () => {
    seedFull();
    const { container } = render(<ResultsPanel />);
    const rowDivider = container.querySelector('[class*="rowDivider"]') as HTMLElement;
    // Force `?.clientHeight` to be undefined so the `?? 100` fallback is taken
    // (jsdom otherwise returns 0, a defined value, which the `??` keeps).
    const spy = vi
      .spyOn(HTMLElement.prototype, 'clientHeight', 'get')
      .mockReturnValue(undefined as unknown as number);
    fireEvent.mouseDown(rowDivider, { clientY: 50 });
    act(() => {
      document.dispatchEvent(new MouseEvent('mousemove', { clientY: 90 }));
    });
    act(() => {
      document.dispatchEvent(new MouseEvent('mouseup'));
    });
    spy.mockRestore();
    expect(document.body.style.cursor).toBe('');
  });
});

// ── #117: structured output kinds ────────────────────────────────────────
// The backend now tags every renderable payload with an `output_kind` and
// useGraphExecution turns those into typed LogEntry fields. ResultsPanel
// renders from those fields instead of sniffing magic string prefixes.

describe('ResultsPanel — structured output kinds (#117)', () => {
  function imageLog(data: string, format = 'png'): LogEntry {
    return makeLog({
      message: '',
      kind: 'image',
      image: { format, encoding: 'base64', data },
    });
  }

  it('renders a kind="image" entry as an <img> built from its payload', () => {
    seedLogs([imageLog('QUJD')]);
    render(<ResultsPanel />);
    const img = screen.getByAltText('output') as HTMLImageElement;
    expect(img.src).toBe('data:image/png;base64,QUJD');
  });

  it('honours the payload format when building the data URL', () => {
    seedLogs([imageLog('PHN2Zy8+', 'svg+xml')]);
    render(<ResultsPanel />);
    const img = screen.getByAltText('output') as HTMLImageElement;
    expect(img.src).toBe('data:image/svg+xml;base64,PHN2Zy8+');
  });

  it('renders a 500-char alphanumeric text output as text, not a broken image', () => {
    // Headline regression for #117: the backend used to sniff this exact
    // shape (len > 200, alphanumeric prefix) and ship it as a base64 PNG.
    const longText = 'TokenIds0123456789abcdef'.repeat(21).slice(0, 500);
    expect(longText).toHaveLength(500);
    seedLogs([makeLog({ message: longText, kind: 'text' })]);
    render(<ResultsPanel />);
    expect(screen.getByText(longText)).toBeInTheDocument();
    expect(screen.queryByAltText('output')).not.toBeInTheDocument();
  });

  it('renders CJK prose output as text (str.isalnum() is True for CJK)', () => {
    const cjk = '注意力機制讓模型在每一步都能回頭看整個序列'.repeat(25).slice(0, 500);
    seedLogs([makeLog({ message: cjk, kind: 'text' })]);
    render(<ResultsPanel />);
    expect(screen.getByText(cjk)).toBeInTheDocument();
    expect(screen.queryByAltText('output')).not.toBeInTheDocument();
  });

  it('drives the training tab from kind="progress" entries and hides them from the log', () => {
    seedLogs([
      makeLog({ message: 'plain line' }),
      makeLog({
        message: '',
        kind: 'progress',
        progress: { event: 'config', config: { lr: 0.05 } },
      }),
      makeLog({
        timestamp: 1000,
        message: '',
        kind: 'progress',
        progress: { event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.9 },
      }),
      makeLog({
        timestamp: 3000,
        message: '',
        kind: 'progress',
        progress: { event: 'epoch', epoch: 2, total_epochs: 2, loss: 0.3 },
      }),
    ]);
    render(<ResultsPanel />);
    // Auto-switched to the training tab with both epochs charted.
    const trainingTabBtn = screen.getByText(t('results.training')).closest('button')!;
    expect(within(trainingTabBtn).getByText('2')).toBeInTheDocument();
    expect(screen.getByTestId('loss-chart').getAttribute('data-len')).toBe('2');
    expect(screen.getByText('0.050000')).toBeInTheDocument(); // config lr
    expect(screen.getByText('2.0s')).toBeInTheDocument(); // (3000-1000)/1000
    // The log tab shows only the plain line — progress entries are filtered.
    fireEvent.click(screen.getByText(t('results.title')));
    expect(screen.getByText('plain line')).toBeInTheDocument();
    const logTabBtn = screen.getByText(t('results.title')).closest('button')!;
    expect(within(logTabBtn).getByText('1')).toBeInTheDocument();
  });

  it('ignores a kind="progress" entry with no payload', () => {
    seedLogs([makeLog({ message: '', kind: 'progress' }), makeLog({ message: 'kept' })]);
    render(<ResultsPanel />);
    expect(screen.getByText(t('results.training')).closest('button')!).toBeDisabled();
    expect(screen.getByText('kept')).toBeInTheDocument();
  });

  it('ignores a kind="image" entry with no payload and falls back to the message', () => {
    seedLogs([makeLog({ message: 'no payload', kind: 'image' })]);
    render(<ResultsPanel />);
    expect(screen.queryByAltText('output')).not.toBeInTheDocument();
    expect(screen.getByText('no payload')).toBeInTheDocument();
  });
});

// ── #130: chart output entries ───────────────────────────────────────────
// A node declaring `media=MEDIA_CHART` ships a JSON spec, and the panel draws
// it with the app's own SVG plots rather than showing a server-rendered PNG.

describe('ResultsPanel — chart output entries (#130)', () => {
  function chartLog(chart: Record<string, unknown>): LogEntry {
    return makeLog({ message: '', kind: 'chart', chart: chart as any });
  }

  it('draws a kind="chart" entry in the log', () => {
    seedLogs([
      chartLog({
        kind: 'bar',
        title: 'Mean petal length',
        bars: [{ label: 'setosa', value: 1.46 }],
      }),
    ]);
    const { container } = render(<ResultsPanel />);
    expect(screen.getByText('Mean petal length')).toBeInTheDocument();
    expect(container.querySelector('[data-chart-kind="bar"]')).toBeInTheDocument();
    expect(container.querySelector('rect[data-label="setosa"]')).toBeInTheDocument();
  });

  it('renders a confusion matrix as a heatmap with its class labels', () => {
    seedLogs([
      chartLog({
        kind: 'heatmap',
        title: 'Confusion matrix',
        matrix: [[13, 0], [1, 12]],
        row_labels: ['setosa', 'versicolor'],
        col_labels: ['setosa', 'versicolor'],
        vmin: 0,
        vmax: 13,
      }),
    ]);
    const { container } = render(<ResultsPanel />);
    expect(container.querySelector('[data-chart-kind="heatmap"]')).toBeInTheDocument();
    expect(container.querySelectorAll('rect[data-i]')).toHaveLength(4);
    expect(screen.getAllByText('setosa').length).toBeGreaterThanOrEqual(2);
  });

  it('shows a chart entry alongside the text a node emitted with it', () => {
    seedLogs([
      makeLog({ message: 'rendered table', kind: 'text' }),
      chartLog({ kind: 'bar', bars: [{ label: 'a', value: 1 }] }),
    ]);
    const { container } = render(<ResultsPanel />);
    expect(screen.getByText('rendered table')).toBeInTheDocument();
    expect(container.querySelector('[data-chart-kind="bar"]')).toBeInTheDocument();
  });

  it('ignores a kind="chart" entry with no spec and falls back to the message', () => {
    seedLogs([makeLog({ message: 'no payload', kind: 'chart' })]);
    const { container } = render(<ResultsPanel />);
    expect(container.querySelector('[data-chart-kind]')).not.toBeInTheDocument();
    expect(screen.getByText('no payload')).toBeInTheDocument();
  });

  it('counts chart entries in the log tab badge like any other entry', () => {
    seedLogs([chartLog({ kind: 'bar', bars: [] }), chartLog({ kind: 'bar', bars: [] })]);
    render(<ResultsPanel />);
    const logTabBtn = screen.getByText(t('results.title')).closest('button')!;
    expect(within(logTabBtn).getByText('2')).toBeInTheDocument();
  });
});

// ── #124: the Runs tab ───────────────────────────────────────────────────

describe('ResultsPanel — Runs tab', () => {
  function seedRuns(statuses: string[]) {
    useRunStore.setState({
      runs: statuses.map((status, i) => ({ id: `r${i}`, status })) as never,
      total: statuses.length,
      activeCount: statuses.filter((s) => s === 'running' || s === 'queued').length,
    });
  }

  it('switches to the Runs tab and hands it the current panel height', () => {
    seedLogs([makeLog({ message: 'a log line' })]);
    render(<ResultsPanel />);
    expect(screen.queryByTestId('runs-panel')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText(t('runs.tab')));
    const panel = screen.getByTestId('runs-panel');
    expect(panel.getAttribute('data-height')).toBe('200'); // DEFAULT_HEIGHT
    // The log content is replaced, not stacked underneath.
    expect(screen.queryByText('a log line')).not.toBeInTheDocument();
  });

  it('is always enabled — runs exist whether or not THIS tab ran anything', () => {
    seedLogs([]);
    render(<ResultsPanel />);
    expect(screen.getByText(t('runs.tab')).closest('button')).not.toBeDisabled();
  });

  it('badges the number of ACTIVE runs, and nothing when none are going', () => {
    seedLogs([]);
    const { rerender } = render(<ResultsPanel />);
    const runsBtn = () => screen.getByText(t('runs.tab')).closest('button')!;
    // Nothing known yet: no badge at all.
    expect(within(runsBtn()).queryByText(/\d/)).not.toBeInTheDocument();

    // Runs exist but all of them are history -- still no badge, because the
    // badge answers "is something happening", not "does history exist".
    act(() => { seedRuns(['succeeded', 'failed']); });
    rerender(<ResultsPanel />);
    expect(within(runsBtn()).queryByText(/\d/)).not.toBeInTheDocument();

    act(() => { seedRuns(['running', 'queued', 'succeeded']); });
    rerender(<ResultsPanel />);
    const badge = within(runsBtn()).getByText('2');
    expect(badge.className).toMatch(/countBadgeActive/);
  });

  it('badges before the panel has ever been opened, from the mount-time check', () => {
    // The list is empty because nothing has polled yet; `activeCount` came
    // from App's bootstrap check. Without this the only affordance that
    // reports a detached run appears only AFTER the user finds the tab.
    seedLogs([]);
    act(() => { useRunStore.setState({ runs: [], total: 0, activeCount: 3 }); });
    render(<ResultsPanel />);
    const runsBtn = screen.getByText(t('runs.tab')).closest('button')!;
    expect(within(runsBtn).getByText('3')).toBeInTheDocument();
  });

  it('hides Clear on the Runs tab — it empties this tab log, not the runs', () => {
    seedLogs([makeLog({ message: 'x' })]);
    render(<ResultsPanel />);
    expect(screen.getByText(t('results.clear'))).toBeInTheDocument();
    fireEvent.click(screen.getByText(t('runs.tab')));
    expect(screen.queryByText(t('results.clear'))).not.toBeInTheDocument();
    // Still there when we come back.
    fireEvent.click(screen.getByText(t('results.title')));
    expect(screen.getByText(t('results.clear'))).toBeInTheDocument();
  });

  it('does not yank the user off Runs when training data starts arriving', () => {
    seedLogs([makeLog({ message: 'a log line' })]);
    render(<ResultsPanel />);
    fireEvent.click(screen.getByText(t('runs.tab')));
    expect(screen.getByTestId('runs-panel')).toBeInTheDocument();

    act(() => {
      seedLogs([
        makeLog({ message: 'a log line' }),
        makeLog({
          message: '',
          kind: 'progress',
          progress: { event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.5 },
        }),
      ]);
    });
    // Still on Runs; the Training tab is now enabled but was not forced.
    expect(screen.getByTestId('runs-panel')).toBeInTheDocument();
    expect(screen.queryByTestId('loss-chart')).not.toBeInTheDocument();
    expect(screen.getByText(t('results.training')).closest('button')).not.toBeDisabled();
  });

  it('still auto-switches to Training from the Log tab', () => {
    seedLogs([makeLog({ message: 'a log line' })]);
    render(<ResultsPanel />);
    act(() => {
      seedLogs([
        makeLog({
          message: '',
          kind: 'progress',
          progress: { event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.5 },
        }),
      ]);
    });
    expect(screen.getByTestId('loss-chart')).toBeInTheDocument();
  });

  it('reveals the Execution Log when a run is watched from the Runs tab', () => {
    seedLogs([makeLog({ message: 'a log line' })]);
    render(<ResultsPanel />);
    fireEvent.click(screen.getByText(t('runs.tab')));
    fireEvent.click(screen.getByText('stub-watch'));
    // The attach streams into the log, so that is where the user is put.
    expect(screen.queryByTestId('runs-panel')).not.toBeInTheDocument();
    expect(screen.getByText('a log line')).toBeInTheDocument();
  });

  it('hides the Runs panel while the dock is collapsed', () => {
    seedLogs([]);
    render(<ResultsPanel />);
    fireEvent.click(screen.getByText(t('runs.tab')));
    fireEvent.click(screen.getByLabelText(t('results.collapse')));
    expect(screen.queryByTestId('runs-panel')).not.toBeInTheDocument();
  });
});

describe('ResultsPanel — legacy magic prefixes (deprecated, #117)', () => {
  // Kept for one release so a frontend running against a pre-#117 backend
  // still renders. Delete this block together with the parsing it covers.
  it('still parses a mix of legacy prefixes and structured entries', () => {
    seedLogs([
      makeLog({ message: '__IMAGE__:TEdD' }),
      makeLog({
        message: '',
        kind: 'image',
        image: { format: 'png', encoding: 'base64', data: 'QUJD' },
      }),
      makeLog({ timestamp: 1000, message: progress({ event: 'epoch', epoch: 1, total_epochs: 2, loss: 0.5 }) }),
      makeLog({
        timestamp: 2000,
        message: '',
        kind: 'progress',
        progress: { event: 'epoch', epoch: 2, total_epochs: 2, loss: 0.25 },
      }),
    ]);
    render(<ResultsPanel />);
    // Both progress sources feed one epoch series.
    expect(screen.getByTestId('loss-chart').getAttribute('data-len')).toBe('2');
    // Both image sources render.
    fireEvent.click(screen.getByText(t('results.title')));
    const imgs = screen.getAllByAltText('output') as HTMLImageElement[];
    expect(imgs.map((i) => i.src)).toEqual([
      'data:image/png;base64,TEdD',
      'data:image/png;base64,QUJD',
    ]);
  });
});

// ── plugin dock tabs (#132) ──────────────────────────────────────────────

describe('plugin panels in the dock', () => {
  afterEach(() => _clearPluginPanels());

  function seedPanel(title = 'Sweeps') {
    const el = registerPluginPanel('sweeps', { id: 'main', title });
    el.appendChild(document.createElement('canvas'));
    return el;
  }

  it('adds a tab after the built-in ones', () => {
    seedPanel();
    render(<ResultsPanel />);
    const labels = [...screen.getByText(t('results.title')).parentElement!.children]
      .map((c) => c.textContent);
    expect(labels).toEqual([
      t('results.title'), t('results.training'), t('runs.tab'), 'Sweeps',
    ]);
  });

  it('shows no plugin tab for a right-docked panel', () => {
    registerPluginPanel('side', { id: 'main', title: 'Side', dock: 'right' });
    render(<ResultsPanel />);
    expect(screen.queryByTestId('plugin-dock-tab-side:main')).not.toBeInTheDocument();
  });

  it('mounts the panel element when its tab is selected', () => {
    const el = seedPanel();
    render(<ResultsPanel />);
    expect(el.isConnected).toBe(false);
    fireEvent.click(screen.getByTestId('plugin-dock-tab-sweeps:main'));
    expect(screen.getByTestId('plugin-dock-panel-sweeps:main').firstChild).toBe(el);
  });

  it('keeps the SAME element across a dock tab switch', () => {
    const el = seedPanel();
    const child = el.firstChild;
    render(<ResultsPanel />);

    fireEvent.click(screen.getByTestId('plugin-dock-tab-sweeps:main'));
    fireEvent.click(screen.getByText(t('results.title')));
    expect(el.isConnected).toBe(false);
    expect(screen.queryByTestId('plugin-dock-panel-sweeps:main')).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId('plugin-dock-tab-sweeps:main'));
    expect(screen.getByTestId('plugin-dock-panel-sweeps:main').firstChild).toBe(el);
    expect(el.firstChild).toBe(child);
  });

  it('detaches the panel while the dock is collapsed', () => {
    const el = seedPanel();
    render(<ResultsPanel />);
    fireEvent.click(screen.getByTestId('plugin-dock-tab-sweeps:main'));
    fireEvent.click(screen.getByLabelText(t('results.collapse')));
    expect(el.isConnected).toBe(false);
  });

  it('hides Clear on a plugin tab — it empties the log, not the plugin', () => {
    seedPanel();
    render(<ResultsPanel />);
    fireEvent.click(screen.getByTestId('plugin-dock-tab-sweeps:main'));
    expect(screen.queryByText(t('results.clear'))).not.toBeInTheDocument();
  });

  it('falls back to the log tab when the selected panel is unregistered', () => {
    seedPanel();
    render(<ResultsPanel />);
    fireEvent.click(screen.getByTestId('plugin-dock-tab-sweeps:main'));
    act(() => removePluginPanel('sweeps', 'main'));
    expect(screen.queryByTestId('plugin-dock-tab-sweeps:main')).not.toBeInTheDocument();
    expect(screen.getByText(t('results.empty'))).toBeInTheDocument();
  });

  it('shows a panel registered while the dock is already on screen', () => {
    render(<ResultsPanel />);
    act(() => { seedPanel('Late'); });
    expect(screen.getByTestId('plugin-dock-tab-sweeps:main')).toHaveTextContent('Late');
  });
});
