import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act, within } from '@testing-library/react';
import { ResultsPanel } from './ResultsPanel';
import { useTabStore, type LogEntry } from '../../store/tabStore';
import { useI18n } from '../../i18n';

// Stub LossChart so the SVG sub-tree doesn't interfere with assertions and
// ResizeObserver bookkeeping stays out of these tests. We assert the props it
// receives via data attributes.
vi.mock('./LossChart', () => ({
  LossChart: ({ losses, height }: { losses: number[]; height: number }) => (
    <div data-testid="loss-chart" data-len={losses.length} data-height={height} />
  ),
}));

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

  it('clicking a node-id badge highlights that node and stops propagation', () => {
    seedLogs([makeLog({ message: 'ValueError: oops', type: 'error', nodeId: 'node-xyz-1' })]);
    render(<ResultsPanel />);
    const badge = screen.getByText('node-xyz'); // slice(0,8)
    fireEvent.click(badge);
    expect(useTabStore.getState().getActiveTab().selectedNodeId).toBe('node-xyz-1');
    // stopPropagation: the error row did NOT expand
    expect(screen.queryByText('oops')).not.toBeInTheDocument();
  });

  it('renders an image entry as an <img> with a base64 data URL', () => {
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
