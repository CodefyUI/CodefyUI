/**
 * Export as Python refused because the graph fails validation.
 *
 * The export route refuses such a graph with the server's English sentence,
 * node ids and all, and the toolbar showed it as an error toast that never
 * timed out and stayed across tab switches. The toolbar now asks the check
 * Run makes and shows its findings the way Run does (utils/validationToasts):
 * in the UI language, naming nodes by title, with Show, cleared by the next
 * Run or export and by a tab switch. The refusal itself is shown only when
 * that check finds nothing or cannot be reached.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Toolbar } from './Toolbar';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import { _resetDeviceOptionsForTesting } from '../../hooks/useDeviceOptions';
import { dismissValidationToasts } from '../../utils/validationToasts';

vi.mock('../../hooks/useGraphExecution', async (importOriginal) => importOriginal());

vi.mock('../../api/rest', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/rest')>()),
  exportGraph: vi.fn(),
  validateGraph: vi.fn(),
  getRun: vi.fn(() => Promise.resolve(null)),
  fetchDevices: vi.fn(() =>
    Promise.resolve({
      default: 'cpu',
      devices: [{ value: 'cpu', label: 'CPU', detail: '', available: true }],
    }),
  ),
}));

const mockedRest = vi.mocked(rest);

const START = '0a9b8c7d-6e5f-4a3b-2c1d-0e9f8a7b6c5d';
const CARD = '755eff7c-1b2a-4c3d-8e9f-0a1b2c3d4e5f';
const REFUSAL =
  `Export failed: Node ${CARD} is triggered, but preset 'T7Empty' has no node to start: it has no nodes`;

/** What `/api/graph/validate` answers for the same graph. */
const INVALID: Awaited<ReturnType<typeof rest.validateGraph>> = {
  valid: false,
  errors: [REFUSAL.slice('Export failed: '.length)],
  issues: [{
    message: REFUSAL.slice('Export failed: '.length),
    code: 'preset_triggered_empty',
    node_id: CARD,
    params: { preset: 'T7Empty' },
  }],
};

function setCanvas() {
  // The store's own first tab, so every field a TabState needs is present.
  // Bound to a saved graph, so an export downloads without a name question.
  const real = useTabStore.getState().tabs[0];
  const tab = (id: string, name: string, nodes: unknown[]) =>
    ({
      ...real, id, name, nodes, edges: [], subgraphs: [], subgraphStack: [],
      currentGraphFile: name, currentGraphName: name, lastRunId: null,
    }) as never;
  useTabStore.setState({
    tabs: [
      tab('tab-1', 'Exam', [
        { id: START, type: 'baseNode', position: { x: 0, y: 0 },
          data: { label: 'Start', type: 'Start', params: {} } },
        { id: CARD, type: 'presetNode', position: { x: 300, y: 0 },
          data: { label: 'Empty card', type: 'preset:T7Empty', params: {} } },
      ]),
      tab('tab-2', 'Other', []),
    ],
    activeTabId: 'tab-1',
  });
}

function clickExportPython() {
  fireEvent.click(screen.getByText('Export'));
  fireEvent.click(screen.getByText('Export as Python'));
}

const errorToasts = () => useToastStore.getState().toasts.filter((toast) => toast.type === 'error');

describe('Toolbar Export as Python: a refused graph', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useToastStore.setState({ toasts: [] });
    setCanvas();
    _resetDeviceOptionsForTesting();
    mockedRest.exportGraph.mockReset();
    mockedRest.validateGraph.mockReset();
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  });

  afterEach(() => {
    dismissValidationToasts();
    vi.restoreAllMocks();
    useToastStore.setState({ toasts: [] });
  });

  it("says what Run's check found, naming the node by its title, with Show", async () => {
    mockedRest.exportGraph.mockRejectedValueOnce(new Error(REFUSAL));
    mockedRest.validateGraph.mockResolvedValueOnce(INVALID);
    render(<Toolbar />);

    clickExportPython();

    await waitFor(() => expect(errorToasts()).toHaveLength(1));
    expect(errorToasts()[0].message).toBe('Empty card is triggered, but preset "T7Empty" has no nodes');
    expect(errorToasts()[0].action?.label).toBe('Show');
    // The same graph the export sent.
    const [nodes, edges] = mockedRest.exportGraph.mock.calls[0];
    expect(mockedRest.validateGraph).toHaveBeenCalledWith(nodes, edges, [], []);
  });

  it('says it in Chinese when the UI is', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    mockedRest.exportGraph.mockRejectedValueOnce(new Error(REFUSAL));
    mockedRest.validateGraph.mockResolvedValueOnce(INVALID);
    render(<Toolbar />);

    fireEvent.click(screen.getByText('匯出'));
    fireEvent.click(screen.getByText('匯出為 Python'));

    await waitFor(() => expect(errorToasts()).toHaveLength(1));
    expect(errorToasts()[0].message).toBe('「Empty card」收到 trigger，但預設模組「T7Empty」裡沒有節點');
    expect(errorToasts()[0].action?.label).toBe('顯示');
  });

  it('shows the refusal when the check finds nothing', async () => {
    mockedRest.exportGraph.mockRejectedValueOnce(new Error('Export failed: server down'));
    mockedRest.validateGraph.mockResolvedValueOnce({ valid: true, errors: [], issues: [] });
    render(<Toolbar />);

    clickExportPython();

    await waitFor(() => expect(errorToasts()).toHaveLength(1));
    expect(errorToasts()[0].message).toBe('Python export failed: Export failed: server down');
  });

  it('shows the refusal when the check cannot be reached', async () => {
    mockedRest.exportGraph.mockRejectedValueOnce(new Error(REFUSAL));
    mockedRest.validateGraph.mockRejectedValueOnce(new Error('Validation failed'));
    render(<Toolbar />);

    clickExportPython();

    await waitFor(() => expect(errorToasts()).toHaveLength(1));
    expect(errorToasts()[0].message).toBe(`Python export failed: ${REFUSAL}`);
  });

  it('is taken down by the next export', async () => {
    mockedRest.exportGraph
      .mockRejectedValueOnce(new Error(REFUSAL))
      .mockResolvedValueOnce({ script: 'print(1)', warnings: [] });
    mockedRest.validateGraph.mockResolvedValueOnce(INVALID);
    render(<Toolbar />);
    clickExportPython();
    await waitFor(() => expect(errorToasts()).toHaveLength(1));

    clickExportPython();

    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(errorToasts()).toEqual([]));
  });

  it('is taken down when another tab comes to the front', async () => {
    mockedRest.exportGraph.mockRejectedValueOnce(new Error(REFUSAL));
    mockedRest.validateGraph.mockResolvedValueOnce(INVALID);
    render(<Toolbar />);
    clickExportPython();
    await waitFor(() => expect(errorToasts()).toHaveLength(1));

    await act(async () => {
      useTabStore.getState().setActiveTab('tab-2');
    });

    expect(errorToasts()).toEqual([]);
  });

  it('never lands on the tab the user switched to while the export was refused', async () => {
    let refuse!: (error: Error) => void;
    mockedRest.exportGraph.mockReturnValueOnce(
      new Promise((_resolve, reject) => { refuse = reject; }),
    );
    mockedRest.validateGraph.mockResolvedValue(INVALID);
    render(<Toolbar />);
    clickExportPython();
    await waitFor(() => expect(mockedRest.exportGraph).toHaveBeenCalledTimes(1));

    await act(async () => {
      useTabStore.getState().setActiveTab('tab-2');
    });
    await act(async () => {
      refuse(new Error(REFUSAL));
    });

    expect(errorToasts()).toEqual([]);
  });
});
