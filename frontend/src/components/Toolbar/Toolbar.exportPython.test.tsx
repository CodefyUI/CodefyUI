/**
 * Export as Python names the file params that tie the script to this
 * computer (#557).
 *
 * The export route decides which params hold an absolute path and answers
 * them in `warnings` beside the script. The toolbar still downloads the
 * script, then raises one warning naming each node and its path, so a student
 * can fix the path before handing the file in.
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

vi.mock('../../hooks/useGraphExecution', () => ({
  useGraphExecution: () => ({ execute: vi.fn(), stop: vi.fn() }),
}));

vi.mock('../../api/rest', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/rest')>()),
  exportGraph: vi.fn(),
  fetchDevices: vi.fn(() =>
    Promise.resolve({
      default: 'cpu',
      devices: [{ value: 'cpu', label: 'CPU', detail: '', available: true }],
    }),
  ),
}));

const mockedRest = vi.mocked(rest);

const WINDOWS_PATH = 'C:\\Users\\student01\\Desktop\\grades.csv';

function absolutePath(overrides: Partial<rest.ExportWarning> = {}): rest.ExportWarning {
  return {
    code: 'absolute_path',
    node_id: 'csv',
    label: 'CSVReader',
    param: 'path',
    value: WINDOWS_PATH,
    containers: [],
    ...overrides,
  };
}

const realAddToast = useToastStore.getState().addToast;
let downloads: string[] = [];

function setCanvas() {
  // The store's own first tab, so every field a TabState needs is present.
  const real = useTabStore.getState().tabs[0];
  const tab = (id: string, name: string, nodes: unknown[]) =>
    ({ ...real, id, name, nodes, edges: [], subgraphs: [], subgraphStack: [] }) as never;
  useTabStore.setState({
    tabs: [
      tab('tab-1', 'Exam', [
        {
          id: 'csv',
          type: 'baseNode',
          position: { x: 0, y: 0 },
          data: { type: 'CSVReader', params: { path: WINDOWS_PATH } },
        },
      ]),
      tab('tab-2', 'Other', []),
    ],
    activeTabId: 'tab-1',
  });
}

/** Opens the Export menu and picks Export as Python, without waiting. */
function clickExportPython(menu = 'Export', item = 'Export as Python') {
  fireEvent.click(screen.getByText(menu));
  fireEvent.click(screen.getByText(item));
}

async function exportPython(menu = 'Export', item = 'Export as Python') {
  const calls = mockedRest.exportGraph.mock.calls.length;
  clickExportPython(menu, item);
  await waitFor(() => expect(mockedRest.exportGraph.mock.calls.length).toBe(calls + 1));
  await waitFor(() => expect(downloads).toHaveLength(calls + 1));
}

/** An export the test answers when it chooses to. */
function pendingExport() {
  let answer!: (result: rest.ExportResult) => void;
  const promise = new Promise<rest.ExportResult>((resolve) => {
    answer = resolve;
  });
  mockedRest.exportGraph.mockReturnValueOnce(promise);
  return (result: rest.ExportResult) => act(async () => answer(result));
}

async function switchTab(id: string) {
  await act(async () => {
    useTabStore.setState({ activeTabId: id });
  });
}

const warningToasts = () =>
  useToastStore.getState().toasts.filter((toast) => toast.type === 'warning');

describe('Toolbar Export as Python: absolute file paths', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useToastStore.setState({ toasts: [], addToast: realAddToast });
    setCanvas();
    _resetDeviceOptionsForTesting();
    mockedRest.exportGraph.mockReset();
    downloads = [];
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      downloads.push(this.download);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    useToastStore.setState({ toasts: [], addToast: realAddToast });
  });

  it('downloads the script, then names the node and its path', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      warnings: [absolutePath()],
    });
    render(<Toolbar />);

    await exportPython();

    expect(downloads).toEqual(['Exam.py']);
    const blob = vi.mocked(URL.createObjectURL).mock.calls[0][0] as Blob;
    expect(blob.type).toBe('text/x-python');
    await waitFor(() => expect(warningToasts()).toHaveLength(1));
    expect(warningToasts()[0].message).toBe(
      `Absolute paths work only on this computer: CSVReader (${WINDOWS_PATH}). `
        + 'Use just the file name instead.',
    );
  });

  // A warning that times out after four seconds is gone before the user is
  // back from the browser's download prompt.
  it('keeps the warning on screen until it is dismissed', async () => {
    const addToast = vi.fn(realAddToast);
    useToastStore.setState({ addToast });
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      warnings: [absolutePath()],
    });
    const { unmount } = render(<Toolbar />);

    await exportPython();

    await waitFor(() => expect(addToast).toHaveBeenCalledTimes(1));
    expect(addToast).toHaveBeenCalledWith(expect.any(String), 'warning', { sticky: true });
    // Before afterEach puts the real `addToast` back: a mounted toolbar
    // re-renders on that swap, outside act().
    unmount();
  });

  it('names a node inside blocks and presets by the levels it sits in, outermost first', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      warnings: [
        absolutePath({
          node_id: 'blk/p1__csv',
          containers: [
            { node_id: 'blk', label: 'Loader' },
            { node_id: 'blk/p1', label: 'Grade Loader' },
          ],
        }),
      ],
    });
    render(<Toolbar />);

    await exportPython();

    await waitFor(() => expect(warningToasts()).toHaveLength(1));
    expect(warningToasts()[0].message).toContain(
      `: Loader ▸ Grade Loader ▸ CSVReader (${WINDOWS_PATH}).`,
    );
  });

  it('names three, counts the rest, and lists a repeated entry once', async () => {
    const at = (label: string, node_id = label) =>
      absolutePath({ label, node_id, value: `/srv/${label}.csv` });
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      // Two instances of one block read the same file: one fact, said once.
      warnings: [at('A', 'one/A'), at('A', 'two/A'), at('B'), at('C'), at('D'), at('E')],
    });
    render(<Toolbar />);

    await exportPython();

    await waitFor(() => expect(warningToasts()).toHaveLength(1));
    expect(warningToasts()[0].message).toContain(
      ': A (/srv/A.csv), B (/srv/B.csv), C (/srv/C.csv), and 2 more.',
    );
  });

  it.each([
    ['an older server that sends no warnings', { script: 'print(1)' }],
    ['nothing to warn about', { script: 'print(1)', warnings: [] }],
    [
      'only a warning code this build does not know',
      { script: 'print(1)', warnings: [absolutePath({ code: 'later_code' })] },
    ],
  ])('downloads and raises no warning for %s', async (_case, result) => {
    mockedRest.exportGraph.mockResolvedValueOnce(result as rest.ExportResult);
    render(<Toolbar />);

    await exportPython();

    expect(downloads).toEqual(['Exam.py']);
    expect(warningToasts()).toEqual([]);
  });

  it('takes the previous warning down on the next export', async () => {
    mockedRest.exportGraph
      .mockResolvedValueOnce({ script: 'print(1)', warnings: [absolutePath()] })
      .mockResolvedValueOnce({ script: 'print(2)', warnings: [] });
    render(<Toolbar />);

    await exportPython();
    await waitFor(() => expect(warningToasts()).toHaveLength(1));
    await exportPython();

    expect(downloads).toEqual(['Exam.py', 'Exam.py']);
    expect(warningToasts()).toEqual([]);
  });

  it("speaks the user's language", async () => {
    useI18n.setState({ locale: 'zh-TW' });
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      warnings: ['A', 'B', 'C', 'D'].map((label) =>
        absolutePath({ label, node_id: label, value: `/srv/${label}.csv` }),
      ),
    });
    render(<Toolbar />);

    await exportPython('匯出', '匯出為 Python');

    await waitFor(() => expect(warningToasts()).toHaveLength(1));
    expect(warningToasts()[0].message).toBe(
      '絕對路徑只在這台電腦有效：A (/srv/A.csv), B (/srv/B.csv), C (/srv/C.csv), 另外 1 個。請只用檔名。',
    );
  });

  // App mounts the toolbar only while a tab is open. Closing the last tab
  // unmounts it; the warning must not stay up on the welcome screen, nor
  // survive into the toolbar the next tab brings back.
  it('takes the warning down with the toolbar, and a later clean export raises none', async () => {
    mockedRest.exportGraph
      .mockResolvedValueOnce({ script: 'print(1)', warnings: [absolutePath()] })
      .mockResolvedValueOnce({ script: 'print(2)', warnings: [] });
    const { unmount } = render(<Toolbar />);
    await exportPython();
    await waitFor(() => expect(warningToasts()).toHaveLength(1));

    unmount();
    expect(warningToasts()).toEqual([]);

    render(<Toolbar />);
    await exportPython();
    expect(warningToasts()).toEqual([]);
  });

  // The user fixes the path and exports again: whatever that export ends in,
  // the old warning names a path that is no longer there.
  it('takes the warning down when the next export fails', async () => {
    mockedRest.exportGraph
      .mockResolvedValueOnce({ script: 'print(1)', warnings: [absolutePath()] })
      .mockRejectedValueOnce(new Error('Export failed: server down'));
    render(<Toolbar />);
    await exportPython();
    await waitFor(() => expect(warningToasts()).toHaveLength(1));

    clickExportPython();

    await waitFor(() =>
      expect(useToastStore.getState().toasts.map((toast) => toast.type)).toEqual(['error']),
    );
  });

  it('takes the warning down when the next export finds an empty canvas', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      warnings: [absolutePath()],
    });
    render(<Toolbar />);
    await exportPython();
    await waitFor(() => expect(warningToasts()).toHaveLength(1));
    await act(async () => {
      useTabStore.setState((s) => ({
        tabs: s.tabs.map((tab) => (tab.id === 'tab-1' ? { ...tab, nodes: [] } : tab)),
      }));
    });

    clickExportPython();

    expect(warningToasts().map((toast) => toast.message)).toEqual([
      'Canvas has no executable nodes — add a node before exporting.',
    ]);
  });

  it('two overlapping exports leave one warning', async () => {
    const answerFirst = pendingExport();
    const answerSecond = pendingExport();
    render(<Toolbar />);
    clickExportPython();
    clickExportPython();

    await answerFirst({ script: 'print(1)', warnings: [absolutePath({ label: 'First' })] });
    await answerSecond({ script: 'print(2)', warnings: [absolutePath({ label: 'Second' })] });

    await waitFor(() => expect(downloads).toEqual(['Exam.py', 'Exam.py']));
    expect(warningToasts().map((toast) => toast.message)).toEqual([
      expect.stringContaining(`Second (${WINDOWS_PATH})`),
    ]);
  });

  // The warning does not say which graph it is about, so it is shown only
  // while that graph's tab is the one in front.
  it('takes the warning down when another tab comes to the front', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      warnings: [absolutePath()],
    });
    render(<Toolbar />);
    await exportPython();
    await waitFor(() => expect(warningToasts()).toHaveLength(1));

    await switchTab('tab-2');

    expect(warningToasts()).toEqual([]);
  });

  it('an export answered after a tab switch still downloads, without the warning', async () => {
    const answer = pendingExport();
    render(<Toolbar />);
    clickExportPython();
    await switchTab('tab-2');

    await answer({ script: 'print(1)', warnings: [absolutePath()] });

    await waitFor(() => expect(downloads).toEqual(['Exam.py']));
    expect(warningToasts()).toEqual([]);
  });

  // The script has downloaded by the time the warning is read, so a warning
  // that cannot be read is dropped; it must not turn the export into a
  // failure.
  it.each([
    ['warnings that are not a list', { warnings: {} }],
    ['containers that are not a list', { warnings: [{ ...absolutePath(), containers: {} }] }],
    ['a container with no label', { warnings: [{ ...absolutePath(), containers: [{ node_id: 'blk' }] }] }],
    ['a value that is not text', { warnings: [{ ...absolutePath(), value: 42 }] }],
    ['an entry that is not an object', { warnings: [null] }],
  ])('drops %s and still reports a successful export', async (_case, body) => {
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      ...body,
    } as unknown as rest.ExportResult);
    render(<Toolbar />);

    await exportPython();

    expect(downloads).toEqual(['Exam.py']);
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it('names the readable entries beside a malformed one', async () => {
    mockedRest.exportGraph.mockResolvedValueOnce({
      script: 'print(1)',
      warnings: [{ ...absolutePath(), containers: {} }, absolutePath({ label: 'Readable' })],
    } as unknown as rest.ExportResult);
    render(<Toolbar />);

    await exportPython();

    await waitFor(() => expect(warningToasts()).toHaveLength(1));
    expect(warningToasts()[0].message).toContain(`: Readable (${WINDOWS_PATH}).`);
    expect(warningToasts()[0].message).not.toContain('CSVReader');
  });
});
