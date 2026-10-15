import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { drainWorker, fetchWorkers, undrainWorker, type WorkerInfo } from '../../api/workers';
import { fetchHealth } from '../../api/rest';
import { useI18n } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { _resetWorkerStoreForTesting, useWorkerStore } from '../../store/workerStore';
import { settle } from '../../test/utils';
import { FINGERPRINT_SHORT_LENGTH, WorkersPanel, formatAge } from './WorkersPanel';
import { trapTab } from './focusTrap';

vi.mock('../../api/workers', () => ({
  fetchWorkers: vi.fn(),
  drainWorker: vi.fn(),
  undrainWorker: vi.fn(),
}));

vi.mock('../../api/rest', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/rest')>()),
  fetchHealth: vi.fn(),
}));

const NOW_S = 1_800_000_000;
const DIGEST = 'abcdef0123456789abcdef0123456789';

function worker(id: string, extra: Partial<WorkerInfo> = {}): WorkerInfo {
  return {
    id,
    name: id,
    status: 'online',
    draining: false,
    version: '3.0.2',
    fingerprint_digest: null,
    platform: {},
    mounts: {},
    devices: [],
    running: [],
    connected_at: NOW_S - 100,
    last_heartbeat_at: NOW_S - 4,
    ...extra,
  };
}

async function openPanel(workers: WorkerInfo[] = [worker('gpu-box')]) {
  vi.mocked(fetchWorkers).mockResolvedValue(workers);
  render(<WorkersPanel />);
  act(() => useWorkerStore.getState().openPanel());
  await settle();
  return screen.getByRole('dialog', { name: 'Workers' });
}

const t = useI18n.getState().t;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetWorkerStoreForTesting();
  vi.spyOn(Date, 'now').mockReturnValue(NOW_S * 1000);
  vi.mocked(fetchWorkers).mockReset().mockResolvedValue([]);
  vi.mocked(fetchHealth).mockReset().mockResolvedValue({ version: '3.0.2' } as never);
  vi.mocked(drainWorker).mockReset();
  vi.mocked(undrainWorker).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('formatAge', () => {
  it('reads seconds, minutes and hours, and never a negative age', () => {
    const now = NOW_S * 1000;
    expect(formatAge(NOW_S - 12, t, now)).toBe('12 s ago');
    expect(formatAge(NOW_S - 185, t, now)).toBe('3 min ago');
    expect(formatAge(NOW_S - 2 * 3600 - 5, t, now)).toBe('2 h ago');
    expect(formatAge(NOW_S + 30, t, now)).toBe('0 s ago');
  });

  it('defaults to the current time', () => {
    expect(formatAge(NOW_S - 7, t)).toBe('7 s ago');
  });
});

describe('trapTab', () => {
  it('does nothing without a container or anything focusable in it', () => {
    const event = { key: 'Tab', shiftKey: false, preventDefault: vi.fn() };
    trapTab(event as never, null);
    trapTab(event as never, document.createElement('div'));
    expect(event.preventDefault).not.toHaveBeenCalled();
  });
});

describe('WorkersPanel', () => {
  it('renders nothing while closed', () => {
    const { container } = render(<WorkersPanel />);
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('shows the loading line until the first read answers', async () => {
    let answer: (value: WorkerInfo[]) => void = () => {};
    vi.mocked(fetchWorkers).mockImplementation(() => new Promise((resolve) => { answer = resolve; }));
    render(<WorkersPanel />);
    act(() => useWorkerStore.getState().openPanel());
    expect(screen.getByText('Loading workers...')).toBeInTheDocument();
    await act(async () => answer([]));
    expect(screen.getByText(/No worker agent has connected/)).toBeInTheDocument();
  });

  it('shows an error instead of the empty line when the read fails', async () => {
    vi.mocked(fetchWorkers).mockRejectedValue(new Error('worker registry is not available'));
    render(<WorkersPanel />);
    act(() => useWorkerStore.getState().openPanel());
    await settle();
    expect(screen.getByRole('alert')).toHaveTextContent('worker registry is not available');
    expect(screen.queryByText(/No worker agent has connected/)).toBeNull();
  });

  it('describes a worker: status, version check, fingerprint, mounts, heartbeat and devices', async () => {
    const dialog = await openPanel([
      worker('gpu-box', {
        fingerprint_digest: DIGEST,
        mounts: { data: '/mnt/nas/data' },
        running: ['run-aaaaaaaa-1111'],
        devices: [
          {
            value: 'cuda:0', kind: 'cuda', index: 0, name: 'RTX 4090',
            memory: { total_bytes: 24 * 1024 ** 3, free_bytes: 20 * 1024 ** 3, source: 'nvml' },
            slots: { limit: 1, taken: 1 },
          },
          {
            value: 'cpu', kind: 'cpu', index: null, name: 'cpu',
            memory: { total_bytes: null, free_bytes: 1024, source: 'psutil' },
          },
          { value: 'mps', kind: 'mps', index: 0, name: '', memory: null },
        ],
      }),
    ]);
    const card = within(dialog).getByRole('article', { name: 'gpu-box' });
    expect(within(card).getByText('Online')).toBeInTheDocument();
    expect(within(card).getByText('matches this server').parentElement)
      .toHaveTextContent('3.0.2 - matches this server');
    const digest = within(card).getByText(DIGEST.slice(0, FINGERPRINT_SHORT_LENGTH));
    expect(digest).toHaveAttribute('title', DIGEST);
    expect(within(card).getByText('data')).toHaveAttribute('title', '/mnt/nas/data');
    expect(within(card).getByText('4 s ago')).toBeInTheDocument();
    // Run ids reported per worker are listed once, under the worker.
    expect(within(card).getByRole('button', { name: 'run-aaaa' })).toBeInTheDocument();
    expect(within(card).getByText('RTX 4090')).toBeInTheDocument();
    expect(within(card).getByText('Slots 1 / 1')).toBeInTheDocument();
    expect(within(card).getByText('20.0 GB free of 24.0 GB')).toHaveAttribute('title', 'nvml');
    // A device whose name is its value, one with half its memory unknown
    // and one with none print neither a second name nor a memory line.
    expect(within(card).getAllByText(/free of/)).toHaveLength(1);
    expect(within(card).getByText('cpu')).toBeInTheDocument();
    expect(within(card).getByText('mps')).toBeInTheDocument();
  });

  it('says when the version differs, and nothing when the server version is unknown', async () => {
    vi.mocked(fetchHealth).mockResolvedValue({ version: '3.1.0' } as never);
    const dialog = await openPanel([worker('a')]);
    expect(within(dialog).getByText('differs from this server (3.1.0)')).toBeInTheDocument();

    act(() => useWorkerStore.setState({ serverVersion: null }));
    expect(within(dialog).queryByText(/this server/)).toBeNull();
  });

  it('reads "not reported", "none" and "no runs" for a bare worker', async () => {
    const dialog = await openPanel([worker('bare', { status: 'disconnected' })]);
    expect(within(dialog).getByText('Disconnected')).toBeInTheDocument();
    expect(within(dialog).getByText('not reported')).toBeInTheDocument();
    expect(within(dialog).getByText('none')).toBeInTheDocument();
    expect(within(dialog).getByText('no runs')).toBeInTheDocument();
  });

  it('lists run ids per device when the server reports them that way', async () => {
    const dialog = await openPanel([
      worker('w', {
        running: ['r-one-0000', 'r-two-0000'],
        devices: [
          { value: 'cuda:0', kind: 'cuda', index: 0, name: 'A', memory: null, running: ['r-one-0000'] },
          { value: 'cuda:1', kind: 'cuda', index: 1, name: 'B', memory: null, running: [] },
        ],
      }),
    ]);
    expect(within(dialog).queryByText('Running')).toBeNull();
    expect(within(dialog).getByRole('button', { name: 'r-one-00' })).toBeInTheDocument();
    expect(within(dialog).queryByRole('button', { name: 'r-two-00' })).toBeNull();
    expect(within(dialog).getByText('no runs')).toBeInTheDocument();
  });

  it('a run link closes the panel, selects the run and asks the dock for the Runs tab', async () => {
    const select = vi.fn(() => Promise.resolve());
    useRunStore.setState({ select });
    const dialog = await openPanel([worker('w', { running: ['run-12345678-abcd'] })]);
    const link = within(dialog).getByRole('button', { name: 'run-1234' });
    expect(link).toHaveAttribute('title', 'Show run run-12345678-abcd in the Runs panel');
    fireEvent.click(link);
    expect(select).toHaveBeenCalledWith('run-12345678-abcd');
    expect(useWorkerStore.getState()).toMatchObject({ panelOpen: false, runsRevealSeq: 1 });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('drains an online worker and undrains a draining one', async () => {
    vi.mocked(drainWorker).mockResolvedValue(worker('a', { draining: true, status: 'draining' }));
    vi.mocked(undrainWorker).mockResolvedValue(worker('b'));
    const dialog = await openPanel([worker('a'), worker('b', { draining: true, status: 'draining' })]);
    const a = within(dialog).getByRole('article', { name: 'a' });
    const b = within(dialog).getByRole('article', { name: 'b' });
    expect(within(b).getByText('Draining')).toBeInTheDocument();

    const drain = within(a).getByRole('button', { name: 'Drain' });
    expect(drain).toHaveAttribute('title', t('workers.drain.title'));
    fireEvent.click(drain);
    await settle();
    expect(drainWorker).toHaveBeenCalledWith('a');
    expect(within(a).getByText('Draining')).toBeInTheDocument();

    fireEvent.click(within(b).getByRole('button', { name: 'Undrain' }));
    await settle();
    expect(undrainWorker).toHaveBeenCalledWith('b');
    expect(within(b).getByText('Online')).toBeInTheDocument();
  });

  it('disables the action while one is in flight', async () => {
    const dialog = await openPanel([worker('a')]);
    act(() => useWorkerStore.setState({ busy: { a: true } }));
    expect(within(dialog).getByRole('button', { name: 'Drain' })).toBeDisabled();
  });

  it('Refresh reads the list again', async () => {
    const dialog = await openPanel([]);
    expect(fetchWorkers).toHaveBeenCalledTimes(1);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Refresh' }));
    await settle();
    expect(fetchWorkers).toHaveBeenCalledTimes(2);
  });

  it('closes on Escape, on the close button and on a press on the backdrop', async () => {
    let dialog = await openPanel([]);
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();

    act(() => useWorkerStore.getState().openPanel());
    await settle();
    dialog = screen.getByRole('dialog', { name: 'Workers' });
    fireEvent.mouseDown(dialog);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.mouseDown(dialog.parentElement as HTMLElement);
    expect(screen.queryByRole('dialog')).toBeNull();

    act(() => useWorkerStore.getState().openPanel());
    await settle();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('takes the focus, keeps Tab inside and gives the focus back on close', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const dialog = await openPanel([]);
    const close = within(dialog).getByRole('button', { name: 'Close' });
    const refresh = within(dialog).getByRole('button', { name: 'Refresh' });
    expect(close).toHaveFocus();

    // The close button is the last control: Tab wraps to the first.
    fireEvent.keyDown(dialog, { key: 'Tab' });
    expect(refresh).toHaveFocus();
    fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true });
    expect(close).toHaveFocus();
    // From the dialog itself (after a click between controls) too.
    dialog.focus();
    fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true });
    expect(close).toHaveFocus();
    // In the middle Tab is left to the browser.
    refresh.focus();
    fireEvent.keyDown(dialog, { key: 'Tab' });
    expect(refresh).toHaveFocus();
    fireEvent.keyDown(dialog, { key: 'a' });

    act(() => useWorkerStore.getState().closePanel());
    expect(opener).toHaveFocus();
    opener.remove();
  });

  it('does not hand the focus back to an element that left the page', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    await openPanel([]);
    opener.remove();
    act(() => useWorkerStore.getState().closePanel());
    expect(opener).not.toHaveFocus();
  });

  it('opens the Add worker dialog over the panel; its Escape closes only itself', async () => {
    const dialog = await openPanel([]);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Add worker' }));
    const add = screen.getByRole('dialog', { name: 'Add a worker' });
    expect(within(add).getByText('cdui agent start')).toBeInTheDocument();
    fireEvent.keyDown(add, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'Add a worker' })).toBeNull();
    expect(screen.getByRole('dialog', { name: 'Workers' })).toBeInTheDocument();
  });

  it('renders in zh-TW', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const dialog = await openPanel([worker('a')]);
    expect(within(dialog).getByText('線上')).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: '排空' })).toBeInTheDocument();
  });
});
