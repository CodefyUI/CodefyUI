import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import {
  addSshWorker,
  deleteSshWorker,
  drainWorker,
  fetchEnrollments,
  fetchWorkerList,
  fetchSshWorkers,
  fetchWorkers,
  removeWorker,
  setSshWorkerEnabled,
  undrainWorker,
  type SshWorker,
  type WorkerInfo,
} from '../../api/workers';
import { useI18n } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { _resetWorkerStoreForTesting, useWorkerStore } from '../../store/workerStore';
import { settle } from '../../test/utils';
import { rememberAppVersion } from '../../utils/appVersion';
import { relativeTime } from '../SourceControl/scm';
import { FINGERPRINT_SHORT_LENGTH, WorkersPanel } from './WorkersPanel';

vi.mock('../../api/workers', async (importOriginal) => ({
  WorkersApiError: (await importOriginal<typeof import('../../api/workers')>()).WorkersApiError,
  fetchWorkers: vi.fn(),
  fetchWorkerList: vi.fn(),
  drainWorker: vi.fn(),
  undrainWorker: vi.fn(),
  removeWorker: vi.fn(),
  fetchEnrollments: vi.fn(),
  createEnrollment: vi.fn(),
  deleteEnrollment: vi.fn(),
  agentCommand: vi.fn(),
  fetchSshWorkers: vi.fn(),
  setSshWorkerEnabled: vi.fn(),
  deleteSshWorker: vi.fn(),
  addSshWorker: vi.fn(),
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

function ssh(id: string, extra: Partial<SshWorker> = {}): SshWorker {
  return {
    id,
    target: `alice@${id}`,
    port: 22,
    identity: null,
    remote_cdui: '~/CodefyUI/cdui',
    remote_port: 18000,
    enabled: true,
    state: 'up',
    last_error: null,
    exit_code: null,
    tail: [],
    attempts: 1,
    retry_at: null,
    up_since: NOW_S - 50,
    ...extra,
  };
}

function sshList(entries: SshWorker[]) {
  return { ssh_workers: entries, ssh_available: true, port_range: [18000, 18100] as [number, number], default_remote_cdui: '~/CodefyUI/cdui' };
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
  vi.mocked(fetchWorkerList).mockReset().mockImplementation(async () => ({ workers: await fetchWorkers(), rejections: [] }));
  rememberAppVersion('3.0.2');
  vi.mocked(drainWorker).mockReset();
  vi.mocked(undrainWorker).mockReset();
  vi.mocked(removeWorker).mockReset().mockResolvedValue(undefined);
  vi.mocked(fetchSshWorkers).mockReset().mockResolvedValue(sshList([]));
  vi.mocked(setSshWorkerEnabled).mockReset();
  vi.mocked(deleteSshWorker).mockReset().mockResolvedValue(undefined);
  vi.mocked(fetchEnrollments).mockReset().mockResolvedValue({
    enrollments: [],
    control_url: 'http://192.168.1.5:8000',
    other_machine_urls: [],
    listening_on_loopback: false,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
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
    expect(within(card).getByText(relativeTime(NOW_S - 4, 'en', NOW_S * 1000))).toBeInTheDocument();
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
    rememberAppVersion('3.1.0');
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

  it('lists run ids per device when the server reports them that way, and the rest under the worker', async () => {
    const dialog = await openPanel([
      worker('w', {
        running: ['r-one-0000', 'r-two-0000'],
        devices: [
          { value: 'cuda:0', kind: 'cuda', index: 0, name: 'A', memory: null, running: ['r-one-0000'] },
          { value: 'cuda:1', kind: 'cuda', index: 1, name: 'B', memory: null, running: [] },
        ],
      }),
    ]);
    const deviceRow = within(dialog).getByText('cuda:0').closest('li') as HTMLElement;
    expect(within(deviceRow).getByRole('button', { name: 'r-one-00' })).toBeInTheDocument();
    // No device lists r-two, so the worker's own row does; r-one is not repeated.
    const running = within(dialog).getByText('Running').nextElementSibling as HTMLElement;
    expect(within(running).getByRole('button', { name: 'r-two-00' })).toBeInTheDocument();
    expect(within(running).queryByRole('button', { name: 'r-one-00' })).toBeNull();
    expect(within(dialog).getByText('no runs')).toBeInTheDocument();
  });

  it('has no worker-level Running row when the devices list every run', async () => {
    const dialog = await openPanel([
      worker('w', {
        running: ['r-one-0000'],
        devices: [
          { value: 'cuda:0', kind: 'cuda', index: 0, name: 'A', memory: null, running: ['r-one-0000'] },
        ],
      }),
    ]);
    expect(within(dialog).queryByText('Running')).toBeNull();
  });

  it('badges a worker connected over plain HTTP from another machine (#786)', async () => {
    const dialog = await openPanel([worker('far', { plain_http: true }), worker('near', { plain_http: false })]);
    const far = within(dialog).getByRole('article', { name: 'far' });
    const near = within(dialog).getByRole('article', { name: 'near' });
    expect(within(far).getByText('Plain HTTP')).toHaveAttribute('title', expect.stringMatching(/unencrypted/));
    expect(within(near).queryByText('Plain HTTP')).toBeNull();
  });

  it('shows a status this build does not know as sent, in a neutral chip', async () => {
    const dialog = await openPanel([worker('w', { status: 'quarantined' as never })]);
    expect(within(dialog).getByText('quarantined')).toBeInTheDocument();
  });

  it('prints a dash for a heartbeat time the server could not give', async () => {
    const dialog = await openPanel([worker('w', { last_heartbeat_at: 0 })]);
    expect(within(dialog).getByText('Last heartbeat').nextElementSibling).toHaveTextContent('-');
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

  it('marks the action unavailable while one is in flight, and a press then does nothing', async () => {
    const dialog = await openPanel([worker('a')]);
    act(() => useWorkerStore.setState({ busy: { a: true } }));
    const drain = within(dialog).getByRole('button', { name: 'Drain' });
    expect(drain).toHaveAttribute('aria-disabled', 'true');
    expect(drain).not.toBeDisabled();
    fireEvent.click(drain);
    await settle();
    expect(drainWorker).not.toHaveBeenCalled();
  });

  it('keeps the focus on the action after Drain, so Escape still closes the panel', async () => {
    vi.mocked(drainWorker).mockResolvedValue(worker('a', { draining: true, status: 'draining' }));
    const dialog = await openPanel([worker('a')]);
    const drain = within(dialog).getByRole('button', { name: 'Drain' });
    drain.focus();
    fireEvent.click(drain);
    await settle();
    const undrain = within(dialog).getByRole('button', { name: 'Undrain' });
    expect(undrain).toHaveFocus();
    fireEvent.keyDown(undrain, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('puts the focus back on the action when it fell to the page during Drain', async () => {
    let answer: (value: WorkerInfo) => void = () => {};
    vi.mocked(drainWorker).mockImplementation(() => new Promise((resolve) => { answer = resolve; }));
    const dialog = await openPanel([worker('a')]);
    const drain = within(dialog).getByRole('button', { name: 'Drain' });
    drain.focus();
    fireEvent.click(drain);
    drain.blur();
    expect(document.body).toHaveFocus();
    await act(async () => answer(worker('a', { draining: true, status: 'draining' })));
    expect(within(dialog).getByRole('button', { name: 'Undrain' })).toHaveFocus();
  });

  it('moves no focus when the panel closed while Drain was in flight', async () => {
    let answer: (value: WorkerInfo) => void = () => {};
    vi.mocked(drainWorker).mockImplementation(() => new Promise((resolve) => { answer = resolve; }));
    const dialog = await openPanel([worker('a')]);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Drain' }));
    act(() => useWorkerStore.getState().closePanel());
    (document.activeElement as HTMLElement | null)?.blur();
    await act(async () => answer(worker('a')));
    expect(document.body).toHaveFocus();
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
    // The dialog is a lazy chunk of its own: it arrives a tick later.
    const add = await screen.findByRole('dialog', { name: 'Add a worker' });
    await settle();
    // It opens on the other-machine tab (#780); the same-host one is a press away.
    expect(add).toHaveAttribute('data-mode', 'remote');
    expect(within(add).getByRole('button', { name: 'Generate token' })).toBeInTheDocument();
    fireEvent.click(within(add).getByRole('button', { name: 'This machine' }));
    expect(within(add).getByText('cdui agent start')).toBeInTheDocument();
    fireEvent.keyDown(add, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'Add a worker' })).toBeNull();
    expect(screen.getByRole('dialog', { name: 'Workers' })).toBeInTheDocument();
  });

  it('removes a worker that is away at once', async () => {
    const dialog = await openPanel([worker('gone', { status: 'offline', connected_at: null }), worker('b')]);
    vi.mocked(fetchWorkers).mockResolvedValue([worker('b')]);
    const card = within(dialog).getByRole('article', { name: 'gone' });
    expect(within(card).getByText('Offline')).toBeInTheDocument();
    const remove = within(card).getByRole('button', { name: 'Remove' });
    remove.focus();
    await act(async () => fireEvent.click(remove));
    expect(removeWorker).toHaveBeenCalledWith('gone');
    expect(within(dialog).queryByRole('article', { name: 'gone' })).toBeNull();
    // The pressed button left with its card: the focus stays in the dialog.
    expect(within(dialog).getByRole('button', { name: 'Close' })).toHaveFocus();
  });

  it('asks again before removing a connected worker', async () => {
    const dialog = await openPanel([worker('live')]);
    const card = within(dialog).getByRole('article', { name: 'live' });
    const remove = within(card).getByRole('button', { name: 'Remove' });
    await act(async () => fireEvent.click(remove));
    expect(removeWorker).not.toHaveBeenCalled();
    expect(remove).toHaveTextContent('Confirm remove');
    expect(remove).toHaveAttribute('title', expect.stringMatching(/end interrupted/));
    // Leaving the button drops the question.
    fireEvent.blur(remove);
    expect(remove).toHaveTextContent('Remove');
    await act(async () => fireEvent.click(remove));
    await act(async () => fireEvent.click(remove));
    expect(removeWorker).toHaveBeenCalledWith('live');
  });

  it('keeps the focus where it is when the remove failed', async () => {
    vi.mocked(removeWorker).mockRejectedValue(new Error('boom'));
    const dialog = await openPanel([worker('gone', { status: 'disconnected' })]);
    const remove = within(dialog).getByRole('button', { name: 'Remove' });
    remove.focus();
    await act(async () => fireEvent.click(remove));
    expect(remove).toHaveFocus();
    expect(within(dialog).getByRole('article', { name: 'gone' })).toBeInTheDocument();
  });

  it('ignores a second remove while one is in flight', async () => {
    let answer: () => void = () => {};
    vi.mocked(removeWorker).mockImplementation(() => new Promise<void>((resolve) => { answer = resolve; }));
    const dialog = await openPanel([worker('gone', { status: 'offline' })]);
    const remove = within(dialog).getByRole('button', { name: 'Remove' });
    fireEvent.click(remove);
    expect(remove).toHaveAttribute('aria-disabled', 'true');
    await act(async () => fireEvent.click(remove));
    expect(removeWorker).toHaveBeenCalledTimes(1);
    await act(async () => answer());
  });

  it('says never for a worker never seen, and opens a re-key dialog for it', async () => {
    const dialog = await openPanel([worker('new-box', { status: 'offline', connected_at: null, last_heartbeat_at: null })]);
    const card = within(dialog).getByRole('article', { name: 'new-box' });
    expect(within(card).getByText('never')).toBeInTheDocument();
    fireEvent.click(within(card).getByRole('button', { name: 'Re-key' }));
    const add = await screen.findByRole('dialog', { name: 'Add a worker' });
    await settle();
    expect(within(add).getByText(/Re-key new-box/)).toBeInTheDocument();
    fireEvent.keyDown(add, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'Add a worker' })).toBeNull();
  });

  it('shows how the node catalog compares: matches, not reported, or the types that differ', async () => {
    const dialog = await openPanel([
      worker('same', { node_catalog: { status: 'match', hash: 'h', missing: [], extra: [] } }),
      worker('old', { node_catalog: { status: 'unknown', hash: null, missing: [], extra: [] } }),
      worker('bare', {
        status: 'degraded',
        degraded_reasons: ['node_catalog'],
        node_catalog: {
          status: 'mismatch', hash: 'x', missing: ['foundations:Edu-KNN', 'foundations:Edu-FFN'], extra: ['mine:Only'],
        },
      }),
      worker('fewer', {
        node_catalog: { status: 'mismatch', hash: 'y', missing: [], extra: ['mine:Only'] },
      }),
    ]);
    const same = within(dialog).getByRole('article', { name: 'same' });
    expect(within(same).getByText('match this server')).toBeInTheDocument();
    const old = within(dialog).getByRole('article', { name: 'old' });
    expect(within(old).getByText(/not reported: runs naming this worker are not checked/)).toBeInTheDocument();
    const bare = within(dialog).getByRole('article', { name: 'bare' });
    expect(within(bare).getByText('Degraded')).toBeInTheDocument();
    expect(within(bare).getByText(/differ from this server/)).toBeInTheDocument();
    expect(within(bare).getByText('foundations:Edu-KNN, foundations:Edu-FFN').parentElement)
      .toHaveTextContent('Missing here (2): foundations:Edu-KNN, foundations:Edu-FFN');
    expect(within(bare).getByText('mine:Only').parentElement).toHaveTextContent('Only on the worker (1): mine:Only');
    const fewer = within(dialog).getByRole('article', { name: 'fewer' });
    expect(within(fewer).queryByText(/Missing here/)).toBeNull();
  });

  it('shows the last refused handshake on a card, and refused names without one on their own', async () => {
    const reason = 'version mismatch: worker 3.0.1, control plane 3.0.2';
    vi.mocked(fetchWorkerList).mockResolvedValue({
      workers: [worker('gpu-box'), worker('busy-box')],
      rejections: [
        { name: 'new-box', reason, at: NOW_S - 60 },
        { name: 'gpu-box', reason: 'version mismatch: worker 1, control plane 2', at: NOW_S - 5, peer: '10.0.0.7' },
        { name: 'busy-box', reason: 'a worker named busy-box is connected with another kind of credential', at: NOW_S - 9, peer: '10.0.0.9', live_owner: true },
      ],
    });
    render(<WorkersPanel />);
    act(() => useWorkerStore.getState().openPanel());
    await settle();
    const dialog = screen.getByRole('dialog', { name: 'Workers' });
    const card = within(dialog).getByRole('article', { name: 'gpu-box' });
    expect(within(card).getByText('Last refused')).toBeInTheDocument();
    expect(within(card).getByText(/version mismatch: worker 1/)).toHaveTextContent(
      `(${relativeTime(NOW_S - 5, 'en', NOW_S * 1000)}, from 10.0.0.7)`,
    );
    // Someone else's attempt on a connected worker's name is not its own.
    const busy = within(dialog).getByRole('article', { name: 'busy-box' });
    expect(within(busy).queryByText('Last refused')).toBeNull();
    const refused = within(dialog).getByRole('region', { name: 'Refused connections' });
    expect(within(refused).getByText('new-box')).toBeInTheDocument();
    expect(within(refused).getByText(reason, { exact: false })).toHaveTextContent(
      `${reason} (${relativeTime(NOW_S - 60, 'en', NOW_S * 1000)})`,
    );
    expect(within(refused).queryByText('gpu-box')).toBeNull();
    expect(within(refused).getByText('busy-box')).toBeInTheDocument();
    expect(within(refused).getByText(/another kind of credential/)).toHaveTextContent('from 10.0.0.9');
  });

  it('lists refused names instead of the empty line when no worker is connected', async () => {
    vi.mocked(fetchWorkerList).mockResolvedValue({
      workers: [],
      rejections: [{ name: 'old-box', reason: 'version mismatch', at: 0 }],
    });
    render(<WorkersPanel />);
    act(() => useWorkerStore.getState().openPanel());
    await settle();
    expect(screen.queryByText(/No worker agent has connected/)).toBeNull();
    expect(screen.getByRole('region', { name: 'Refused connections' })).toHaveTextContent('old-box');
  });

  it('renders in zh-TW', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const dialog = await openPanel([worker('a')]);
    expect(within(dialog).getByText('線上')).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: '排空' })).toBeInTheDocument();
  });

  // ── SSH workers (#784) ──

  it('shows an SSH worker\'s tunnel on its card, with Disable', async () => {
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([ssh('gpu-box')]));
    vi.mocked(setSshWorkerEnabled).mockResolvedValue(ssh('gpu-box', { enabled: false, state: 'stopped' }));
    const dialog = await openPanel([worker('gpu-box')]);
    const card = within(dialog).getByRole('article', { name: 'gpu-box' });
    expect(within(card).getByText('SSH tunnel')).toBeInTheDocument();
    expect(within(card).getByText('Up')).toBeInTheDocument();
    expect(within(card).getByText('alice@gpu-box')).toBeInTheDocument();
    expect(within(card).getByText('remote port 18000')).toBeInTheDocument();
    expect(within(card).queryByRole('status')).toBeNull();
    expect(within(card).queryByText('Recent ssh output')).toBeNull();
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([ssh('gpu-box', { enabled: false, state: 'stopped' })]));
    vi.mocked(fetchWorkers).mockResolvedValue([worker('gpu-box', { status: 'disconnected', draining: true })]);
    await act(async () => fireEvent.click(within(card).getByRole('button', { name: 'Disable' })));
    expect(setSshWorkerEnabled).toHaveBeenCalledWith('gpu-box', false);
    // The card flips at once, not at the next poll.
    expect(within(card).getByText('Disconnected')).toBeInTheDocument();
    expect(within(card).getByText('Stopped')).toBeInTheDocument();
    expect(within(card).getByRole('button', { name: 'Enable' })).toBeInTheDocument();
  });

  it('shows why a tunnel failed, and the recent output', async () => {
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([ssh('gpu-box', {
      state: 'failed',
      port: 2222,
      identity: '/home/me/.ssh/key',
      remote_port: null,
      last_error: 'alice@gpu-box: Permission denied (publickey).',
      exit_code: 255,
      tail: ['debug1: hello', 'alice@gpu-box: Permission denied (publickey).'],
    })]));
    const dialog = await openPanel([worker('gpu-box', { status: 'offline' })]);
    const card = within(dialog).getByRole('article', { name: 'gpu-box' });
    expect(within(card).getByText('Failed')).toBeInTheDocument();
    expect(within(card).getByText('alice@gpu-box:2222')).toHaveAttribute('title', '/home/me/.ssh/key');
    expect(within(card).queryByText(/remote port/)).toBeNull();
    expect(within(card).getByRole('status')).toHaveTextContent('Permission denied (publickey).');
    expect(within(card).getByText('Recent ssh output')).toBeInTheDocument();
    expect(within(card).getByText(/debug1: hello/)).toBeInTheDocument();
  });

  it('lists an SSH worker that has not connected yet, with Delete', async () => {
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([ssh('new-box', { state: 'connecting', up_since: null })]));
    const dialog = await openPanel([]);
    expect(within(dialog).queryByText(t('workers.empty'))).toBeNull();
    const card = within(dialog).getByRole('article', { name: 'new-box' });
    expect(within(card).getByText('Not connected yet')).toBeInTheDocument();
    expect(within(card).getByText('Connecting')).toBeInTheDocument();
    await act(async () => fireEvent.click(within(card).getByRole('button', { name: 'Delete' })));
    expect(deleteSshWorker).toHaveBeenCalledWith('new-box');
    expect(within(dialog).queryByRole('article', { name: 'new-box' })).toBeNull();
  });

  it('does nothing on Enable or Delete while an action on it is in flight', async () => {
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([ssh('new-box', { enabled: false, state: 'stopped' })]));
    const dialog = await openPanel([]);
    act(() => useWorkerStore.setState({ busy: { 'new-box': true } }));
    const card = within(dialog).getByRole('article', { name: 'new-box' });
    fireEvent.click(within(card).getByRole('button', { name: 'Enable' }));
    fireEvent.click(within(card).getByRole('button', { name: 'Delete' }));
    expect(setSshWorkerEnabled).not.toHaveBeenCalled();
    expect(deleteSshWorker).not.toHaveBeenCalled();
  });

  it('shows a tunnel state this build does not know in a neutral chip', async () => {
    const odd = ssh('gpu-box', { state: 'paused' as SshWorker['state'] });
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([odd]));
    const dialog = await openPanel([worker('gpu-box')]);
    expect(within(dialog).getByText('paused')).toBeInTheDocument();
  });

  it('reads the list again after the dialog added an SSH worker', async () => {
    vi.mocked(addSshWorker).mockResolvedValue(ssh('gpu-box', { state: 'connecting' }));
    const dialog = await openPanel([]);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Add worker' }));
    const add = await screen.findByRole('dialog', { name: 'Add a worker' });
    await settle();
    fireEvent.click(within(add).getByRole('button', { name: 'Over SSH' }));
    await settle();
    fireEvent.change(within(add).getByLabelText(/SSH target/), { target: { value: 'alice@gpu-box' } });
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([ssh('gpu-box', { state: 'connecting' })]));
    const before = vi.mocked(fetchWorkers).mock.calls.length;
    await act(async () => fireEvent.click(within(add).getByRole('button', { name: 'Add and connect' })));
    await settle();
    expect(screen.queryByRole('dialog', { name: 'Add a worker' })).toBeNull();
    expect(vi.mocked(fetchWorkers).mock.calls.length).toBeGreaterThan(before);
    expect(within(dialog).getByRole('article', { name: 'gpu-box' })).toBeInTheDocument();
  });

  it('shows the remote agent pid, Retry for a tunnel that gave up, and a failed SSH read', async () => {
    vi.mocked(fetchSshWorkers).mockResolvedValue(sshList([
      ssh('gpu-box', { remote_pid: 4242 }),
      ssh('back-box', { remote_pid: 77, reattached: true }),
      ssh('stuck', { state: 'failed', gave_up: true, last_error: 'administratively prohibited' }),
    ]));
    vi.mocked(setSshWorkerEnabled).mockResolvedValue(ssh('stuck', { state: 'connecting' }));
    const dialog = await openPanel([worker('gpu-box')]);
    expect(within(dialog).getByText('remote agent pid 4242')).toBeInTheDocument();
    expect(within(dialog).getByText('remote agent pid 77, re-attached after a dropped session')).toBeInTheDocument();
    const stuck = within(dialog).getByRole('article', { name: 'stuck' });
    await act(async () => fireEvent.click(within(stuck).getByRole('button', { name: 'Retry' })));
    expect(setSshWorkerEnabled).toHaveBeenCalledWith('stuck', true);
    act(() => useWorkerStore.setState({ busy: { stuck: true } }));
    fireEvent.click(within(stuck).getByRole('button', { name: 'Retry' }));
    expect(setSshWorkerEnabled).toHaveBeenCalledTimes(1);
    act(() => useWorkerStore.setState({ sshError: 'Missing or invalid token' }));
    expect(within(dialog).getByText(/Could not read the SSH workers/)).toBeInTheDocument();
  });
});
