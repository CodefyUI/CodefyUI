import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import {
  agentCommand,
  addSshWorker,
  createEnrollment,
  deleteEnrollment,
  fetchEnrollments,
  fetchSshWorkers,
  type CreatedEnrollment,
  type Enrollment,
  type EnrollmentList,
  type SshWorker,
  type SshWorkerList,
} from '../../api/workers';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { settle } from '../../test/utils';
import {
  AGENT_START_COMMAND,
  AGENT_STOP_COMMAND,
  AddWorkerDialog,
  isHttpsUrl,
  isLoopbackUrl,
  workerNameProblem,
} from './AddWorkerDialog';

vi.mock('../../api/workers', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/workers')>()),
  fetchEnrollments: vi.fn(),
  createEnrollment: vi.fn(),
  deleteEnrollment: vi.fn(),
  agentCommand: vi.fn(),
  fetchSshWorkers: vi.fn(),
  addSshWorker: vi.fn(),
}));

const TOKEN = 'abcdef.0123456789abcdef';

function listOf(extra: Partial<EnrollmentList> = {}): EnrollmentList {
  return {
    enrollments: [],
    control_url: 'http://192.168.1.5:8000',
    other_machine_urls: ['http://192.168.1.5:8000'],
    listening_on_loopback: false,
    ...extra,
  };
}

function enrollment(id: string, extra: Partial<Enrollment> = {}): Enrollment {
  return {
    id,
    label: null,
    worker: null,
    created_at: '2026-10-15T00:00:00.000000Z',
    expires_at: '2026-10-16T00:00:00.000000Z',
    uses_left: 1,
    state: 'pending',
    ...extra,
  };
}

function created(extra: Partial<CreatedEnrollment> = {}): CreatedEnrollment {
  return {
    ...enrollment('abcdef'),
    token: TOKEN,
    control_url: 'http://192.168.1.5:8000',
    command: `cdui agent start --control http://192.168.1.5:8000 --token ${TOKEN}`,
    ...extra,
  };
}

/** The fingerprint the fake server pins in its commands (#786). */
let serverFp: string | null = null;

/** The fake server's command builder (`routes_workers.agent_command`). */
function serverCommand(url: string, token: string, worker?: string | null): string {
  let command = `cdui agent start --control ${url} --token ${token}`;
  if (worker) command += ` --name ${worker}`;
  if (serverFp && url.startsWith('https:')) command += ` --ca-fingerprint ${serverFp}`;
  return command;
}

const writeText = vi.fn();

beforeEach(() => {
  serverFp = null;
  vi.mocked(fetchEnrollments).mockReset().mockResolvedValue(listOf());
  vi.mocked(createEnrollment).mockReset().mockImplementation(async (options) => {
    const url = options.control_url ?? 'http://192.168.1.5:8000';
    return created({ control_url: url, worker: options.worker ?? null, command: serverCommand(url, TOKEN, options.worker) });
  });
  vi.mocked(agentCommand).mockReset().mockImplementation(
    async ({ control_url, token, worker }) => serverCommand(control_url, token, worker),
  );
  vi.mocked(deleteEnrollment).mockReset().mockResolvedValue(undefined);
  vi.mocked(fetchSshWorkers).mockReset().mockResolvedValue(sshInfo());
  vi.mocked(addSshWorker).mockReset().mockResolvedValue(sshEntry());
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  writeText.mockReset().mockResolvedValue(undefined);
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
});

afterEach(() => {
  Reflect.deleteProperty(navigator, 'clipboard');
});

function renderDialog(onClose = vi.fn()) {
  render(<AddWorkerDialog mode="same-host" onClose={onClose} />);
  return { onClose, dialog: screen.getByRole('dialog', { name: 'Add a worker' }) };
}

describe('AddWorkerDialog (same-host)', () => {
  it('prints the agent start and stop commands and the options', () => {
    const { dialog } = renderDialog();
    expect(dialog).toHaveAttribute('data-mode', 'same-host');
    expect(within(dialog).getByText(AGENT_START_COMMAND)).toBeInTheDocument();
    expect(within(dialog).getByText(AGENT_STOP_COMMAND)).toBeInTheDocument();
    expect(within(dialog).getByText('--name <name>')).toBeInTheDocument();
    expect(within(dialog).getByText('--devices cuda:0,cuda:1')).toBeInTheDocument();
    expect(within(dialog).getByText('--foreground')).toBeInTheDocument();
  });

  it('copies a command and says so', async () => {
    const { dialog } = renderDialog();
    const [copyStart, copyStop] = within(dialog).getAllByRole('button', { name: 'Copy' });
    await act(async () => fireEvent.click(copyStart));
    expect(writeText).toHaveBeenCalledWith('cdui agent start');
    await act(async () => fireEvent.click(copyStop));
    expect(writeText).toHaveBeenLastCalledWith('cdui agent stop');
    expect(useToastStore.getState().toasts.map((t) => t.message)).toEqual([
      'Command copied', 'Command copied',
    ]);
  });

  it('asks for a manual copy when the clipboard refuses', async () => {
    writeText.mockRejectedValue(new Error('denied'));
    const { dialog } = renderDialog();
    await act(async () => fireEvent.click(within(dialog).getAllByRole('button', { name: 'Copy' })[0]));
    const [toast] = useToastStore.getState().toasts;
    expect(toast).toMatchObject({ type: 'error', message: expect.stringMatching(/copy it by hand/) });
  });

  it('closes on Escape, the close button and a press on its backdrop only', () => {
    const { dialog, onClose } = renderDialog();
    expect(within(dialog).getByRole('button', { name: 'Close' })).toHaveFocus();
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.mouseDown(dialog);
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.mouseDown(dialog.parentElement as HTMLElement);
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it('keeps Tab inside and does not let its keys reach the panel', () => {
    const outer = vi.fn();
    render(
      <div onKeyDown={outer}>
        <AddWorkerDialog mode="same-host" onClose={vi.fn()} />
      </div>,
    );
    const dialog = screen.getByRole('dialog', { name: 'Add a worker' });
    const close = within(dialog).getByRole('button', { name: 'Close' });
    const copies = within(dialog).getAllByRole('button', { name: 'Copy' });
    copies[1].focus();
    fireEvent.keyDown(dialog, { key: 'Tab' });
    expect(close).toHaveFocus();
    expect(outer).not.toHaveBeenCalled();
  });

  it('gives the focus back on close, unless the opener left the page', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const first = render(<AddWorkerDialog mode="same-host" onClose={vi.fn()} />);
    first.unmount();
    expect(opener).toHaveFocus();

    const second = render(<AddWorkerDialog mode="same-host" onClose={vi.fn()} />);
    opener.remove();
    second.unmount();
    expect(opener).not.toHaveFocus();
  });

  it('renders in zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<AddWorkerDialog mode="same-host" onClose={vi.fn()} />);
    expect(screen.getByRole('dialog', { name: '新增 worker' })).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: '複製' })).toHaveLength(2);
  });
});

async function renderRemote() {
  render(<AddWorkerDialog mode="remote" onClose={vi.fn()} />);
  await settle();
  return screen.getByRole('dialog');
}

describe('AddWorkerDialog (another machine, #780)', () => {
  it('switches between the two tabs', async () => {
    const dialog = await renderRemote();
    expect(dialog).toHaveAttribute('data-mode', 'remote');
    const remote = within(dialog).getByRole('button', { name: 'Another machine' });
    const local = within(dialog).getByRole('button', { name: 'This machine' });
    expect(remote).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(local);
    expect(dialog).toHaveAttribute('data-mode', 'same-host');
    expect(within(dialog).getByText(AGENT_START_COMMAND)).toBeInTheDocument();
    fireEvent.click(remote);
    await settle();
    expect(within(dialog).getByRole('button', { name: 'Generate token' })).toBeInTheDocument();
  });

  it('starts from the address this page uses and lists no tokens yet', async () => {
    const dialog = await renderRemote();
    expect(within(dialog).getByRole('textbox', { name: /Server address/ })).toHaveValue('http://192.168.1.5:8000');
    expect(within(dialog).getByText('No enrollment tokens.')).toBeInTheDocument();
    // Plain HTTP to another machine (#786): one note, the warning.
    expect(within(dialog).getByRole('note')).toHaveTextContent(/plain HTTP: the worker token and the secrets/);
    expect(within(dialog).queryByText(/only works on this machine/)).toBeNull();
  });

  it('warns about a loopback address and offers the others', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({
      control_url: 'http://127.0.0.1:8000',
      other_machine_urls: ['http://10.0.0.7:8000', 'http://192.168.1.5:8000'],
    }));
    const dialog = await renderRemote();
    expect(within(dialog).getByText(/127\.0\.0\.1:8000 only works on this machine/)).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: 'http://10.0.0.7:8000' }));
    expect(within(dialog).getByRole('textbox', { name: /Server address/ })).toHaveValue('http://10.0.0.7:8000');
    expect(within(dialog).queryByText(/only works on this machine/)).toBeNull();
  });

  it('opens on Over SSH when the server cannot be reached from another machine at all (#840)', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({
      control_url: 'http://localhost:8000', other_machine_urls: [], listening_on_loopback: true,
    }));
    const dialog = await renderRemote();
    expect(dialog).toHaveAttribute('data-mode', 'ssh');
    expect(within(dialog).getByRole('button', { name: 'Over SSH' })).toHaveAttribute('aria-pressed', 'true');
    // Another machine still says why, with no sentence left hanging, and
    // points back to SSH.
    fireEvent.click(within(dialog).getByRole('button', { name: 'Another machine' }));
    await settle();
    expect(within(dialog).getAllByRole('note')[0]).toHaveTextContent(/cdui start --host 0\.0\.0\.0/);
    expect(within(dialog).getByText(/localhost:8000 only works on this machine: type an address/)).toBeInTheDocument();
    expect(within(dialog).queryByText(/Use an address the worker can reach:$/)).toBeNull();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Connect over SSH instead' }));
    await settle();
    expect(dialog).toHaveAttribute('data-mode', 'ssh');
  });

  it('says it is loading while it asks the server where it listens, and keeps a tab picked meanwhile', async () => {
    let answer: (value: EnrollmentList) => void = () => {};
    vi.mocked(fetchEnrollments).mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }));
    render(<AddWorkerDialog mode="remote" onClose={vi.fn()} />);
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText('Loading workers...')).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: 'This machine' }));
    await act(async () => answer(listOf({ listening_on_loopback: true })));
    expect(dialog).toHaveAttribute('data-mode', 'same-host');
  });

  it('ignores the answer when it closed before the server answered', async () => {
    let answer: (value: EnrollmentList) => void = () => {};
    let fail: (err: Error) => void = () => {};
    vi.mocked(fetchEnrollments)
      .mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }))
      .mockImplementationOnce(() => new Promise((_, reject) => { fail = reject; }));
    render(<AddWorkerDialog mode="remote" onClose={vi.fn()} />);
    cleanup();
    await act(async () => answer(listOf()));
    render(<AddWorkerDialog mode="remote" onClose={vi.fn()} />);
    cleanup();
    await act(async () => fail(new Error('gone')));
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('generates a token with the chosen lifetime and uses, and builds the command', async () => {
    const dialog = await renderRemote();
    fireEvent.change(within(dialog).getByRole('combobox', { name: 'Valid for' }), { target: { value: '3600' } });
    const uses = within(dialog).getByRole('spinbutton', { name: 'Uses' });
    fireEvent.change(uses, { target: { value: '3' } });
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({ enrollments: [enrollment('abcdef', { uses_left: 3 })] }));
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    expect(createEnrollment).toHaveBeenCalledWith({ ttl_s: 3600, uses: 3, control_url: 'http://192.168.1.5:8000' });
    const command = `cdui agent start --control http://192.168.1.5:8000 --token ${TOKEN}`;
    expect(within(dialog).getByText(command)).toBeInTheDocument();
    expect(within(dialog).getByText('Pending')).toBeInTheDocument();
    expect(within(dialog).getByText(/3 use\(s\) left/)).toBeInTheDocument();

    // Editing the address rebuilds the command; a trailing slash is dropped.
    const address = within(dialog).getByRole('textbox', { name: /Server address/ });
    await act(async () => fireEvent.change(address, { target: { value: ' http://box.lan:8000/ ' } }));
    expect(within(dialog).getByText(`cdui agent start --control http://box.lan:8000 --token ${TOKEN}`))
      .toBeInTheDocument();
    // An emptied field falls back to the address the token was made for.
    await act(async () => fireEvent.change(address, { target: { value: '' } }));
    expect(within(dialog).getByText(command)).toBeInTheDocument();

    await act(async () => fireEvent.click(within(dialog).getAllByRole('button', { name: 'Copy' })[0]));
    expect(writeText).toHaveBeenCalledWith(command);
  });

  it('keeps the uses within 1 to 100', async () => {
    const dialog = await renderRemote();
    const uses = within(dialog).getByRole('spinbutton', { name: 'Uses' });
    fireEvent.change(uses, { target: { value: '500' } });
    expect(uses).toHaveValue(100);
    fireEvent.change(uses, { target: { value: '0' } });
    expect(uses).toHaveValue(1);
    fireEvent.change(uses, { target: { value: '' } });
    expect(uses).toHaveValue(1);
  });

  it('ignores a second press while a token is being generated', async () => {
    let answer: (value: CreatedEnrollment) => void = () => {};
    vi.mocked(createEnrollment).mockImplementation(() => new Promise((resolve) => { answer = resolve; }));
    const dialog = await renderRemote();
    const generate = within(dialog).getByRole('button', { name: 'Generate token' });
    fireEvent.click(generate);
    expect(generate).toHaveAttribute('aria-disabled', 'true');
    fireEvent.click(generate);
    expect(createEnrollment).toHaveBeenCalledTimes(1);
    await act(async () => answer(created()));
    expect(generate).toHaveAttribute('aria-disabled', 'false');
  });

  it('lists the tokens and revokes or deletes them', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({
      enrollments: [
        enrollment('abcdef', { label: 'lab box' }),
        enrollment('old111', { state: 'expired' }),
        enrollment('use222', { state: 'used', uses_left: 0 }),
      ],
    }));
    const dialog = await renderRemote();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    const shown = /--token abcdef\./;
    expect(within(dialog).getByText(shown)).toBeInTheDocument();
    expect(within(dialog).getByText('lab box')).toBeInTheDocument();
    expect(within(dialog).getByText('Expired')).toBeInTheDocument();
    expect(within(dialog).getByText('Used')).toBeInTheDocument();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Delete old111' })));
    expect(deleteEnrollment).toHaveBeenCalledWith('old111');
    // Another token's revoke leaves the command shown.
    expect(within(dialog).getByText(shown)).toBeInTheDocument();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Revoke abcdef' })));
    expect(deleteEnrollment).toHaveBeenLastCalledWith('abcdef');
    // Revoking the token just shown takes its command away too.
    expect(within(dialog).queryByText(shown)).toBeNull();
    expect(fetchEnrollments).toHaveBeenCalledTimes(4);
  });

  it('shows what failed', async () => {
    // The dialog's own read fails, so it opens on this tab, whose read fails too.
    vi.mocked(fetchEnrollments)
      .mockRejectedValueOnce(new Error('Missing or invalid X-CodefyUI-Token header'))
      .mockRejectedValueOnce(new Error('Missing or invalid X-CodefyUI-Token header'));
    const dialog = await renderRemote();
    expect(within(dialog).getByRole('alert')).toHaveTextContent(/X-CodefyUI-Token/);

    vi.mocked(createEnrollment).mockRejectedValueOnce('worker enrollment is not available');
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    expect(within(dialog).getByRole('alert')).toHaveTextContent(/worker enrollment is not available/);
  });

  it('shows a failed revoke and clears the error on the next good read', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({ enrollments: [enrollment('abcdef')] }));
    vi.mocked(deleteEnrollment).mockRejectedValueOnce(new Error("no enrollment token 'abcdef'"));
    const dialog = await renderRemote();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Revoke abcdef' })));
    expect(within(dialog).getByRole('alert')).toHaveTextContent(/no enrollment token/);
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Revoke abcdef' })));
    expect(within(dialog).queryByRole('alert')).toBeNull();
  });

  it('renders in zh-TW', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const dialog = await renderRemote();
    expect(within(dialog).getByRole('button', { name: '產生 token' })).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: '另一台機器' })).toHaveAttribute('aria-pressed', 'true');
  });
});

describe('AddWorkerDialog (trust, #786)', () => {
  const FP = `sha256:${'ab'.repeat(32)}`;
  const OPENSSL = 'openssl x509 -in /path/to/fullchain.pem -pubkey -noout | openssl dgst -sha256';

  async function type(dialog: HTMLElement, value: string) {
    await act(async () => {
      fireEvent.change(within(dialog).getByRole('textbox', { name: /Server address/ }), { target: { value } });
    });
  }

  it('over https without a readable certificate shows the openssl command', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({
      control_url: 'https://cp.lab', tls_fingerprint: null, fingerprint_command: OPENSSL,
    }));
    const dialog = await renderRemote();
    expect(within(dialog).getByRole('note')).toHaveTextContent(/compute its fingerprint on the proxy host/);
    expect(within(dialog).getByText(OPENSSL)).toBeInTheDocument();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    expect(createEnrollment).toHaveBeenCalledWith({ ttl_s: 86400, uses: 1, control_url: 'https://cp.lab' });
    expect(within(dialog).getByText(`cdui agent start --control https://cp.lab --token ${TOKEN}`)).toBeInTheDocument();
  });

  it('shows the command the server built, and asks it again for an edited address', async () => {
    serverFp = FP;
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({
      control_url: 'https://cp.lab', tls_fingerprint: FP, fingerprint_command: OPENSSL,
    }));
    const dialog = await renderRemote();
    expect(within(dialog).getByRole('note')).toHaveTextContent(FP);
    expect(within(dialog).queryByText(OPENSSL)).toBeNull();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    expect(within(dialog).getByText(
      `cdui agent start --control https://cp.lab --token ${TOKEN} --ca-fingerprint ${FP}`,
    )).toBeInTheDocument();
    expect(agentCommand).not.toHaveBeenCalled();

    // A plain http address: the server leaves the pin out; the note warns.
    await type(dialog, 'http://10.0.0.7:8000');
    expect(agentCommand).toHaveBeenLastCalledWith({ control_url: 'http://10.0.0.7:8000', token: TOKEN, worker: null });
    expect(within(dialog).getByText(`cdui agent start --control http://10.0.0.7:8000 --token ${TOKEN}`))
      .toBeInTheDocument();
    expect(within(dialog).getByRole('note')).toHaveTextContent(/plain HTTP/);

    // Half-typed: no request, and no stale command.
    await type(dialog, 'http');
    expect(agentCommand).toHaveBeenCalledTimes(1);
    expect(within(dialog).queryByText(/cdui agent start --control/)).toBeNull();
  });

  it('judges the address the command uses: the server\'s own when the field is empty', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({ control_url: 'https://cp.lab', fingerprint_command: OPENSSL }));
    const dialog = await renderRemote();
    await type(dialog, '');
    expect(within(dialog).getByRole('note')).toHaveTextContent(/compute its fingerprint/);
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    // No address typed: the server's default, which the token's command uses.
    expect(createEnrollment).toHaveBeenLastCalledWith({ ttl_s: 86400, uses: 1 });
    expect(within(dialog).getByText(`cdui agent start --control http://192.168.1.5:8000 --token ${TOKEN}`))
      .toBeInTheDocument();
    expect(within(dialog).getByRole('note')).toHaveTextContent(/plain HTTP/);
  });

  it('shows a refused address', async () => {
    const dialog = await renderRemote();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    vi.mocked(agentCommand).mockRejectedValue(new Error('control_url must be http(s)://host[:port][/path]'));
    await type(dialog, 'http://bad;host');
    expect(within(dialog).getByRole('alert')).toHaveTextContent(/control_url must be/);
  });

  it('drops an answer for an address that changed meanwhile', async () => {
    const dialog = await renderRemote();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    let answer: (value: string) => void = () => {};
    let refuse: (reason: Error) => void = () => {};
    let fresh: (value: string) => void = () => {};
    vi.mocked(agentCommand)
      .mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }))
      .mockImplementationOnce(() => new Promise((_, reject) => { refuse = reject; }))
      .mockImplementationOnce(() => new Promise((resolve) => { fresh = resolve; }));
    await type(dialog, 'http://a.lan:8000');
    await type(dialog, 'http://b.lan:8000');
    await type(dialog, 'http://c.lan:8000');
    await act(async () => {
      answer('stale command');
      refuse(new Error('stale error'));
      fresh('fresh command');
    });
    expect(within(dialog).queryByText('stale command')).toBeNull();
    expect(within(dialog).queryByRole('alert')).toBeNull();
    expect(within(dialog).getByText('fresh command')).toBeInTheDocument();
  });

  it('over https before the list arrives shows the advice without a command', async () => {
    vi.mocked(fetchEnrollments).mockRejectedValue(new Error('offline'));
    render(<AddWorkerDialog mode="remote" onClose={vi.fn()} />);
    await settle();
    const dialog = screen.getByRole('dialog');
    await type(dialog, 'https://cp.lab');
    expect(within(dialog).getByRole('note')).toHaveTextContent(/compute its fingerprint/);
    expect(within(dialog).queryByRole('button', { name: 'Copy' })).toBeNull();
  });

  it('says nothing about trust without any address', async () => {
    vi.mocked(fetchEnrollments).mockRejectedValue(new Error('offline'));
    render(<AddWorkerDialog mode="remote" onClose={vi.fn()} />);
    await settle();
    expect(within(screen.getByRole('dialog')).queryByRole('note')).toBeNull();
  });

  it('renders in zh-TW', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({ control_url: 'https://cp.lab', tls_fingerprint: FP }));
    const dialog = await renderRemote();
    expect(within(dialog).getByRole('note')).toHaveTextContent(/釘選/);
  });
});

describe('isHttpsUrl', () => {
  it('knows an https address', () => {
    expect(isHttpsUrl('https://cp.lab')).toBe(true);
    expect(isHttpsUrl('HTTPS://cp.lab:8443')).toBe(true);
    expect(isHttpsUrl('http://cp.lab')).toBe(false);
    expect(isHttpsUrl('')).toBe(false);
  });
});

describe('workerNameProblem', () => {
  it('reads a name as the server does', () => {
    expect(workerNameProblem('')).toBeNull();
    expect(workerNameProblem('  gpu-1 ')).toBeNull();
    expect(workerNameProblem('GPU-1')).toBeNull();
    expect(workerNameProblem('-gpu')).toBe('workers.add.ssh.badName');
    expect(workerNameProblem('a'.repeat(64))).toBe('workers.add.ssh.badName');
    expect(workerNameProblem('gpu_1')).toBe('workers.add.ssh.badName');
    expect(workerNameProblem('any')).toBe('workers.add.ssh.reservedName');
    expect(workerNameProblem('Enrollments')).toBe('workers.add.ssh.reservedName');
  });
});

describe('isLoopbackUrl', () => {
  it('knows the addresses that only work on this machine', () => {
    expect(isLoopbackUrl('http://127.0.0.1:8000')).toBe(true);
    expect(isLoopbackUrl('http://127.1.2.3:8000')).toBe(true);
    expect(isLoopbackUrl('http://localhost:8000')).toBe(true);
    expect(isLoopbackUrl('http://[::1]:8000')).toBe(true);
    expect(isLoopbackUrl('http://10.0.0.7:8000')).toBe(false);
    expect(isLoopbackUrl('not a url')).toBe(false);
    expect(isLoopbackUrl('')).toBe(false);
  });
});

describe('AddWorkerDialog (re-key, #780)', () => {
  it('generates a token bound to the worker and names it in the command', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({ enrollments: [enrollment('abcdef', { worker: 'gpu-box' })] }));
    render(<AddWorkerDialog mode="remote" rekey="gpu-box" onClose={vi.fn()} />);
    await settle();
    // Its own title, no mode switch (#840).
    const dialog = screen.getByRole('dialog', { name: 'Re-key gpu-box' });
    expect(within(dialog).queryByRole('button', { name: 'Over SSH' })).toBeNull();
    expect(within(dialog).getByText(/Re-key gpu-box: generate/)).toBeInTheDocument();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    expect(createEnrollment).toHaveBeenCalledWith({
      ttl_s: 86400, uses: 1, worker: 'gpu-box', control_url: 'http://192.168.1.5:8000',
    });
    expect(within(dialog).getByText(
      `cdui agent start --control http://192.168.1.5:8000 --token ${TOKEN} --name gpu-box`,
    )).toBeInTheDocument();
    expect(within(dialog).getByText('re-keys gpu-box')).toBeInTheDocument();
  });
});

function sshInfo(extra: Partial<SshWorkerList> = {}): SshWorkerList {
  return { ssh_workers: [], ssh_available: true, port_range: [18000, 18100], default_remote_cdui: '~/CodefyUI/cdui', ...extra };
}

function sshEntry(extra: Partial<SshWorker> = {}): SshWorker {
  return {
    id: 'gpu-box',
    target: 'alice@gpu-box',
    port: 22,
    identity: null,
    remote_cdui: '~/CodefyUI/cdui',
    remote_port: 18000,
    enabled: true,
    state: 'connecting',
    last_error: null,
    exit_code: null,
    tail: [],
    attempts: 1,
    retry_at: null,
    up_since: null,
    ...extra,
  };
}

async function renderSsh(props: { onClose?: () => void; onAdded?: () => void } = {}) {
  const onClose = props.onClose ?? vi.fn();
  render(<AddWorkerDialog mode="ssh" onClose={onClose} onAdded={props.onAdded} />);
  await settle();
  return { onClose, dialog: screen.getByRole('dialog') };
}

describe('AddWorkerDialog (over SSH, #784)', () => {
  it('opens on the SSH form, and the mode switch reaches it from another tab', async () => {
    const { dialog } = await renderSsh();
    expect(dialog).toHaveAttribute('data-mode', 'ssh');
    expect(within(dialog).getByRole('button', { name: 'Over SSH' })).toHaveAttribute('aria-pressed', 'true');
    expect(within(dialog).getByLabelText(/SSH target/)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/^Port/)).toHaveValue(22);
    expect(within(dialog).getByLabelText(/cdui on the remote host/)).toHaveAttribute('placeholder', '~/CodefyUI/cdui');
    expect(within(dialog).queryByRole('note')).toBeNull();
    fireEvent.click(within(dialog).getByRole('button', { name: 'This machine' }));
    expect(dialog).toHaveAttribute('data-mode', 'same-host');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Over SSH' }));
    await settle();
    expect(dialog).toHaveAttribute('data-mode', 'ssh');
  });

  it('checks the worker name before it sends anything, quoting it as typed (#840)', async () => {
    const { dialog } = await renderSsh();
    fireEvent.change(within(dialog).getByLabelText(/SSH target/), { target: { value: 'alice@gpu-box' } });
    const name = within(dialog).getByLabelText(/Worker name/);
    fireEvent.change(name, { target: { value: ' Bad Name! ' } });
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Add and connect' })));
    expect(within(dialog).getByRole('alert')).toHaveTextContent('Worker name "Bad Name!" may use only letters');
    fireEvent.change(name, { target: { value: 'Local' } });
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Add and connect' })));
    expect(within(dialog).getByRole('alert')).toHaveTextContent('"Local" is reserved');
    expect(addSshWorker).not.toHaveBeenCalled();
    // Upper case is not a fault: the server stores it lower-cased.
    fireEvent.change(name, { target: { value: 'GPU-1' } });
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Add and connect' })));
    expect(addSshWorker).toHaveBeenCalledWith({ target: 'alice@gpu-box', port: 22, name: 'GPU-1' });
  });

  it('says a bad name in zh-TW', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const { dialog } = await renderSsh();
    fireEvent.change(within(dialog).getByLabelText(/SSH 目標/), { target: { value: 'alice@gpu-box' } });
    fireEvent.change(within(dialog).getByLabelText(/Worker 名稱/), { target: { value: 'Bad Name!' } });
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: '新增並連線' })));
    expect(within(dialog).getByRole('alert')).toHaveTextContent('Worker 名稱「Bad Name!」只能使用');
  });

  it('asks for a target before it sends anything', async () => {
    const { dialog } = await renderSsh();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Add and connect' })));
    expect(within(dialog).getByRole('alert')).toHaveTextContent('Enter the SSH target');
    expect(addSshWorker).not.toHaveBeenCalled();
  });

  it('sends every field it was given, then says so, refreshes and closes', async () => {
    const onAdded = vi.fn();
    const { dialog, onClose } = await renderSsh({ onAdded });
    fireEvent.change(within(dialog).getByLabelText(/SSH target/), { target: { value: ' alice@gpu-box ' } });
    fireEvent.change(within(dialog).getByLabelText(/^Port/), { target: { value: '2222' } });
    fireEvent.change(within(dialog).getByLabelText(/Worker name/), { target: { value: ' gpu-1 ' } });
    fireEvent.change(within(dialog).getByLabelText(/Identity file/), { target: { value: '~/.ssh/lab' } });
    fireEvent.change(within(dialog).getByLabelText(/cdui on the remote host/), { target: { value: '/opt/CodefyUI/cdui' } });
    vi.mocked(addSshWorker).mockResolvedValue(sshEntry({ id: 'gpu-1' }));
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Add and connect' })));
    expect(addSshWorker).toHaveBeenCalledWith({
      target: 'alice@gpu-box', port: 2222, name: 'gpu-1', identity: '~/.ssh/lab', remote_cdui: '/opt/CodefyUI/cdui',
    });
    expect(useToastStore.getState().toasts.map((t) => t.message)).toEqual(['SSH worker gpu-1 added; connecting']);
    expect(onAdded).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('sends only the target and port 22 by default, and works without onAdded', async () => {
    const { dialog, onClose } = await renderSsh();
    fireEvent.change(within(dialog).getByLabelText(/SSH target/), { target: { value: 'gpu-box' } });
    fireEvent.change(within(dialog).getByLabelText(/^Port/), { target: { value: '' } });
    await act(async () => fireEvent.submit(within(dialog).getByLabelText(/SSH target/)));
    expect(addSshWorker).toHaveBeenCalledWith({ target: 'gpu-box', port: 22 });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('shows the server\'s refusal and stays open; a second press while sending does nothing', async () => {
    const { dialog, onClose } = await renderSsh();
    fireEvent.change(within(dialog).getByLabelText(/SSH target/), { target: { value: 'gpu-box' } });
    let fail: (err: Error) => void = () => {};
    vi.mocked(addSshWorker).mockImplementationOnce(() => new Promise((_, reject) => { fail = reject; }));
    const button = within(dialog).getByRole('button', { name: 'Add and connect' });
    await act(async () => fireEvent.click(button));
    expect(button).toHaveAttribute('aria-disabled', 'true');
    await act(async () => fireEvent.click(button));
    expect(addSshWorker).toHaveBeenCalledTimes(1);
    await act(async () => fail(new Error('a worker named gpu-box exists already')));
    expect(within(dialog).getByRole('alert')).toHaveTextContent('exists already');
    expect(onClose).not.toHaveBeenCalled();
    expect(button).toHaveAttribute('aria-disabled', 'false');
  });

  it('warns when the server has no ssh, and shows a failed read', async () => {
    vi.mocked(fetchSshWorkers).mockResolvedValueOnce(sshInfo({ ssh_available: false, default_remote_cdui: '/srv/cdui' }));
    const { dialog } = await renderSsh();
    expect(within(dialog).getByRole('note')).toHaveTextContent('No ssh command was found');
    expect(within(dialog).getByLabelText(/cdui on the remote host/)).toHaveAttribute('placeholder', '/srv/cdui');
    cleanup();
    vi.mocked(fetchSshWorkers).mockRejectedValueOnce(new Error('SSH workers are not available'));
    const again = await renderSsh();
    expect(within(again.dialog).getByRole('alert')).toHaveTextContent('SSH workers are not available');
  });

  it('renders in zh-TW', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    const { dialog } = await renderSsh();
    expect(within(dialog).getByRole('button', { name: '透過 SSH' })).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: '新增並連線' })).toBeInTheDocument();
  });
});
