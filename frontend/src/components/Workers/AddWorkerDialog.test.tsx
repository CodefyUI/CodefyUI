import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import {
  createEnrollment,
  deleteEnrollment,
  fetchEnrollments,
  type CreatedEnrollment,
  type Enrollment,
  type EnrollmentList,
} from '../../api/workers';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { settle } from '../../test/utils';
import { AGENT_START_COMMAND, AGENT_STOP_COMMAND, AddWorkerDialog, isLoopbackUrl } from './AddWorkerDialog';

vi.mock('../../api/workers', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/workers')>()),
  fetchEnrollments: vi.fn(),
  createEnrollment: vi.fn(),
  deleteEnrollment: vi.fn(),
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

const writeText = vi.fn();

beforeEach(() => {
  vi.mocked(fetchEnrollments).mockReset().mockResolvedValue(listOf());
  vi.mocked(createEnrollment).mockReset().mockResolvedValue(created());
  vi.mocked(deleteEnrollment).mockReset().mockResolvedValue(undefined);
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
    expect(within(dialog).queryByRole('note')).toBeNull();
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

  it('says so when the server cannot be reached from another machine at all', async () => {
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({
      control_url: 'http://localhost:8000', other_machine_urls: [], listening_on_loopback: true,
    }));
    const dialog = await renderRemote();
    expect(within(dialog).getByRole('note')).toHaveTextContent(/cdui start --host 0\.0\.0\.0/);
    expect(within(dialog).getByText(/localhost:8000 only works/)).toBeInTheDocument();
  });

  it('generates a token with the chosen lifetime and uses, and builds the command', async () => {
    const dialog = await renderRemote();
    fireEvent.change(within(dialog).getByRole('combobox', { name: 'Valid for' }), { target: { value: '3600' } });
    const uses = within(dialog).getByRole('spinbutton', { name: 'Uses' });
    fireEvent.change(uses, { target: { value: '3' } });
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({ enrollments: [enrollment('abcdef', { uses_left: 3 })] }));
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    expect(createEnrollment).toHaveBeenCalledWith({ ttl_s: 3600, uses: 3 });
    const command = `cdui agent start --control http://192.168.1.5:8000 --token ${TOKEN}`;
    expect(within(dialog).getByText(command)).toBeInTheDocument();
    expect(within(dialog).getByText('Pending')).toBeInTheDocument();
    expect(within(dialog).getByText(/3 use\(s\) left/)).toBeInTheDocument();

    // Editing the address rebuilds the command; a trailing slash is dropped.
    const address = within(dialog).getByRole('textbox', { name: /Server address/ });
    fireEvent.change(address, { target: { value: ' http://box.lan:8000/ ' } });
    expect(within(dialog).getByText(`cdui agent start --control http://box.lan:8000 --token ${TOKEN}`))
      .toBeInTheDocument();
    // An emptied field falls back to the server's own idea of its address.
    fireEvent.change(address, { target: { value: '' } });
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
    vi.mocked(fetchEnrollments).mockRejectedValueOnce(new Error('Missing or invalid X-CodefyUI-Token header'));
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
    vi.mocked(createEnrollment).mockResolvedValue(created({ worker: 'gpu-box' }));
    vi.mocked(fetchEnrollments).mockResolvedValue(listOf({ enrollments: [enrollment('abcdef', { worker: 'gpu-box' })] }));
    render(<AddWorkerDialog mode="remote" rekey="gpu-box" onClose={vi.fn()} />);
    await settle();
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText(/Re-key gpu-box/)).toBeInTheDocument();
    await act(async () => fireEvent.click(within(dialog).getByRole('button', { name: 'Generate token' })));
    expect(createEnrollment).toHaveBeenCalledWith({ ttl_s: 86400, uses: 1, worker: 'gpu-box' });
    expect(within(dialog).getByText(
      `cdui agent start --control http://192.168.1.5:8000 --token ${TOKEN} --name gpu-box`,
    )).toBeInTheDocument();
    expect(within(dialog).getByText('re-keys gpu-box')).toBeInTheDocument();
  });
});
