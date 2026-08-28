import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { PackGpuInfo, PackSummary } from '../../api/rest';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { confirm } from '../../utils/dialog';
import { GpuPackDetails } from './GpuPackDetails';

// The confirm is an in-app modal driven by a promise; mocking the helper keeps
// these tests about the CARD's decisions rather than about the dialog.
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const confirmMock = vi.mocked(confirm);

function pack(over: Partial<PackSummary> = {}): PackSummary {
  return {
    id: 'gpu-torch',
    title: 'GPU PyTorch',
    description: 'Switch the wheel',
    install_mode: 'restart',
    status: 'not_installed',
    pip_ready: false,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: 'cdui install --gpu cu128',
    ...over,
  };
}

function gpu(over: Partial<PackGpuInfo> = {}): PackGpuInfo {
  return {
    detected_label: 'NVIDIA GeForce RTX 4080',
    recommended_variant: 'cu128',
    installed_variant: 'cpu',
    variants: ['cu128', 'cu126', 'cpu'],
    install_command: 'cdui install --gpu auto',
    ...over,
  };
}

let writeText: ReturnType<typeof vi.fn>;

function renderCard(props: Partial<Parameters<typeof GpuPackDetails>[0]> = {}) {
  const onInstall = vi.fn();
  render(
    <GpuPackDetails
      pack={pack()}
      gpu={gpu()}
      launchMode="start"
      canInstall
      busy={false}
      restartAvailable={false}
      onInstall={onInstall}
      {...props}
    />,
  );
  return onInstall;
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  confirmMock.mockReset();
  confirmMock.mockResolvedValue(true);
  writeText = vi.fn(async () => undefined);
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText },
    configurable: true,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('GpuPackDetails — what this machine has', () => {
  it('names the detected GPU and both builds', () => {
    renderCard();
    expect(
      screen.getByText('Detected GPU: NVIDIA GeForce RTX 4080'),
    ).toBeInTheDocument();
    expect(screen.getByText('Installed build: cpu')).toBeInTheDocument();
    expect(screen.getByText('Recommended build: cu128')).toBeInTheDocument();
  });

  it('says so when there is no GPU, and stays quiet about a build it cannot name', () => {
    renderCard({
      gpu: gpu({ detected_label: null, installed_variant: null, recommended_variant: null }),
    });
    expect(
      screen.getByText('No GPU detected. The CPU build of PyTorch is already installed.'),
    ).toBeInTheDocument();
    // `installed_variant: null` is "cannot tell", not "none" — so no line at all.
    expect(screen.queryByText(/Installed build/)).toBeNull();
    expect(screen.queryByText(/Recommended build/)).toBeNull();
  });

  it('offers the build choice, and hides the select when there is only one', () => {
    renderCard();
    const select = screen.getByLabelText('PyTorch build') as HTMLSelectElement;
    expect(select.value).toBe('cu128');
    expect(Array.from(select.options).map((o) => o.value)).toEqual([
      'cu128',
      'cu126',
      'cpu',
    ]);
  });

  it('hides the select when the machine offers one build', () => {
    renderCard({ gpu: gpu({ variants: ['cpu'], recommended_variant: 'cpu' }) });
    expect(screen.queryByLabelText('PyTorch build')).toBeNull();
  });
});

describe('GpuPackDetails — the command block', () => {
  it('explains why the app cannot do it and shows the command in start mode', () => {
    renderCard({ launchMode: 'start' });
    expect(
      screen.getByText(
        'Switching the PyTorch build from inside the app is not available yet. Run this in a terminal with the server stopped:',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('cdui install --gpu cu128')).toBeInTheDocument();
    // No install button: this build cannot start a restart-mode job.
    expect(screen.queryByRole('button', { name: 'Install and restart' })).toBeNull();
  });

  it('names cdui dev as the reason in dev mode', () => {
    renderCard({ launchMode: 'dev' });
    expect(
      screen.getByText(
        'You started CodefyUI with cdui dev, so the server cannot restart itself. Run this in the backend terminal, then start it again:',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('cdui install --gpu cu128')).toBeInTheDocument();
  });

  it('gives an unknown launch mode the neutral sentence, not the dev one', () => {
    // `unknown` means no catalog has answered (or a server too old to say).
    // Telling a reader they started CodefyUI with `cdui dev` when we do not
    // know that is a guess; the start-mode sentence is true either way.
    renderCard({ launchMode: 'unknown' });
    expect(
      screen.getByText(
        'Switching the PyTorch build from inside the app is not available yet. Run this in a terminal with the server stopped:',
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/cdui dev/)).toBeNull();
  });

  it('falls back to the GPU-wide command when the pack has none', () => {
    renderCard({ pack: pack({ install_command: null }) });
    expect(screen.getByText('cdui install --gpu auto')).toBeInTheDocument();
  });

  it('says the server gave no command when neither side has one', () => {
    renderCard({
      pack: pack({ install_command: null }),
      gpu: gpu({ install_command: null }),
    });
    expect(
      screen.getByText(
        'The server did not provide an install command. See the README for the GPU install steps.',
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Copy command' })).toBeNull();
  });

  it('copies the command and says so', async () => {
    renderCard();
    fireEvent.click(screen.getByRole('button', { name: 'Copy command' }));
    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith('cdui install --gpu cu128'),
    );
    await waitFor(() =>
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: 'Copied to clipboard.',
        type: 'success',
      }),
    );
  });

  it('tells the user to copy it by hand when the clipboard refuses', async () => {
    writeText.mockRejectedValue(new Error('denied'));
    renderCard();
    fireEvent.click(screen.getByRole('button', { name: 'Copy command' }));
    await waitFor(() =>
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: 'Could not copy. Select the text and copy it by hand.',
        type: 'error',
      }),
    );
  });

  it('survives a browser with no clipboard API at all', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: undefined,
      configurable: true,
    });
    renderCard();
    fireEvent.click(screen.getByRole('button', { name: 'Copy command' }));
    await waitFor(() =>
      expect(useToastStore.getState().toasts[0]).toMatchObject({
        message: 'Could not copy. Select the text and copy it by hand.',
      }),
    );
  });
});

describe('GpuPackDetails — the restart path PR 5 turns on', () => {
  it('offers install-and-restart, and asks first', async () => {
    const onInstall = renderCard({ restartAvailable: true });
    expect(
      screen.getByText('The server restarts after this install. Running graphs will stop.'),
    ).toBeInTheDocument();
    // The command block is what the app shows INSTEAD of this, not alongside.
    expect(screen.queryByRole('button', { name: 'Copy command' })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Install and restart' }));
    await waitFor(() => expect(onInstall).toHaveBeenCalledWith('cu128'));
    expect(confirmMock).toHaveBeenCalledWith(
      expect.objectContaining({
        message: 'Install cu128 and restart the server?',
        variant: 'danger',
      }),
    );
  });

  it('installs the build the user picked, not the recommended one', async () => {
    const onInstall = renderCard({ restartAvailable: true });
    fireEvent.change(screen.getByLabelText('PyTorch build'), {
      target: { value: 'cu126' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Install and restart' }));
    await waitFor(() => expect(onInstall).toHaveBeenCalledWith('cu126'));
  });

  it('does nothing when the confirm is declined', async () => {
    confirmMock.mockResolvedValue(false);
    const onInstall = renderCard({ restartAvailable: true });
    fireEvent.click(screen.getByRole('button', { name: 'Install and restart' }));
    await waitFor(() => expect(confirmMock).toHaveBeenCalled());
    expect(onInstall).not.toHaveBeenCalled();
  });

  it('keeps the command block in dev mode even once restarts are supported', () => {
    renderCard({ restartAvailable: true, launchMode: 'dev' });
    // Nothing supervises a `cdui dev` server, so there is nothing to restart.
    expect(screen.queryByRole('button', { name: 'Install and restart' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Copy command' })).toBeInTheDocument();
  });

  it('disables install-and-restart when remote installs are refused', () => {
    renderCard({ restartAvailable: true, canInstall: false });
    const button = screen.getByRole('button', { name: 'Install and restart' });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute(
      'title',
      'Installing is only allowed from the computer that runs the server.',
    );
  });

  it('disables install-and-restart while this pack already has a request in flight', () => {
    renderCard({ restartAvailable: true, busy: true });
    expect(screen.getByRole('button', { name: 'Install and restart' })).toBeDisabled();
  });
});
