import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import type { PackGpuInfo, PackItem, PackSummary } from '../../api/rest';
import { useDialogStore } from '../../store/dialogStore';
import {
  emptyPackJob,
  usePackStore,
  _resetPackStoreForTesting,
  type PackJob,
} from '../../store/packStore';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { STICK_PX } from './PackLogTail';
import { HIGHLIGHT_MS, PackCenterModal } from './PackCenterModal';

function item(over: Partial<PackItem> & { id: string }): PackItem {
  return {
    kind: 'hf',
    repo_id: `sentence-transformers/${over.id}`,
    url: null,
    size_bytes: 90 * 1024 * 1024,
    license: 'apache-2.0',
    status: 'missing',
    ...over,
  };
}

function pack(over: Partial<PackSummary> & { id: string }): PackSummary {
  return {
    title: `server title for ${over.id}`,
    description: 'server description',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: false,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...over,
  };
}

const embeddings = pack({
  id: 'sentence-embeddings',
  status: 'partial',
  size_bytes_total: 100 * 1024 * 1024,
  items: [
    item({ id: 'all-MiniLM-L6-v2', status: 'missing', size_bytes: 90 * 1024 * 1024 }),
    item({ id: 'labse', status: 'present', size_bytes: 10 * 1024 * 1024 }),
  ],
});

const words = pack({
  id: 'word-vectors',
  status: 'installed',
  items: [item({ id: 'glove-6b-50d', status: 'present' })],
});

const gpuInfo: PackGpuInfo = {
  detected_label: 'NVIDIA GeForce RTX 4080',
  recommended_variant: 'cu128',
  installed_variant: 'cpu',
  variants: ['cu128', 'cpu'],
  install_command: 'cdui install --gpu auto',
};

/** Fresh mocks for every action the panel is allowed to call. */
function makeActions() {
  return {
    refresh: vi.fn(async () => {}),
    install: vi.fn(async () => {}),
    cancel: vi.fn(async () => {}),
    removeItem: vi.fn(async () => {}),
    dismissJob: vi.fn(() => {}),
    stopFollowing: vi.fn(() => {}),
  };
}

let actions: ReturnType<typeof makeActions>;

/**
 * Seed the store the panel reads, with fresh mock actions installed through
 * `setState`. Never `vi.spyOn(usePackStore.getState(), ...)`: that spies on a
 * snapshot object, and the history leaks between cases.
 */
function seed(state: Partial<ReturnType<typeof usePackStore.getState>> = {}) {
  actions = makeActions();
  const packs = state.packs ?? [];
  usePackStore.setState({
    packs,
    byId: Object.fromEntries(packs.map((p) => [p.id, p])),
    loading: false,
    loaded: true,
    unsupported: false,
    error: null,
    remoteInstallAllowed: true,
    launchMode: 'start',
    // A server that has NOT said it can restart itself, so every case that
    // wants the restart path has to say so.
    restartAvailable: false,
    gpu: null,
    job: null,
    busy: {},
    cancelling: false,
    ...actions,
    ...state,
  });
}

function job(over: Partial<PackJob> = {}): PackJob {
  return { ...emptyPackJob('j1', 'sentence-embeddings'), ...over };
}

function open(packId?: string) {
  useUIStore.getState().openPackCenter(packId);
}

const cardFor = (packId: string) =>
  document.querySelector(`[data-pack-id="${packId}"]`) as HTMLElement;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  useDialogStore.setState({ active: null });
  useUIStore.setState({
    packCenterOpen: false,
    packCenterFocusPackId: null,
    shortcutsModalOpen: false,
  });
  _resetPackStoreForTesting();
  seed();
});

afterEach(() => {
  // Inside act(): this hook runs BEFORE Testing Library's own cleanup, so the
  // panel is still mounted and subscribed when `packCenterOpen` goes false —
  // closing it here is a real React update, and unwrapped it printed an
  // "update was not wrapped in act(...)" line for every case in this file.
  act(() => {
    useUIStore.setState({
      packCenterOpen: false,
      packCenterFocusPackId: null,
      shortcutsModalOpen: false,
    });
    useDialogStore.setState({ active: null });
  });
  vi.restoreAllMocks();
});

// ── Mounting ────────────────────────────────────────────────────────────────

describe('PackCenterModal — mounting', () => {
  it('renders nothing, and asks the server nothing, while it is closed', () => {
    render(<PackCenterModal />);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(actions.refresh).not.toHaveBeenCalled();
  });

  it('reads the catalog once when it opens', () => {
    seed({ packs: [words] });
    open();
    render(<PackCenterModal />);
    expect(screen.getByRole('dialog', { name: 'Package Center' })).toBeInTheDocument();
    expect(actions.refresh).toHaveBeenCalledTimes(1);
  });

  it('re-reads on the refresh button', () => {
    seed({ packs: [words] });
    open();
    render(<PackCenterModal />);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh pack status' }));
    expect(actions.refresh).toHaveBeenCalledTimes(2);
  });
});

// ── The list ────────────────────────────────────────────────────────────────

describe('PackCenterModal — the pack list', () => {
  it('names every pack in the reader language, with its state and its size', () => {
    seed({ packs: [embeddings, words] });
    open();
    render(<PackCenterModal />);

    expect(screen.getByRole('region', { name: 'Pack list' })).toBeInTheDocument();
    expect(
      screen.getByRole('complementary', { name: 'Install activity' }),
    ).toBeInTheDocument();

    expect(screen.getByText('Sentence embeddings')).toBeInTheDocument();
    expect(screen.getByText('Word vectors (GloVe)')).toBeInTheDocument();
    expect(screen.getByText('Partly installed')).toBeInTheDocument();
    expect(screen.getByText('Installed')).toBeInTheDocument();
    expect(screen.getByText('Download size: 100 MB')).toBeInTheDocument();
  });

  it('says it is loading before the first answer', () => {
    seed({ loading: true, loaded: false, packs: [] });
    open();
    render(<PackCenterModal />);
    expect(screen.getByText('Loading packs...')).toBeInTheDocument();
  });

  it('reports a failed read and offers to try again', () => {
    seed({ loaded: false, error: 'connection refused', packs: [] });
    open();
    render(<PackCenterModal />);
    expect(
      screen.getByText('Failed to load packs: connection refused'),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(actions.refresh).toHaveBeenCalledTimes(2);
  });

  it('keeps the rows it has when a refresh fails, and still says it failed', () => {
    seed({ packs: [words], error: 'connection refused' });
    open();
    render(<PackCenterModal />);
    // One dropped packet must not empty the Package Center.
    expect(screen.getByText('Word vectors (GloVe)')).toBeInTheDocument();
    expect(
      screen.getByText('Failed to load packs: connection refused'),
    ).toBeInTheDocument();
  });

  it('explains a server that predates the Package Center', () => {
    seed({ unsupported: true, packs: [] });
    open();
    render(<PackCenterModal />);
    expect(
      screen.getByText(
        'This server does not support the Package Center. Update CodefyUI and restart it.',
      ),
    ).toBeInTheDocument();
  });

  it('says so when the server ships no optional packs', () => {
    seed({ packs: [] });
    open();
    render(<PackCenterModal />);
    expect(screen.getByText('No optional packs are available')).toBeInTheDocument();
  });
});

// ── Installing ──────────────────────────────────────────────────────────────

describe('PackCenterModal — installing', () => {
  it('starts with the missing items ticked and installs exactly those', () => {
    seed({ packs: [embeddings] });
    open();
    render(<PackCenterModal />);

    expect(screen.getByLabelText('sentence-transformers/all-MiniLM-L6-v2')).toBeChecked();
    fireEvent.click(screen.getByRole('button', { name: 'Install selected' }));
    expect(actions.install).toHaveBeenCalledWith('sentence-embeddings', {
      items: ['all-MiniLM-L6-v2'],
      mode: 'live',
    });
  });

  it('removes a downloaded item through the store, which asks first', () => {
    seed({ packs: [embeddings] });
    open();
    render(<PackCenterModal />);
    fireEvent.click(
      screen.getByRole('button', { name: 'Remove sentence-transformers/labse' }),
    );
    expect(actions.removeItem).toHaveBeenCalledWith('sentence-embeddings', 'labse');
  });

  it('refuses and explains when the server only allows local installs', () => {
    seed({ packs: [embeddings], remoteInstallAllowed: false });
    open();
    render(<PackCenterModal />);

    const button = screen.getByRole('button', { name: 'Install selected' });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute(
      'title',
      'Installing is only allowed from the computer that runs the server.',
    );
    // And said once in the footer, where it does not need a hover to find.
    expect(
      screen.getByText('Installing is only allowed from the computer that runs the server.'),
    ).toBeInTheDocument();
    expect(actions.install).not.toHaveBeenCalled();
  });

  it('hands over the command for the GPU pack, which no server can install to itself', () => {
    seed({
      packs: [
        pack({
          id: 'gpu-torch',
          install_mode: 'restart',
          install_command: 'cdui install --gpu cu128',
        }),
      ],
      gpu: gpuInfo,
    });
    open();
    render(<PackCenterModal />);

    // This server did not say it can restart itself, so the card is the
    // command: a button here would post a restart the backend answers 409.
    expect(screen.getByText('cdui install --gpu cu128')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Install and restart' })).toBeNull();
  });

  it('installs and restarts the GPU pack when the server says it can', async () => {
    seed({
      packs: [
        pack({
          id: 'gpu-torch',
          install_mode: 'restart',
          install_command: 'cdui install --gpu cu128',
        }),
      ],
      gpu: gpuInfo,
      restartAvailable: true,
    });
    open();
    render(<PackCenterModal />);

    fireEvent.click(screen.getByRole('button', { name: 'Install and restart' }));

    // The real in-app confirm, not a mock: this is the one place the panel
    // asks before it takes the server away.
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    expect(useDialogStore.getState().active).toMatchObject({
      message: 'Install cu128 and restart the server?',
      variant: 'danger',
    });
    act(() => useDialogStore.getState().close(true));

    await waitFor(() =>
      expect(actions.install).toHaveBeenCalledWith('gpu-torch', {
        mode: 'restart',
        variant: 'cu128',
      }),
    );
  });
});

// ── Dependencies ────────────────────────────────────────────────────────────

describe('PackCenterModal — a pack that needs another one', () => {
  it('names the blocker and jumps to its card', () => {
    const rag = pack({
      id: 'rag',
      depends_on: ['sentence-embeddings'],
      blocked_by: ['sentence-embeddings'],
      items: [item({ id: 'qwen2.5-0.5b' })],
    });
    seed({ packs: [embeddings, rag] });
    open();
    render(<PackCenterModal />);
    const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView');

    const card = cardFor('rag');
    expect(within(card).getByRole('button', { name: 'Install selected' })).toBeDisabled();

    fireEvent.click(
      within(card).getByRole('button', { name: 'Install Sentence embeddings first' }),
    );

    expect(scroll).toHaveBeenCalled();
    expect(cardFor('sentence-embeddings').className).toContain('cardHighlighted');
    // The request is consumed, so a later unrelated render does not re-fire it.
    expect(useUIStore.getState().packCenterFocusPackId).toBeNull();
  });

  it('scrolls to the pack it was opened for', () => {
    seed({ packs: [embeddings, words] });
    const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView');
    open('word-vectors');
    render(<PackCenterModal />);

    expect(scroll).toHaveBeenCalledWith({ block: 'nearest' });
    expect(cardFor('word-vectors').className).toContain('cardHighlighted');
    expect(cardFor('sentence-embeddings').className).not.toContain('cardHighlighted');
  });

  it('drops the ring after HIGHLIGHT_MS so it reads as a pointer, not a state', () => {
    seed({ packs: [embeddings, words] });
    vi.useFakeTimers();
    try {
      open('word-vectors');
      render(<PackCenterModal />);
      expect(cardFor('word-vectors').className).toContain('cardHighlighted');

      // The constant, not a copy of it: a ring that outlived its exported
      // duration would still pass a hard-coded 2000.
      act(() => {
        vi.advanceTimersByTime(HIGHLIGHT_MS - 1);
      });
      // One millisecond short: still on. The pair of assertions is what
      // makes this about the constant rather than about "eventually".
      expect(cardFor('word-vectors').className).toContain('cardHighlighted');

      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(cardFor('word-vectors').className).not.toContain('cardHighlighted');
    } finally {
      vi.useRealTimers();
    }
  });

  it('finds no card for a pack id that names an Object.prototype member', () => {
    // `cardRefs` is a bare object keyed by pack id: a plain index would hand
    // back a FUNCTION for `constructor` — truthy, and `scrollIntoView` is not
    // one of its methods, so the panel would die of a TypeError.
    seed({ packs: [embeddings, words] });
    const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView');
    // `scrollIntoView` is stubbed on the shared prototype, and spying on an
    // already-spied method hands back the SAME mock — history and all. A
    // "was never called" assertion has to start from zero to mean anything.
    scroll.mockClear();
    open('constructor');
    render(<PackCenterModal />);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(scroll).not.toHaveBeenCalled();
    expect(cardFor('word-vectors').className).not.toContain('cardHighlighted');
    // Unconsumed: the request is kept in case the card turns up later.
    expect(useUIStore.getState().packCenterFocusPackId).toBe('constructor');
  });

  it('waits for the catalog before deciding it cannot find the pack', () => {
    seed({ loading: true, loaded: false, packs: [] });
    open('word-vectors');
    render(<PackCenterModal />);
    expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');

    act(() => {
      usePackStore.setState({ loading: false, loaded: true, packs: [words] });
    });
    expect(cardFor('word-vectors').className).toContain('cardHighlighted');
  });
});

// ── The activity pane ───────────────────────────────────────────────────────

describe('PackCenterModal — the activity pane', () => {
  it('says nothing is happening when nothing is', () => {
    seed({ packs: [words] });
    open();
    render(<PackCenterModal />);
    expect(screen.getByText('Nothing is installing right now.')).toBeInTheDocument();
    expect(
      screen.getByText('Pick a pack on the left. Downloads keep going if you close this window.'),
    ).toBeInTheDocument();
  });

  it('shows the step, the overall bar and the log, and cancels', () => {
    seed({
      packs: [embeddings],
      job: job({
        steps: [
          { step: 'pip', label: 'pip install', state: 'done' },
          { step: 'download:all-MiniLM-L6-v2', label: 'Downloading model', state: 'running' },
        ],
        items: {
          'all-MiniLM-L6-v2': {
            bytesDone: 45 * 1024 * 1024,
            bytesTotal: 90 * 1024 * 1024,
            percent: 50,
          },
        },
        log: [
          { seq: 1, ts: null, kind: 'step', text: 'pip install' },
          { seq: 2, ts: null, kind: 'log', text: 'Collecting sentence-transformers' },
        ],
      }),
    });
    open();
    render(<PackCenterModal />);

    expect(screen.getByText('Installing Sentence embeddings')).toBeInTheDocument();
    // Translated off the step ID, not off the server's English label.
    expect(screen.getByText('Step 2: Downloading all-MiniLM-L6-v2')).toBeInTheDocument();

    const overall = screen.getByRole('progressbar', { name: 'Install progress' });
    expect(overall).toHaveAttribute('aria-valuenow', '50');

    const log = screen.getByRole('log', { name: 'Install log' });
    expect(within(log).getByText('Collecting sentence-transformers')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Cancel install' }));
    expect(actions.cancel).toHaveBeenCalledTimes(1);
  });

  it('announces the step and a rounded percent, and no more often than that', () => {
    seed({
      packs: [embeddings],
      job: job({
        steps: [{ step: 'verify', label: 'verify', state: 'running' }],
        items: {
          'all-MiniLM-L6-v2': { bytesDone: 9 * 1024 * 1024, bytesTotal: null, percent: 10 },
        },
      }),
    });
    open();
    render(<PackCenterModal />);
    // The panel portals to <body>, so the RTL container is not its parent.
    const live = () => document.querySelector('[aria-live="polite"][aria-atomic="true"]')!;
    expect(live().textContent).toBe('Step 1: Verifying the installation 10%');

    // One more megabyte is not news: a polite region re-read on every progress
    // frame would talk over itself for the whole download.
    act(() => {
      usePackStore.setState({
        job: job({
          steps: [{ step: 'verify', label: 'verify', state: 'running' }],
          items: {
            'all-MiniLM-L6-v2': {
              bytesDone: 10 * 1024 * 1024,
              bytesTotal: null,
              percent: 11,
            },
          },
        }),
      });
    });
    expect(live().textContent).toBe('Step 1: Verifying the installation 10%');

    act(() => {
      usePackStore.setState({
        job: job({
          steps: [{ step: 'verify', label: 'verify', state: 'running' }],
          items: {
            'all-MiniLM-L6-v2': {
              bytesDone: 18 * 1024 * 1024,
              bytesTotal: null,
              percent: 20,
            },
          },
        }),
      });
    });
    expect(live().textContent).toBe('Step 1: Verifying the installation 20%');
  });

  it('announces that an install started before the first step lands', () => {
    seed({
      packs: [embeddings],
      job: job({
        items: { 'all-MiniLM-L6-v2': { bytesDone: 0, bytesTotal: null, percent: 0 } },
      }),
    });
    open();
    render(<PackCenterModal />);
    // The seconds between "accepted" and the first step event are silent
    // otherwise, and a bare "0%" says nothing about what is installing.
    expect(
      document.querySelector('[aria-live="polite"][aria-atomic="true"]')!.textContent,
    ).toBe('Installing Sentence embeddings 0%');
  });

  it('reports a failure with the server message and its hint', () => {
    seed({
      packs: [embeddings],
      job: job({
        status: 'failed',
        error: { message: 'No space left on device', hint: 'Free 4 GB and try again.' },
      }),
    });
    open();
    render(<PackCenterModal />);

    const banner = screen.getByRole('status');
    expect(banner).toHaveAttribute('data-tone', 'error');
    expect(within(banner).getByText('Install failed: No space left on device')).toBeInTheDocument();
    expect(within(banner).getByText('Free 4 GB and try again.')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(actions.dismissJob).toHaveBeenCalledTimes(1);
  });

  it('reports a pack that is installed but needs the server restarted', () => {
    seed({
      packs: [pack({ id: 'gpu-torch', install_mode: 'restart' })],
      gpu: gpuInfo,
      job: job({
        packId: 'gpu-torch',
        status: 'needs_restart',
        restartCommand: 'cdui install --gpu cu128',
      }),
    });
    open();
    render(<PackCenterModal />);

    const banner = screen.getByRole('status');
    expect(banner).toHaveAttribute('data-tone', 'warning');
    expect(
      within(banner).getByText(
        'Installed. The server has to restart before GPU PyTorch can be used.',
      ),
    ).toBeInTheDocument();
    expect(within(banner).getByText('cdui install --gpu cu128')).toBeInTheDocument();
    expect(within(banner).getByRole('button', { name: 'Copy command' })).toBeInTheDocument();
    // No retry: this job never said a restart would finish it.
    expect(
      within(banner).queryByRole('button', { name: 'Restart the server and install' }),
    ).toBeNull();
  });

  it('offers to restart and install when a live install stopped and both sides agree', async () => {
    // The one shape that earns the button: a LIVE install the resolver
    // stopped (`retryMode`), on a server that can restart itself
    // (`restartAvailable`). Either half alone would offer a 409.
    seed({
      packs: [pack({ id: 'rag' })],
      restartAvailable: true,
      job: job({
        packId: 'rag',
        status: 'needs_restart',
        restartCommand: 'cdui packs install rag --restart',
        retryMode: 'restart',
      }),
    });
    open();
    render(<PackCenterModal />);

    const banner = screen.getByRole('status');
    // The command stays: the same install, by hand, for whoever prefers it.
    expect(within(banner).getByText('cdui packs install rag --restart')).toBeInTheDocument();

    fireEvent.click(
      within(banner).getByRole('button', { name: 'Restart the server and install' }),
    );

    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    // What the restart actually does, said before the server goes away: the
    // helper installs packages, and the models are a second install.
    expect(useDialogStore.getState().active).toMatchObject({
      message:
        'The server restarts to install the Python packages; download the models afterwards with a normal install.',
      variant: 'danger',
    });
    act(() => useDialogStore.getState().close(true));

    await waitFor(() =>
      expect(actions.install).toHaveBeenCalledWith('rag', { mode: 'restart' }),
    );
  });

  it('retries the pack whose banner asked, not whatever job arrived meanwhile', async () => {
    // A confirm stays open across catalog polls, and `refresh` adopts a job
    // started in another tab the moment it sees one. The pack the user
    // agreed to reinstall is the one they were reading about.
    seed({
      packs: [pack({ id: 'rag' }), embeddings],
      restartAvailable: true,
      job: job({ packId: 'rag', status: 'needs_restart', retryMode: 'restart' }),
    });
    open();
    render(<PackCenterModal />);

    fireEvent.click(screen.getByRole('button', { name: 'Restart the server and install' }));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());

    act(() => {
      usePackStore.setState({
        job: { ...job({ packId: 'sentence-embeddings' }), jobId: 'j-elsewhere' },
      });
    });
    act(() => useDialogStore.getState().close(true));

    await waitFor(() => expect(actions.install).toHaveBeenCalled());
    expect(actions.install).toHaveBeenCalledWith('rag', { mode: 'restart' });
  });

  it('installs nothing when the restart retry is declined', async () => {
    seed({
      packs: [pack({ id: 'rag' })],
      restartAvailable: true,
      job: job({ packId: 'rag', status: 'needs_restart', retryMode: 'restart' }),
    });
    open();
    render(<PackCenterModal />);

    fireEvent.click(screen.getByRole('button', { name: 'Restart the server and install' }));
    await waitFor(() => expect(useDialogStore.getState().active).not.toBeNull());
    act(() => useDialogStore.getState().close(false));

    await waitFor(() => expect(useDialogStore.getState().active).toBeNull());
    expect(actions.install).not.toHaveBeenCalled();
  });

  it('hides the retry when the job never said a restart would finish it', () => {
    // A restart-capable server and a `needs_restart` job that carried no
    // `retry_mode`: the GPU pack's own ending, which is not a retry of
    // anything — the install got as far as a restart-mode install goes, and
    // offering to run it again would start the whole thing over.
    seed({
      packs: [pack({ id: 'gpu-torch', install_mode: 'restart' })],
      gpu: gpuInfo,
      restartAvailable: true,
      job: job({
        packId: 'gpu-torch',
        status: 'needs_restart',
        restartCommand: 'cdui install --gpu cu128',
      }),
    });
    open();
    render(<PackCenterModal />);

    const banner = screen.getByRole('status');
    expect(
      within(banner).queryByRole('button', { name: 'Restart the server and install' }),
    ).toBeNull();
    expect(within(banner).getByText('cdui install --gpu cu128')).toBeInTheDocument();
  });

  it('disables the retry while this pack already has a request in flight', () => {
    seed({
      packs: [pack({ id: 'rag' })],
      restartAvailable: true,
      busy: { rag: true },
      job: job({ packId: 'rag', status: 'needs_restart', retryMode: 'restart' }),
    });
    open();
    render(<PackCenterModal />);

    expect(
      screen.getByRole('button', { name: 'Restart the server and install' }),
    ).toBeDisabled();
  });

  it('disables the retry for a browser the server refuses installs from', () => {
    seed({
      packs: [pack({ id: 'rag' })],
      restartAvailable: true,
      remoteInstallAllowed: false,
      job: job({ packId: 'rag', status: 'needs_restart', retryMode: 'restart' }),
    });
    open();
    render(<PackCenterModal />);

    const button = screen.getByRole('button', { name: 'Restart the server and install' });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute(
      'title',
      'Installing is only allowed from the computer that runs the server.',
    );
  });

  it('hides the retry when the server cannot restart itself', () => {
    // Same job, same event, a server that never claimed a restart: the
    // command block is the whole banner again.
    seed({
      packs: [pack({ id: 'rag' })],
      restartAvailable: false,
      job: job({
        packId: 'rag',
        status: 'needs_restart',
        restartCommand: 'cdui packs install rag --restart',
        retryMode: 'restart',
      }),
    });
    open();
    render(<PackCenterModal />);

    const banner = screen.getByRole('status');
    expect(
      within(banner).queryByRole('button', { name: 'Restart the server and install' }),
    ).toBeNull();
    expect(within(banner).getByText('cdui packs install rag --restart')).toBeInTheDocument();
  });

  it('says it has lost track rather than claiming the job is still running', () => {
    seed({ packs: [embeddings], job: job({ status: 'lost' }) });
    open();
    render(<PackCenterModal />);
    expect(
      within(screen.getByRole('status')).getByText(
        'Lost contact with the server. Refresh to check the pack status.',
      ),
    ).toBeInTheDocument();
  });

  it('reports a finished install, and an install the user stopped', () => {
    seed({ packs: [embeddings], job: job({ status: 'done' }) });
    open();
    const view = render(<PackCenterModal />);

    const banner = screen.getByRole('status');
    expect(banner).toHaveAttribute('data-tone', 'success');
    expect(
      within(banner).getByText('Installed Sentence embeddings.'),
    ).toBeInTheDocument();

    view.unmount();
    seed({ packs: [embeddings], job: job({ status: 'cancelled' }) });
    open();
    render(<PackCenterModal />);
    const stopped = screen.getByRole('status');
    // Neutral, not an error: the user asked for this one.
    expect(stopped).toHaveAttribute('data-tone', 'neutral');
    expect(within(stopped).getByText('Install cancelled.')).toBeInTheDocument();
  });

  it('announces how a job ENDED, not just how it was going', () => {
    // The banner mounts with its text already in it, which is not reliably
    // read; `cancelled` and `lost` were silent because the live region was
    // unmounted with the progress bar. One region, for the job's whole life.
    seed({ packs: [embeddings], job: job({ status: 'cancelled' }) });
    open();
    render(<PackCenterModal />);
    expect(
      document.querySelector('[aria-live="polite"][aria-atomic="true"]')!.textContent,
    ).toBe('Install cancelled.');
  });

  it('says the log is waiting rather than showing an empty box', () => {
    seed({ packs: [embeddings], job: job({ log: [] }) });
    open();
    render(<PackCenterModal />);
    const log = screen.getByRole('log', { name: 'Install log' });
    expect(within(log).getByText('Waiting for the first message...')).toBeInTheDocument();
  });

  it('follows the log while the reader is at the bottom, and stops when they are not', () => {
    const line = (seq: number) => ({
      seq, ts: null, kind: 'log' as const, text: `line ${seq}`,
    });
    seed({ packs: [embeddings], job: job({ log: [line(1)] }) });
    open();
    render(<PackCenterModal />);

    // jsdom does no layout, so the box's geometry is stated outright.
    const box = screen.getByRole('log', { name: 'Install log' });
    for (const [prop, value] of Object.entries({
      scrollHeight: 500, clientHeight: 100, scrollTop: 400,
    })) {
      Object.defineProperty(box, prop, { value, writable: true, configurable: true });
    }

    act(() => {
      usePackStore.setState({ job: job({ log: [line(1), line(2)] }) });
    });
    expect(box.scrollTop).toBe(500);

    // Scrolled up to read the line that failed: the next message must not
    // yank the reader back down.
    box.scrollTop = 500 - 100 - STICK_PX - 1;
    fireEvent.scroll(box);
    act(() => {
      usePackStore.setState({ job: job({ log: [line(1), line(2), line(3)] }) });
    });
    expect(box.scrollTop).toBe(500 - 100 - STICK_PX - 1);

    // Back within STICK_PX of the bottom: following resumes.
    box.scrollTop = 500 - 100 - STICK_PX;
    fireEvent.scroll(box);
    act(() => {
      usePackStore.setState({ job: job({ log: [line(1), line(2), line(3), line(4)] }) });
    });
    expect(box.scrollTop).toBe(500);
  });

  it('keeps the job on screen across a close and a reopen, and touches no follower', () => {
    seed({
      packs: [embeddings],
      job: job({ steps: [{ step: 'pip', label: 'pip', state: 'running' }] }),
    });
    open();
    render(<PackCenterModal />);
    expect(screen.getByText('Step 1: Installing Python packages')).toBeInTheDocument();

    act(() => {
      useUIStore.getState().closePackCenter();
    });
    expect(screen.queryByRole('dialog')).toBeNull();

    act(() => {
      open();
    });
    expect(screen.getByText('Step 1: Installing Python packages')).toBeInTheDocument();
    // The download outlives the window it was started from: the panel neither
    // starts nor stops the follower that is feeding it.
    expect(actions.stopFollowing).not.toHaveBeenCalled();
    expect(usePackStore.getState().job).not.toBeNull();
  });
});

// ── Closing ─────────────────────────────────────────────────────────────────

describe('PackCenterModal — closing', () => {
  beforeEach(() => {
    seed({ packs: [words] });
    open();
    render(<PackCenterModal />);
  });

  it('closes on Escape', () => {
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().packCenterOpen).toBe(false);
  });

  it('closes on the backdrop but not on a press inside the surface', () => {
    const dialog = screen.getByRole('dialog');
    fireEvent.mouseDown(dialog);
    expect(useUIStore.getState().packCenterOpen).toBe(true);

    fireEvent.mouseDown(dialog.parentElement!);
    expect(useUIStore.getState().packCenterOpen).toBe(false);
  });

  it('closes on the close button', () => {
    fireEvent.click(screen.getByRole('button', { name: 'Close Package Center' }));
    expect(useUIStore.getState().packCenterOpen).toBe(false);
  });

  it('leaves Escape alone while something is stacked above it', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().packCenterOpen).toBe(true);
    useUIStore.setState({ shortcutsModalOpen: false });

    useDialogStore.setState({ active: { kind: 'confirm', title: 'x' } as never });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().packCenterOpen).toBe(true);
    useDialogStore.setState({ active: null });

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().packCenterOpen).toBe(false);
  });

  it('refuses to close while the server is restarting', () => {
    act(() => {
      usePackStore.setState({
        restart: { phase: 'waiting', packId: 'gpu-torch', startedAt: Date.now(), command: null },
      });
    });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().packCenterOpen).toBe(true);

    // Nothing behind the overlay is true while the server is going away, so
    // the backdrop is dead too.
    fireEvent.mouseDown(screen.getByRole('dialog').parentElement!);
    expect(useUIStore.getState().packCenterOpen).toBe(true);
  });

  it('hands focus back to whatever had it', () => {
    // The modal took focus on open; closing it must not leave focus on <body>.
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveFocus();

    act(() => {
      useUIStore.getState().closePackCenter();
    });
    expect(document.activeElement).not.toBe(dialog);
  });
});
