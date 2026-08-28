import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, within } from '@testing-library/react';
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
import { PackCenterModal } from './PackCenterModal';

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
  useUIStore.setState({
    packCenterOpen: false,
    packCenterFocusPackId: null,
    shortcutsModalOpen: false,
  });
  useDialogStore.setState({ active: null });
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
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
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

    // PR 1's backend refuses a restart-mode install with 409 and this command,
    // so until PR 5 the card is the command, not a button.
    expect(screen.getByText('cdui install --gpu cu128')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Install and restart' })).toBeNull();
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
  });

  it('says it has lost track rather than claiming the job is still running', () => {
    seed({ packs: [embeddings], job: job({ status: 'lost' }) });
    open();
    render(<PackCenterModal />);
    expect(
      screen.getByText('Lost contact with the server. Refresh to check the pack status.'),
    ).toBeInTheDocument();
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
