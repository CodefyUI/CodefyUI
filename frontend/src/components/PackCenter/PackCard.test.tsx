import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { PackGpuInfo, PackItem, PackSummary } from '../../api/rest';
import { emptyPackJob, type PackJob } from '../../store/packStore';
import { useI18n } from '../../i18n';
import type { PackIndex } from '../../utils/packAvailability';
import { PackCard, StatusPill } from './PackCard';

function item(over: Partial<PackItem> & { id: string }): PackItem {
  return {
    kind: 'hf',
    repo_id: `sentence-transformers/${over.id}`,
    url: null,
    size_bytes: 1024 * 1024,
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

function gpu(over: Partial<PackGpuInfo> = {}): PackGpuInfo {
  return {
    detected_label: 'NVIDIA GeForce RTX 4080',
    recommended_variant: 'cu128',
    installed_variant: 'cpu',
    variants: ['cu128', 'cpu'],
    install_command: 'cdui install --gpu auto',
    ...over,
  };
}

function index(...packs: PackSummary[]): PackIndex {
  return Object.fromEntries(packs.map((p) => [p.id, p]));
}

type CardProps = Parameters<typeof PackCard>[0];

function renderCard(over: Partial<CardProps> = {}) {
  const onInstall = vi.fn();
  const onRemoveItem = vi.fn();
  const onFocusPack = vi.fn();
  const props: CardProps = {
    pack: pack({ id: 'word-vectors' }),
    byId: {},
    job: null,
    busy: false,
    highlighted: false,
    canInstall: true,
    launchMode: 'start',
    gpu: null,
    onInstall,
    onRemoveItem,
    onFocusPack,
    ...over,
  };
  const view = render(<PackCard {...props} />);
  return {
    onInstall,
    onRemoveItem,
    onFocusPack,
    container: view.container,
    rerender: (next: Partial<CardProps>) =>
      view.rerender(<PackCard {...props} {...next} />),
  };
}

const installBtn = () => screen.getByRole('button', { name: 'Install selected' });

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('StatusPill', () => {
  it('names the status and tones it', () => {
    const { rerender } = render(<StatusPill status="installed" />);
    expect(screen.getByText('Installed')).toHaveAttribute('data-tone', 'success');

    rerender(<StatusPill status="partial" />);
    expect(screen.getByText('Partly installed')).toHaveAttribute('data-tone', 'warning');

    rerender(<StatusPill status="needs_restart" />);
    expect(screen.getByText('Restart needed')).toHaveAttribute('data-tone', 'warning');

    rerender(<StatusPill status="not_installed" />);
    expect(screen.getByText('Not installed')).toHaveAttribute('data-tone', 'neutral');
  });

  it('marks the installing pill as live with a dot', () => {
    const { container } = render(<StatusPill status="installing" />);
    expect(screen.getByText('Installing')).toHaveAttribute('data-tone', 'info');
    expect(container.querySelector('[data-role="pulse"]')).not.toBeNull();
  });
});

describe('PackCard — what it says about a pack', () => {
  it("prefers this build's copy over the server's", () => {
    renderCard({ pack: pack({ id: 'word-vectors', size_bytes_total: 66 * 1024 * 1024 }) });
    expect(screen.getByText('Word vectors (GloVe)')).toBeInTheDocument();
    expect(
      screen.getByText(
        'Real 400k-word GloVe-50d table for WordVector; no Python packages needed',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('Download size: 66 MB')).toBeInTheDocument();
  });

  it('shows the server copy for a pack this build predates', () => {
    renderCard({ pack: pack({ id: 'pack-from-the-future' }) });
    expect(screen.getByText('server title for pack-from-the-future')).toBeInTheDocument();
    expect(screen.getByText('server description')).toBeInTheDocument();
  });

  it('carries the status onto the element, for the left-edge tint', () => {
    const { container } = renderCard({ pack: pack({ id: 'rag', status: 'partial' }) });
    expect(container.querySelector('[data-status="partial"]')).not.toBeNull();
    expect(screen.getByText('Partly installed')).toBeInTheDocument();
  });

  it('lists the python packages a pack needs', () => {
    renderCard({
      pack: pack({ id: 'rag', pip: [{ spec: 'transformers>=4.44' }, { spec: 'accelerate' }] }),
    });
    expect(
      screen.getByText('Python packages: transformers>=4.44, accelerate'),
    ).toBeInTheDocument();
  });

  it('says nothing about python for a pack that is only files', () => {
    renderCard({ pack: pack({ id: 'word-vectors' }) });
    expect(screen.queryByText(/^Python packages:/)).toBeNull();
  });
});

describe('PackCard — choosing what to install', () => {
  const p = pack({
    id: 'sentence-embeddings',
    items: [
      item({ id: 'all-MiniLM-L6-v2', status: 'missing', size_bytes: 90 * 1024 * 1024 }),
      item({ id: 'bge-small-zh', status: 'present', size_bytes: 100 * 1024 * 1024 }),
      item({ id: 'labse', status: 'missing', size_bytes: 10 * 1024 * 1024 }),
    ],
  });

  it('starts with everything that is not downloaded ticked', () => {
    renderCard({ pack: p });
    expect(screen.getByLabelText('sentence-transformers/all-MiniLM-L6-v2')).toBeChecked();
    expect(screen.getByLabelText('sentence-transformers/labse')).toBeChecked();
    // A downloaded item has no checkbox at all: there is nothing to fetch.
    expect(screen.queryByLabelText('sentence-transformers/bge-small-zh')).toBeNull();
    expect(screen.getByText('100 MB selected')).toBeInTheDocument();
  });

  it('installs exactly the ticked items', () => {
    const { onInstall } = renderCard({ pack: p });
    fireEvent.click(screen.getByLabelText('sentence-transformers/labse'));
    expect(screen.getByText('90 MB selected')).toBeInTheDocument();

    fireEvent.click(installBtn());
    expect(onInstall).toHaveBeenCalledWith(['all-MiniLM-L6-v2'], 'live');
  });

  it('re-ticks everything missing on Select all', () => {
    const { onInstall } = renderCard({ pack: p });
    fireEvent.click(screen.getByLabelText('sentence-transformers/labse'));
    fireEvent.click(screen.getByLabelText('sentence-transformers/all-MiniLM-L6-v2'));
    expect(installBtn()).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Select all missing' }));
    fireEvent.click(installBtn());
    // Catalog order, not click order: the request reads like the card.
    expect(onInstall).toHaveBeenCalledWith(['all-MiniLM-L6-v2', 'labse'], 'live');
  });

  it('reseeds the ticks when the catalog says an item has landed', () => {
    const { rerender } = renderCard({ pack: p });
    fireEvent.click(screen.getByLabelText('sentence-transformers/all-MiniLM-L6-v2'));
    expect(screen.getByText('10 MB selected')).toBeInTheDocument();

    // The install finished: MiniLM is on disk now, and the stale un-tick must
    // not survive into a selection that no longer mentions it.
    rerender({
      pack: pack({
        ...p,
        items: [
          item({ id: 'all-MiniLM-L6-v2', status: 'present', size_bytes: 90 * 1024 * 1024 }),
          item({ id: 'bge-small-zh', status: 'present', size_bytes: 100 * 1024 * 1024 }),
          item({ id: 'labse', status: 'missing', size_bytes: 10 * 1024 * 1024 }),
        ],
      }),
    });
    expect(screen.getByText('10 MB selected')).toBeInTheDocument();
    expect(screen.getByLabelText('sentence-transformers/labse')).toBeChecked();
    expect(screen.queryByLabelText('sentence-transformers/all-MiniLM-L6-v2')).toBeNull();
  });

  it('offers to remove an item that is already on disk', () => {
    const { onRemoveItem } = renderCard({ pack: p });
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
    expect(onRemoveItem).toHaveBeenCalledWith('bge-small-zh');
  });

  it('names the licence on the row and in the name tooltip', () => {
    renderCard({ pack: p });
    expect(screen.getByText('sentence-transformers/labse')).toHaveAttribute(
      'title',
      'License: apache-2.0',
    );
    expect(screen.getAllByText('apache-2.0').length).toBe(3);
  });

  it('names a plain-URL item by its file', () => {
    renderCard({
      pack: pack({
        id: 'word-vectors',
        items: [
          item({
            id: 'glove-6b-50d',
            kind: 'asset',
            repo_id: null,
            url: 'https://example.test/data/glove.6B.50d.zip?v=2',
          }),
        ],
      }),
    });
    expect(screen.getByLabelText('glove.6B.50d.zip')).toBeInTheDocument();
  });
});

describe('PackCard — when it cannot install', () => {
  const p = pack({
    id: 'sentence-embeddings',
    items: [item({ id: 'all-MiniLM-L6-v2' })],
  });

  it('explains a refused remote install on the disabled button', () => {
    renderCard({ pack: p, canInstall: false });
    expect(installBtn()).toBeDisabled();
    expect(installBtn()).toHaveAttribute(
      'title',
      'Installing is only allowed from the computer that runs the server.',
    );
  });

  it('says what to tick when nothing is', () => {
    renderCard({ pack: p });
    fireEvent.click(screen.getByLabelText('sentence-transformers/all-MiniLM-L6-v2'));
    expect(installBtn()).toBeDisabled();
    // An instruction, not the neighbouring button's label: "Select all
    // missing" on a dead Install button reads as a description of the button
    // rather than of what is stopping it.
    expect(installBtn()).toHaveAttribute('title', 'Tick at least one item to install');
  });

  it('points at the missing dependency and jumps to its card', () => {
    const { onInstall, onFocusPack } = renderCard({
      pack: pack({
        id: 'rag',
        depends_on: ['sentence-embeddings'],
        blocked_by: ['sentence-embeddings'],
        items: [item({ id: 'qwen2.5-0.5b' })],
      }),
      byId: index(pack({ id: 'sentence-embeddings', status: 'partial' })),
    });

    // The dependency is named WITH its own state, so "why is this blocked" is
    // answered on the card instead of in a toast after a refused click.
    expect(screen.getByText('Requires: Sentence embeddings')).toBeInTheDocument();
    expect(screen.getByText('Partly installed')).toBeInTheDocument();
    expect(installBtn()).toBeDisabled();
    expect(installBtn()).toHaveAttribute('title', 'Install Sentence embeddings first');

    fireEvent.click(
      screen.getByRole('button', { name: 'Install Sentence embeddings first' }),
    );
    expect(onFocusPack).toHaveBeenCalledWith('sentence-embeddings');
    expect(onInstall).not.toHaveBeenCalled();
  });

  it('disables the button while this pack already has a request in flight', () => {
    renderCard({ pack: p, busy: true });
    expect(installBtn()).toBeDisabled();
  });
});

describe('PackCard — while a job is running', () => {
  it('puts a bar and a byte count on each downloading item', () => {
    const job: PackJob = {
      ...emptyPackJob('j1', 'sentence-embeddings'),
      items: {
        'all-MiniLM-L6-v2': {
          bytesDone: 45 * 1024 * 1024,
          bytesTotal: 90 * 1024 * 1024,
          percent: 50,
        },
      },
    };
    renderCard({
      pack: pack({
        id: 'sentence-embeddings',
        status: 'installing',
        items: [
          item({ id: 'all-MiniLM-L6-v2', status: 'downloading', size_bytes: 90 * 1024 * 1024 }),
          item({ id: 'labse', status: 'missing' }),
        ],
      }),
      job,
    });

    const bar = screen.getByRole('progressbar', {
      name: 'sentence-transformers/all-MiniLM-L6-v2',
    });
    expect(bar).toHaveAttribute('aria-valuenow', '50');
    expect(screen.getByText('45 MB / 90 MB')).toBeInTheDocument();
    // Nothing has been said about the other item yet, so it has no bar.
    expect(
      screen.queryByRole('progressbar', { name: 'sentence-transformers/labse' }),
    ).toBeNull();
  });

  it('locks the tick boxes so the selection cannot drift under a running job', () => {
    renderCard({
      pack: pack({ id: 'sentence-embeddings', items: [item({ id: 'labse' })] }),
      job: emptyPackJob('j1', 'sentence-embeddings'),
    });
    expect(screen.getByLabelText('sentence-transformers/labse')).toBeDisabled();
    expect(installBtn()).toBeDisabled();
  });

  it('locks Remove too, so nothing deletes a file the job is writing', () => {
    renderCard({
      pack: pack({
        id: 'sentence-embeddings',
        items: [item({ id: 'labse', status: 'present' })],
      }),
      job: emptyPackJob('j1', 'sentence-embeddings'),
    });
    expect(screen.getByRole('button', { name: 'Remove' })).toBeDisabled();
  });

  it('leaves another pack alone while this one installs', () => {
    renderCard({
      pack: pack({ id: 'word-vectors', items: [item({ id: 'glove-6b-50d' })] }),
      job: emptyPackJob('j1', 'sentence-embeddings'),
    });
    expect(screen.getByLabelText('sentence-transformers/glove-6b-50d')).toBeEnabled();
  });
});

describe('PackCard — the GPU pack', () => {
  it('shows the wheel-swap card instead of item rows', () => {
    renderCard({
      pack: pack({
        id: 'gpu-torch',
        install_mode: 'restart',
        install_command: 'cdui install --gpu cu128',
      }),
      gpu: gpu(),
    });
    expect(screen.getByText('GPU PyTorch')).toBeInTheDocument();
    expect(screen.getByText('Detected GPU: NVIDIA GeForce RTX 4080')).toBeInTheDocument();
    expect(screen.getByText('cdui install --gpu cu128')).toBeInTheDocument();
    // No selection UI: there is nothing to tick on a wheel swap.
    expect(screen.queryByRole('button', { name: 'Install selected' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Select all missing' })).toBeNull();
  });
});
