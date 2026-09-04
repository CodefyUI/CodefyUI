import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, within } from '@testing-library/react';
import type { PluginCatalogEntry, PluginInspection } from '../../api/rest';
import { useDialogStore } from '../../store/dialogStore';
import {
  emptyPluginJob,
  usePluginStore,
  _resetPluginStoreForTesting,
  type InspectionState,
  type PluginJob,
} from '../../store/pluginStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { HIGHLIGHT_MS } from '../PackCenter/PackCenterModal';
import { PluginCenterModal } from './PluginCenterModal';

function entry(over: Partial<PluginCatalogEntry> & { id: string }): PluginCatalogEntry {
  return {
    name: over.id,
    description: '',
    kind: 'builtin',
    official: true,
    status: 'available',
    source_kind: null,
    source: over.id,
    repo: null,
    ref: null,
    sha: null,
    url: null,
    homepage: '',
    version: null,
    installed_at: null,
    enabled: false,
    chapters: [],
    lessons: [],
    tags: [],
    nodes: [],
    node_count: 0,
    capabilities: [],
    trusted_modules: [],
    python_deps: {},
    has_frontend: false,
    consent_required: false,
    frontend_entry: null,
    job: null,
    ...over,
  };
}

/** A built-in teaching pack, installed: the common row. */
const edu = entry({
  id: 'edu',
  name: 'EDU teaching nodes',
  description: 'Hands-on teaching nodes for the labs.',
  status: 'installed',
  source_kind: 'builtin',
  enabled: true,
  version: '0.1.0',
  chapters: ['I1', 'I2'],
  node_count: 8,
  // Both shapes a manifest writes: a constraint that leads with an operator,
  // and a bare version the installer has to pin with `==`.
  python_deps: { model2vec: '>=0.8.0', torch: '2.1.0' },
});

/** A third-party plugin off GitHub, installed and pinned. */
const demo = entry({
  id: 'demo',
  name: 'Demo plugin',
  kind: 'github',
  official: false,
  status: 'installed',
  source_kind: 'github_url',
  enabled: true,
  version: '1.2.0',
  repo: 'owner/demo',
  url: 'https://github.com/owner/demo',
  ref: 'v1.2.0',
  sha: 'abcdef1234567890abcdef1234567890abcdef12',
});

/** A built-in pack nobody has installed. */
const stats = entry({ id: 'stats', name: 'Stats nodes' });

/** An inspection as the server would answer one. */
function inspection(over: Partial<PluginInspection> = {}): PluginInspection {
  return {
    inspection_id: 'insp-1',
    expires_at: '2026-01-01T00:00:00Z',
    kind: 'github',
    mode: 'install',
    plugin_id: 'demo',
    catalog_id: null,
    official: false,
    source: 'owner/demo',
    url: 'https://github.com/owner/demo',
    ref: null,
    sha: null,
    name: 'Demo plugin',
    version: '1.2.0',
    description: '',
    homepage: '',
    manifest: {},
    capabilities: [],
    allowed_modules: [],
    python_deps: {},
    has_frontend: false,
    chapters: [],
    lessons: [],
    consent_required: false,
    installed: null,
    up_to_date: false,
    capabilities_added: [],
    allowed_modules_added: [],
    warnings: [],
    ...over,
  };
}

/**
 * A review the server has answered, in the shape the store keeps it.
 *
 * The narrow variant rather than `InspectionState`, so a case that adds an
 * `error` to it can spread this instead of casting the union back down.
 */
function ready(
  over: Partial<PluginInspection> = {},
  /** The row whose button raised it; null for a source somebody typed. */
  forPluginId: string | null = null,
): Extract<InspectionState, { phase: 'ready' }> {
  const data = inspection(over);
  return {
    phase: 'ready', data, source: data.source, forPluginId, kind: data.mode, error: null,
  };
}

/** Fresh mocks for every action the panel is allowed to call. */
function makeActions() {
  return {
    refresh: vi.fn(async () => {}),
    install: vi.fn(async () => {}),
    inspect: vi.fn(async () => {}),
    installInspected: vi.fn(async () => {}),
    clearInspection: vi.fn(() => {}),
    update: vi.fn(async () => {}),
    uninstall: vi.fn(async () => {}),
    setEnabled: vi.fn(async () => {}),
  };
}

let actions: ReturnType<typeof makeActions>;

/**
 * Seed the store the panel reads, with fresh mock actions installed through
 * `setState`. Never `vi.spyOn(usePluginStore.getState(), ...)`: that spies on a
 * snapshot object, and the history leaks between cases.
 */
function seed(state: Partial<ReturnType<typeof usePluginStore.getState>> = {}) {
  actions = makeActions();
  const plugins = state.plugins ?? [];
  usePluginStore.setState({
    plugins,
    byId: Object.fromEntries(plugins.map((p) => [p.id, p])),
    loading: false,
    loaded: true,
    unsupported: false,
    error: null,
    remoteInstallAllowed: true,
    job: null,
    busy: {},
    cancelling: false,
    inspection: { phase: 'idle' },
    ...actions,
    ...state,
  });
}

function job(over: Partial<PluginJob> = {}): PluginJob {
  return { ...emptyPluginJob('j1', 'demo'), ...over };
}

function open(pluginId?: string) {
  useUIStore.getState().openPluginCenter(pluginId);
}

const cardFor = (pluginId: string) =>
  document.querySelector(`[data-plugin-id="${pluginId}"]`) as HTMLElement;

/** Every button on one row, by its label. */
const buttonsOn = (pluginId: string) =>
  within(cardFor(pluginId))
    .queryAllByRole('button')
    .map((button) => button.textContent);

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useDialogStore.setState({ active: null });
  useUIStore.setState({
    pluginCenterOpen: false,
    pluginCenterFocusPluginId: null,
    packCenterOpen: false,
    shortcutsModalOpen: false,
  });
  _resetPluginStoreForTesting();
  seed();
});

afterEach(() => {
  // Inside act(): this hook runs BEFORE Testing Library's own cleanup, so the
  // panel is still mounted and subscribed when `pluginCenterOpen` goes false —
  // closing it here is a real React update, and unwrapped it printed an
  // "update was not wrapped in act(...)" line for every case in this file.
  act(() => {
    useUIStore.setState({
      pluginCenterOpen: false,
      pluginCenterFocusPluginId: null,
      packCenterOpen: false,
      shortcutsModalOpen: false,
    });
    useDialogStore.setState({ active: null });
  });
  vi.restoreAllMocks();
});

// ── Mounting ────────────────────────────────────────────────────────────────

describe('PluginCenterModal — mounting', () => {
  it('renders nothing, and asks the server nothing, while it is closed', () => {
    render(<PluginCenterModal />);
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(actions.refresh).not.toHaveBeenCalled();
  });

  it('reads the catalog once when it opens', () => {
    seed({ plugins: [edu] });
    open();
    render(<PluginCenterModal />);
    expect(screen.getByRole('dialog', { name: 'Plugin Center' })).toBeInTheDocument();
    expect(actions.refresh).toHaveBeenCalledTimes(1);
  });

  it('re-reads on the refresh button', () => {
    seed({ plugins: [edu] });
    open();
    render(<PluginCenterModal />);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh plugin status' }));
    expect(actions.refresh).toHaveBeenCalledTimes(2);
  });
});

// ── The list ────────────────────────────────────────────────────────────────

describe('PluginCenterModal — the plugin list', () => {
  it('names a plugin, where it came from and what it brings', () => {
    seed({ plugins: [edu] });
    open();
    render(<PluginCenterModal />);

    expect(screen.getByRole('region', { name: 'Plugin list' })).toBeInTheDocument();
    // The activity pane's own element exists from the start, so the pane can
    // move in later without the list reflowing around a new column.
    expect(
      screen.getByRole('complementary', { name: 'Install activity' }),
    ).toBeInTheDocument();

    const card = within(cardFor('edu'));
    expect(card.getByText('EDU teaching nodes')).toBeInTheDocument();
    // Scoped to the card: "Installed" is also the middle filter button.
    expect(card.getByText('Installed')).toBeInTheDocument();
    expect(card.getByText('v0.1.0')).toBeInTheDocument();
    expect(card.getByText('Built-in')).toBeInTheDocument();
    expect(card.getByText('Lessons: I1, I2')).toBeInTheDocument();
    expect(card.getByText('8 nodes')).toBeInTheDocument();
    // The specs as the installer would write them, not prettier strings that
    // would install something else.
    expect(
      card.getByText('Python packages: model2vec>=0.8.0, torch==2.1.0'),
    ).toBeInTheDocument();
  });

  it('links a GitHub plugin to its repository and says which commit is here', () => {
    seed({ plugins: [demo] });
    open();
    render(<PluginCenterModal />);

    const card = within(cardFor('demo'));
    const link = card.getByRole('link', { name: 'owner/demo' });
    expect(link).toHaveAttribute('href', 'https://github.com/owner/demo');
    // Seven characters of the sha: enough to check against a repository,
    // short enough to sit on a meta line. The ref is `v1.2.0` and the header
    // already says `v1.2.0`, so the pin drops it rather than printing one
    // release twice down one card.
    expect(card.getByText('abcdef1')).toBeInTheDocument();
    expect(card.getAllByText('v1.2.0')).toHaveLength(1);
    // A plain third-party repository wears no origin chip — the row already
    // prints owner/repo, and "GitHub" over a GitHub link says it twice.
    expect(card.queryByText('Official')).toBeNull();
  });

  it('keeps the ref on a plugin pinned to something other than its version', () => {
    seed({ plugins: [entry({ ...demo, ref: 'main' })] });
    open();
    render(<PluginCenterModal />);

    expect(within(cardFor('demo')).getByText('main @ abcdef1')).toBeInTheDocument();
  });

  it('says it is loading before the first answer', () => {
    seed({ loading: true, loaded: false, plugins: [] });
    open();
    render(<PluginCenterModal />);
    expect(screen.getByText('Loading plugins...')).toBeInTheDocument();
  });

  it('reports a failed read and offers to try again', () => {
    seed({ loaded: false, error: 'connection refused', plugins: [] });
    open();
    render(<PluginCenterModal />);
    expect(
      screen.getByText('Failed to load plugins: connection refused'),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(actions.refresh).toHaveBeenCalledTimes(2);
  });

  it('keeps the rows it has when a refresh fails, and still says it failed', () => {
    seed({ plugins: [edu], error: 'connection refused' });
    open();
    render(<PluginCenterModal />);
    // One dropped packet must not empty the Plugin Center.
    expect(cardFor('edu')).not.toBeNull();
    expect(
      screen.getByText('Failed to load plugins: connection refused'),
    ).toBeInTheDocument();
  });

  it('explains a server that predates the Plugin Center, and offers it nothing', () => {
    seed({ unsupported: true, plugins: [] });
    open();
    render(<PluginCenterModal />);
    expect(
      screen.getByText('This server has no Plugin Center. Update CodefyUI and restart it.'),
    ).toBeInTheDocument();
    // No filter and no source box over a server that cannot answer either.
    expect(screen.queryByRole('button', { name: 'All' })).toBeNull();
    expect(screen.queryByLabelText('Install from GitHub')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Review' })).toBeNull();
  });

  it('says so when the server lists no plugins at all', () => {
    seed({ plugins: [] });
    open();
    render(<PluginCenterModal />);
    expect(screen.getByText('No plugins are available')).toBeInTheDocument();
  });
});

// ── A row per state ─────────────────────────────────────────────────────────

describe('PluginCenterModal — what a row offers', () => {
  it('offers Install, and nothing else, on a plugin that is not here', () => {
    seed({ plugins: [stats] });
    open();
    render(<PluginCenterModal />);

    expect(within(cardFor('stats')).getByText('Not installed')).toBeInTheDocument();
    expect(buttonsOn('stats')).toEqual(['Install']);
  });

  it('switches off, updates or removes an installed GitHub plugin', () => {
    seed({ plugins: [demo] });
    open();
    render(<PluginCenterModal />);
    expect(buttonsOn('demo')).toEqual(['Disable', 'Update', 'Uninstall']);
  });

  it('has nothing to update a built-in from', () => {
    seed({ plugins: [edu] });
    open();
    render(<PluginCenterModal />);
    // It ships with the server: the update route would have no source to read.
    expect(buttonsOn('edu')).toEqual(['Disable', 'Uninstall']);
  });

  it('offers Enable and Uninstall on a plugin that is switched off', () => {
    seed({ plugins: [entry({ id: 'demo', status: 'disabled', source_kind: 'github_url' })] });
    open();
    render(<PluginCenterModal />);

    expect(within(cardFor('demo')).getByText('Disabled')).toBeInTheDocument();
    expect(buttonsOn('demo')).toEqual(['Enable', 'Uninstall']);
  });

  it('only switches a linked folder, never installs or removes one', () => {
    seed({
      plugins: [
        entry({
          id: 'wip',
          name: 'Work in progress',
          kind: 'external',
          official: false,
          status: 'installed',
          source_kind: 'local',
        }),
      ],
    });
    open();
    render(<PluginCenterModal />);

    // `cdui plugin link` owns the directory; the panel would be removing
    // somebody's working copy.
    expect(within(cardFor('wip')).getByText('Linked folder')).toBeInTheDocument();
    expect(buttonsOn('wip')).toEqual(['Disable']);
  });

  it('offers to put a removed plugin back', () => {
    seed({ plugins: [entry({ id: 'stats', status: 'removed' })] });
    open();
    render(<PluginCenterModal />);

    expect(within(cardFor('stats')).getByText('Removed')).toBeInTheDocument();
    // A tombstone has nothing left to uninstall.
    expect(buttonsOn('stats')).toEqual(['Install']);
  });

  it('offers to repair or clear a plugin whose files are gone', () => {
    seed({ plugins: [entry({ id: 'demo', status: 'missing_files', source_kind: 'github_url' })] });
    open();
    render(<PluginCenterModal />);

    expect(within(cardFor('demo')).getByText('Files missing')).toBeInTheDocument();
    // The lockfile still has it, so clearing the record is a real answer.
    expect(buttonsOn('demo')).toEqual(['Install', 'Uninstall']);
  });

  it('says a row is installing, and offers no button while it is', () => {
    seed({ plugins: [entry({ id: 'demo', status: 'installing' })] });
    open();
    render(<PluginCenterModal />);

    const card = cardFor('demo');
    expect(within(card).getByText('Installing')).toBeInTheDocument();
    expect(card.querySelector('[data-role="pulse"]')).not.toBeNull();
    expect(buttonsOn('demo')).toEqual([]);
  });

  it('says so from the job alone, before the catalog has caught up', () => {
    // A job adopted from another tab lands before the poll that would have
    // set the row's status: the row this job names is installing either way.
    seed({ plugins: [demo], job: job({ pluginId: 'demo' }) });
    open();
    render(<PluginCenterModal />);

    expect(within(cardFor('demo')).getByText('Installing')).toBeInTheDocument();
    expect(buttonsOn('demo')).toEqual([]);
  });

  it('leaves a row alone while one of its own requests is in flight', () => {
    seed({ plugins: [demo], busy: { demo: true } });
    open();
    render(<PluginCenterModal />);

    // The buttons stay — this is a request, not an install job — but every
    // one of them is dead until it answers.
    for (const button of within(cardFor('demo')).getAllByRole('button')) {
      expect(button).toBeDisabled();
    }
  });
});

// ── The actions ─────────────────────────────────────────────────────────────

describe('PluginCenterModal — the actions', () => {
  it('hands every change to the store, which owns the confirm and the job', () => {
    seed({ plugins: [stats, demo] });
    open();
    render(<PluginCenterModal />);

    fireEvent.click(within(cardFor('stats')).getByRole('button', { name: 'Install' }));
    expect(actions.install).toHaveBeenCalledWith('stats');

    fireEvent.click(within(cardFor('demo')).getByRole('button', { name: 'Update' }));
    expect(actions.update).toHaveBeenCalledWith('demo');

    // No second confirm here: `pluginStore.uninstall` asks first, with the
    // danger dialog and the sentence about the Python packages that stay.
    fireEvent.click(within(cardFor('demo')).getByRole('button', { name: 'Uninstall' }));
    expect(actions.uninstall).toHaveBeenCalledWith('demo');

    fireEvent.click(within(cardFor('demo')).getByRole('button', { name: 'Disable' }));
    expect(actions.setEnabled).toHaveBeenCalledWith('demo', false);
  });

  it('switches a disabled plugin back on', () => {
    seed({ plugins: [entry({ id: 'demo', status: 'disabled' })] });
    open();
    render(<PluginCenterModal />);

    fireEvent.click(screen.getByRole('button', { name: 'Enable' }));
    expect(actions.setEnabled).toHaveBeenCalledWith('demo', true);
  });
});

// ── A server that only installs locally ─────────────────────────────────────

describe('PluginCenterModal — remote installs refused', () => {
  it('says why once, and disables exactly what the server would refuse', () => {
    seed({ plugins: [stats, demo], remoteInstallAllowed: false });
    open();
    render(<PluginCenterModal />);

    const sentence = 'Installing is only allowed from the computer that runs the server.';
    // Once, in the footer. On the buttons it is a `title`, not a fourth copy
    // of the same sentence down the card.
    expect(screen.getAllByText(sentence)).toHaveLength(1);

    const install = within(cardFor('stats')).getByRole('button', { name: 'Install' });
    expect(install).toBeDisabled();
    expect(install).toHaveAttribute('title', sentence);

    const update = within(cardFor('demo')).getByRole('button', { name: 'Update' });
    expect(update).toBeDisabled();
    expect(within(cardFor('demo')).getByRole('button', { name: 'Uninstall' })).toBeDisabled();

    // Switching a plugin off changes nothing on disk, so the token-only gate
    // does not cover it and neither does this.
    expect(within(cardFor('demo')).getByRole('button', { name: 'Disable' })).toBeEnabled();
  });
});

// ── The source box ──────────────────────────────────────────────────────────

describe('PluginCenterModal — the source box', () => {
  /** Type *value* into the box and press Review. */
  function review(value: string) {
    fireEvent.change(screen.getByLabelText('Install from GitHub'), {
      target: { value },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Review' }));
  }

  it('offers a box for a plugin the catalog does not list', () => {
    seed({ plugins: [edu] });
    open();
    render(<PluginCenterModal />);

    expect(screen.getByLabelText('Install from GitHub')).toHaveAttribute(
      'placeholder', 'owner/repo[@ref] or GitHub URL',
    );
    // Nothing to review until something is typed.
    expect(screen.getByRole('button', { name: 'Review' })).toBeDisabled();
  });

  it('refuses a source that is not one without asking the server', () => {
    seed();
    open();
    render(<PluginCenterModal />);

    review('not a repo!');

    expect(
      screen.getByText('Enter a catalog name, owner/repo[@ref] or a GitHub URL.'),
    ).toBeInTheDocument();
    // The server would answer 400 to the same string; this build knows that
    // without the round trip, so it does not make one.
    expect(actions.inspect).not.toHaveBeenCalled();
  });

  it('asks the server to read a source it can parse, trimmed', () => {
    seed();
    open();
    render(<PluginCenterModal />);

    review('  owner/demo@v1  ');

    expect(actions.inspect).toHaveBeenCalledWith('owner/demo@v1');
    expect(
      screen.queryByText('Enter a catalog name, owner/repo[@ref] or a GitHub URL.'),
    ).toBeNull();
  });

  it('says it is reading the source while it waits', () => {
    seed({ inspection: { phase: 'inspecting', source: 'owner/demo' } });
    open();
    render(<PluginCenterModal />);

    expect(screen.getByRole('button', { name: 'Downloading...' })).toBeDisabled();
  });

  it('withdraws the inline refusal as soon as the box changes', () => {
    seed();
    open();
    render(<PluginCenterModal />);

    review('not a repo!');
    const input = screen.getByLabelText('Install from GitHub');
    expect(input).toHaveAttribute('aria-invalid', 'true');

    // The refusal was earned by the STRING, so it is withdrawn by the
    // keystroke that changes it rather than by another press of Review.
    fireEvent.change(input, { target: { value: 'owner/demo' } });
    expect(
      screen.queryByText('Enter a catalog name, owner/repo[@ref] or a GitHub URL.'),
    ).toBeNull();
    expect(input).not.toHaveAttribute('aria-invalid');
  });

  it('reports a source the server could not read', () => {
    // The store has already turned the code into a sentence; what is left is
    // naming the source it was about.
    seed({
      inspection: {
        phase: 'error',
        source: 'owner/demo',
        failure: { message: 'Could not reach GitHub.', code: 'github_unreachable', detail: null },
      },
    });
    open();
    render(<PluginCenterModal />);

    expect(
      screen.getByText('Could not fetch owner/demo: Could not reach GitHub.'),
    ).toBeInTheDocument();
    // `role="alert"` says it once when it arrives; this is what says it again
    // to somebody who tabs back into the field.
    expect(screen.getByLabelText('Install from GitHub')).toHaveAttribute(
      'aria-describedby', screen.getByRole('alert').id,
    );
  });

  it('does not carry a refusal over into the next time the panel is opened', () => {
    seed({
      inspection: {
        phase: 'error',
        source: 'owner/demo',
        failure: { message: 'Could not reach GitHub.', code: 'github_unreachable', detail: null },
      },
    });
    open();
    render(<PluginCenterModal />);

    act(() => {
      useUIStore.setState({ pluginCenterOpen: false });
    });

    // A refusal belongs to the box it was typed into; reopening the panel
    // over it would explain a request nobody made.
    expect(actions.clearInspection).toHaveBeenCalled();
  });

  it('keeps a review that is still waiting for an answer', () => {
    seed({ inspection: ready() });
    open();
    render(<PluginCenterModal />);

    act(() => {
      useUIStore.setState({ pluginCenterOpen: false });
    });

    // A decision, not a complaint: the install it leads to outlives the
    // window, so the card that offers it does too.
    expect(actions.clearInspection).not.toHaveBeenCalled();
  });

  it('names the id a reserved-id refusal was about', () => {
    // The refusal carries no sentence at all — only the id it is about, and
    // that is the whole useful part.
    seed({
      inspection: {
        phase: 'error',
        source: 'owner/edu',
        failure: {
          message: 'reserved_id',
          code: 'reserved_id',
          detail: { code: 'reserved_id', id: 'edu' },
        },
      },
    });
    open();
    render(<PluginCenterModal />);

    expect(
      screen.getByText('The id "edu" is reserved for a built-in pack.'),
    ).toBeInTheDocument();
    expect(screen.queryByText(/reserved_id/)).toBeNull();
  });

  it('lists the names a catalog miss offered instead', () => {
    seed({
      inspection: {
        phase: 'error',
        source: 'c9',
        failure: {
          message: 'unknown_catalog_name',
          code: 'unknown_catalog_name',
          detail: { code: 'unknown_catalog_name', known: ['edu', 'stats'] },
        },
      },
    });
    open();
    render(<PluginCenterModal />);

    expect(screen.getByText('No plugin is called "c9".')).toBeInTheDocument();
    // The list only exists in the refusal's body: no sentence written on the
    // server could have said which names would have worked.
    expect(screen.getByText('This server can install: edu, stats')).toBeInTheDocument();
  });

  it('will not review from a browser the server refuses installs from', () => {
    seed({ remoteInstallAllowed: false });
    open();
    render(<PluginCenterModal />);

    fireEvent.change(screen.getByLabelText('Install from GitHub'), {
      target: { value: 'owner/demo' },
    });
    const button = screen.getByRole('button', { name: 'Review' });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute(
      'title', 'Installing is only allowed from the computer that runs the server.',
    );
  });
});

// ── The review ──────────────────────────────────────────────────────────────

describe('PluginCenterModal — the review', () => {
  const card = () => screen.getByRole('region', { name: 'Review before installing' });

  it('stays out of the way until the server has answered', () => {
    seed({ plugins: [edu] });
    open();
    render(<PluginCenterModal />);
    expect(screen.queryByRole('region', { name: 'Review before installing' })).toBeNull();

    act(() => {
      usePluginStore.setState({ inspection: { phase: 'inspecting', source: 'owner/demo' } });
    });
    // Nothing has been read yet, and the button already says so: an empty
    // card here would be the same fact twice and a second layout jump.
    expect(screen.queryByRole('region', { name: 'Review before installing' })).toBeNull();
  });

  it('stays away from an install a row started that asks for nothing', () => {
    // `install()` inspects before it installs, and the review is READY
    // between those two round trips. Without this, pressing Install on a
    // built-in pack — the commonest thing in this panel — would flash a
    // consent card at the top of the list and scroll away from the row that
    // was clicked, for as long as the install request takes.
    seed({
      plugins: [edu],
      inspection: ready({ plugin_id: 'edu', consent_required: false }, 'edu'),
    });
    open();
    render(<PluginCenterModal />);

    expect(screen.queryByRole('region', { name: 'Review before installing' })).toBeNull();
  });

  it('answers a row whose plugin does ask for something', () => {
    seed({
      plugins: [demo],
      inspection: ready(
        { plugin_id: 'demo', consent_required: true, capabilities: ['network'] }, 'demo',
      ),
    });
    open();
    render(<PluginCenterModal />);

    // Which row it is about: the store's `forPluginId` never reaches the
    // card, so this attribute is the only thing tying the two together.
    expect(card()).toHaveAttribute('data-review-for', 'demo');
  });

  it('comes back for a row when the server refused the install', () => {
    // `already_installed` leaves `consent_required` false — the refusal is
    // the whole reason there is a card, and Reinstall is what answers it.
    seed({
      inspection: {
        ...ready({ plugin_id: 'demo', consent_required: false }, 'demo'),
        error: {
          message: 'already_installed', code: 'already_installed', detail: null,
        },
      },
    });
    open();
    render(<PluginCenterModal />);

    expect(within(card()).getByRole('button', { name: 'Reinstall' })).toBeInTheDocument();
  });

  it('brings itself into view, however far down the row that asked was', () => {
    // A row's own Install button raises a review, and this card renders at
    // the top of the list: without the scroll, clicking Install on the eighth
    // plugin would look like nothing happened.
    const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView');
    seed({ inspection: ready() });
    open();
    render(<PluginCenterModal />);

    expect(scroll).toHaveBeenCalledWith({ block: 'nearest' });
  });

  it('shows what was found at the source, and what installing it would cost', () => {
    seed({
      plugins: [entry({ id: 'demo', nodes: ['demo:Split', 'demo:Join'] })],
      inspection: ready({
        name: 'Demo plugin',
        version: '1.2.0',
        description: 'Two nodes and a panel.',
        manifest: { plugin: { authors: ['Ada', 'Grace'] } },
        homepage: 'https://example.com/demo',
        python_deps: { torch: '2.1.0' },
        ref: 'main',
        sha: 'abcdef1234567890',
        capabilities: ['network', 'gpu'],
        allowed_modules: ['subprocess'],
        has_frontend: true,
        consent_required: true,
      }),
    });
    open();
    render(<PluginCenterModal />);

    const review = within(card());
    expect(review.getByText('Demo plugin')).toBeInTheDocument();
    expect(review.getByText('v1.2.0')).toBeInTheDocument();
    expect(review.getByText('Two nodes and a panel.')).toBeInTheDocument();
    expect(review.getByText('Author: Ada, Grace')).toBeInTheDocument();
    // The catalog's row is where node names come from: an inspection cannot
    // say without importing the code it is asking about.
    expect(review.getByText('Nodes: demo:Split, demo:Join')).toBeInTheDocument();
    expect(review.getByText('Python packages: torch==2.1.0')).toBeInTheDocument();
    expect(review.getByText('main @ abcdef1')).toBeInTheDocument();
    expect(review.getByRole('link', { name: 'Homepage' }))
      .toHaveAttribute('href', 'https://example.com/demo');

    // One line per capability, saying what granting it costs — and the raw
    // id for one this build has no line for, rather than a silent gap.
    expect(review.getByText(/^network: reach any host/)).toBeInTheDocument();
    expect(review.getByText('gpu')).toBeInTheDocument();
    expect(review.getByText(/Granting is a declaration, not a sandbox/))
      .toBeInTheDocument();
    expect(review.getByText('I trust this author. Allows: subprocess'))
      .toBeInTheDocument();
    expect(
      review.getByText('Ships JavaScript that runs in this editor with full access.'),
    ).toBeInTheDocument();
  });

  it('asks nothing of a manifest that declares nothing', () => {
    seed({ inspection: ready() });
    open();
    render(<PluginCenterModal />);

    const review = within(card());
    expect(review.queryByRole('checkbox')).toBeNull();
    expect(review.queryByText('This plugin asks for:')).toBeNull();
    // Nothing to agree to, so nothing to wait for.
    expect(review.getByRole('button', { name: 'Install' })).toBeEnabled();
  });

  it('keeps Install dead until every box the manifest asks for is ticked', () => {
    seed({
      inspection: ready({
        capabilities: ['network'], allowed_modules: ['subprocess'], consent_required: true,
      }),
    });
    open();
    render(<PluginCenterModal />);

    const review = within(card());
    const install = review.getByRole('button', { name: 'Install' });
    expect(install).toBeDisabled();

    fireEvent.click(review.getByRole('checkbox', { name: 'Grant these capabilities' }));
    // One of two is not consent.
    expect(install).toBeDisabled();

    fireEvent.click(
      review.getByRole('checkbox', { name: 'I trust this author. Allows: subprocess' }),
    );
    expect(install).toBeEnabled();
  });

  it('sends the capabilities and the trust the user actually ticked', () => {
    seed({
      inspection: ready({
        capabilities: ['network'], allowed_modules: ['subprocess'], consent_required: true,
      }),
    });
    open();
    render(<PluginCenterModal />);

    const review = within(card());
    fireEvent.click(review.getByRole('checkbox', { name: 'Grant these capabilities' }));
    fireEvent.click(
      review.getByRole('checkbox', { name: 'I trust this author. Allows: subprocess' }),
    );
    fireEvent.click(review.getByRole('button', { name: 'Install' }));

    // The store turns the tick into the DECLARED list on the wire; what the
    // card promises is that it never sends a trust nobody gave.
    expect(actions.installInspected).toHaveBeenCalledWith({
      acceptCapabilities: true, trustAuthor: true,
    });
  });

  it('trusts no author when there was none to trust', () => {
    seed({ inspection: ready({ capabilities: ['network'], consent_required: true }) });
    open();
    render(<PluginCenterModal />);

    const review = within(card());
    fireEvent.click(review.getByRole('checkbox', { name: 'Grant these capabilities' }));
    fireEvent.click(review.getByRole('button', { name: 'Install' }));

    expect(actions.installInspected).toHaveBeenCalledWith({
      acceptCapabilities: true, trustAuthor: false,
    });
  });

  it('offers to reinstall a plugin the server says is already here', () => {
    seed({
      inspection: {
        ...ready(),
        error: {
          message: 'already_installed',
          code: 'already_installed',
          detail: { code: 'already_installed', plugin_id: 'demo' },
        },
      },
    });
    open();
    render(<PluginCenterModal />);

    // The 409 was an offer, not a failure: the review is still spendable and
    // the button is what accepts it.
    fireEvent.click(within(card()).getByRole('button', { name: 'Reinstall' }));
    expect(actions.installInspected).toHaveBeenCalledWith({
      acceptCapabilities: true, trustAuthor: false, force: true,
    });
  });

  it('says Update when the review is of a version over one already installed', () => {
    seed({ inspection: ready({ mode: 'update' }) });
    open();
    render(<PluginCenterModal />);

    expect(within(card()).getByRole('button', { name: 'Update' })).toBeInTheDocument();
  });

  it('will not state the installed nodes on a card about the next version', () => {
    // The catalog knows what is registered TODAY, which on an update is the
    // version being replaced — and this is the one screen whose whole job is
    // saying what is about to arrive.
    seed({
      plugins: [entry({ id: 'demo', nodes: ['demo:Split', 'demo:Join'] })],
      inspection: ready({ plugin_id: 'demo', mode: 'update' }),
    });
    open();
    render(<PluginCenterModal />);

    expect(within(card()).queryByText(/^Nodes:/)).toBeNull();
  });

  it('puts the review away when it is cancelled', () => {
    seed({ inspection: ready() });
    open();
    render(<PluginCenterModal />);

    fireEvent.click(within(card()).getByRole('button', { name: 'Cancel' }));
    expect(actions.clearInspection).toHaveBeenCalled();
  });

  it('will not link a homepage that is not a web address', () => {
    // A manifest at a source nobody has installed yet is untrusted text, and
    // `javascript:` in the one link on a consent screen is a script that
    // runs in this editor.
    seed({ inspection: ready({ homepage: 'javascript:alert(1)' }) });
    open();
    render(<PluginCenterModal />);

    expect(within(card()).queryByRole('link')).toBeNull();
  });

  it('will not install from a browser the server refuses installs from', () => {
    seed({ inspection: ready(), remoteInstallAllowed: false });
    open();
    render(<PluginCenterModal />);

    expect(within(card()).getByRole('button', { name: 'Install' })).toBeDisabled();
    // Putting the review away is not an install, and never was refused.
    expect(within(card()).getByRole('button', { name: 'Cancel' })).toBeEnabled();
  });
});

// ── The filter ──────────────────────────────────────────────────────────────

describe('PluginCenterModal — the filter', () => {
  it('shows one half of the catalog at a time', () => {
    seed({ plugins: [edu, stats] });
    open();
    render(<PluginCenterModal />);

    const all = screen.getByRole('button', { name: 'All' });
    const installed = screen.getByRole('button', { name: 'Installed' });
    const available = screen.getByRole('button', { name: 'Available' });
    expect(all).toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(installed);
    expect(installed).toHaveAttribute('aria-pressed', 'true');
    expect(all).toHaveAttribute('aria-pressed', 'false');
    expect(cardFor('edu')).not.toBeNull();
    expect(cardFor('stats')).toBeNull();

    fireEvent.click(available);
    expect(cardFor('edu')).toBeNull();
    expect(cardFor('stats')).not.toBeNull();

    fireEvent.click(all);
    expect(cardFor('edu')).not.toBeNull();
    expect(cardFor('stats')).not.toBeNull();
  });

  it('offers no filter over a catalog with nothing in it', () => {
    seed({ plugins: [] });
    open();
    render(<PluginCenterModal />);
    expect(screen.queryByRole('button', { name: 'All' })).toBeNull();
  });
});

// ── The deep link ───────────────────────────────────────────────────────────

describe('PluginCenterModal — the deep link', () => {
  it('scrolls to the plugin it was opened for, and consumes the request', () => {
    seed({ plugins: [edu, stats] });
    const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView');
    open('stats');
    render(<PluginCenterModal />);

    expect(scroll).toHaveBeenCalledWith({ block: 'nearest' });
    expect(cardFor('stats').className).toContain('cardHighlighted');
    expect(cardFor('edu').className).not.toContain('cardHighlighted');
    // Consumed, so a later unrelated render does not re-fire the jump.
    expect(useUIStore.getState().pluginCenterFocusPluginId).toBeNull();
  });

  it('widens the filter rather than sending a jump to a hidden row', () => {
    seed({ plugins: [edu, stats] });
    vi.spyOn(HTMLElement.prototype, 'scrollIntoView');
    open();
    render(<PluginCenterModal />);

    fireEvent.click(screen.getByRole('button', { name: 'Available' }));
    expect(cardFor('edu')).toBeNull();

    // A toast about an installed plugin, arriving while the list is narrowed
    // to what is not installed.
    act(() => {
      useUIStore.getState().openPluginCenter('edu');
    });

    expect(screen.getByRole('button', { name: 'All' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    expect(cardFor('edu').className).toContain('cardHighlighted');
    expect(useUIStore.getState().pluginCenterFocusPluginId).toBeNull();
  });

  it('drops the ring after HIGHLIGHT_MS so it reads as a pointer, not a state', () => {
    seed({ plugins: [edu, stats] });
    vi.useFakeTimers();
    try {
      open('stats');
      render(<PluginCenterModal />);
      expect(cardFor('stats').className).toContain('cardHighlighted');

      // The constant, not a copy of it: a ring that outlived its exported
      // duration would still pass a hard-coded 2000.
      act(() => {
        vi.advanceTimersByTime(HIGHLIGHT_MS - 1);
      });
      expect(cardFor('stats').className).toContain('cardHighlighted');

      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(cardFor('stats').className).not.toContain('cardHighlighted');
    } finally {
      vi.useRealTimers();
    }
  });
});

// ── Closing ─────────────────────────────────────────────────────────────────

describe('PluginCenterModal — closing', () => {
  beforeEach(() => {
    seed({ plugins: [edu] });
    open();
    render(<PluginCenterModal />);
  });

  it('closes on Escape', () => {
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().pluginCenterOpen).toBe(false);
  });

  it('closes on the backdrop but not on a press inside the surface', () => {
    const dialog = screen.getByRole('dialog');
    fireEvent.mouseDown(dialog);
    expect(useUIStore.getState().pluginCenterOpen).toBe(true);

    fireEvent.mouseDown(dialog.parentElement!);
    expect(useUIStore.getState().pluginCenterOpen).toBe(false);
  });

  it('closes on the close button', () => {
    fireEvent.click(screen.getByRole('button', { name: 'Close Plugin Center' }));
    expect(useUIStore.getState().pluginCenterOpen).toBe(false);
  });

  it('leaves Escape alone while something is stacked above it', () => {
    useUIStore.setState({ shortcutsModalOpen: true });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().pluginCenterOpen).toBe(true);
    useUIStore.setState({ shortcutsModalOpen: false });

    useDialogStore.setState({ active: { kind: 'confirm', title: 'x' } as never });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().pluginCenterOpen).toBe(true);
    useDialogStore.setState({ active: null });

    // The Package Center answers the same key. One press, one window.
    useUIStore.setState({ packCenterOpen: true });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().pluginCenterOpen).toBe(true);
    useUIStore.setState({ packCenterOpen: false });

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(useUIStore.getState().pluginCenterOpen).toBe(false);
  });

  it('hands focus back to whatever had it', () => {
    // The modal took focus on open; closing it must not leave focus on <body>.
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveFocus();

    act(() => {
      useUIStore.getState().closePluginCenter();
    });
    expect(document.activeElement).not.toBe(dialog);
  });
});
