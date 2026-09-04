import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CustomTab } from './CustomTab';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import { _resetPluginStoreForTesting, usePluginStore } from '../../store/pluginStore';
import * as rest from '../../api/rest';
import type {
  CustomNodeInfo,
  PackCatalog,
  PackSummary,
  PluginCatalog,
  PluginCatalogEntry,
} from '../../api/rest';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return {
    ...actual,
    listCustomNodes: vi.fn(),
    // Both catalogs are read through the stores, and both stores read them
    // through these calls; the tab's Refresh button asks for a re-read.
    listPacks: vi.fn(),
    listPluginCatalog: vi.fn(),
  };
});

// The manager modal is the existing CustomNodeManager; this tab only owns
// opening and closing it, so a stub keeps its own fetches out of these tests.
vi.mock('../CustomNodeManager/CustomNodeManager', () => ({
  CustomNodeManager: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="custom-node-manager">
      <button type="button" onClick={onClose}>close manager</button>
    </div>
  ),
}));

const mockedRest = vi.mocked(rest);

function customNode(overrides: Partial<CustomNodeInfo> = {}): CustomNodeInfo {
  return { filename: 'my_node.py', enabled: true, nodes: ['MyNode'], ...overrides };
}

/** One installed, enabled plugin, as the catalog answers with it. */
function plugin(overrides: Partial<PluginCatalogEntry> = {}): PluginCatalogEntry {
  return {
    id: 'c1',
    name: 'Chapter 1',
    description: 'Intro nodes',
    kind: 'builtin',
    official: true,
    status: 'installed',
    source_kind: 'builtin',
    source: 'c1',
    repo: null,
    ref: null,
    sha: null,
    url: null,
    homepage: '',
    version: '1.0.0',
    installed_at: null,
    enabled: true,
    chapters: [],
    lessons: [],
    tags: [],
    nodes: ['EduAdd', 'EduMul'],
    node_count: 2,
    capabilities: [],
    trusted_modules: [],
    python_deps: {},
    has_frontend: false,
    consent_required: false,
    frontend_entry: null,
    job: null,
    ...overrides,
  };
}

function packSummary(overrides: Partial<PackSummary> = {}): PackSummary {
  return {
    id: 'word-vectors',
    title: 'Word vectors',
    description: 'A GloVe table',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: true,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...overrides,
  };
}

const EMPTY_CATALOG: PackCatalog = {
  packs: [],
  active_job: null,
  last_restart_job: null,
  remote_install_allowed: true,
  launch_mode: 'start',
  restart_available: false,
  gpu: null,
};

const EMPTY_PLUGIN_CATALOG: PluginCatalog = {
  entries: [],
  active_job: null,
  remote_install_allowed: true,
  generation: 0,
};

// `_reset*ForTesting` restores a store's DATA, not its actions, so the cases
// that install a fake `refresh` would otherwise leave it in place for every
// case after them.
const realPackRefresh = usePackStore.getState().refresh;
const realPluginRefresh = usePluginStore.getState().refresh;

/** Put a catalog in the store, the way a finished `refresh()` would. */
function seedPacks(packs: PackSummary[], unsupported = false) {
  usePackStore.setState({
    loaded: true,
    unsupported,
    packs,
    byId: Object.fromEntries(packs.map((pack) => [pack.id, pack])),
  });
}

/**
 * The same for plugins. The section is a pure VIEW of this store now — the
 * tab has no `listPlugins` call of its own — so every case that wants a
 * plugin row puts one here.
 */
function seedPlugins(
  plugins: PluginCatalogEntry[],
  extra: Partial<ReturnType<typeof usePluginStore.getState>> = {},
) {
  usePluginStore.setState({
    loaded: true,
    loading: false,
    unsupported: false,
    error: null,
    plugins,
    byId: Object.fromEntries(plugins.map((entry) => [entry.id, entry])),
    ...extra,
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({
    packCenterOpen: false,
    packCenterFocusPackId: null,
    pluginCenterOpen: false,
    pluginCenterFocusPluginId: null,
  });
  _resetPackStoreForTesting();
  _resetPluginStoreForTesting();
  usePackStore.setState({ refresh: realPackRefresh });
  usePluginStore.setState({ refresh: realPluginRefresh });
  mockedRest.listCustomNodes.mockReset();
  mockedRest.listPacks.mockReset();
  mockedRest.listPluginCatalog.mockReset();
  mockedRest.listCustomNodes.mockResolvedValue([]);
  mockedRest.listPacks.mockResolvedValue(EMPTY_CATALOG);
  mockedRest.listPluginCatalog.mockResolvedValue(EMPTY_PLUGIN_CATALOG);
  seedPacks([]);
  seedPlugins([]);
});

afterEach(() => {
  // Both stores are reset in `beforeEach`, NOT here: the tab is still mounted
  // at this point, and writing to a store it subscribes to would re-render it
  // outside `act`.
  vi.restoreAllMocks();
});

describe('CustomTab', () => {
  it('shows the loading state, then both section headers', async () => {
    render(<CustomTab />);
    expect(screen.getByText('Loading...')).toBeTruthy();
    await waitFor(() => expect(screen.queryByText('Loading...')).toBeNull());
    expect(screen.getByText('Custom & Plugins')).toBeTruthy();
    expect(screen.getByText('Custom Nodes')).toBeTruthy();
    expect(screen.getByText('Plugins')).toBeTruthy();
  });

  it('shows an empty state per section, and no hint that repeats a button', async () => {
    render(<CustomTab />);
    await screen.findByText('No custom nodes yet');
    expect(screen.getByText('No plugins installed')).toBeTruthy();
    // The button one line above the empty state IS the Plugin Center, so the
    // hint that named it as a destination is gone. The packs one stays: it
    // says what a pack is, which no button can.
    expect(screen.getByRole('button', { name: 'Plugin Center...' })).toBeTruthy();
    expect(screen.queryByText('Install plugins from the Plugin Center')).toBeNull();
    expect(
      screen.getByText(
        'Models and libraries for LLM nodes are installed from the Package Center',
      ),
    ).toBeTruthy();
  });

  it('lists custom node files with their node names and enabled chip', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([
      customNode({ filename: 'a.py', nodes: ['Alpha', 'Beta'] }),
      customNode({ filename: 'b.py', enabled: false, nodes: [] }),
    ]);
    render(<CustomTab />);
    await screen.findByText('a.py');
    expect(screen.getByText('Alpha, Beta')).toBeTruthy();
    expect(screen.getByText('Enabled')).toBeTruthy();
    expect(screen.getByText('Disabled')).toBeTruthy();
    // A file with no parsed nodes renders no node-name line.
    expect(screen.getByText('b.py')).toBeTruthy();
  });

  it('lists plugin packs with version, description and node count', async () => {
    seedPlugins([plugin()]);
    render(<CustomTab />);
    await screen.findByText('Chapter 1');
    expect(screen.getByText('v1.0.0')).toBeTruthy();
    expect(screen.getByText('Intro nodes')).toBeTruthy();
    expect(screen.getByText('2 nodes')).toBeTruthy();
    expect(screen.getByText('Enabled')).toBeTruthy();
  });

  it('greys out a disabled plugin and omits absent optional fields', async () => {
    seedPlugins([
      plugin({
        id: 'c2',
        name: 'Chapter 2',
        status: 'disabled',
        version: null,
        description: '',
        enabled: false,
        nodes: [],
      }),
    ]);
    const { container } = render(<CustomTab />);
    await screen.findByText('Chapter 2');
    expect(screen.getByText('Disabled')).toBeTruthy();
    expect(container.querySelector('[data-disabled="true"]')).toBeTruthy();
    expect(container.querySelectorAll('[class*="rowVersion"]')).toHaveLength(0);
    expect(container.querySelectorAll('[class*="rowDesc"]')).toHaveLength(0);
    expect(container.querySelectorAll('[class*="rowMeta"]')).toHaveLength(0);
  });

  it('lists what is installed, and leaves the rest of the catalog to the panel', async () => {
    // The catalog is everything a server COULD have; this section has always
    // answered "what have I got". An install still running and a lockfile
    // entry whose files are gone are in neither half.
    seedPlugins([
      plugin({ id: 'c1', name: 'Chapter 1', status: 'installed' }),
      plugin({ id: 'c2', name: 'Chapter 2', status: 'disabled', enabled: false }),
      plugin({ id: 'c3', name: 'Chapter 3', status: 'available', enabled: false }),
      plugin({ id: 'c4', name: 'Chapter 4', status: 'removed', enabled: false }),
      plugin({ id: 'c5', name: 'Chapter 5', status: 'installing', enabled: false }),
      plugin({ id: 'c6', name: 'Chapter 6', status: 'missing_files', enabled: false }),
    ]);
    const { container } = render(<CustomTab />);

    await screen.findByText('Chapter 1');
    expect(screen.getByText('Chapter 2')).toBeTruthy();
    for (const absent of ['Chapter 3', 'Chapter 4', 'Chapter 5', 'Chapter 6']) {
      expect(screen.queryByText(absent)).toBeNull();
    }
    // ...and the header count agrees with the rows it is counting.
    const counts = Array.from(container.querySelectorAll('[class*="sectionCount"]')).map(
      (el) => el.textContent,
    );
    expect(counts).toEqual(['0', '0', '2']);
  });

  it('shows the empty state on a server with no Plugin Center at all', async () => {
    // `unsupported` and a stale catalog can be true together; the server's
    // verdict wins, and no row claims a plugin this server cannot report on.
    seedPlugins([plugin()], { unsupported: true });
    render(<CustomTab />);

    expect(await screen.findByText('No plugins installed')).toBeTruthy();
    expect(screen.queryByText('Chapter 1')).toBeNull();
  });

  it('opens the Plugin Center from the section header', async () => {
    render(<CustomTab />);
    await screen.findByText('No plugins installed');

    fireEvent.click(screen.getByText('Plugin Center...'));
    expect(useUIStore.getState().pluginCenterOpen).toBe(true);
    // No plugin is focused: the handler must not pass its click event
    // through as a plugin id.
    expect(useUIStore.getState().pluginCenterFocusPluginId).toBeNull();
  });

  it('shows the section counts', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    seedPlugins([plugin(), plugin({ id: 'c2', name: 'Chapter 2' })]);
    const { container } = render(<CustomTab />);
    await screen.findByText('a.py');
    const counts = Array.from(container.querySelectorAll('[class*="sectionCount"]')).map(
      (el) => el.textContent,
    );
    // Nodes, packs, plugins — the packs section sits between the other two.
    expect(counts).toEqual(['1', '0', '2']);
  });

  it('opens the custom node manager and re-fetches when it closes', async () => {
    render(<CustomTab />);
    await screen.findByText('No custom nodes yet');
    expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText('Manage...'));
    expect(screen.getByTestId('custom-node-manager')).toBeTruthy();

    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'uploaded.py' })]);
    fireEvent.click(screen.getByText('close manager'));
    expect(screen.queryByTestId('custom-node-manager')).toBeNull();
    await screen.findByText('uploaded.py');
  });

  it('shows the error state and retries on click', async () => {
    // A catalog read that failed is still the tab's error to report: the
    // Plugins section is one of the two lists this tab is about, and half a
    // tab behind an error the other half is not in reads as a broken list.
    seedPlugins([], { error: 'backend gone', loaded: false });
    render(<CustomTab />);
    await screen.findByText('Failed to load: backend gone');

    // A fresh action installed through setState, never a spy on the live one:
    // a spy taken from getState() outlives the store it was taken from.
    usePluginStore.setState({
      refresh: vi.fn(async () => {
        seedPlugins([plugin({ name: 'Recovered' })]);
      }),
    });
    fireEvent.click(screen.getByText('Retry'));
    await screen.findByText('Recovered');
  });

  it('keeps the tab when a catalog it already has fails to refresh', async () => {
    // The plugin store is SHARED, and its error is sticky until the next
    // catalog lands: a refresh the Plugin Center asked for, over rows this
    // tab is already showing, must not replace the custom nodes and the packs
    // with somebody else's dropped packet.
    seedPlugins([plugin({ name: 'Chapter 1' })], { error: 'connection refused' });
    render(<CustomTab />);
    await screen.findByText('Chapter 1');

    expect(screen.queryByText('Failed to load: connection refused')).toBeNull();
  });

  it('re-fetches from the refresh button', async () => {
    render(<CustomTab />);
    await screen.findByText('No plugins installed');
    usePluginStore.setState({
      refresh: vi.fn(async () => {
        seedPlugins([plugin({ name: 'Just installed' })]);
      }),
    });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await screen.findByText('Just installed');
  });

  // ── Optional packs (the Package Center entry point) ───────────────

  it('lists packs with status pills and opens the Package Center', async () => {
    seedPacks([
      packSummary({ id: 'word-vectors', status: 'installed' }),
      packSummary({ id: 'from-the-future', title: 'Newer pack', status: 'partial' }),
    ]);
    const { container } = render(<CustomTab />);

    expect(await screen.findByText('Optional packs')).toBeTruthy();
    // Named from THIS build's catalog copy where it has one, and from the
    // server's own title where it does not.
    expect(screen.getByText('Word vectors (GloVe)')).toBeTruthy();
    expect(screen.getByText('Newer pack')).toBeTruthy();
    expect(screen.getByText('Installed')).toBeTruthy();
    expect(screen.getByText('Partly installed')).toBeTruthy();
    // The same pill the modal uses, so a status reads the same in both.
    expect(container.querySelectorAll('[class*="pill"][data-tone]')).toHaveLength(2);
    // The count is the ROWS listed, like both sibling sections: "1" over
    // two listed packs reads as a list that failed to load half of itself.
    const counts = Array.from(container.querySelectorAll('[class*="sectionCount"]')).map(
      (el) => el.textContent,
    );
    expect(counts).toEqual(['0', '2', '0']);

    fireEvent.click(screen.getByText('Package Center...'));
    expect(useUIStore.getState().packCenterOpen).toBe(true);
    // No pack is focused: the handler must not pass its click event through
    // as a pack id.
    expect(useUIStore.getState().packCenterFocusPackId).toBeNull();
  });

  it('describes a pack from this build, and falls back to the server text', async () => {
    seedPacks([
      packSummary({ id: 'word-vectors' }),
      packSummary({ id: 'from-the-future', description: 'Whatever the server says' }),
      packSummary({ id: 'no-words', description: '' }),
    ]);
    const { container } = render(<CustomTab />);

    expect(
      await screen.findByText(
        'Real 400k-word GloVe-50d table for WordVector; no Python packages needed',
      ),
    ).toBeTruthy();
    expect(screen.getByText('Whatever the server says')).toBeTruthy();
    // A pack with nothing to say gets no empty line under its name.
    expect(container.querySelectorAll('[class*="rowDesc"]')).toHaveLength(2);
  });

  it('shows the empty hint when the server has no packs', async () => {
    render(<CustomTab />);

    expect(await screen.findByText('No optional packs available')).toBeTruthy();
    expect(
      screen.getByText(
        'Models and libraries for LLM nodes are installed from the Package Center',
      ),
    ).toBeTruthy();
  });

  it('shows the same hint on a server with no Package Center at all', async () => {
    seedPacks([packSummary({ id: 'word-vectors', status: 'installed' })], true);
    const { container } = render(<CustomTab />);

    // `unsupported` and a stale catalog can be true together; the server's
    // verdict wins, and no row claims a pack this server cannot install.
    expect(await screen.findByText('No optional packs available')).toBeTruthy();
    expect(screen.queryByText('Word vectors (GloVe)')).toBeNull();
    // ...and the header count agrees with the rows it is counting.
    const counts = Array.from(container.querySelectorAll('[class*="sectionCount"]')).map(
      (el) => el.textContent,
    );
    expect(counts).toEqual(['0', '0', '0']);
  });

  it('the refresh button re-reads the pack catalog too', async () => {
    render(<CustomTab />);
    await screen.findByText('No optional packs available');
    // A fresh action installed through setState, never a spy on the live one:
    // a spy taken from getState() outlives the store it was taken from.
    const refresh = vi.fn().mockResolvedValue(undefined);
    usePackStore.setState({ refresh });

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    expect(refresh).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(2));
  });

  it('the refresh button re-reads the plugin catalog too', async () => {
    render(<CustomTab />);
    await screen.findByText('No plugins installed');
    const refresh = vi.fn().mockResolvedValue(undefined);
    usePluginStore.setState({ refresh });

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));

    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it('does not read either catalog on mount — the sidebar shell already has', async () => {
    render(<CustomTab />);
    await screen.findByText('No optional packs available');
    expect(mockedRest.listPacks).not.toHaveBeenCalled();
    expect(mockedRest.listPluginCatalog).not.toHaveBeenCalled();
  });
});
