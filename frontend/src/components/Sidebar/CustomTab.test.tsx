import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CustomTab } from './CustomTab';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import * as rest from '../../api/rest';
import type {
  CustomNodeInfo,
  PackCatalog,
  PackSummary,
  PluginSummary,
} from '../../api/rest';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return {
    ...actual,
    listCustomNodes: vi.fn(),
    listPlugins: vi.fn(),
    // The packs section is a view of `packStore`; the tab's Refresh button
    // asks it to re-read the catalog through this call.
    listPacks: vi.fn(),
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

function plugin(overrides: Partial<PluginSummary> = {}): PluginSummary {
  return {
    id: 'c1',
    name: 'Chapter 1',
    version: '1.0.0',
    description: 'Intro nodes',
    enabled: true,
    nodes: ['EduAdd', 'EduMul'],
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
  gpu: null,
};

// `_resetPackStoreForTesting` restores the store's DATA, not its actions, so
// the one case that installs a fake `refresh` would otherwise leave it in
// place for every case after it.
const realPackRefresh = usePackStore.getState().refresh;

/** Put a catalog in the store, the way a finished `refresh()` would. */
function seedPacks(packs: PackSummary[], unsupported = false) {
  usePackStore.setState({
    loaded: true,
    unsupported,
    packs,
    byId: Object.fromEntries(packs.map((pack) => [pack.id, pack])),
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useUIStore.setState({ packCenterOpen: false, packCenterFocusPackId: null });
  _resetPackStoreForTesting();
  usePackStore.setState({ refresh: realPackRefresh });
  mockedRest.listCustomNodes.mockReset();
  mockedRest.listPlugins.mockReset();
  mockedRest.listPacks.mockReset();
  mockedRest.listCustomNodes.mockResolvedValue([]);
  mockedRest.listPlugins.mockResolvedValue([]);
  mockedRest.listPacks.mockResolvedValue(EMPTY_CATALOG);
  seedPacks([]);
});

afterEach(() => {
  // The pack store is reset in `beforeEach`, NOT here: the tab is still
  // mounted at this point, and writing to a store it subscribes to would
  // re-render it outside `act`.
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

  it('shows an empty state per section, with the install hint for plugins', async () => {
    render(<CustomTab />);
    await screen.findByText('No custom nodes yet');
    expect(screen.getByText('No plugins installed')).toBeTruthy();
    expect(screen.getByText('Install packs with the cdui plugin CLI')).toBeTruthy();
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
    mockedRest.listPlugins.mockResolvedValue([plugin()]);
    render(<CustomTab />);
    await screen.findByText('Chapter 1');
    expect(screen.getByText('v1.0.0')).toBeTruthy();
    expect(screen.getByText('Intro nodes')).toBeTruthy();
    expect(screen.getByText('2 nodes')).toBeTruthy();
    expect(screen.getByText('Enabled')).toBeTruthy();
  });

  it('greys out a disabled plugin and omits absent optional fields', async () => {
    mockedRest.listPlugins.mockResolvedValue([
      plugin({ id: 'c2', name: 'Chapter 2', version: '', description: '', enabled: false, nodes: [] }),
    ]);
    const { container } = render(<CustomTab />);
    await screen.findByText('Chapter 2');
    expect(screen.getByText('Disabled')).toBeTruthy();
    expect(container.querySelector('[data-disabled="true"]')).toBeTruthy();
    expect(container.querySelectorAll('[class*="rowVersion"]')).toHaveLength(0);
    expect(container.querySelectorAll('[class*="rowDesc"]')).toHaveLength(0);
    expect(container.querySelectorAll('[class*="rowMeta"]')).toHaveLength(0);
  });

  it('shows the section counts', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    mockedRest.listPlugins.mockResolvedValue([plugin(), plugin({ id: 'c2', name: 'Chapter 2' })]);
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
    mockedRest.listPlugins.mockRejectedValueOnce(new Error('backend gone'));
    render(<CustomTab />);
    await screen.findByText('Failed to load: backend gone');

    mockedRest.listPlugins.mockResolvedValue([plugin({ name: 'Recovered' })]);
    fireEvent.click(screen.getByText('Retry'));
    await screen.findByText('Recovered');
  });

  it('re-fetches from the refresh button', async () => {
    render(<CustomTab />);
    await screen.findByText('No plugins installed');
    mockedRest.listPlugins.mockResolvedValue([plugin({ name: 'Just installed' })]);
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
    await waitFor(() => expect(mockedRest.listPlugins).toHaveBeenCalledTimes(2));
  });

  it('does not read the catalog on mount — the sidebar shell already has', async () => {
    render(<CustomTab />);
    await screen.findByText('No optional packs available');
    expect(mockedRest.listPacks).not.toHaveBeenCalled();
  });
});
