import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as rest from '../api/rest';
import type {
  JobEventsPage,
  PluginCatalog,
  PluginCatalogEntry,
  PluginInspection,
} from '../api/rest';
import { ApiError } from '../api/rest';

// Partial mock: `ApiError` is a real class the store narrows on with
// `instanceof`, so only the network calls are stubbed.
vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listPluginCatalog: vi.fn(),
    inspectPluginSource: vi.fn(),
    installPlugin: vi.fn(),
    updatePlugin: vi.fn(),
    uninstallPlugin: vi.fn(),
    setPluginEnabled: vi.fn(),
    getPluginJobEvents: vi.fn(),
    cancelPluginJob: vi.fn(),
  };
});

// The uninstall confirmation is an in-app modal driven by a promise; mocking
// the helper keeps these tests about the STORE's decisions.
vi.mock('../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

// The real one imports plugin bundles over the network and mounts widgets.
// What matters here is only THAT it ran, and when.
vi.mock('../plugins/PluginHost', () => ({
  reloadPluginFrontends: vi.fn(),
}));

import {
  _resetPluginStoreForTesting,
  emptyPluginJob,
  parseGitHubSource,
  usePluginStore,
} from './pluginStore';
import { useNodeDefStore } from './nodeDefStore';
import { useToastStore } from './toastStore';
import { useUIStore } from './uiStore';
import { useI18n } from '../i18n';
import { confirm } from '../utils/dialog';
import { reloadPluginFrontends } from '../plugins/PluginHost';

const api = vi.mocked(rest);
const confirmMock = vi.mocked(confirm);
const reloadMock = vi.mocked(reloadPluginFrontends);

/**
 * What ran, in the order it ran.
 *
 * The three-step refresh is a promise of ORDER as much as of contents — the
 * catalog before the definitions before the frontends — and three separate
 * `toHaveBeenCalled` assertions cannot say that.
 */
const order: string[] = [];

/** The most recent toast. `Array.prototype.at` is outside the project lib. */
function lastToast() {
  const { toasts } = useToastStore.getState();
  return toasts[toasts.length - 1];
}

function entry(
  partial: Partial<PluginCatalogEntry> & { id: string },
): PluginCatalogEntry {
  return {
    name: partial.id,
    description: '',
    kind: 'builtin',
    official: true,
    status: 'available',
    source_kind: null,
    source: partial.id,
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
    ...partial,
  };
}

function catalog(partial: Partial<PluginCatalog> = {}): PluginCatalog {
  return {
    entries: [],
    active_job: null,
    remote_install_allowed: true,
    generation: 1,
    ...partial,
  };
}

function inspection(partial: Partial<PluginInspection> = {}): PluginInspection {
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
    sha: 'a'.repeat(40),
    name: 'Demo plugin',
    version: '1.0.0',
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
    ...partial,
  };
}

function eventsPage(partial: Partial<JobEventsPage> = {}): JobEventsPage {
  return { job_id: 'j1', status: 'running', events: [], cursor: 0, ...partial };
}

/** A refusal shaped the way FastAPI sends one: the code lives under `detail`. */
function refused(
  status: number, code: string, extra: Record<string, unknown> = {},
): ApiError {
  return new ApiError(status, code, { detail: { code, ...extra } });
}

/** A parked long poll — the server holding the connection open. */
function parked(): Promise<JobEventsPage> {
  return new Promise<JobEventsPage>(() => {});
}

/**
 * Answer every catalog read with *value*, recording the call in `order`.
 *
 * Always through this rather than `mockResolvedValue`, which would replace the
 * recording implementation and quietly drop the catalog step out of the order
 * the three-step refresh is asserted on.
 */
function serveCatalog(value: PluginCatalog = catalog()): void {
  api.listPluginCatalog.mockImplementation(async () => {
    order.push('catalog');
    return value;
  });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  _resetPluginStoreForTesting();
  order.length = 0;

  serveCatalog();
  api.inspectPluginSource.mockResolvedValue(inspection());
  api.installPlugin.mockResolvedValue({ job_id: 'j1' });
  api.updatePlugin.mockResolvedValue({ kind: 'job', job_id: 'j1' });
  api.uninstallPlugin.mockResolvedValue({
    id: 'demo',
    removed: true,
    tombstoned: false,
    files_removed: true,
    python_deps_left: [],
    uninstall_command: null,
    reinstall_hint: '',
  });
  api.setPluginEnabled.mockImplementation(async (id, enabled) => ({ id, enabled }));
  // Running-and-idle by default: an accidental follower parks instead of
  // settling and firing toasts into an unrelated test.
  api.getPluginJobEvents.mockResolvedValue(eventsPage());
  api.cancelPluginJob.mockResolvedValue({ job_id: 'j1', cancelled: true });
  confirmMock.mockResolvedValue(true);
  reloadMock.mockImplementation(async () => {
    order.push('frontends');
    return [];
  });
  // A fresh `vi.fn()` through `setState`, never `vi.spyOn` on an action read
  // off `getState()`: the spy would keep a stale object and carry its call
  // history into the next case.
  useNodeDefStore.setState({
    fetchDefinitions: vi.fn(async () => { order.push('definitions'); }),
  });
});

afterEach(() => {
  _resetPluginStoreForTesting();
  // The panel is not this file's store, but a toast action opens it — and an
  // open panel inherited by the next case is a state nothing here set.
  useUIStore.setState({ pluginCenterOpen: false, pluginCenterFocusPluginId: null });
  vi.useRealTimers();
  vi.clearAllMocks();
});

// ── source parsing ───────────────────────────────────────────────────────

describe('parseGitHubSource', () => {
  it('reads the three forms the CLI accepts', () => {
    expect(parseGitHubSource('owner/repo')).toEqual({
      kind: 'github', owner: 'owner', repo: 'repo', ref: null,
    });
    expect(parseGitHubSource('owner/repo@v1.2')).toEqual({
      kind: 'github', owner: 'owner', repo: 'repo', ref: 'v1.2',
    });
    expect(parseGitHubSource('https://github.com/owner/repo')).toEqual({
      kind: 'github', owner: 'owner', repo: 'repo', ref: null,
    });
  });

  it('accepts the shapes a pasted URL actually has', () => {
    // What comes off the GitHub UI's clone button and address bar.
    expect(parseGitHubSource('https://github.com/owner/repo.git')).toMatchObject({
      owner: 'owner', repo: 'repo',
    });
    expect(parseGitHubSource('https://www.github.com/owner/repo/')).toMatchObject({
      owner: 'owner', repo: 'repo',
    });
    expect(parseGitHubSource('http://github.com/owner/repo@main')).toMatchObject({
      repo: 'repo', ref: 'main',
    });
  });

  it('reads a bare word as a catalog name and trims what was typed', () => {
    expect(parseGitHubSource('  c1  ')).toEqual({ kind: 'catalog', id: 'c1' });
  });

  it('refuses anything that is neither', () => {
    for (const bad of ['', '   ', 'Not A Source', 'https://gitlab.com/o/r', 'owner/']) {
      expect(parseGitHubSource(bad), bad).toBeNull();
    }
  });
});

// ── the catalog ──────────────────────────────────────────────────────────

describe('pluginStore — refresh', () => {
  it('reads a 404 as a server without a Plugin Center', async () => {
    api.listPluginCatalog.mockRejectedValue(new ApiError(404, 'Not Found'));

    await usePluginStore.getState().refresh();

    const state = usePluginStore.getState();
    expect(state.unsupported).toBe(true);
    expect(state.loaded).toBe(true);
    expect(state.plugins).toEqual([]);
    // Not an error the user did anything about, so nothing reports one.
    expect(state.error).toBeNull();
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it('keeps the rows it has when the network drops', async () => {
    api.listPluginCatalog.mockResolvedValueOnce(
      catalog({ entries: [entry({ id: 'demo', name: 'Demo plugin' })], generation: 7 }),
    );
    await usePluginStore.getState().refresh();

    api.listPluginCatalog.mockRejectedValue(new Error('Failed to fetch'));
    await usePluginStore.getState().refresh();

    const state = usePluginStore.getState();
    expect(state.plugins).toHaveLength(1);
    expect(state.byId.demo.name).toBe('Demo plugin');
    expect(state.generation).toBe(7);
    expect(state.error).toBe('Failed to fetch');
    // A dropped packet is not an answer: `unsupported` would blank the panel.
    expect(state.unsupported).toBe(false);
  });

  it('adopts the server active job, and only follows it once', async () => {
    serveCatalog(catalog({
      active_job: { job_id: 'j9', plugin_id: 'demo', kind: 'update' },
    }));
    api.getPluginJobEvents.mockImplementation(parked);

    await usePluginStore.getState().refresh();
    await usePluginStore.getState().refresh();

    expect(usePluginStore.getState().job).toMatchObject({
      jobId: 'j9', pluginId: 'demo', kind: 'update', status: 'running',
    });
    // Adoption runs on every poll; restarting the follower each time would
    // abort a long poll that was about to answer.
    expect(api.getPluginJobEvents).toHaveBeenCalledTimes(1);
  });

  it('marks a running job lost when the server no longer knows it', async () => {
    usePluginStore.setState({ job: emptyPluginJob('j1', 'demo') });

    await usePluginStore.getState().refresh();

    expect(usePluginStore.getState().job!.status).toBe('lost');
  });

  it('leaves a running job alone while a follower is still on it', async () => {
    api.getPluginJobEvents.mockImplementation(parked);
    usePluginStore.getState().followJob('j1', 'demo');

    await usePluginStore.getState().refresh();

    // `active_job` goes null the moment a job ends, and the follower's next
    // page has the real answer; `lost` from here is a race.
    expect(usePluginStore.getState().job!.status).toBe('running');
  });
});

// ── inspecting and installing ────────────────────────────────────────────

describe('pluginStore — install', () => {
  it('installs straight away when the inspection asks for no consent', async () => {
    api.inspectPluginSource.mockResolvedValue(inspection({
      plugin_id: 'c1', consent_required: false, capabilities: [], mode: 'install',
    }));

    await usePluginStore.getState().install('c1');

    expect(api.inspectPluginSource).toHaveBeenCalledWith('c1');
    expect(api.installPlugin).toHaveBeenCalledWith({
      inspection_id: 'insp-1', accept_capabilities: [],
    });
    expect(usePluginStore.getState().job).toMatchObject({
      jobId: 'j1', pluginId: 'c1', kind: 'install',
    });
    // The review is spent: the card must not stay on screen behind the job.
    expect(usePluginStore.getState().inspection).toEqual({ phase: 'idle' });
    expect(usePluginStore.getState().busy.c1).toBeFalsy();
  });

  it('stops at the review when the plugin asks for something', async () => {
    api.inspectPluginSource.mockResolvedValue(inspection({
      plugin_id: 'demo', consent_required: true, capabilities: ['net'],
    }));

    await usePluginStore.getState().install('demo');

    expect(api.installPlugin).not.toHaveBeenCalled();
    expect(usePluginStore.getState().inspection).toMatchObject({
      phase: 'ready',
      source: 'demo',
      // The row it came from, so the panel can show the review beside it.
      forPluginId: 'demo',
      kind: 'install',
      error: null,
    });
  });

  it('shows the row as installing the moment the job is accepted', async () => {
    serveCatalog(
      catalog({ entries: [entry({ id: 'demo', status: 'available' })] }),
    );
    await usePluginStore.getState().refresh();
    api.inspectPluginSource.mockResolvedValue(inspection({ plugin_id: 'demo' }));

    await usePluginStore.getState().install('demo');

    expect(usePluginStore.getState().byId.demo.status).toBe('installing');
    expect(usePluginStore.getState().plugins[0].status).toBe('installing');
  });

  it('refuses a second install while a job is running', async () => {
    usePluginStore.setState({ job: emptyPluginJob('j1', 'other') });

    await usePluginStore.getState().install('demo');

    expect(api.inspectPluginSource).not.toHaveBeenCalled();
    expect(lastToast().message).toBe('Another install is already running.');
  });

  it('ignores a second click while the first request is in flight', async () => {
    let release: (value: PluginInspection) => void = () => {};
    api.inspectPluginSource.mockReturnValue(
      new Promise<PluginInspection>((resolve) => { release = resolve; }),
    );

    const first = usePluginStore.getState().install('demo');
    expect(usePluginStore.getState().busy.demo).toBe(true);
    await usePluginStore.getState().install('demo');
    expect(api.inspectPluginSource).toHaveBeenCalledTimes(1);

    release(inspection({ plugin_id: 'demo' }));
    await first;
    expect(usePluginStore.getState().busy.demo).toBeFalsy();
  });
});

describe('pluginStore — inspect', () => {
  it('refuses an unparseable source without asking the server', async () => {
    await usePluginStore.getState().inspect('not a source!');

    expect(api.inspectPluginSource).not.toHaveBeenCalled();
    expect(usePluginStore.getState().inspection).toEqual({
      phase: 'error',
      source: 'not a source!',
      message: 'Enter owner/repo[@ref] or a GitHub URL.',
    });
  });

  it('records a typed source with no row of its own', async () => {
    api.inspectPluginSource.mockResolvedValue(inspection({ mode: 'install' }));

    await usePluginStore.getState().inspect('  owner/demo  ');

    expect(api.inspectPluginSource).toHaveBeenCalledWith('owner/demo');
    expect(usePluginStore.getState().inspection).toMatchObject({
      phase: 'ready', source: 'owner/demo', forPluginId: null, kind: 'install',
    });
  });

  it('keeps the server refusal as the review error', async () => {
    api.inspectPluginSource.mockRejectedValue(
      refused(400, 'unparseable_source'),
    );

    await usePluginStore.getState().inspect('owner/demo');

    expect(usePluginStore.getState().inspection).toEqual({
      phase: 'error', source: 'owner/demo', message: 'unparseable_source',
    });
  });

  it('clears the review on request', async () => {
    await usePluginStore.getState().inspect('owner/demo');
    usePluginStore.getState().clearInspection();

    expect(usePluginStore.getState().inspection).toEqual({ phase: 'idle' });
  });
});

describe('pluginStore — installInspected', () => {
  /** Put a ready review in front of the store, the way `inspect` would. */
  async function ready(partial: Partial<PluginInspection> = {}): Promise<void> {
    api.inspectPluginSource.mockResolvedValue(inspection({
      plugin_id: 'demo', consent_required: true, capabilities: ['net', 'fs'],
      allowed_modules: ['requests'], ...partial,
    }));
    await usePluginStore.getState().inspect('owner/demo');
  }

  it('sends the capabilities the manifest declared, never a blanket yes', async () => {
    await ready();

    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: true,
    });

    expect(api.installPlugin).toHaveBeenCalledWith({
      inspection_id: 'insp-1',
      accept_capabilities: ['net', 'fs'],
      trust_author: true,
    });
    const body = api.installPlugin.mock.calls[0][0];
    expect(Array.isArray(body.accept_capabilities)).toBe(true);
    // Never sent unless a caller asks for it: the inspection's own mode is
    // what makes an update an update.
    expect(body.force).toBeUndefined();
  });

  it('omits what the user did not accept', async () => {
    await ready();

    await usePluginStore.getState().installInspected({
      acceptCapabilities: false, trustAuthor: false,
    });

    expect(api.installPlugin).toHaveBeenCalledWith({ inspection_id: 'insp-1' });
  });

  it('sends force only when it is asked for', async () => {
    await ready();

    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: false, force: true,
    });

    expect(api.installPlugin.mock.calls[0][0].force).toBe(true);
  });

  it('seeds the job and starts following on a 202', async () => {
    await ready();
    api.getPluginJobEvents.mockImplementation(parked);

    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: false,
    });

    expect(usePluginStore.getState().job).toMatchObject({
      jobId: 'j1', pluginId: 'demo', kind: 'install', status: 'running', cursor: 0,
    });
    expect(api.getPluginJobEvents).toHaveBeenCalledWith(
      'j1', expect.objectContaining({ cursor: 0 }),
    );
  });

  it('keeps the review open when the consent was incomplete', async () => {
    await ready();
    api.installPlugin.mockRejectedValue(
      refused(400, 'consent_required', { missing_capabilities: ['net'] }),
    );

    await usePluginStore.getState().installInspected({
      acceptCapabilities: false, trustAuthor: false,
    });

    const state = usePluginStore.getState();
    // The fix is a tick box on the card in front of the user, not a new
    // inspection: the review stays, and grows a message.
    expect(state.inspection).toMatchObject({
      phase: 'ready', forPluginId: null, error: 'consent_required',
    });
    expect(state.job).toBeNull();
  });

  it('keeps the review open when the author was not trusted', async () => {
    await ready();
    api.installPlugin.mockRejectedValue(
      refused(400, 'trust_author_required', { allowed_modules: ['requests'] }),
    );

    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: false,
    });

    expect(usePluginStore.getState().inspection).toMatchObject({
      phase: 'ready', error: 'trust_author_required',
    });
  });

  it('toasts and refreshes when the server is already busy', async () => {
    await ready();
    api.installPlugin.mockRejectedValue(refused(409, 'busy', { job_id: 'j2' }));

    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: false,
    });

    expect(lastToast().message).toBe('Another install is already running.');
    // Whatever the server IS running is more useful than the refusal.
    expect(api.listPluginCatalog).toHaveBeenCalled();
  });

  it('says so when the server refuses a remote install', async () => {
    await ready();
    api.installPlugin.mockRejectedValue(new ApiError(403, 'Forbidden'));

    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: false,
    });

    expect(lastToast().message).toBe(
      'Installing is only allowed from the computer that runs the server.',
    );
    expect(lastToast().type).toBe('error');
  });

  it('reports anything else as a failed install', async () => {
    await ready();
    api.installPlugin.mockRejectedValue(new Error('Failed to fetch'));

    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: false,
    });

    expect(lastToast().message).toBe('Install failed: Failed to fetch');
  });

  it('does nothing without a review to install', async () => {
    await usePluginStore.getState().installInspected({
      acceptCapabilities: true, trustAuthor: false,
    });

    expect(api.installPlugin).not.toHaveBeenCalled();
  });
});

// ── update ───────────────────────────────────────────────────────────────

describe('pluginStore — update', () => {
  beforeEach(async () => {
    serveCatalog(catalog({
      entries: [entry({ id: 'demo', name: 'Demo plugin', status: 'installed' })],
    }));
    await usePluginStore.getState().refresh();
  });

  it('says so when there was nothing to update', async () => {
    api.updatePlugin.mockResolvedValue({ kind: 'up_to_date', sha: 'a'.repeat(40) });

    await usePluginStore.getState().update('demo');

    expect(lastToast().message).toBe('Demo plugin is up to date.');
    expect(usePluginStore.getState().job).toBeNull();
  });

  it('opens the review when the new version asks for more', async () => {
    api.updatePlugin.mockResolvedValue({
      kind: 'needs_consent',
      inspection: inspection({ plugin_id: 'demo', mode: 'update' }),
      capabilities_added: ['net'],
      allowed_modules_added: [],
    });

    await usePluginStore.getState().update('demo');

    expect(usePluginStore.getState().inspection).toMatchObject({
      phase: 'ready', forPluginId: 'demo', kind: 'update', source: 'demo',
    });
    expect(usePluginStore.getState().job).toBeNull();
  });

  it('follows the job an accepted update starts', async () => {
    api.getPluginJobEvents.mockImplementation(parked);

    await usePluginStore.getState().update('demo');

    expect(usePluginStore.getState().job).toMatchObject({
      jobId: 'j1', pluginId: 'demo', kind: 'update',
    });
    expect(usePluginStore.getState().byId.demo.status).toBe('installing');
    expect(api.getPluginJobEvents).toHaveBeenCalledTimes(1);
  });

  it('reports a refused update', async () => {
    api.updatePlugin.mockRejectedValue(new ApiError(502, 'GitHub is down'));

    await usePluginStore.getState().update('demo');

    expect(lastToast().message).toBe('Update failed: GitHub is down');
    expect(usePluginStore.getState().busy.demo).toBeFalsy();
  });
});

// ── uninstall, enable, disable ───────────────────────────────────────────

describe('pluginStore — uninstall', () => {
  beforeEach(async () => {
    serveCatalog(catalog({
      entries: [entry({ id: 'demo', name: 'Demo plugin', status: 'installed' })],
    }));
    await usePluginStore.getState().refresh();
    order.length = 0;
  });

  it('asks first, and does nothing when the answer is no', async () => {
    confirmMock.mockResolvedValue(false);

    await usePluginStore.getState().uninstall('demo');

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(api.uninstallPlugin).not.toHaveBeenCalled();
  });

  it('names the plugin in the question it asks', async () => {
    await usePluginStore.getState().uninstall('demo');

    expect(confirmMock.mock.calls[0][0]).toMatchObject({
      message:
        'Uninstall Demo plugin? Graphs that use its nodes will stop running. '
        + 'Its Python packages stay installed.',
      variant: 'danger',
    });
  });

  it('removes it, then re-reads the catalog, the nodes and the frontends', async () => {
    await usePluginStore.getState().uninstall('demo');

    expect(api.uninstallPlugin).toHaveBeenCalledWith('demo');
    expect(order).toEqual(['catalog', 'definitions', 'frontends']);
    expect(lastToast().message).toBe('Demo plugin uninstalled.');
    expect(lastToast().type).toBe('success');
  });

  it('carries the server hint when the files could not be deleted', async () => {
    api.uninstallPlugin.mockRejectedValue(refused(409, 'files_locked', {
      error: 'WinError 32', hint: 'Close the app that is holding the file.',
    }));

    await usePluginStore.getState().uninstall('demo');

    expect(lastToast().message).toBe(
      'Could not remove Demo plugin: Close the app that is holding the file.',
    );
    expect(lastToast().type).toBe('warning');
  });

  it('reports any other refusal against the plugin name', async () => {
    api.uninstallPlugin.mockRejectedValue(new Error('Failed to fetch'));

    await usePluginStore.getState().uninstall('demo');

    expect(lastToast().message).toBe('Could not remove Demo plugin: Failed to fetch');
    expect(usePluginStore.getState().busy.demo).toBeFalsy();
  });
});

describe('pluginStore — setEnabled', () => {
  beforeEach(async () => {
    serveCatalog(catalog({
      entries: [entry({ id: 'demo', name: 'Demo plugin', status: 'installed' })],
    }));
    await usePluginStore.getState().refresh();
    order.length = 0;
  });

  it('enables, then re-reads everything the switch changed', async () => {
    await usePluginStore.getState().setEnabled('demo', true);

    expect(api.setPluginEnabled).toHaveBeenCalledWith('demo', true);
    expect(order).toEqual(['catalog', 'definitions', 'frontends']);
    expect(lastToast().message).toBe('Demo plugin enabled.');
  });

  it('disables the same way', async () => {
    await usePluginStore.getState().setEnabled('demo', false);

    expect(api.setPluginEnabled).toHaveBeenCalledWith('demo', false);
    expect(order).toEqual(['catalog', 'definitions', 'frontends']);
    expect(lastToast().message).toBe('Demo plugin disabled.');
  });

  it('reports a refused switch and clears the row', async () => {
    api.setPluginEnabled.mockRejectedValue(new Error('nope'));

    await usePluginStore.getState().setEnabled('demo', true);

    expect(lastToast().message).toBe('Could not change Demo plugin: nope');
    expect(usePluginStore.getState().busy.demo).toBeFalsy();
  });

  it('survives a step of the refresh failing, and still runs the rest', async () => {
    reloadMock.mockRejectedValue(new Error('bundle gone'));

    await usePluginStore.getState().setEnabled('demo', true);

    // The definitions were still fetched, and the failure was reported.
    expect(order).toEqual(['catalog', 'definitions']);
    expect(useToastStore.getState().toasts.map((toast) => toast.message)).toContain(
      'Install failed: bundle gone',
    );
  });
});

// ── the follower and its endings ─────────────────────────────────────────

describe('pluginStore — the follower', () => {
  /**
   * Timers are faked throughout: the loop's only real waits are its idle and
   * retry sleeps, and asserting "it did NOT poll again" by sleeping 10 ms of
   * wall clock proves nothing when the next turn was 500 ms away regardless.
   */
  beforeEach(() => {
    vi.useFakeTimers();
  });

  /** Let queued microtasks (a resolved fetch, a `set`) run to completion. */
  const settle = () => vi.advanceTimersByTimeAsync(0);

  /** Follow a job whose pages end in *pages*. */
  function follow(kind: 'install' | 'update' = 'install'): void {
    usePluginStore.getState().followJob('j1', 'demo', kind, 0);
  }

  it('folds pages into the job and settles it once', async () => {
    api.getPluginJobEvents
      .mockResolvedValueOnce(eventsPage({
        cursor: 2,
        events: [
          { type: 'step_started', cursor: 1, ts: 't', step: 'download', label: 'Downloading' },
          { type: 'log', cursor: 2, ts: 't', line: 'unpacking' },
        ],
      }))
      .mockResolvedValue(eventsPage({
        status: 'done', cursor: 3, events: [{ type: 'job_done', cursor: 3, ts: 't' }],
      }));

    follow();
    await settle();

    const job = usePluginStore.getState().job!;
    expect(job.status).toBe('done');
    expect(job.cursor).toBe(3);
    expect(job.steps).toEqual([
      { step: 'download', label: 'Downloading', state: 'done' },
    ]);
    expect(job.log.map((line) => line.text)).toEqual([
      'Downloading', 'unpacking', 'done',
    ]);
  });

  it('runs the catalog, the nodes and the frontends in that order when one lands', async () => {
    api.getPluginJobEvents.mockResolvedValue(eventsPage({ status: 'done', cursor: 0 }));

    follow();
    await settle();

    expect(order).toEqual(['catalog', 'definitions', 'frontends']);
    expect(lastToast().message).toBe('demo installed.');
    expect(lastToast().type).toBe('success');
  });

  it('says updated for an update job', async () => {
    api.getPluginJobEvents.mockResolvedValue(eventsPage({ status: 'done', cursor: 0 }));

    follow('update');
    await settle();

    expect(lastToast().message).toBe('demo updated.');
  });

  it('reports a failure with the message the job carried', async () => {
    api.getPluginJobEvents.mockResolvedValue(eventsPage({
      status: 'failed',
      cursor: 1,
      events: [{
        type: 'job_failed', cursor: 1, ts: 't', message: 'pip exited 1', hint: null,
      }],
    }));

    follow();
    await settle();

    expect(lastToast().message).toBe('Install failed: pip exited 1');
    expect(lastToast().type).toBe('error');
    // Nothing landed, so only the catalog is stale.
    expect(order).toEqual(['catalog']);
  });

  it('puts an Open Plugin Center button on the failure, pointed at the row', async () => {
    api.getPluginJobEvents.mockResolvedValue(eventsPage({ status: 'failed', cursor: 0 }));

    follow();
    await settle();

    expect(lastToast().action!.label).toBe('Open Plugin Center');
    lastToast().action!.onClick();
    expect(useUIStore.getState().pluginCenterOpen).toBe(true);
    expect(useUIStore.getState().pluginCenterFocusPluginId).toBe('demo');
  });

  it('reports a cancelled job', async () => {
    api.getPluginJobEvents.mockResolvedValue(eventsPage({ status: 'cancelled', cursor: 0 }));

    follow();
    await settle();

    expect(lastToast().message).toBe('Install cancelled.');
    expect(order).toEqual(['catalog']);
  });

  it('keeps a needs_restart job on screen and says what to do', async () => {
    api.getPluginJobEvents.mockResolvedValue(eventsPage({
      status: 'needs_restart',
      cursor: 1,
      events: [{
        type: 'needs_restart', cursor: 1, ts: 't', command: 'cdui start',
      }],
    }));

    follow();
    await settle();

    expect(lastToast().message).toBe('Restart the server to load demo.');
    expect(lastToast().type).toBe('warning');
    const job = usePluginStore.getState().job!;
    expect(job.status).toBe('needs_restart');
    expect(job.restartCommand).toBe('cdui start');
  });

  it('settles a job exactly once, however often it is adopted', async () => {
    api.getPluginJobEvents.mockResolvedValue(eventsPage({ status: 'done', cursor: 0 }));

    follow();
    await settle();
    follow();
    await settle();

    expect(
      useToastStore.getState().toasts.filter((t) => t.message === 'demo installed.'),
    ).toHaveLength(1);
  });

  it('marks the job lost after the retries run out', async () => {
    api.getPluginJobEvents.mockRejectedValue(new Error('Failed to fetch'));

    follow();
    await settle();
    await vi.advanceTimersByTimeAsync(2000 * 5);

    // No restart handshake to read into the silence: `lost` is the honest
    // answer, and nothing is toasted about it.
    expect(usePluginStore.getState().job!.status).toBe('lost');
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it('stops following on request', async () => {
    api.getPluginJobEvents.mockImplementation(parked);

    follow();
    await settle();
    usePluginStore.getState().stopFollowing();

    await vi.advanceTimersByTimeAsync(60_000);
    expect(api.getPluginJobEvents).toHaveBeenCalledTimes(1);
  });
});

// ── cancel and dismiss ───────────────────────────────────────────────────

describe('pluginStore — cancel and dismiss', () => {
  it('asks the server to cancel the open job', async () => {
    usePluginStore.setState({ job: emptyPluginJob('j5', 'demo') });

    await usePluginStore.getState().cancel();

    expect(api.cancelPluginJob).toHaveBeenCalledWith('j5');
    // Cooperative: the FOLLOWER records the outcome, not this call.
    expect(usePluginStore.getState().job!.status).toBe('running');
    expect(usePluginStore.getState().cancelling).toBe(false);
  });

  it('reports a cancel that did not go through', async () => {
    usePluginStore.setState({ job: emptyPluginJob('j5', 'demo') });
    api.cancelPluginJob.mockRejectedValue(new Error('gone'));

    await usePluginStore.getState().cancel();

    expect(lastToast().message).toBe('Could not cancel the install: gone');
    expect(usePluginStore.getState().cancelling).toBe(false);
  });

  it('has nothing to cancel when no job is running', async () => {
    await usePluginStore.getState().cancel();
    usePluginStore.setState({
      job: { ...emptyPluginJob('j5', 'demo'), status: 'done' },
    });
    await usePluginStore.getState().cancel();

    expect(api.cancelPluginJob).not.toHaveBeenCalled();
  });

  it('refuses to dismiss a job that is still running', () => {
    usePluginStore.setState({ job: emptyPluginJob('j5', 'demo') });

    usePluginStore.getState().dismissJob();
    expect(usePluginStore.getState().job).not.toBeNull();

    usePluginStore.setState({
      job: { ...emptyPluginJob('j5', 'demo'), status: 'failed' },
    });
    usePluginStore.getState().dismissJob();
    expect(usePluginStore.getState().job).toBeNull();
  });
});

// ── the boot read ────────────────────────────────────────────────────────

describe('pluginStore — checkInProgress', () => {
  it('adopts a job that outlived the page, and says so', async () => {
    serveCatalog(catalog({
      entries: [entry({ id: 'demo', name: 'Demo plugin' })],
      active_job: { job_id: 'j9', plugin_id: 'demo', kind: 'install' },
    }));
    api.getPluginJobEvents.mockImplementation(parked);

    await usePluginStore.getState().checkInProgress();

    expect(lastToast().message).toBe(
      'A plugin is still installing. Open the Plugin Center to watch it.',
    );
    expect(lastToast().action!.label).toBe('Open Plugin Center');
    expect(usePluginStore.getState().job!.jobId).toBe('j9');
  });

  it('says nothing when there was no job to adopt', async () => {
    await usePluginStore.getState().checkInProgress();

    expect(useToastStore.getState().toasts).toEqual([]);
    expect(usePluginStore.getState().loaded).toBe(true);
  });

  it('reads the catalog once per page load', async () => {
    await usePluginStore.getState().checkInProgress();
    await usePluginStore.getState().checkInProgress();

    expect(api.listPluginCatalog).toHaveBeenCalledTimes(1);
  });

  it('lets a later mount try again when nothing answered', async () => {
    api.listPluginCatalog.mockRejectedValueOnce(new Error('Failed to fetch'));

    await usePluginStore.getState().checkInProgress();
    expect(usePluginStore.getState().loaded).toBe(false);

    await usePluginStore.getState().checkInProgress();
    expect(api.listPluginCatalog).toHaveBeenCalledTimes(2);
    expect(usePluginStore.getState().loaded).toBe(true);
  });

  it('stays quiet on a server without the route', async () => {
    api.listPluginCatalog.mockRejectedValue(new ApiError(404, 'Not Found'));

    await usePluginStore.getState().checkInProgress();

    expect(useToastStore.getState().toasts).toEqual([]);
    expect(usePluginStore.getState().unsupported).toBe(true);
  });
});

// ── the test hatch ───────────────────────────────────────────────────────

describe('_resetPluginStoreForTesting', () => {
  it('puts the store and its schedulers back to boot state', async () => {
    api.getPluginJobEvents.mockImplementation(parked);
    serveCatalog(catalog({
      entries: [entry({ id: 'demo' })], generation: 4,
    }));
    await usePluginStore.getState().checkInProgress();
    usePluginStore.getState().followJob('j1', 'demo');
    usePluginStore.setState({ busy: { demo: true }, cancelling: true });

    _resetPluginStoreForTesting();

    expect(usePluginStore.getState()).toMatchObject({
      plugins: [], byId: {}, loaded: false, unsupported: false, error: null,
      generation: 0, job: null, busy: {}, cancelling: false,
      inspection: { phase: 'idle' },
    });
    // The once-per-page guard is released too, or every case after the first
    // would silently skip its boot read.
    await usePluginStore.getState().checkInProgress();
    expect(api.listPluginCatalog).toHaveBeenCalledTimes(2);
  });
});
