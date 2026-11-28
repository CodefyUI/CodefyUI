import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import * as projectsApi from '../../api/projects';
import type { ProjectsOverview } from '../../api/projects';
import { PROJECT_CHANGED_EVENT } from '../../api/_auth';
import * as projectSwitch from '../../utils/projectSwitch';
import { useGitStore } from '../../store/gitStore';
import { useProjectStore } from '../../store/projectStore';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import { renderSettled, settle } from '../../test/utils';
import { ProjectChip } from './ProjectChip';
import { ProjectDialog } from './ProjectDialog';
import { ProjectNotices } from './ProjectNotices';
import { ModsNotLoadedBanner } from './ModsNotLoadedBanner';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';

vi.mock('../../api/projects', async (importOriginal) => {
  const actual = await importOriginal<typeof projectsApi>();
  return {
    ...actual,
    openProject: vi.fn(),
    createProject: vi.fn(),
    cloneProject: vi.fn(),
    trustProjectEnv: vi.fn(),
    removeRecentProject: vi.fn(),
    browseFolders: vi.fn(),
    getImportPlan: vi.fn(),
    importFlatGraphs: vi.fn(),
    getProjects: vi.fn(),
    getProjectMods: vi.fn(),
    trustProjectMods: vi.fn(),
  };
});

vi.mock('../../utils/projectSwitch', async (importOriginal) => {
  const actual = await importOriginal<typeof projectSwitch>();
  return {
    ...actual,
    confirmLeavingUnsavedTabs: vi.fn(async () => true),
    finishSwitch: vi.fn(async () => undefined),
    takeAfterSwitchNotice: vi.fn(() => null),
  };
});

const api = vi.mocked(projectsApi);
const switching = vi.mocked(projectSwitch);

const ENV_NONE = {
  state: 'none' as const, names: [], blocked: [], applied: [], kept: [],
  restart_recommended: [], sha256: null, needs_consent: false,
};

function overview(patch: Partial<ProjectsOverview> = {}): ProjectsOverview {
  return {
    current: { path: '/p/alpha', name: 'alpha', project_id: 'a'.repeat(16) },
    project_id: 'a'.repeat(16),
    projects_root: '/p',
    recent: [
      { path: '/p/alpha', name: 'alpha', last_opened: '', missing: false },
      { path: '/p/beta', name: 'beta', last_opened: '', missing: false },
      { path: '/p/gone', name: 'gone', last_opened: '', missing: true },
    ],
    env: ENV_NONE,
    startup: null,
    flat_graphs: 0,
    browse_allowed: false,
    ...patch,
  };
}

function switched(name = 'beta') {
  return {
    project: { path: `/p/${name}`, name, project_id: 'b'.repeat(16) },
    project_id: 'b'.repeat(16),
    env: ENV_NONE,
    env_restart_recommended: [],
  };
}

beforeEach(() => {
  useProjectStore.setState({
    projectDir: '/p/alpha', projectName: 'alpha', loaded: true,
    overview: overview(), stale: false,
  });
  useGitStore.setState({
    repoState: 'ready',
    status: {
      branch: 'main', detached: false, head: null, unborn: false, upstream: null,
      ahead: null, behind: null, upstream_gone: false, staged: [],
      unstaged: [{ path: 'graphs/a.json' } as never], untracked: [], conflicted: [],
      stash_count: 0, merge_in_progress: false, rebase_in_progress: false,
    },
  });
  useToastStore.setState({ toasts: [] });
});

afterEach(() => {
  vi.clearAllMocks();
  try {
    sessionStorage.clear();
  } catch {
    /* jsdom always has it */
  }
});

describe('ProjectChip', () => {
  it('names the project and its branch, with a dot for uncommitted changes', async () => {
    await renderSettled(<ProjectChip />);
    expect(screen.getByTestId('project-chip-name').textContent).toBe('alpha');
    expect(screen.getByText('main')).toBeTruthy();
    expect(screen.getByRole('img', { name: 'Uncommitted changes' })).toBeTruthy();
  });

  it('lists the other recent projects and switches to one', async () => {
    api.openProject.mockResolvedValue(switched());
    await renderSettled(<ProjectChip />);
    fireEvent.click(screen.getByRole('button', { name: /Project: alpha/ }));
    const menu = screen.getByRole('menu');
    expect(within(menu).queryByRole('menuitem', { name: 'Open alpha' })).toBeNull();
    const missing = within(menu).getByRole('menuitem', { name: /Open gone/ });
    expect(missing.getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Open beta' }));
    await waitFor(() => expect(switching.finishSwitch).toHaveBeenCalled());
    expect(switching.confirmLeavingUnsavedTabs).toHaveBeenCalled();
    expect(api.openProject).toHaveBeenCalledWith('/p/beta');
  });

  it('shows a refused switch as the server worded it', async () => {
    api.openProject.mockRejectedValue(new projectsApi.ProjectError(
      409, 'runs_active', 'Cannot switch projects: 1 run is running in this project'));
    await renderSettled(<ProjectChip />);
    fireEvent.click(screen.getByRole('button', { name: /Project: alpha/ }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Open beta' }));
    await waitFor(() => expect(useToastStore.getState().toasts[0]?.message)
      .toBe('Cannot switch projects: 1 run is running in this project'));
    expect(switching.finishSwitch).not.toHaveBeenCalled();
  });

  it('offers to import the flat graphs only in flat mode', async () => {
    useProjectStore.setState({
      projectDir: null, projectName: null,
      overview: overview({ current: null, project_id: 'flat', flat_graphs: 3 }),
    });
    await renderSettled(<ProjectChip />);
    expect(screen.getByTestId('project-chip-name').textContent).toBe('No project');
    fireEvent.click(screen.getByRole('button', { name: /Project: No project/ }));
    expect(screen.getByRole('menuitem', { name: /3 graph\(s\)/ })).toBeTruthy();
    expect(screen.queryByRole('menuitem', { name: 'Show in Source Control' })).toBeNull();
  });

  it('says the branch first in the menu and opens Source Control from it', async () => {
    await renderSettled(<ProjectChip />);
    const trigger = screen.getByRole('button', { name: /Project: alpha/ });
    // The tooltip keeps what a narrow window leaves off the chip.
    expect(trigger.getAttribute('title'))
      .toBe('Project: alpha · branch main · uncommitted changes');
    fireEvent.click(trigger);
    const rows = within(screen.getByRole('menu')).getAllByRole('menuitem');
    expect(rows[0].textContent).toContain('On branch main');
    expect(rows[0].textContent).toContain('with uncommitted changes');
    expect(within(screen.getByRole('menu'))
      .queryByRole('menuitem', { name: 'Show in Source Control' })).toBeNull();
    fireEvent.click(rows[0]);
    expect(useUIStore.getState().sidebarTab).toBe('git');
  });

  it('offers Show in Source Control when there is no branch to name', async () => {
    useGitStore.setState({ status: null });
    await renderSettled(<ProjectChip />);
    const trigger = screen.getByRole('button', { name: /Project: alpha/ });
    expect(trigger.getAttribute('title')).toBe('Project: alpha');
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole('menuitem', { name: 'Show in Source Control' }));
    expect(useUIStore.getState().sidebarTab).toBe('git');
  });
});

describe('ProjectDialog', () => {
  it.each([
    ['has space', 'no spaces'],
    ['.hidden', 'cannot start with'],
    ['trailing.', 'cannot end with'],
    ['CON', 'Windows keeps'],
    ['lpt1.txt', 'Windows keeps'],
    ['a/b', 'no spaces'],
  ])('says while typing that %s cannot be a project name', async (name, says) => {
    await renderSettled(<ProjectDialog mode="create" onClose={() => {}} />);
    const submit = screen.getByRole('button', { name: 'Create and open' });
    expect(submit).toHaveProperty('disabled', true); // nothing typed yet
    const input = screen.getByLabelText('Project name');
    fireEvent.change(input, { target: { value: name } });
    const note = screen.getByTestId('project-name-note');
    expect(note.textContent).toContain(says);
    expect(note.textContent).not.toContain('It will be created at');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(submit).toHaveProperty('disabled', true);
    fireEvent.change(input, { target: { value: 'lab' } });
    expect(screen.getByTestId('project-name-note').textContent).toContain('/p/lab');
    expect(submit).toHaveProperty('disabled', false);
  });

  it('checks a new project name before asking the server', async () => {
    await renderSettled(<ProjectDialog mode="create" onClose={() => {}} />);
    const input = screen.getByLabelText('Project name');
    fireEvent.change(input, { target: { value: 'has space' } });
    fireEvent.submit(input.closest('form')!);
    expect(screen.getByRole('alert').textContent).toContain('no spaces');
    expect(api.createProject).not.toHaveBeenCalled();

    api.createProject.mockResolvedValue(switched('lab'));
    fireEvent.change(input, { target: { value: 'lab' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create and open' }));
    await waitFor(() => expect(switching.finishSwitch).toHaveBeenCalled());
    expect(api.createProject).toHaveBeenCalledWith('lab');
  });

  it('shows the server refusal and its hint in the dialog', async () => {
    api.cloneProject.mockRejectedValue(new projectsApi.ProjectError(
      400, 'not_a_project', 'not a CodefyUI project; the clone was removed',
      'Add the manifest to the repository'));
    await renderSettled(<ProjectDialog mode="clone" onClose={() => {}} />);
    const input = screen.getByLabelText('Repository URL (https)');
    fireEvent.change(input, { target: { value: 'http://x/y.git' } });
    fireEvent.click(screen.getByRole('button', { name: 'Clone and open' }));
    expect(screen.getByRole('alert').textContent).toContain('https://');
    fireEvent.change(input, { target: { value: 'https://git.example/team/y.git' } });
    fireEvent.click(screen.getByRole('button', { name: 'Clone and open' }));
    await waitFor(() => expect(screen.getByRole('alert').textContent)
      .toContain('the clone was removed'));
    expect(screen.getByRole('alert').textContent).toContain('Add the manifest');
    expect(switching.finishSwitch).not.toHaveBeenCalled();
  });

  it('does nothing when the user keeps the unsaved tabs', async () => {
    switching.confirmLeavingUnsavedTabs.mockResolvedValueOnce(false);
    await renderSettled(<ProjectDialog mode="open" onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/Project folder/), { target: { value: '/p/beta' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open' }));
    await settle();
    expect(api.openProject).not.toHaveBeenCalled();
  });

  it('lists recents (with Remove) and refuses to browse off this machine', async () => {
    api.removeRecentProject.mockResolvedValue({ removed: '/p/beta' });
    api.getProjects.mockResolvedValue(overview());
    await renderSettled(<ProjectDialog mode="open" onClose={() => {}} />);
    expect(screen.getByText(/only on the machine the server runs on/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Remove beta from the recent list' }));
    await waitFor(() => expect(api.removeRecentProject).toHaveBeenCalledWith('/p/beta'));
    expect(api.browseFolders).not.toHaveBeenCalled();
  });

  it('browses folders when the server allows it', async () => {
    useProjectStore.setState({ overview: overview({ browse_allowed: true }) });
    api.browseFolders.mockResolvedValue({
      dir: '/p', parent: '/', is_project: false, truncated: false,
      entries: [{ name: 'beta', path: '/p/beta', is_project: true },
        { name: 'notes', path: '/p/notes', is_project: false }],
    });
    await renderSettled(<ProjectDialog mode="open" onClose={() => {}} />);
    expect(screen.getByText('Folders in /p')).toBeTruthy();
    expect(screen.getAllByText('project')).toHaveLength(1);
  });

  it('lists each file an import copies, with sizes and the total', async () => {
    api.getImportPlan.mockResolvedValue({
      source: '/install/backend/data',
      graphs: [{ name: 'train', file: 'train.json', size: 2048, readable: true }],
      graphs_bytes: 2048,
      assets: [
        { path: 'models/weights.pt', kind: 'models', size: 5 * 1024 * 1024, files: 1, is_dir: false },
        { path: 'images/cats', kind: 'images', size: 3000, files: 2, is_dir: true },
      ],
      assets_bytes: 5 * 1024 * 1024 + 3000,
    });
    api.importFlatGraphs.mockResolvedValue(switched('workspace'));
    await renderSettled(<ProjectDialog mode="import" onClose={() => {}} />);
    expect(screen.getByText('1 graph(s), 2.0 KB')).toBeTruthy();
    expect(screen.getByText('models/weights.pt')).toBeTruthy();
    expect(screen.getByText('5.0 MB')).toBeTruthy();
    expect(screen.getByText('images/cats/')).toBeTruthy();
    expect(screen.getByText('2.9 KB · 2 file(s)')).toBeTruthy();
    const box = screen.getByRole('checkbox') as HTMLInputElement;
    expect(box.checked).toBe(true);
    expect(box.parentElement!.textContent).toContain('(5.0 MB in all)');
    expect(screen.getByText(/nothing in \/install\/backend\/data is moved/)).toBeTruthy();

    fireEvent.click(box);
    fireEvent.click(screen.getByRole('button', { name: 'Copy into the project and open it' }));
    await waitFor(() => expect(api.importFlatGraphs)
      .toHaveBeenCalledWith({ name: 'workspace' }, false));
  });
});

describe('ProjectDialog keyboard', () => {
  it('puts the focus back in the field after a refusal, and Escape still closes', async () => {
    api.openProject.mockRejectedValue(new projectsApi.ProjectError(
      403, 'path_refused', '/private/etc is inside the system directory /etc'));
    const onClose = vi.fn();
    await renderSettled(<ProjectDialog mode="open" onClose={onClose} />);
    const input = screen.getByLabelText(/Project folder/) as HTMLInputElement;
    fireEvent.change(input, { target: { value: '/etc' } });
    const submit = screen.getByRole('button', { name: 'Open' });
    submit.focus();
    fireEvent.click(submit);
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('/etc'));
    expect(document.activeElement).toBe(input);
    expect([input.selectionStart, input.selectionEnd]).toEqual([0, 4]);

    // Even from <body> (where a disabled button leaves the focus).
    input.blur();
    expect(document.activeElement).toBe(document.body);
    fireEvent.keyDown(document.body, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('keeps Tab inside the dialog, also when the focus fell out of it', async () => {
    await renderSettled(<ProjectDialog mode="create" onClose={() => {}} />);
    const dialog = screen.getByRole('dialog');
    (document.activeElement as HTMLElement | null)?.blur();
    fireEvent.keyDown(document.body, { key: 'Tab' });
    expect(dialog.contains(document.activeElement)).toBe(true);
    const close = screen.getByRole('button', { name: 'Close' });
    close.focus();
    fireEvent.keyDown(close, { key: 'Tab', shiftKey: true });
    expect(dialog.contains(document.activeElement)).toBe(true);
    expect(document.activeElement).not.toBe(close);
  });

  it('leaves Escape to a confirm opened over it', async () => {
    const onClose = vi.fn();
    await renderSettled(<ProjectDialog mode="create" onClose={onClose} />);
    const over = document.createElement('div');
    over.setAttribute('aria-modal', 'true');
    const button = document.createElement('button');
    over.appendChild(button);
    document.body.appendChild(over);
    button.focus();
    fireEvent.keyDown(button, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
    over.remove();
  });

  it('says a refusal without variables in the page language, the server text below', async () => {
    api.openProject.mockRejectedValue(new projectsApi.ProjectError(
      400, 'not_a_project', '/p/x has no codefyui.project.toml; it is not a CodefyUI project'));
    await renderSettled(<ProjectDialog mode="open" onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/Project folder/), { target: { value: '/p/x' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open' }));
    const alert = await screen.findByRole('alert');
    await waitFor(() => expect(alert.textContent).toContain('That folder is not a CodefyUI project'));
    expect(alert.textContent).toContain('/p/x has no codefyui.project.toml');
  });
});

describe('ProjectNotices', () => {
  it('makes the page behind the stale banner inert until reload', async () => {
    const root = document.createElement('div');
    root.id = 'root';
    document.body.appendChild(root);
    try {
      await renderSettled(<ProjectNotices />);
      expect(root.hasAttribute('inert')).toBe(false);
      act(() => useProjectStore.getState().markStale());
      expect(root.hasAttribute('inert')).toBe(true);
      const banner = screen.getByTestId('project-stale-banner');
      expect(root.contains(banner)).toBe(false);
    } finally {
      root.remove();
    }
  });

  it('shows the stale banner when another window switches the server', async () => {
    await renderSettled(<ProjectNotices />);
    expect(screen.queryByTestId('project-stale-banner')).toBeNull();
    act(() => {
      window.dispatchEvent(new CustomEvent(PROJECT_CHANGED_EVENT, { detail: { projectId: 'x' } }));
    });
    expect(screen.getByTestId('project-stale-banner').textContent)
      .toContain('no longer open');
    expect(screen.getByRole('button', { name: 'Reload' })).toBeTruthy();
  });

  it('asks before loading an untrusted .env, names only, blocked ones marked', async () => {
    useProjectStore.setState({
      overview: overview({
        env: {
          state: 'untrusted', names: ['HF_HOME', 'LD_PRELOAD', 'MY_TOKEN'],
          blocked: ['LD_PRELOAD'], applied: [], kept: [], restart_recommended: [],
          sha256: 'f'.repeat(64), needs_consent: true,
        },
      }),
    });
    api.trustProjectEnv.mockResolvedValue({
      ...switched('alpha'),
      env: { ...ENV_NONE, state: 'trusted', names: ['HF_HOME', 'LD_PRELOAD', 'MY_TOKEN'],
        applied: ['HF_HOME', 'MY_TOKEN'], restart_recommended: ['HF_HOME'] },
      env_restart_recommended: ['HF_HOME'],
    });
    await renderSettled(<ProjectNotices />);
    const dialog = await screen.findByTestId('project-env-consent');
    expect(within(dialog).getByText('LD_PRELOAD').parentElement!.textContent)
      .toContain('never loaded');
    expect(within(dialog).getByText(/values never leave the server/)).toBeTruthy();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Load 2 variable(s)' }));
    await waitFor(() => expect(api.trustProjectEnv).toHaveBeenCalledWith('f'.repeat(64)));
    await waitFor(() => expect(screen.queryByTestId('project-env-consent')).toBeNull());
    const messages = useToastStore.getState().toasts.map((t) => t.message);
    expect(messages).toContain('Loaded 2 variable(s) from .env.');
    expect(messages.some((m) => m.includes('after the server restarts: HF_HOME'))).toBe(true);
  });

  it('points out the names that choose a server, and Escape says not now', async () => {
    useProjectStore.setState({
      overview: overview({
        env: {
          state: 'untrusted', names: ['MY_TOKEN', 'OPENAI_BASE_URL', 'PIP_INDEX_URL'],
          blocked: ['PIP_INDEX_URL'], applied: [], kept: [], restart_recommended: [],
          endpoints: ['OPENAI_BASE_URL'], sha256: 'e'.repeat(64), needs_consent: true,
        },
      }),
    });
    await renderSettled(<ProjectNotices />);
    const dialog = await screen.findByTestId('project-env-consent');
    const marks = within(dialog).getAllByTestId('env-endpoint');
    expect(marks).toHaveLength(1);
    expect(marks[0].parentElement!.textContent).toContain('OPENAI_BASE_URL');
    expect(marks[0].textContent).toBe('chooses a server');
    expect(marks[0].querySelector('svg')).not.toBeNull(); // not colour alone
    expect(dialog.textContent).toContain('check their values before you agree');
    (document.activeElement as HTMLElement | null)?.blur();
    fireEvent.keyDown(document.body, { key: 'Escape' });
    expect(screen.queryByTestId('project-env-consent')).toBeNull();
    expect(sessionStorage.getItem('codefyui-env-declined')).toBe('e'.repeat(64));
  });

  it('does not ask again this session after "Not now"', async () => {
    const env = {
      state: 'changed' as const, names: ['A'], blocked: [], applied: [], kept: [],
      restart_recommended: [], sha256: 'e'.repeat(64), needs_consent: true,
    };
    useProjectStore.setState({ overview: overview({ env }) });
    const first = await renderSettled(<ProjectNotices />);
    expect(await screen.findByText(/changed since you last agreed/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(screen.queryByTestId('project-env-consent')).toBeNull();
    first.unmount();
    await renderSettled(<ProjectNotices />);
    expect(screen.queryByTestId('project-env-consent')).toBeNull();
    expect(api.trustProjectEnv).not.toHaveBeenCalled();
  });

  it('says after the reload what the switch left to say', async () => {
    switching.takeAfterSwitchNotice.mockReturnValueOnce({
      project: 'vision', restart: ['CUDA_VISIBLE_DEVICES'], usesLfs: true,
      imported: 2, copied: 3, skipped: 1,
    });
    await renderSettled(<ProjectNotices />);
    const messages = useToastStore.getState().toasts.map((t) => t.message);
    expect(messages).toContain(
      'Copied 2 graph(s) and 3 file(s) into the project. 1 item(s) the project already had were kept.');
    expect(messages.some((m) => m.includes('CUDA_VISIBLE_DEVICES'))).toBe(true);
    expect(messages.some((m) => m.includes('git lfs pull'))).toBe(true);
  });
});


// ── phase 4C: the mods/ trust prompt ─────────────────────────────────────

const MODS_DIGEST = 'd'.repeat(64);

function modsScan(patch: Partial<projectsApi.ModsScan> = {}): projectsApi.ModsScan {
  return {
    state: 'untrusted', loaded: false, needs_consent: true, restart_recommended: false,
    digest: MODS_DIGEST,
    files: [
      { path: 'nodes/loss.py', size: 420, role: 'node', sha256: '1'.repeat(64), lines: 18,
        finding: null, needs: [], nodes: ['FocalLoss'] },
      { path: 'nodes/fetch.py', size: 900, role: 'node', sha256: '2'.repeat(64), lines: 30,
        finding: { line: 3, kind: 'import', module: 'socket', capability: 'network',
          message: 'line 3: imports socket: needs network' },
        needs: [{ capability: 'network', summary: 'open network connections' }],
        nodes: ['Fetch'] },
      { path: 'presets/aug.json', size: 120, role: 'preset', sha256: '3'.repeat(64) },
    ],
    refused: [], duplicates: {},
    notice: 'These files are Python.',
    trust: { state: 'untrusted', digest: MODS_DIGEST, changed: null },
    ...patch,
  };
}

const MODS_NEEDS = {
  state: 'untrusted' as const, loaded: false, needs_consent: true, digest: MODS_DIGEST,
};

describe('Mods consent', () => {
  it('lists each file with its scan, warns plainly, and trusts the digest it showed', async () => {
    useProjectStore.setState({ overview: overview({ mods: MODS_NEEDS }) });
    api.getProjectMods.mockResolvedValue(modsScan());
    api.trustProjectMods.mockResolvedValue({
      ...switched('alpha'),
      mods: { state: 'trusted', loaded: true, needs_consent: false,
        nodes: ['project:FocalLoss'], skipped: [{ path: 'nodes/fetch.py', reason: 'finding' }] },
    });
    const fetchDefinitions = vi.spyOn(useNodeDefStore.getState(), 'fetchDefinitions')
      .mockResolvedValue(undefined);
    await renderSettled(<ProjectNotices />);
    const dialog = await screen.findByTestId('project-mods-consent');
    await within(dialog).findAllByTestId('mods-file');
    expect(dialog.textContent).toContain(
      'These files are Python. Trusting them runs them in this CodefyUI server with your '
      + 'permissions. The scan result is a hint, not a sandbox.');
    const rows = within(dialog).getAllByTestId('mods-file');
    expect(rows.map((r) => r.querySelector('code')!.textContent)).toEqual([
      'mods/nodes/loss.py', 'mods/nodes/fetch.py', 'mods/presets/aug.json']);
    expect(rows[0].textContent).toContain('18 lines');
    expect(rows[0].textContent).toContain('no findings');
    expect(rows[0].textContent).toContain('project:FocalLoss');
    // In mods' words, the capability in the page's language; never the
    // plugin gate's text about a plugin manifest.
    expect(rows[1].textContent).toContain('Line 3 imports socket: needs network');
    expect(rows[1].textContent).not.toContain('manifest');
    fireEvent.click(within(dialog).getByTestId('mods-trust'));
    await waitFor(() => expect(api.trustProjectMods).toHaveBeenCalledWith(MODS_DIGEST, false));
    await waitFor(() => expect(screen.queryByTestId('project-mods-consent')).toBeNull());
    expect(fetchDefinitions).toHaveBeenCalled();
    const messages = useToastStore.getState().toasts.map((t) => t.message);
    expect(messages).toContain('Loaded 1 node(s) from mods/.');
    expect(messages.some((m) => m.includes('nodes/fetch.py'))).toBe(true);
    expect(useProjectStore.getState().overview!.mods!.needs_consent).toBe(false);
    fetchDefinitions.mockRestore();
  });

  it('"trust these files completely" is asked for, and sent', async () => {
    useProjectStore.setState({ overview: overview({ mods: MODS_NEEDS }) });
    api.getProjectMods.mockResolvedValue(modsScan());
    api.trustProjectMods.mockRejectedValue(
      new projectsApi.ProjectError(409, 'mods_changed', 'changed'));
    await renderSettled(<ProjectNotices />);
    const dialog = await screen.findByTestId('project-mods-consent');
    fireEvent.click(await within(dialog).findByTestId('mods-trust-full'));
    fireEvent.click(within(dialog).getByTestId('mods-trust'));
    await waitFor(() => expect(api.trustProjectMods).toHaveBeenCalledWith(MODS_DIGEST, true));
    // A tree that changed since it was shown: said, not loaded.
    expect((await within(dialog).findByRole('alert')).textContent)
      .toContain('changed while this was open');
  });

  it('not now is kept for this digest, and a changed tree asks again', async () => {
    useProjectStore.setState({ overview: overview({ mods: MODS_NEEDS }) });
    api.getProjectMods.mockResolvedValue(modsScan());
    const first = await renderSettled(<ProjectNotices />);
    const dialog = await screen.findByTestId('project-mods-consent');
    await within(dialog).findAllByTestId('mods-file');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Not now' }));
    await waitFor(() => expect(screen.queryByTestId('project-mods-consent')).toBeNull());
    expect(api.trustProjectMods).not.toHaveBeenCalled();
    first.unmount();
    // The same tree, next mount: not asked.
    await renderSettled(<ProjectNotices />);
    expect(screen.queryByTestId('project-mods-consent')).toBeNull();
    // Another tree: asked.
    act(() => useProjectStore.setState({
      overview: overview({ mods: { ...MODS_NEEDS, digest: 'e'.repeat(64) } }) }));
    expect(await screen.findByTestId('project-mods-consent')).toBeTruthy();
  });

  it('a blocked tree lists what is refused and offers no trust', async () => {
    useProjectStore.setState({ overview: overview() });
    api.getProjectMods.mockResolvedValue(modsScan({
      state: 'blocked', needs_consent: false,
      refused: [{ path: 'nodes/__pycache__', reason: 'compiled bytecode is never loaded' }],
    }));
    await renderSettled(<ProjectNotices />);
    act(() => useProjectStore.getState().openModsReview());
    const dialog = await screen.findByTestId('project-mods-consent');
    expect((await within(dialog).findByTestId('mods-refused')).textContent)
      .toContain('compiled bytecode is never loaded');
    expect(within(dialog).queryByTestId('mods-trust')).toBeNull();
  });

  it('the chip offers Review mods and says when a restart is recommended', async () => {
    useProjectStore.setState({ overview: overview({
      mods: { state: 'trusted', loaded: true, needs_consent: false, restart_recommended: true },
    }) });
    await renderSettled(<ProjectChip />);
    // An icon beside the name, never text that squeezes it (phase 4C round 1).
    const icon = screen.getByTestId('project-chip-restart');
    expect(icon.getAttribute('aria-label')).toBe('restart recommended');
    expect(icon.textContent).toBe('');
    expect(screen.getByTestId('project-chip-name').textContent).toBe('alpha');
    fireEvent.click(screen.getByRole('button', { name: /alpha/ }));
    const rows = await screen.findAllByRole('menuitem');
    expect(rows[0].textContent).toContain('Another project’s mods ran in this server');
    expect(rows[0].getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Review mods/' }));
    expect(useProjectStore.getState().modsReviewOpen).toBe(true);
    act(() => useProjectStore.getState().closeModsReview());
  });

  it('the tab banner counts project nodes the server does not have', async () => {
    useNodeDefStore.setState({ definitions: [
      { node_name: 'Conv2d', category: 'CNN', description: '', inputs: [], outputs: [], params: [] },
    ] });
    useTabStore.setState({ tabs: [], activeTabId: null as unknown as string });
    useTabStore.getState().addTab('t');
    useTabStore.getState().setNodes([
      { id: 'a', type: 'baseNode', position: { x: 0, y: 0 },
        data: { label: 'a', type: 'project:FocalLoss', params: {} } },
      { id: 'b', type: 'baseNode', position: { x: 0, y: 0 },
        data: { label: 'b', type: 'Conv2d', params: {} } },
    ] as never);
    await renderSettled(<ModsNotLoadedBanner />);
    const banner = screen.getByTestId('mods-not-loaded-banner');
    expect(banner.textContent).toContain('uses 1 node(s) from this project’s mods');
    fireEvent.click(within(banner).getByRole('button', { name: 'Review mods' }));
    expect(useProjectStore.getState().modsReviewOpen).toBe(true);
    act(() => {
      useProjectStore.getState().closeModsReview();
      useNodeDefStore.setState({ definitions: [
        ...useNodeDefStore.getState().definitions,
        { node_name: 'project:FocalLoss', category: 'Loss', description: '', inputs: [],
          outputs: [], params: [] },
      ] });
    });
    expect(screen.queryByTestId('mods-not-loaded-banner')).toBeNull();
  });
});


describe('Mods round 2', () => {
  it('marks the files that never load, after the ones that do', async () => {
    useProjectStore.setState({ overview: overview({ mods: MODS_NEEDS }) });
    api.getProjectMods.mockResolvedValue(modsScan({
      files: [
        { path: 'nodes/loss.py', size: 420, role: 'node', sha256: '1'.repeat(64), lines: 18,
          finding: null, needs: [], nodes: ['FocalLoss'], loads: true },
        { path: 'conftest.py', size: 30, role: 'other', sha256: '4'.repeat(64), loads: false },
        { path: 'nodes/__init__.py', size: 10, role: 'other', sha256: '5'.repeat(64), loads: false },
      ],
    }));
    await renderSettled(<ProjectNotices />);
    const dialog = await screen.findByTestId('project-mods-consent');
    const rows = await within(dialog).findAllByTestId('mods-file');
    expect(within(rows[0]).queryByTestId('mods-never-loaded')).toBeNull();
    for (const row of rows.slice(1)) {
      expect(within(row).getByTestId('mods-never-loaded').textContent).toBe('not loaded');
    }
  });

  it('says when mods/ changed on disk, and checks only when asked', async () => {
    useProjectStore.setState({ overview: overview({
      mods: { state: 'trusted', loaded: true, needs_consent: false, disk_changed: true },
    }) });
    const reload = vi.spyOn(useNodeDefStore.getState(), 'reload').mockResolvedValue(undefined);
    await renderSettled(<ProjectNotices />);
    const banner = screen.getByTestId('mods-disk-changed');
    expect(banner.textContent).toContain('What runs is still the version loaded before');
    expect(reload).not.toHaveBeenCalled();
    fireEvent.click(within(banner).getByRole('button', { name: 'Reload and check' }));
    expect(reload).toHaveBeenCalledTimes(1);
    reload.mockRestore();
  });
});
