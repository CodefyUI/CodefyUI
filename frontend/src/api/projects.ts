/**
 * `/api/projects`: which project is open, and switching it (phase 4A).
 *
 * Every switch answers with the new project; the caller then reloads the
 * page, so every store, socket and cache starts again on the new project.
 * The server refuses a switch with a 409 whose message names what is busy
 * (a run, a study, an install, a git operation, a worker's mount): that
 * message is shown as it is.
 */

import { apiFetch } from './_auth';

const BASE = '/api/projects';

export interface ProjectRef {
  path: string;
  name: string;
  project_id: string;
}

export interface RecentProject {
  path: string;
  name: string;
  last_opened: string;
  /** The folder is gone or no longer has a manifest. */
  missing: boolean;
}

/** The open project's `.env`: variable NAMES only, never values. */
export interface ProjectEnv {
  state: 'none' | 'trusted' | 'untrusted' | 'changed';
  names: string[];
  /** Names that are never loaded at run time (PATH, PYTHON*, LD_*, ...). */
  blocked: string[];
  applied: string[];
  /** Names the server process already had; the process value was kept. */
  kept: string[];
  /** Names that only take effect after a server restart. */
  restart_recommended: string[];
  /**
   * Loadable names that choose a server (`*_URL`, `*_BASE`, `*_ENDPOINT`,
   * `*_HOST`, `*_API_BASE`): the consent prompt points them out.
   */
  endpoints?: string[];
  /** sha256 of the file that was read; consent is for exactly this file. */
  sha256: string | null;
  needs_consent: boolean;
}

/** What happened to the open project's `mods/` (phase 4C). */
export interface ModsStatus {
  /** none: no Python under mods/; blocked: something there is never loaded. */
  state: 'none' | 'blocked' | 'trusted' | 'untrusted' | 'changed';
  loaded: boolean;
  /** The `project:<NODE_NAME>` types that are registered. */
  nodes?: string[];
  presets?: string[];
  skipped?: { path: string; reason: string }[];
  /** Another project's mods ran in this server: unloading is not isolation. */
  restart_recommended?: boolean;
  needs_consent: boolean;
  /** With needs_consent: the tree's digest, what "not now" is kept against. */
  digest?: string | null;
}

/** One file under `mods/`, as the trust prompt lists it. */
export interface ModsFile {
  path: string;
  size: number;
  role: 'node' | 'preset' | 'other';
  sha256: string;
  lines?: number;
  finding?: { line: number | null; message: string } | null;
  needs?: { capability: string; summary: string }[];
  nodes?: string[];
  duplicates?: string[];
}

/** `GET /api/projects/mods`: the scan, never an import. */
export interface ModsScan extends ModsStatus {
  digest: string | null;
  files: ModsFile[];
  refused: { path: string; reason: string }[];
  duplicates: Record<string, string[]>;
  notice: string;
  trust: {
    state: ModsStatus['state'];
    digest: string | null;
    changed: { added: string[]; removed: string[]; changed: string[] } | null;
  };
}

export interface ProjectsOverview {
  current: ProjectRef | null;
  project_id: string;
  projects_root: string;
  recent: RecentProject[];
  env: ProjectEnv;
  /** Absent from a server older than phase 4C. */
  mods?: ModsStatus;
  startup: { reason: string; created: boolean; warnings: string[] } | null;
  /** Graphs in the pre-project folder, offered for import in flat mode. */
  flat_graphs: number;
  browse_allowed: boolean;
}

export interface SwitchResult {
  project: ProjectRef | null;
  project_id: string;
  env: ProjectEnv;
  mods?: ModsStatus;
  env_restart_recommended: string[];
  unchanged?: boolean;
  created?: string | null;
  cloned?: string;
  uses_lfs?: boolean;
  imported?: string[];
  skipped?: { file: string; reason: string }[];
  copied?: string[];
  kept?: string[];
}

export interface BrowseEntry {
  name: string;
  path: string;
  is_project: boolean;
}

export interface BrowseResult {
  dir: string;
  parent: string | null;
  is_project: boolean;
  entries: BrowseEntry[];
  truncated: boolean;
}

export interface ImportAsset {
  path: string;
  kind: 'models' | 'images' | 'media';
  size: number;
  files: number;
  is_dir: boolean;
}

export interface ImportPlan {
  source: string;
  graphs: { name: string; file: string; size: number; readable: boolean }[];
  graphs_bytes: number;
  assets: ImportAsset[];
  assets_bytes: number;
}

/** A refusal from the server: its stable `code` and its sentence. */
export class ProjectError extends Error {
  readonly code: string;
  readonly status: number;
  readonly hint: string | null;

  constructor(status: number, code: string, message: string, hint: string | null = null) {
    super(message);
    this.name = 'ProjectError';
    this.status = status;
    this.code = code;
    this.hint = hint;
  }
}

async function answer<T>(res: Response): Promise<T> {
  if (res.ok) return res.json() as Promise<T>;
  const body = await res.json().catch(() => null);
  const detail = body?.detail;
  if (detail && typeof detail === 'object') {
    const hint = typeof detail.details === 'string' ? detail.details : null;
    throw new ProjectError(res.status, String(detail.code ?? 'error'),
      String(detail.message ?? res.statusText), hint);
  }
  throw new ProjectError(res.status, 'error',
    typeof detail === 'string' ? detail : `${res.status} ${res.statusText}`);
}

function post<T>(path: string, body: unknown, method = 'POST'): Promise<T> {
  return apiFetch(`${BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((res) => answer<T>(res));
}

export function getProjects(): Promise<ProjectsOverview> {
  return apiFetch(BASE).then((res) => answer<ProjectsOverview>(res));
}

export function openProject(path: string): Promise<SwitchResult> {
  return post('/open', { path });
}

export function createProject(name: string, parent?: string): Promise<SwitchResult> {
  return post('/create', parent ? { name, parent } : { name });
}

export function cloneProject(url: string, parent?: string): Promise<SwitchResult> {
  return post('/clone', parent ? { url, parent } : { url });
}

export function trustProjectEnv(sha256: string): Promise<SwitchResult> {
  return post('/trust', { sha256 });
}

export function getProjectMods(): Promise<ModsScan> {
  return apiFetch(`${BASE}/mods`).then((res) => answer<ModsScan>(res));
}

/**
 * Trust the `mods/` tree whose digest the user was shown. A tree that
 * changed since is refused (409 `mods_changed`). *full* also runs the files
 * the scan had findings for.
 */
export function trustProjectMods(sha256: string, full: boolean): Promise<SwitchResult> {
  return post('/trust-mods', { sha256, full });
}

export function removeRecentProject(path: string): Promise<{ removed: string }> {
  return post('/recent', { path }, 'DELETE');
}

export function browseFolders(dir?: string, hidden = false): Promise<BrowseResult> {
  const params = new URLSearchParams();
  if (dir) params.set('dir', dir);
  if (hidden) params.set('hidden', 'true');
  const query = params.toString();
  return apiFetch(`${BASE}/browse${query ? `?${query}` : ''}`)
    .then((res) => answer<BrowseResult>(res));
}

export function getImportPlan(): Promise<ImportPlan> {
  return apiFetch(`${BASE}/import`).then((res) => answer<ImportPlan>(res));
}

export function importFlatGraphs(
  target: { name: string } | { path: string },
  includeAssets: boolean,
): Promise<SwitchResult> {
  return post('/import', { ...target, include_assets: includeAssets });
}

/** `1.2 MB`-style size for the import dialog. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}
