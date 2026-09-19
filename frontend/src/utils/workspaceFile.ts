// Type-only imports: erased at build time, so this module pulls in no store
// and no DOM, and the format stays testable on its own.
import type { Locale } from '../i18n';
import type { TabRunSettings } from '../store/tabStore';
import type { EdgeStyle, FontSize, UIPreferences } from '../store/uiStore';

/**
 * The `.cduiworkspace` format: every open tab in one file.
 *
 * PURE on purpose. `exportWorkspace.ts` reads the stores and calls
 * `buildWorkspaceFile`; `importWorkspaceFile.ts` applies what
 * `parseWorkspaceFile` returns. Nothing here knows either exists.
 *
 * Naming: the plugin API's `WorkspaceSnapshot` means ONE tab, so no type in
 * this file is called a snapshot.
 */

/** How an importer recognises the file: by this value, never by the extension. */
export const WORKSPACE_FORMAT = 'codefyui-workspace';
/** The envelope version this build writes, and the newest it reads. */
export const WORKSPACE_VERSION = 1;
export const WORKSPACE_EXTENSION = '.cduiworkspace';

/**
 * One tab's run settings, under the names `TabState` uses for them.
 *
 * Declared here rather than imported from `TabRunSettings`: the FILE's shape
 * must not change because the app's did. The `satisfies` tables below are what
 * make a change to the app a compile error instead of a silent drop.
 */
export interface WorkspaceRunSettings {
  /**
   * `null` is "no seed". A number is NOT bounded here, on the way in or out:
   * the backend refuses a seed outside 0..4294967295 when the run is
   * submitted, with a message the canvas shows, and that is the one place the
   * range is stated.
   */
  seed: number | null;
  deterministic: boolean;
  recordOutputs: boolean;
  verboseMode: boolean;
  weightsPersistent: boolean;
  backwardMode: boolean;
  autoBackward: boolean;
}

/**
 * `tabs[i].graph`: exactly what Export JSON writes, plus `format_version`.
 * One of these blocks is a valid Export-JSON graph and imports on its own.
 * The lists are `unknown[]` because this module never looks inside them.
 */
export interface WorkspaceGraph {
  name: string;
  description: string;
  nodes: unknown[];
  edges: unknown[];
  presets: unknown[];
  segmentGroups: unknown[];
  subgraphs: unknown[];
  /**
   * Present only when the graph assigns a device. Spelled out here rather
   * than imported from the app's `GraphSettings`, ON PURPOSE: a field added
   * to that type must not change what this format claims to hold.
   */
  settings?: { device?: string };
  /** `GRAPH_FORMAT_VERSION` at export time. */
  format_version: number;
}

export interface WorkspaceTabEntry {
  /** The tab label. No length bound here: the tab strip clips a long one. */
  title: string;
  graph: WorkspaceGraph;
  run: WorkspaceRunSettings;
}

/**
 * Every key optional: an importer applies what is there and valid. Declared
 * independently of `UIPreferences`, for the reason `WorkspaceRunSettings` is.
 */
export interface WorkspacePreferences {
  locale?: Locale;
  fontSize?: FontSize;
  edgeStyle?: EdgeStyle;
  gridSnap?: boolean;
  tooltips?: boolean;
  beginnerMode?: boolean;
}

export interface WorkspaceFile {
  format: typeof WORKSPACE_FORMAT;
  version: number;
  /** The server version the exporter last heard from /api/health; null when unknown. */
  app_version: string | null;
  exported_at: string;
  /** Index into `tabs` of the tab that was active; null if it was not exported. */
  active: number | null;
  tabs: WorkspaceTabEntry[];
  preferences?: WorkspacePreferences;
}

/** What `buildWorkspaceFile` is handed. The caller supplies the clock. */
export interface WorkspaceBuildInput {
  appVersion: string | null;
  /** An ISO timestamp. */
  exportedAt: string;
  /** Index into `tabs`, already re-indexed after skipping; null when the active tab is not in it. */
  active: number | null;
  tabs: WorkspaceTabEntry[];
  preferences?: WorkspacePreferences;
}

/** One entry as read back. `title` and `graph` are untrusted file contents. */
export interface ParsedWorkspaceTab {
  /** The importer falls back to a default tab name for a blank or non-string one. */
  title: unknown;
  /** The importer checks it is a graph before anything reads it. */
  graph: unknown;
  /** Only the values that passed their type check; a missing key keeps the tab's default. */
  run: Partial<WorkspaceRunSettings>;
}

export interface ParsedWorkspace {
  version: number;
  appVersion: string | null;
  exportedAt: string | null;
  /**
   * A valid index into `tabs`, or null.
   *
   * It indexes the FILE's tabs, entries the importer will refuse included --
   * `tabs` below is positional and keeps them. An importer that drops entries
   * must therefore map this through its own positions, never count what it
   * opened.
   */
  active: number | null;
  /** Never empty. Positional: `tabs[i]` is the file's `tabs[i]`. */
  tabs: ParsedWorkspaceTab[];
  /** Only the keys that held an allowed value. */
  preferences: WorkspacePreferences;
}

export type WorkspaceParseFailure =
  /** `format` is not `codefyui-workspace`: some other JSON. */
  | 'not_workspace'
  /** `version` is not an integer >= 1, or `tabs` is not a non-empty array. */
  | 'invalid'
  /** `version` is newer than this build reads. Nothing may be created from it. */
  | 'too_new';

export type WorkspaceParseResult =
  | { ok: true; workspace: ParsedWorkspace }
  | { ok: false; reason: WorkspaceParseFailure };

/** Assemble the file. Keys are written in the order the format documents them. */
export function buildWorkspaceFile(input: WorkspaceBuildInput): WorkspaceFile {
  const { active, tabs, preferences } = input;
  const activeInRange =
    active !== null && Number.isInteger(active) && active >= 0 && active < tabs.length;
  return {
    format: WORKSPACE_FORMAT,
    version: WORKSPACE_VERSION,
    app_version: input.appVersion,
    exported_at: input.exportedAt,
    active: activeInRange ? active : null,
    tabs,
    ...(preferences && Object.keys(preferences).length > 0 ? { preferences } : {}),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Does this parsed JSON claim to be a workspace file? Content, not extension. */
export function isWorkspaceFile(data: unknown): boolean {
  return isRecord(data) && data.format === WORKSPACE_FORMAT;
}

// Exhaustive against the app's own run settings: an eighth one added to
// `TabRunSettings` is a compile error here, not a field this parser drops
// without a word. The type import is TYPE-ONLY, so the module stays pure.
const RUN_FLAGS = {
  deterministic: true,
  recordOutputs: true,
  verboseMode: true,
  weightsPersistent: true,
  backwardMode: true,
  autoBackward: true,
} satisfies Record<Exclude<keyof TabRunSettings, 'seed'>, true>;

function readRunSettings(raw: unknown): Partial<WorkspaceRunSettings> {
  if (!isRecord(raw)) return {};
  const run: Partial<WorkspaceRunSettings> = {};
  const seed = raw.seed;
  if (seed === null || (typeof seed === 'number' && Number.isFinite(seed))) run.seed = seed;
  for (const flag of Object.keys(RUN_FLAGS) as (keyof typeof RUN_FLAGS)[]) {
    const value = raw[flag];
    if (typeof value === 'boolean') run[flag] = value;
  }
  return run;
}

// `satisfies` makes each table exhaustive: adding a locale, a font size or an
// edge style without listing it here is a compile error, not a silent skip.
const LOCALES = { en: true, 'zh-TW': true } satisfies Record<Locale, true>;
const FONT_SIZES = { small: true, default: true, large: true } satisfies Record<FontSize, true>;
const EDGE_STYLES = { circuit: true, curve: true } satisfies Record<EdgeStyle, true>;
// And exhaustive against the app's own preferences, the way `RUN_FLAGS` is.
const PREFERENCE_FLAGS = {
  gridSnap: true,
  tooltips: true,
  beginnerMode: true,
} satisfies Record<Exclude<keyof UIPreferences, 'fontSize' | 'edgeStyle'>, true>;

/** `value` when it is one of the table's OWN keys (so `'toString'` is not one). */
function oneOf<T extends string>(allowed: Record<T, true>, value: unknown): T | undefined {
  return typeof value === 'string' && Object.keys(allowed).includes(value)
    ? (value as T)
    : undefined;
}

function readPreferences(raw: unknown): WorkspacePreferences {
  if (!isRecord(raw)) return {};
  const preferences: WorkspacePreferences = {};
  const locale = oneOf(LOCALES, raw.locale);
  if (locale !== undefined) preferences.locale = locale;
  const fontSize = oneOf(FONT_SIZES, raw.fontSize);
  if (fontSize !== undefined) preferences.fontSize = fontSize;
  const edgeStyle = oneOf(EDGE_STYLES, raw.edgeStyle);
  if (edgeStyle !== undefined) preferences.edgeStyle = edgeStyle;
  for (const flag of Object.keys(PREFERENCE_FLAGS) as (keyof typeof PREFERENCE_FLAGS)[]) {
    const value = raw[flag];
    if (typeof value === 'boolean') preferences[flag] = value;
  }
  return preferences;
}

/**
 * Read already-parsed JSON as a workspace file.
 *
 * Validates the ENVELOPE only. Each tab entry is kept positional and
 * untrusted, so the importer can refuse one entry without sinking the rest.
 */
export function parseWorkspaceFile(data: unknown): WorkspaceParseResult {
  if (!isRecord(data) || data.format !== WORKSPACE_FORMAT) {
    return { ok: false, reason: 'not_workspace' };
  }
  const version = data.version;
  if (typeof version !== 'number' || !Number.isInteger(version) || version < 1) {
    return { ok: false, reason: 'invalid' };
  }
  // Before `tabs`: a newer envelope may shape its tabs differently, and we
  // cannot know what we would drop.
  if (version > WORKSPACE_VERSION) return { ok: false, reason: 'too_new' };

  const rawTabs = data.tabs;
  if (!Array.isArray(rawTabs) || rawTabs.length === 0) return { ok: false, reason: 'invalid' };
  const tabs: ParsedWorkspaceTab[] = rawTabs.map((raw: unknown) =>
    isRecord(raw)
      ? { title: raw.title, graph: raw.graph, run: readRunSettings(raw.run) }
      : { title: undefined, graph: undefined, run: {} },
  );

  const active = data.active;
  return {
    ok: true,
    workspace: {
      version,
      appVersion: typeof data.app_version === 'string' ? data.app_version : null,
      exportedAt: typeof data.exported_at === 'string' ? data.exported_at : null,
      active:
        typeof active === 'number' && Number.isInteger(active) && active >= 0 && active < tabs.length
          ? active
          : null,
      tabs,
      preferences: readPreferences(data.preferences),
    },
  };
}

/** `workspace-YYYY-MM-DD.cduiworkspace`, from the LOCAL date. */
export function workspaceFileName(now: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  return `workspace-${date}${WORKSPACE_EXTENSION}`;
}
