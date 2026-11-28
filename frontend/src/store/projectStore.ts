import { create } from 'zustand';
import { getProjects, type ProjectsOverview } from '../api/projects';
import { setProjectId } from '../api/_auth';

/** Basename of a project dir path (POSIX or Windows, trailing seps tolerated). */
function baseName(dir: string): string {
  const cleaned = dir.replace(/[\\/]+$/, '');
  const parts = cleaned.split(/[\\/]/);
  return parts[parts.length - 1] || cleaned;
}

/**
 * sessionStorage key for what a switch wants said after the reload it
 * triggers (the variables that need a restart, an LFS reminder).
 */
export const AFTER_SWITCH_KEY = 'codefyui-after-switch';

interface ProjectState {
  /** Absolute project dir from /api/health, or null in non-project mode. */
  projectDir: string | null;
  /** Basename of projectDir, shown in the editor header badge. */
  projectName: string | null;
  /** True once /api/health has resolved (avoids acting on the unknown state). */
  loaded: boolean;
  /** `GET /api/projects`: recents, the .env status, flat graphs. */
  overview: ProjectsOverview | null;
  /**
   * Another window switched the server to another project (or a write was
   * refused with `project_changed`): this window shows a project that is
   * not open, and offers a reload.
   */
  stale: boolean;
  setProject: (dir: string | null) => void;
  /** Read `GET /api/projects` and send its id with every write from now. */
  loadOverview: () => Promise<ProjectsOverview | null>;
  setOverview: (overview: ProjectsOverview) => void;
  markStale: () => void;
  /** The mods trust prompt is open (from the notice, a banner or a badge). */
  modsReviewOpen: boolean;
  openModsReview: () => void;
  closeModsReview: () => void;
}

export const useProjectStore = create<ProjectState>((set) => ({
  projectDir: null,
  projectName: null,
  loaded: false,
  overview: null,
  stale: false,
  setProject: (dir) =>
    set({ projectDir: dir, projectName: dir ? baseName(dir) : null, loaded: true }),
  loadOverview: async () => {
    try {
      const overview = await getProjects();
      setProjectId(overview.project_id);
      set({
        overview,
        projectName: overview.current?.name
          ?? useProjectStore.getState().projectName,
      });
      return overview;
    } catch {
      // The server is unreachable or older than projects: the editor keeps
      // working in whatever mode /api/health reported.
      return null;
    }
  },
  setOverview: (overview) => set({ overview }),
  markStale: () => set({ stale: true }),
  modsReviewOpen: false,
  openModsReview: () => set({ modsReviewOpen: true }),
  closeModsReview: () => set({ modsReviewOpen: false }),
}));

/** The registry prefix of a project's own mods nodes (phase 4C). */
export const PROJECT_NODE_PREFIX = 'project:';

export function isProjectNodeType(type: string | undefined | null): boolean {
  return typeof type === 'string' && type.startsWith(PROJECT_NODE_PREFIX);
}

export function isProjectMode(): boolean {
  return useProjectStore.getState().projectDir !== null;
}
