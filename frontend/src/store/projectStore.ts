import { create } from 'zustand';

/** Basename of a project dir path (POSIX or Windows, trailing seps tolerated). */
function baseName(dir: string): string {
  const cleaned = dir.replace(/[\\/]+$/, '');
  const parts = cleaned.split(/[\\/]/);
  return parts[parts.length - 1] || cleaned;
}

interface ProjectState {
  /** Absolute project dir from /api/health, or null in non-project mode. */
  projectDir: string | null;
  /** Basename of projectDir, shown in the editor header badge. */
  projectName: string | null;
  /** True once /api/health has resolved (avoids acting on the unknown state). */
  loaded: boolean;
  setProject: (dir: string | null) => void;
}

export const useProjectStore = create<ProjectState>((set) => ({
  projectDir: null,
  projectName: null,
  loaded: false,
  setProject: (dir) =>
    set({ projectDir: dir, projectName: dir ? baseName(dir) : null, loaded: true }),
}));

export function isProjectMode(): boolean {
  return useProjectStore.getState().projectDir !== null;
}
