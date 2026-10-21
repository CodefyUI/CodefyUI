import { create } from 'zustand';
import {
  StudiesApiError,
  cancelStudy,
  getStudy,
  getStudyEvents,
  listStudies,
  resumeStudy,
  type StudyBudget,
  type StudyDetail,
  type StudyEvent,
  type StudySummary,
} from '../api/studies';

/** How often the open panel reads the list, or the open study, again. */
export const STUDY_POLL_MS = 2000;
/** The newest events kept for the open study's log. */
export const STUDY_EVENTS_KEPT = 200;

/**
 * The Studies panel (#343): the list of durable studies and the one open.
 *
 * Everything here is read from the server, so closing the tab that submitted
 * a study changes nothing: any browser that opens the panel reattaches to
 * the same studies. `panelOpen` lives here, not in the panel, because the
 * Runs panel opens it and `isAnyModalOpen` reads it.
 */
interface StudyState {
  panelOpen: boolean;
  studies: StudySummary[];
  total: number;
  loaded: boolean;
  listError: string | null;
  selectedId: string | null;
  detail: StudyDetail | null;
  detailError: string | null;
  /** The server answered 404 for the open study. */
  notFound: boolean;
  /** The open study's audit log, oldest first, the newest STUDY_EVENTS_KEPT. */
  events: StudyEvent[];
  cursor: number;
  /** A cancel or resume in flight. */
  pending: 'cancel' | 'resume' | null;
  /** Why the last cancel or resume was refused. */
  actionError: string | null;

  openPanel: (studyId?: string) => void;
  closePanel: () => void;
  openStudy: (studyId: string) => Promise<void>;
  closeStudy: () => void;
  /** Read the open study, or the list when none is open. */
  refresh: () => Promise<void>;
  cancel: () => Promise<void>;
  resume: (budget?: StudyBudget) => Promise<void>;
  /** Poll while the caller is mounted; returns the release. */
  watch: () => () => void;
}

const initialState = {
  panelOpen: false,
  studies: [] as StudySummary[],
  total: 0,
  loaded: false,
  listError: null,
  selectedId: null,
  detail: null,
  detailError: null,
  notFound: false,
  events: [] as StudyEvent[],
  cursor: 0,
  pending: null,
  actionError: null,
};

const unselected = {
  detail: null,
  detailError: null,
  notFound: false,
  events: [] as StudyEvent[],
  cursor: 0,
  pending: null,
  actionError: null,
};

/** Bumped when the open study changes, so a late reply cannot land on another. */
let generation = 0;
let watchers = 0;
let timer: ReturnType<typeof setInterval> | null = null;

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export const useStudyStore = create<StudyState>()((set, get) => {
  const refreshList = async () => {
    const expected = generation;
    try {
      const list = await listStudies();
      if (generation !== expected || get().selectedId !== null) return;
      set({ studies: list.studies, total: list.total, loaded: true, listError: null });
    } catch (error) {
      if (generation !== expected) return;
      set({ loaded: true, listError: messageOf(error) });
    }
  };

  const refreshStudy = async (studyId: string) => {
    const expected = generation;
    const current = () => generation === expected && get().selectedId === studyId;
    try {
      const detail = await getStudy(studyId);
      if (!current()) return;
      set({ detail, detailError: null, notFound: false });
      const page = await getStudyEvents(studyId, get().cursor);
      if (!current()) return;
      // Two reads in flight can fetch the same page: keep each cursor once.
      set((state) => {
        const fresh = page.events.filter((event) => event.cursor > state.cursor);
        if (fresh.length === 0) return state;
        return {
          events: [...state.events, ...fresh].slice(-STUDY_EVENTS_KEPT),
          cursor: fresh[fresh.length - 1].cursor,
        };
      });
    } catch (error) {
      if (!current()) return;
      set({
        detailError: messageOf(error),
        notFound: error instanceof StudiesApiError && error.status === 404,
      });
    }
  };

  const act = async (kind: 'cancel' | 'resume', run: (study: StudyDetail) => Promise<StudyDetail>) => {
    const { detail, pending } = get();
    if (!detail || pending !== null) return;
    const expected = generation;
    set({ pending: kind, actionError: null });
    try {
      const updated = await run(detail);
      if (generation !== expected) return;
      set({ detail: updated });
      await refreshStudy(detail.study_id);
    } catch (error) {
      if (generation === expected) set({ actionError: messageOf(error) });
    } finally {
      if (generation === expected) set({ pending: null });
    }
  };

  return {
    ...initialState,

    openPanel: (studyId) => {
      set({ panelOpen: true });
      if (studyId) void get().openStudy(studyId);
    },

    closePanel: () => {
      generation += 1;
      set({ panelOpen: false, selectedId: null, ...unselected });
    },

    openStudy: async (studyId) => {
      generation += 1;
      set({ selectedId: studyId, ...unselected });
      await refreshStudy(studyId);
    },

    closeStudy: () => {
      generation += 1;
      set({ selectedId: null, ...unselected });
      void refreshList();
    },

    refresh: async () => {
      const studyId = get().selectedId;
      if (studyId !== null) await refreshStudy(studyId);
      else await refreshList();
    },

    // No expected_revision from the panel: it acts on the latest state it
    // polled, and the server refuses a cancel or resume that the study's
    // status does not allow.
    cancel: () => act('cancel', (study) => cancelStudy(study.study_id)),

    resume: (budget) => act('resume', (study) => resumeStudy(study.study_id, budget ? { budget } : {})),

    watch: () => {
      watchers += 1;
      if (watchers === 1) {
        void get().refresh();
        timer = setInterval(() => void get().refresh(), STUDY_POLL_MS);
      }
      let released = false;
      return () => {
        if (released) return;
        released = true;
        watchers -= 1;
        if (watchers === 0 && timer !== null) {
          clearInterval(timer);
          timer = null;
        }
      };
    },
  };
});

/** Test isolation for the module-level poller. */
export function _resetStudyStoreForTesting(): void {
  generation += 1;
  watchers = 0;
  if (timer !== null) clearInterval(timer);
  timer = null;
  useStudyStore.setState({ ...initialState });
}
