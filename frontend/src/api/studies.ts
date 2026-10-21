import { apiFetch } from './_auth';

/**
 * Durable studies (#343): `/api/studies`, for the Studies panel.
 *
 * A study is server-owned, so any browser can list them and follow one: the
 * panel reads the same rows whichever tab, plugin or client submitted them.
 */

const BASE_URL = '/api/studies';

export type StudyStatus =
  | 'queued'
  | 'running'
  | 'cancel_requested'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted'
  | 'halted';

export type AttemptStatus =
  | 'pending'
  | 'submitting'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted'
  | 'lost'
  | 'timed_out';

export interface StudyBudget {
  max_runs: number | null;
  max_parallel_runs: number;
  max_run_seconds?: number | null;
  max_active_seconds?: number | null;
}

/** Which budget halted the study, its limit and what was used. */
export interface StudyHalt {
  budget: 'max_runs' | 'max_active_seconds' | string;
  limit: number;
  used: number;
  unfinished_repetitions: number;
  at: string;
}

export interface StudyBudgetUsage {
  runs: { used: number; limit: number | null };
  parallel: { active: number; limit: number };
  active_seconds: { used: number; limit: number | null };
  run_seconds: { longest_active: number; limit: number | null };
  unfinished_repetitions: number;
  exhausted: 'max_runs' | 'max_active_seconds' | null;
}

export interface StudyObjective {
  node_id: string;
  metric: string;
  direction: 'minimize' | 'maximize';
}

export interface StudySummary {
  study_id: string;
  name: string | null;
  status: StudyStatus | string;
  revision: number;
  objective: StudyObjective;
  repetitions: number;
  seed: number;
  budget: StudyBudget;
  resume_policy: 'explicit' | 'auto';
  idempotency_key: string | null;
  error: string | null;
  halt: StudyHalt | null;
  /** The plugin that submitted it through its jobs routes; null = the core API. */
  owner: string | null;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
  /** Current attempts by status. */
  counts: Partial<Record<AttemptStatus | string, number>>;
}

export interface StudyAttempt {
  attempt_id: string;
  variant_index: number;
  repetition: number;
  attempt: number;
  seed: number;
  run_id: string | null;
  status: AttemptStatus | string;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  objective_value: number | null;
  stop_reason?: string | null;
}

export interface StudyVariant {
  index: number;
  label: string;
  repetitions: Array<{ repetition: number; seed: number | null; current: StudyAttempt | null }>;
  attempts: StudyAttempt[];
  aggregate: { count: number; mean: number | null; std: number | null; values: Array<number | null> };
  eligible: boolean;
  ineligible_reasons: string[];
  rank: number | null;
  tie: boolean;
  tied_with: number[];
}

export interface StudyRanking {
  rank: number;
  variant_index: number;
  label: string;
  mean: number;
  std: number | null;
  count: number;
  tie: boolean;
  tied_with: number[];
}

export interface StudyDetail extends StudySummary {
  budget_usage: StudyBudgetUsage;
  attempts_submitted: number;
  variants: StudyVariant[];
  rankings: StudyRanking[];
  last_cursor: number;
}

export interface StudyEvent {
  cursor: number;
  type: string;
  payload: Record<string, unknown> | null;
  ts: string;
}

export interface StudyList {
  studies: StudySummary[];
  total: number;
}

/** An API failure: the server's `detail` (or *fallback*), and the HTTP status. */
export class StudiesApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'StudiesApiError';
  }
}

async function failure(res: Response, fallback: string): Promise<StudiesApiError> {
  const body = await res.json().catch(() => ({}));
  return new StudiesApiError(
    typeof body?.detail === 'string' ? body.detail : `${fallback}: ${res.statusText}`,
    res.status,
  );
}

/** Newest first. */
export async function listStudies(signal?: AbortSignal): Promise<StudyList> {
  const res = await fetch(`${BASE_URL}?limit=100`, signal ? { signal } : undefined);
  if (!res.ok) throw await failure(res, 'Studies failed');
  const body = await res.json();
  return {
    studies: Array.isArray(body.studies) ? body.studies : [],
    total: typeof body.total === 'number' ? body.total : 0,
  };
}

export async function getStudy(studyId: string, signal?: AbortSignal): Promise<StudyDetail> {
  const res = await fetch(`${BASE_URL}/${encodeURIComponent(studyId)}`, signal ? { signal } : undefined);
  if (!res.ok) throw await failure(res, 'Study failed');
  return res.json();
}

/** The audit log strictly after *cursor*, oldest first. */
export async function getStudyEvents(
  studyId: string,
  cursor: number,
  signal?: AbortSignal,
): Promise<{ events: StudyEvent[]; cursor: number }> {
  const res = await fetch(
    `${BASE_URL}/${encodeURIComponent(studyId)}/events?cursor=${cursor}`,
    signal ? { signal } : undefined,
  );
  if (!res.ok) throw await failure(res, 'Study events failed');
  const body = await res.json();
  return {
    events: Array.isArray(body.events) ? body.events : [],
    cursor: typeof body.cursor === 'number' ? body.cursor : cursor,
  };
}

async function post(path: string, body: Record<string, unknown>, fallback: string): Promise<StudyDetail> {
  const res = await apiFetch(`${BASE_URL}/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await failure(res, fallback);
  return res.json();
}

/** Stop scheduling and cancel the children in flight; `expectedRevision` makes a race a 409. */
export function cancelStudy(studyId: string, expectedRevision?: number): Promise<StudyDetail> {
  return post(
    `${encodeURIComponent(studyId)}/cancel`,
    expectedRevision === undefined ? {} : { expected_revision: expectedRevision },
    'Cancel failed',
  );
}

/** Continue the study, with `budget` replacing its budget when given. */
export function resumeStudy(
  studyId: string,
  options: { budget?: StudyBudget; expectedRevision?: number } = {},
): Promise<StudyDetail> {
  const body: Record<string, unknown> = {};
  if (options.budget) body.budget = options.budget;
  if (options.expectedRevision !== undefined) body.expected_revision = options.expectedRevision;
  return post(`${encodeURIComponent(studyId)}/resume`, body, 'Resume failed');
}
