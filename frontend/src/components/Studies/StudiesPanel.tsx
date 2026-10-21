import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import type {
  StudyAttempt,
  StudyBudget,
  StudyDetail,
  StudyHalt,
  StudySummary,
  StudyVariant,
} from '../../api/studies';
import { useI18n, type TranslationKey } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { useStudyStore } from '../../store/studyStore';
import { useWorkerStore } from '../../store/workerStore';
import { friendlyError } from '../../utils/errorMessages';
import { trapTab } from '../shared/focusTrap';
import styles from './StudiesPanel.module.css';

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

/** Study statuses a cancel applies to, and the ones a resume continues. */
const CANCELLABLE = new Set(['queued', 'running', 'interrupted', 'failed', 'halted']);
const RESUMABLE = new Set(['interrupted', 'cancelled', 'failed', 'halted']);
/** Attempt statuses whose repetition is done. */
const FINAL = new Set(['succeeded', 'failed', 'timed_out']);

/**
 * The Studies panel (#343): every durable study on the server, and one study
 * with its progress, budget, rankings and audit log. Reading only server
 * state, it reattaches to any study from any browser.
 */
export function StudiesPanel() {
  const open = useStudyStore((s) => s.panelOpen);
  if (!open) return null;
  return <StudiesDialog />;
}

function StudiesDialog() {
  const { t } = useI18n();
  const titleId = useId();
  const selectedId = useStudyStore((s) => s.selectedId);
  const closePanel = useStudyStore((s) => s.closePanel);
  const closeStudy = useStudyStore((s) => s.closeStudy);
  const refresh = useStudyStore((s) => s.refresh);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => useStudyStore.getState().watch(), []);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      // Escape steps back one level: from a study to the list, then out.
      if (selectedId !== null) closeStudy();
      else closePanel();
      return;
    }
    trapTab(event, dialogRef.current);
  };

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) closePanel();
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={handleKeys}
      >
        <header className={styles.header}>
          <h2 id={titleId}>{t('studies.title')}</h2>
          <button type="button" className={styles.btn} onClick={() => void refresh()}>
            {t('studies.refresh')}
          </button>
          <button
            ref={closeRef}
            type="button"
            className={styles.close}
            onClick={closePanel}
            aria-label={t('studies.close')}
          >
            &#215;
          </button>
        </header>
        <div className={styles.body}>
          {selectedId === null ? <StudyList /> : <StudyView />}
        </div>
      </div>
    </div>,
    document.body,
  );
}

function statusLabel(t: Translate, status: string, prefix: 'studies.status' | 'studies.attempt'): string {
  const key = `${prefix}.${status}` as TranslationKey;
  const text = t(key);
  // A status from a newer server falls through as its own token.
  return text === key ? status : text;
}

function Chip({ status, prefix }: { status: string; prefix: 'studies.status' | 'studies.attempt' }) {
  const { t } = useI18n();
  return (
    <span className={[styles.chip, styles[`chip_${status}`]].filter(Boolean).join(' ')}>
      {statusLabel(t, status, prefix)}
    </span>
  );
}

function progress(counts: StudySummary['counts']): { done: number; total: number } {
  let done = 0;
  let total = 0;
  for (const [status, count] of Object.entries(counts) as Array<[string, number]>) {
    total += count;
    if (FINAL.has(status)) done += count;
  }
  return { done, total };
}

function formatStamp(stamp: string | null): string {
  if (!stamp) return '-';
  const date = new Date(stamp);
  return Number.isNaN(date.getTime()) ? stamp : date.toLocaleString();
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return '-';
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
}

function StudyList() {
  const { t } = useI18n();
  const studies = useStudyStore((s) => s.studies);
  const loaded = useStudyStore((s) => s.loaded);
  const listError = useStudyStore((s) => s.listError);
  const openStudy = useStudyStore((s) => s.openStudy);

  return (
    <>
      {listError !== null && <div className={styles.error} role="alert">{friendlyError(listError)}</div>}
      {!loaded ? (
        <p className={styles.muted}>{t('studies.loading')}</p>
      ) : studies.length === 0 ? (
        <p className={styles.muted}>{t('studies.empty')}</p>
      ) : (
        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>{t('studies.col.name')}</th>
                <th>{t('studies.col.status')}</th>
                <th>{t('studies.col.owner')}</th>
                <th>{t('studies.col.progress')}</th>
                <th>{t('studies.col.created')}</th>
              </tr>
            </thead>
            <tbody>
              {studies.map((study) => {
                const name = study.name || t('studies.unnamed');
                const { done, total } = progress(study.counts);
                return (
                  <tr key={study.study_id} data-testid={`study-row-${study.study_id}`}>
                    <td>
                      <button
                        type="button"
                        className={styles.linkBtn}
                        title={t('studies.action.openTitle', { name })}
                        onClick={() => void openStudy(study.study_id)}
                      >
                        {name}
                      </button>
                    </td>
                    <td><Chip status={study.status} prefix="studies.status" /></td>
                    <td>{study.owner ?? t('studies.owner.core')}</td>
                    <td className={styles.num}>{t('studies.progress', { done, total })}</td>
                    <td className={styles.num}>{formatStamp(study.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function haltText(t: Translate, halt: StudyHalt): string {
  const values = {
    used: formatNumber(halt.used),
    limit: formatNumber(halt.limit),
    unfinished: halt.unfinished_repetitions,
    budget: halt.budget,
  };
  if (halt.budget === 'max_runs') return t('studies.halt.max_runs', values);
  if (halt.budget === 'max_active_seconds') return t('studies.halt.max_active_seconds', values);
  return t('studies.halt.other', values);
}

function StudyView() {
  const { t } = useI18n();
  const detail = useStudyStore((s) => s.detail);
  const detailError = useStudyStore((s) => s.detailError);
  const notFound = useStudyStore((s) => s.notFound);
  const closeStudy = useStudyStore((s) => s.closeStudy);
  const backRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    backRef.current?.focus();
  }, []);

  const back = (
    <button ref={backRef} type="button" className={styles.btn} onClick={closeStudy}>
      {t('studies.back')}
    </button>
  );
  if (!detail) {
    return (
      <>
        <div className={styles.detailHeader}>{back}</div>
        {notFound ? (
          <p className={styles.muted}>{t('studies.detail.notFound')}</p>
        ) : detailError !== null ? (
          <div className={styles.error} role="alert">{friendlyError(detailError)}</div>
        ) : (
          <p className={styles.muted}>{t('studies.detail.loading')}</p>
        )}
      </>
    );
  }
  return <StudyDetailBody detail={detail} back={back} />;
}

function StudyDetailBody({ detail, back }: { detail: StudyDetail; back: ReactNode }) {
  const { t } = useI18n();
  const detailError = useStudyStore((s) => s.detailError);
  const pending = useStudyStore((s) => s.pending);
  const actionError = useStudyStore((s) => s.actionError);
  const cancel = useStudyStore((s) => s.cancel);
  const resume = useStudyStore((s) => s.resume);
  const name = detail.name || t('studies.unnamed');
  const status = detail.status;

  return (
    <>
      <div className={styles.detailHeader}>
        {back}
        <h3>{name}</h3>
        <Chip status={status} prefix="studies.status" />
      </div>
      <div className={styles.meta}>
        <span>{t('studies.detail.objective', {
          metric: detail.objective.metric,
          node: detail.objective.node_id,
          direction: t(`studies.direction.${detail.objective.direction}`),
        })}</span>
        <span>{t('studies.detail.meta', {
          variants: detail.variants.length,
          repetitions: detail.repetitions,
          seed: detail.seed,
        })}</span>
        {detail.owner && <span>{t('studies.detail.owner', { owner: detail.owner })}</span>}
      </div>

      {detail.halt && <div className={styles.notice} role="status">{haltText(t, detail.halt)}</div>}
      {status === 'interrupted' && <div className={styles.notice} role="status">{t('studies.detail.interrupted')}</div>}
      {detail.error && <div className={styles.error}>{friendlyError(detail.error)}</div>}
      {detailError !== null && <div className={styles.error} role="alert">{friendlyError(detailError)}</div>}
      {actionError !== null && (
        <div className={styles.error} role="alert">{t('studies.action.failed', { error: friendlyError(actionError) })}</div>
      )}

      <div className={styles.actions}>
        {CANCELLABLE.has(status) && (
          <button type="button" className={`${styles.btn} ${styles.danger}`} disabled={pending !== null} onClick={() => void cancel()}>
            {t('studies.action.cancel')}
          </button>
        )}
        {RESUMABLE.has(status) && status !== 'halted' && (
          <button type="button" className={`${styles.btn} ${styles.accent}`} disabled={pending !== null} onClick={() => void resume()}>
            {t('studies.action.resume')}
          </button>
        )}
      </div>
      {status === 'halted' && (
        <BudgetEditor key={detail.revision} budget={detail.budget} halt={detail.halt} disabled={pending !== null} onResume={(budget) => void resume(budget)} />
      )}

      <BudgetFacts detail={detail} />
      <Rankings detail={detail} />
      <Repetitions detail={detail} />
      <EventLog />
    </>
  );
}

function BudgetFacts({ detail }: { detail: StudyDetail }) {
  const { t } = useI18n();
  const usage = detail.budget_usage;
  const ofLimit = (used: string, limit: number | null, unit: (value: number) => string = formatNumber) => (
    limit === null || limit === undefined
      ? `${used} (${t('studies.budget.unlimited')})`
      : t('studies.budget.ofLimit', { used, limit: unit(limit) }));
  const exhausted = (key: 'max_runs' | 'max_active_seconds') => usage.exhausted === key && (
    <span className={styles.exhausted}> {t('studies.budget.exhausted')}</span>
  );
  const seconds = (value: number) => t('studies.budget.seconds', { value: formatNumber(value) });
  return (
    <section className={styles.section} aria-label={t('studies.budget.title')}>
      <h4>{t('studies.budget.title')}</h4>
      <dl className={styles.facts}>
        <div>
          <dt>{t('studies.budget.runs')}</dt>
          <dd className={styles.num}>{ofLimit(String(usage.runs.used), usage.runs.limit)}{exhausted('max_runs')}</dd>
        </div>
        <div>
          <dt>{t('studies.budget.parallel')}</dt>
          <dd className={styles.num}>{ofLimit(String(usage.parallel.active), usage.parallel.limit)}</dd>
        </div>
        <div>
          <dt>{t('studies.budget.activeSeconds')}</dt>
          <dd className={styles.num}>{ofLimit(seconds(usage.active_seconds.used), usage.active_seconds.limit, seconds)}{exhausted('max_active_seconds')}</dd>
        </div>
        <div>
          <dt>{t('studies.budget.runSeconds')}</dt>
          <dd className={styles.num}>
            {usage.run_seconds.limit === null || usage.run_seconds.limit === undefined
              ? t('studies.budget.unlimited')
              : seconds(usage.run_seconds.limit)}
            {usage.run_seconds.longest_active > 0 && (
              <span className={styles.muted}> ({t('studies.budget.longest', { value: formatNumber(usage.run_seconds.longest_active) })})</span>
            )}
          </dd>
        </div>
      </dl>
    </section>
  );
}

/** Parse an optional limit field: '' is no limit, otherwise a positive number. */
function parseLimit(raw: string, integer: boolean): number | null | undefined {
  const text = raw.trim();
  if (text === '') return null;
  const value = Number(text);
  if (!Number.isFinite(value) || value <= 0 || (integer && !Number.isInteger(value))) return undefined;
  return value;
}

function BudgetEditor({ budget, halt, disabled, onResume }: {
  budget: StudyBudget;
  halt: StudyHalt | null;
  disabled: boolean;
  onResume: (budget: StudyBudget) => void;
}) {
  const { t } = useI18n();
  const [maxRuns, setMaxRuns] = useState(budget.max_runs === null ? '' : String(budget.max_runs));
  const [maxActive, setMaxActive] = useState(
    budget.max_active_seconds === null || budget.max_active_seconds === undefined ? '' : String(budget.max_active_seconds),
  );
  const runs = parseLimit(maxRuns, true);
  const active = parseLimit(maxActive, false);
  const invalid = runs === undefined || active === undefined;
  const runsId = useId();
  const activeId = useId();
  const submit = () => {
    if (invalid) return;
    onResume({ ...budget, max_runs: runs, max_active_seconds: active });
  };
  return (
    <form
      className={styles.budgetForm}
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      data-halted-by={halt?.budget}
    >
      <label htmlFor={runsId}>
        {t('studies.budget.maxRuns')}
        <input id={runsId} inputMode="numeric" value={maxRuns} onChange={(event) => setMaxRuns(event.target.value)} />
      </label>
      <label htmlFor={activeId}>
        {t('studies.budget.maxActiveSeconds')}
        <input id={activeId} inputMode="decimal" value={maxActive} onChange={(event) => setMaxActive(event.target.value)} />
      </label>
      <button type="submit" className={`${styles.btn} ${styles.accent}`} disabled={disabled || invalid}>
        {t('studies.action.resumeBudget')}
      </button>
      {invalid && <span className={styles.exhausted} role="alert">{t('studies.budget.invalid')}</span>}
    </form>
  );
}

function Rankings({ detail }: { detail: StudyDetail }) {
  const { t } = useI18n();
  const labels = new Map(detail.variants.map((variant) => [variant.index, variant.label]));
  return (
    <section className={styles.section} aria-label={t('studies.rankings.title')}>
      <h4>{t('studies.rankings.title')}</h4>
      {detail.rankings.length === 0 ? (
        <p className={styles.muted}>{t('studies.rankings.empty')}</p>
      ) : (
        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>{t('studies.col.rank')}</th>
                <th>{t('studies.col.variant')}</th>
                <th>{t('studies.col.mean')}</th>
                <th>{t('studies.col.std')}</th>
                <th>{t('studies.col.count')}</th>
                <th>{t('studies.col.tie')}</th>
              </tr>
            </thead>
            <tbody>
              {detail.rankings.map((row) => (
                <tr key={row.variant_index} data-testid={`study-rank-${row.variant_index}`}>
                  <td className={styles.num}>{row.rank}</td>
                  <td>{row.label}</td>
                  <td className={styles.num}>{formatNumber(row.mean)}</td>
                  <td className={styles.num}>{formatNumber(row.std)}</td>
                  <td className={styles.num}>{row.count}</td>
                  <td>{row.tie
                    ? t('studies.tie', { labels: row.tied_with.map((index) => labels.get(index) ?? String(index)).join(', ') })
                    : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function openRun(runId: string): void {
  // The run opens in the Runs panel, which the dock reveals.
  useStudyStore.getState().closePanel();
  void useRunStore.getState().select(runId);
  useWorkerStore.getState().revealRuns();
}

function AttemptCell({ attempt, label, repetition }: { attempt: StudyAttempt | null; label: string; repetition: number }) {
  const { t } = useI18n();
  if (attempt === null) return <td>-</td>;
  return (
    <td title={t('studies.attempt.title', { attempt: attempt.attempt, seed: attempt.seed, run: attempt.run_id ?? '-' })}>
      <div className={styles.cell}>
        <Chip status={attempt.status} prefix="studies.attempt" />
        <span className={styles.num}>{formatNumber(attempt.objective_value)}</span>
        {attempt.attempt > 1 && <span className={styles.muted}>{t('studies.attempt.retries', { attempt: attempt.attempt })}</span>}
        {attempt.run_id && (
          <button
            type="button"
            className={styles.linkBtn}
            aria-label={t('studies.attempt.openRun', { label, n: repetition + 1 })}
            onClick={() => openRun(attempt.run_id!)}
          >
            {attempt.run_id.slice(0, 8)}
          </button>
        )}
      </div>
    </td>
  );
}

/**
 * Why a variant is not ranked, worded here from its repetitions (the
 * server's `ineligible_reasons` are English): the study rule ranks a variant
 * only when every repetition succeeded and recorded the objective.
 */
function ineligibleReasons(t: Translate, variant: StudyVariant): string[] {
  const reasons: string[] = [];
  for (const { repetition, current } of variant.repetitions) {
    const n = repetition + 1;
    if (current === null) reasons.push(t('studies.reason.status', { n, status: '-' }));
    else if (current.status !== 'succeeded') {
      reasons.push(t('studies.reason.status', { n, status: statusLabel(t, current.status, 'studies.attempt') }));
    } else if (current.objective_value === null) reasons.push(t('studies.reason.noValue', { n }));
  }
  return reasons;
}

function Repetitions({ detail }: { detail: StudyDetail }) {
  const { t } = useI18n();
  const repetitions = Array.from({ length: detail.repetitions }, (_, index) => index);
  return (
    <section className={styles.section} aria-label={t('studies.variants.title')}>
      <h4>{t('studies.variants.title')}</h4>
      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>{t('studies.col.variant')}</th>
              {repetitions.map((index) => <th key={index}>{t('studies.col.repetition', { n: index + 1 })}</th>)}
              <th>{t('studies.col.mean')}</th>
              <th>{t('studies.col.std')}</th>
              <th>{t('studies.col.count')}</th>
              <th>{t('studies.col.eligible')}</th>
            </tr>
          </thead>
          <tbody>
            {detail.variants.map((variant) => (
              <tr key={variant.index} data-testid={`study-variant-${variant.index}`}>
                <td>{variant.label}</td>
                {repetitions.map((index) => (
                  <AttemptCell
                    key={index}
                    attempt={variant.repetitions[index]?.current ?? null}
                    label={variant.label}
                    repetition={index}
                  />
                ))}
                <td className={styles.num}>{formatNumber(variant.aggregate.mean)}</td>
                <td className={styles.num}>{formatNumber(variant.aggregate.std)}</td>
                <td className={styles.num}>{variant.aggregate.count}</td>
                <td>{variant.eligible
                  ? t('studies.variant.eligible')
                  : t('studies.variant.ineligible', { reasons: ineligibleReasons(t, variant).join('; ') })}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EventLog() {
  const { t } = useI18n();
  const events = useStudyStore((s) => s.events);
  return (
    <section className={styles.section} aria-label={t('studies.events.title')}>
      <h4>{t('studies.events.title')}</h4>
      {events.length === 0 ? (
        <p className={styles.muted}>{t('studies.events.empty')}</p>
      ) : (
        <ol className={styles.events}>
          {[...events].reverse().map((event) => (
            <li key={event.cursor}>
              <span>{event.cursor}</span>
              <span>{formatStamp(event.ts)}</span>
              <span>{event.type}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
