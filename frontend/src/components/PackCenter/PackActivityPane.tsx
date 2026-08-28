import { useMemo } from 'react';
import type { PackSummary } from '../../api/rest';
import type { PackJob, PackJobPhase } from '../../store/packStore';
import { useI18n } from '../../i18n';
import { ProgressBar } from '../shared/ProgressBar';
import { CommandBlock } from './CommandBlock';
import { PackLogTail } from './PackLogTail';
import {
  catalogKey,
  currentStep,
  jobOverallPercent,
  stepLabel,
  type Translate,
} from './packStatus';
import styles from './PackCenterModal.module.css';

export interface PackActivityPaneProps {
  job: PackJob | null;
  /** The catalog row for `job.packId`, for the byte weights and the name. */
  pack: PackSummary | undefined;
  onCancel: () => void;
  onDismiss: () => void;
  cancelling: boolean;
}

type BannerTone = 'success' | 'warning' | 'error' | 'neutral';

/**
 * The right-hand column: what is installing, how far it has got, and what it
 * left behind when it stopped.
 *
 * It is a pure VIEW of `packStore.job`. It starts no follower and stops none —
 * a multi-gigabyte download has to survive the modal being closed, the tab
 * being switched and the page being reloaded, so the job's whole lifecycle
 * lives in the store and this component only renders it.
 */
export function PackActivityPane({
  job,
  pack,
  onCancel,
  onDismiss,
  cancelling,
}: PackActivityPaneProps) {
  const { t, locale } = useI18n();

  const percent = jobOverallPercent(job, pack);
  const step = job === null ? null : currentStep(job);
  const title = job === null ? '' : packName(t, job.packId, pack);
  const stepText =
    step === null
      ? null
      : t('packs.activity.step', {
          index: step.index,
          label: stepLabel(t, step.step, step.label),
        });

  // Announced on a STEP change and every ten percent, never on every byte: a
  // polite live region re-read on each of the thousands of progress frames a
  // 2 GB download emits would talk over itself for the whole install.
  //
  // Falls back to the job's own headline for the second or two between "the
  // install was accepted" and the first step event, so a screen reader hears
  // that something started rather than a bare "0%".
  const bucket = percent === null ? -1 : Math.floor(percent / 10);
  const announcement = useMemo(
    () =>
      [
        stepText ?? t('packs.activity.job', { pack: title }),
        percent === null ? null : `${Math.round(percent)}%`,
      ]
        .filter(Boolean)
        .join(' '),
    // Intentionally narrow: `stepText`, `title` and `percent` are read at the
    // moment one of these changes, which IS the throttle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [job?.jobId, step?.step, bucket, locale],
  );

  if (job === null) {
    return (
      <>
        <div className={styles.activityTitle}>{t('packs.activity.idle')}</div>
        <div className={styles.activityHint}>{t('packs.activity.idleHint')}</div>
      </>
    );
  }

  const running = job.status === 'running';

  return (
    <>
      <div className={styles.activityTitle}>
        {t('packs.activity.job', { pack: title })}
      </div>

      {running ? (
        <>
          {stepText !== null && <div className={styles.stepLine}>{stepText}</div>}
          <div className={styles.srOnly} aria-live="polite" aria-atomic="true">
            {announcement}
          </div>
          <div className={styles.overallLabel}>{t('packs.activity.overall')}</div>
          <ProgressBar
            tone="info"
            showValue
            label={t('packs.activity.progressAria')}
            value={percent}
          />
        </>
      ) : (
        <ResultBanner job={job} title={title} />
      )}

      <PackLogTail lines={job.log} ariaLabel={t('packs.activity.log')} />

      <div className={styles.cardActions}>
        {running ? (
          <button
            type="button"
            className={styles.secondaryBtn}
            disabled={cancelling}
            onClick={onCancel}
          >
            {cancelling ? t('packs.cancelling') : t('packs.cancel')}
          </button>
        ) : (
          <button type="button" className={styles.secondaryBtn} onClick={onDismiss}>
            {t('packs.activity.dismiss')}
          </button>
        )}
      </div>
    </>
  );
}

/** How a finished job is reported: one toned banner, and what to do next. */
function ResultBanner({ job, title }: { job: PackJob; title: string }) {
  const { t } = useI18n();
  const tone = BANNER_TONE[job.status] ?? 'neutral';

  return (
    <div className={styles.resultBanner} role="status" data-tone={tone}>
      {job.status === 'done' && <span>{t('packs.activity.done', { pack: title })}</span>}

      {job.status === 'failed' && (
        <>
          <span>
            {t('packs.activity.failed', { message: job.error?.message ?? '' })}
          </span>
          {/* The server's hint is the actionable half — "free 4 GB", "install
              build tools" — and it is written by the step that failed. */}
          {job.error?.hint && (
            <span className={styles.resultHint}>{job.error.hint}</span>
          )}
        </>
      )}

      {job.status === 'cancelled' && <span>{t('packs.activity.cancelled')}</span>}

      {job.status === 'needs_restart' && (
        <>
          <span>{t('packs.activity.needsRestart', { pack: title })}</span>
          {job.restartCommand !== null && <CommandBlock command={job.restartCommand} />}
        </>
      )}

      {job.status === 'lost' && <span>{t('packs.activity.lost')}</span>}
    </div>
  );
}

const BANNER_TONE: Record<PackJobPhase, BannerTone> = {
  running: 'neutral',
  done: 'success',
  failed: 'error',
  cancelled: 'neutral',
  needs_restart: 'warning',
  lost: 'warning',
};

/** The pack's name, preferring this build's copy, then the server's, then id. */
function packName(t: Translate, packId: string, pack: PackSummary | undefined): string {
  const key = catalogKey(packId, 'title');
  if (key !== null) return t(key);
  return pack?.title ?? packId;
}
