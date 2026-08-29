import { useCallback, useMemo } from 'react';
import type { PackSummary } from '../../api/rest';
import type { PackJob, PackJobPhase } from '../../store/packStore';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { localizedPackTitle } from '../../utils/packAvailability';
import { ProgressBar } from '../shared/ProgressBar';
import { CommandBlock } from './CommandBlock';
import { PackLogTail } from './PackLogTail';
import {
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
  /** The server's own answer to "can I install this and come back?". */
  restartAvailable: boolean;
  /** False when the server refuses installs from this browser (remote). */
  canInstall: boolean;
  /** This job's pack already has an install request in flight. */
  busy: boolean;
  onCancel: () => void;
  onDismiss: () => void;
  /**
   * Install *packId* again in restart mode. Only ever called from the
   * `needs_restart` banner, and only when the retry was offered.
   *
   * The id is an ARGUMENT rather than something the handler looks up, because
   * a confirm dialog is open across several catalog polls and `refresh` can
   * adopt another tab's job while it is: the pack that gets installed has to
   * be the one whose banner the user was reading.
   */
  onRestartInstall: (packId: string) => void;
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
  restartAvailable,
  canInstall,
  busy,
  onCancel,
  onDismiss,
  onRestartInstall,
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

  // How a job that has stopped is reported, or null while it is running.
  const result = job === null ? null : resultSentence(t, job, title);

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
      result ?? [
        stepText ?? t('packs.activity.job', { pack: title }),
        percent === null ? null : `${Math.round(percent)}%`,
      ]
        .filter(Boolean)
        .join(' '),
    // Intentionally narrow: `stepText`, `title` and `percent` are read at the
    // moment one of these changes, which IS the throttle. `result` is a
    // string, so it compares by value and the ending announces itself even
    // when the step and the bucket both stayed put.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [job?.jobId, step?.step, bucket, locale, result],
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

      {/* Mounted for the job's whole life, not just while it runs: a live
          region that appears WITH its text already in it is not reliably
          announced, and unmounting this one at the end is what made
          `cancelled` and `lost` silent. One region, from "installing" to the
          sentence that ends it. */}
      <div className={styles.srOnly} aria-live="polite" aria-atomic="true">
        {announcement}
      </div>

      {running ? (
        <>
          {stepText !== null && <div className={styles.stepLine}>{stepText}</div>}
          <div className={styles.overallLabel}>{t('packs.activity.overall')}</div>
          <ProgressBar
            tone="info"
            showValue
            label={t('packs.activity.progressAria')}
            value={percent}
          />
        </>
      ) : (
        <ResultBanner
          job={job}
          title={title}
          restartAvailable={restartAvailable}
          canInstall={canInstall}
          busy={busy}
          onRestartInstall={onRestartInstall}
        />
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
function ResultBanner({
  job,
  title,
  restartAvailable,
  canInstall,
  busy,
  onRestartInstall,
}: {
  job: PackJob;
  title: string;
  restartAvailable: boolean;
  canInstall: boolean;
  busy: boolean;
  onRestartInstall: (packId: string) => void;
}) {
  const { t } = useI18n();
  const tone = BANNER_TONE[job.status] ?? 'neutral';

  // The same sentence the live region announces, by construction rather than
  // by two copies of the same five `t()` calls.
  const sentence = resultSentence(t, job, title);

  // BOTH flags, and they answer different questions. `retryMode` is this
  // job's own: the server said a restart-mode install is what would get past
  // whatever stopped this one (a resolver conflict against the constraints
  // file). `restartAvailable` is the server's: it can actually perform one.
  // Either alone would offer a button that ends in a 409.
  const canRetry = job.status === 'needs_restart'
    && job.retryMode === 'restart'
    && restartAvailable;

  // The body is about what the restart DOES, not just that it happens: the
  // helper installs packages only — it runs from an interpreter with none of
  // this app's downloader in it — so a pack with models comes back with them
  // still missing, and an ordinary install afterwards is what fetches them.
  // Being told that before the server goes away is the difference between a
  // second install and a bug report.
  const retry = useCallback(async () => {
    const ok = await confirm({
      title: t('packs.activity.restartAndInstall'),
      message: t('packs.activity.restartAndInstallNote'),
      confirmText: t('packs.activity.restartAndInstall'),
      variant: 'danger',
    });
    if (!ok) return;
    // This banner's own pack, captured before the dialog opened — not
    // whatever job the store holds now. A `refresh` between the click and the
    // answer can adopt an install started in another tab, and reinstalling
    // THAT one is not what the user just agreed to.
    onRestartInstall(job.packId);
  }, [job.packId, onRestartInstall, t]);

  return (
    <div className={styles.resultBanner} role="status" data-tone={tone}>
      {sentence !== null && <span>{sentence}</span>}

      {/* The server's hint is the actionable half — "free 4 GB", "install
          build tools" — and it is written by the step that failed. */}
      {job.status === 'failed' && job.error?.hint && (
        <span className={styles.resultHint}>{job.error.hint}</span>
      )}

      {/* Kept even when the button is there: the command is the same install
          in a terminal, and a user who was just told the server will restart
          is entitled to do it themselves instead. */}
      {job.status === 'needs_restart' && job.restartCommand !== null && (
        <CommandBlock command={job.restartCommand} />
      )}

      {canRetry && (
        <div className={styles.cardActions}>
          {/* Disabled on the same two counts as the GPU card's button: a
              remote browser, whose install the server refuses from anywhere
              but its own machine, and a request already in flight for this
              pack — a live button in either case would spend a confirm and a
              restart warning on a refusal. */}
          <button
            type="button"
            className={styles.primaryBtn}
            disabled={!canInstall || busy}
            title={canInstall ? undefined : t('packs.remoteDisabled')}
            onClick={() => void retry()}
          >
            {t('packs.activity.restartAndInstall')}
          </button>
        </div>
      )}
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

/**
 * The pack's name, by the one rule every surface uses.
 *
 * `pack` is this job's own catalog row, so a one-entry index answers exactly
 * what the whole catalog would — and the pane has no reason to hold the rest
 * of it.
 */
function packName(t: Translate, packId: string, pack: PackSummary | undefined): string {
  return localizedPackTitle(t, pack ? { [packId]: pack } : {}, packId);
}

/**
 * The one sentence a finished job is reported with.
 *
 * Shared by the banner and the live region on purpose: the banner is what a
 * sighted reader sees, and a `role="status"` element that MOUNTS with its text
 * already in it is not reliably announced — `cancelled` and `lost` were
 * silent. The announcer stays mounted for the whole job and this sentence
 * lands in it as a change, which is the thing screen readers do read.
 */
function resultSentence(t: Translate, job: PackJob, title: string): string | null {
  switch (job.status) {
    case 'done':
      return t('packs.activity.done', { pack: title });
    case 'failed':
      return t('packs.activity.failed', { message: job.error?.message ?? '' });
    case 'cancelled':
      return t('packs.activity.cancelled');
    case 'needs_restart':
      // One status, two stories, and "Installed." is only true for one of
      // them. A RESTART-mode job really did install: the helper is running
      // and the server is on its way out. A LIVE job with this status stopped
      // dead on a resolver conflict having changed nothing — telling that
      // user their pack is installed is how they come back to report it
      // missing. `retryMode` is the server's own word for "a restart is what
      // would finish this", and a job whose mode is not `restart` is a live
      // one whatever else it says.
      return job.retryMode === 'restart' || job.mode !== 'restart'
        ? t('packs.activity.needsRestartConflict')
        : t('packs.activity.needsRestart', { pack: title });
    case 'lost':
      return t('packs.activity.lost');
    default:
      return null;
  }
}
