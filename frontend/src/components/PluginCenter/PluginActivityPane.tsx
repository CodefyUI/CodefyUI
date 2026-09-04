import { useMemo } from 'react';
import type { PluginCatalogEntry } from '../../api/rest';
import type { JobPhase } from '../../store/jobFollower';
import type { PluginJob } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import { ProgressBar } from '../shared/ProgressBar';
// The pack panel's own pieces rather than copies of them: an install
// transcript and a command with a copy button are the same objects in both
// windows, and two of each would drift the day one is fixed.
import { CommandBlock } from '../PackCenter/CommandBlock';
import { PackLogTail } from '../PackCenter/PackLogTail';
import {
  cliInstallCommand,
  currentStep,
  jobOverallPercent,
  stepLabel,
  type Translate,
} from './pluginStatus';
import styles from '../PackCenter/PackCenterModal.module.css';

type BannerTone = 'success' | 'warning' | 'error' | 'neutral';

const BANNER_TONE: Record<JobPhase, BannerTone> = {
  running: 'neutral',
  done: 'success',
  failed: 'error',
  cancelled: 'neutral',
  needs_restart: 'warning',
  lost: 'warning',
};

export interface PluginActivityPaneProps {
  job: PluginJob | null;
  /**
   * The catalog row for `job.pluginId`: its name, the source the terminal
   * fallback would install from, and how long an install of it should take
   * (a built-in pack has no download and four steps; a repository has eight).
   *
   * Undefined when the catalog has no row for this job -- a plugin installed
   * from a URL this registry does not list, or a job adopted before the first
   * catalog read answers. The pane then names the job by its plugin id,
   * offers no command, and counts it as the longer install, rather than
   * guessing at any of the three.
   */
  entry: PluginCatalogEntry | undefined;
  /** A cancel request is in flight: the button says so and takes no second press. */
  cancelling: boolean;
  onCancel: () => void;
  onDismiss: () => void;
  /** Re-read the catalog, for the one banner that cannot say what happened. */
  onRefresh: () => void;
}

/**
 * The right-hand column of the Plugin Center: what is installing, how far it
 * has got, and what it left behind when it stopped.
 *
 * A pure VIEW of `pluginStore.job`, exactly like the Package Center's pane and
 * for the same reason: an install has to survive this window being closed, a
 * second tab and a page reload, so the job's whole lifecycle lives in the
 * store and this component only renders it.
 *
 * Its markup and roles ARE the pack pane's -- the same title, the same live
 * region, the same bar, the same log and the same two buttons -- so the two
 * panes read identically to a screen reader and neither has to be learned
 * twice. What differs is only what a plugin job is: install or update, one
 * tarball rather than several models, and no restart-mode retry (this store
 * has no restart mode; a plugin never swaps a wheel out from under the
 * running interpreter).
 */
export function PluginActivityPane({
  job,
  entry,
  cancelling,
  onCancel,
  onDismiss,
  onRefresh,
}: PluginActivityPaneProps) {
  const { t, locale } = useI18n();

  // The row as well as the job: how many steps an install takes depends on
  // where it comes from, and a built-in pack has no download at all.
  const percent = jobOverallPercent(job, entry);
  const current = job === null ? null : currentStep(job.steps);
  // The catalog row's name, falling back to the id -- the store's own rule
  // (`pluginStore.ts: pluginName`), so a toast and this pane call a plugin
  // the same thing.
  const title = job === null ? '' : (entry?.name || job.pluginId);
  const headline = job === null
    ? ''
    : t(job.kind === 'update'
      ? 'pluginCenter.activity.updating'
      : 'pluginCenter.activity.installing', { plugin: title });
  const stepText = current === null
    ? null
    : t('packs.activity.step', {
      index: current.index,
      label: stepLabel(t, current.step.step, current.step.label),
    });

  // How a job that has stopped is reported, or null while it is running.
  const result = job === null ? null : resultSentence(t, job, title);

  // Announced on a STEP change and every ten percent, never on every byte: a
  // polite live region re-read on each progress frame would talk over itself
  // for the whole install.
  //
  // Falls back to the job's own headline for the second or two between "the
  // install was accepted" and the first step event, so a screen reader hears
  // that something started rather than a bare "0%".
  const bucket = percent === null ? -1 : Math.floor(percent / 10);
  const announcement = useMemo(
    () =>
      result ?? [
        stepText ?? headline,
        percent === null ? null : `${Math.round(percent)}%`,
      ]
        .filter(Boolean)
        .join(' '),
    // Intentionally narrow: `stepText`, `headline` and `percent` are read at
    // the moment one of these changes, which IS the throttle. `result` is a
    // string, so it compares by value and the ending announces itself even
    // when the step and the bucket both stayed put.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [job?.jobId, current?.step.step, bucket, locale, result],
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
      {/* "Installing X" only while it IS installing: a stopped job is titled
          by its plugin, and the banner underneath is what happened to it. */}
      <div className={styles.activityTitle}>{running ? headline : title}</div>

      {/* Mounted for the job's whole life, not just while it runs: a live
          region that appears WITH its text already in it is not reliably
          announced, and that is what makes `cancelled` and `lost` silent. One
          region, from "installing" to the sentence that ends it. */}
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
        <ResultBanner job={job} title={title} entry={entry} onRefresh={onRefresh} />
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
  entry,
  onRefresh,
}: {
  job: PluginJob;
  title: string;
  entry: PluginCatalogEntry | undefined;
  onRefresh: () => void;
}) {
  const { t } = useI18n();
  const tone = BANNER_TONE[job.status] ?? 'neutral';

  // The same sentence the live region announces, by construction rather than
  // by two copies of the same five `t()` calls.
  const sentence = resultSentence(t, job, title);

  // The same install in a terminal, for a job the panel could not finish.
  //
  // NOT offered for a LINKED FOLDER: `cdui plugin install` takes a name or a
  // repository, and a linked row's source is a path on this machine, so the
  // command would be one the CLI refuses. A folder is put back with `cdui
  // plugin link`, which is a different sentence and outside this panel.
  //
  // Nor for a job whose row the catalog does not have: what to install is
  // exactly what is missing, and `cdui plugin install <id>` on a plugin that
  // is not in the registry is a command that fails.
  const fallback = job.status === 'failed'
    && entry !== undefined
    && entry.source_kind !== 'local'
    ? cliInstallCommand(entry)
    : null;

  return (
    <div className={styles.resultBanner} role="status" data-tone={tone}>
      {sentence !== null && <span>{sentence}</span>}

      {/* The server's hint is the actionable half -- "check the repository
          name", "the tarball is larger than this build will download" -- and
          it is written by the step that failed. */}
      {job.status === 'failed' && job.error?.hint && (
        <span className={styles.resultHint}>{job.error.hint}</span>
      )}

      {/* The command the server cannot run for itself, which the sentence
          above ends in a colon for. A plugin job reaches this status from one
          place only -- `deps.install_deps_step` refusing to replace a package
          the running interpreter has already loaded -- so what it carries is
          the `uv pip install` line to run with the server stopped. */}
      {job.status === 'needs_restart' && job.restartCommand !== null && (
        <CommandBlock command={job.restartCommand} />
      )}

      {fallback !== null && (
        <>
          <span className={styles.resultHint}>{t('pluginCenter.activity.cliFallback')}</span>
          <CommandBlock command={fallback} />
        </>
      )}

      {/* The one ending nobody can report: the follower gave up, so what the
          job did is a question only the catalog can answer. */}
      {job.status === 'lost' && (
        <div className={styles.cardActions}>
          {/* "Refresh", not `pluginCenter.refresh`: the sentence above the
              button already ends in "Refresh to check the plugin status", and
              that key is also the header icon's name -- two controls with one
              name, one of them repeating the line it sits under. */}
          <button type="button" className={styles.primaryBtn} onClick={onRefresh}>
            {t('sidebar.refresh')}
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * The one sentence a finished job is reported with.
 *
 * Shared by the banner and the live region on purpose: the banner is what a
 * sighted reader sees, and a `role="status"` element that MOUNTS with its text
 * already in it is not reliably announced. The announcer stays mounted for the
 * whole job and this sentence lands in it as a change, which is the thing
 * screen readers do read.
 *
 * A failure is worded by KIND, out of the two keys that already exist for it
 * -- the same pair the store's toast uses -- rather than printing the server's
 * message bare: `error.message` is often a fragment ("HTTP 404"), and a red
 * box containing only that does not say what failed.
 */
function resultSentence(t: Translate, job: PluginJob, title: string): string | null {
  const update = job.kind === 'update';
  switch (job.status) {
    case 'done':
      return t(
        update ? 'pluginCenter.activity.updated' : 'pluginCenter.activity.installed',
        { plugin: title },
      );
    case 'failed':
      // The PANE's key, not the toast's twin of it: the two are the same
      // string in both locales today, and keeping this one on the pane's
      // namespace is what stops a later edit to the toast rewording a banner.
      // There is no pane-side key for an update, so that arm keeps its own.
      return t(
        update ? 'pluginCenter.updateFailed' : 'packs.activity.failed',
        { message: job.error?.message ?? '' },
      );
    case 'cancelled':
      return t('packs.activity.cancelled');
    case 'needs_restart':
      return t('pluginCenter.activity.needsRestart', { plugin: title });
    case 'lost':
      return t('pluginCenter.activity.lost');
    default:
      return null;
  }
}
