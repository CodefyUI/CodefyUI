import { useMemo } from 'react';
import type { PluginCatalogEntry } from '../../api/rest';
import type { JobPhase } from '../../store/jobFollower';
import type { PluginJob, PluginRemoval } from '../../store/pluginStore';
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
   * The last uninstall, when there is one and no job. An uninstall is a
   * single request rather than a job to follow, so it has no steps and no
   * transcript -- but it is the last thing that happened in this panel, and
   * the pane is where the panel says so.
   */
  removal: PluginRemoval | null;
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
  removal,
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

  // Has the job stopped? Then the banner below is what reports it.
  const ended = job !== null && job.status !== 'running';

  // Announced on a STEP change and every ten percent, never on every byte: a
  // polite live region re-read on each progress frame would talk over itself
  // for the whole install.
  //
  // Falls back to the job's own headline for the second or two between "the
  // install was accepted" and the first step event, so a screen reader hears
  // that something started rather than a bare "0%".
  //
  // And it goes QUIET at the end. The result banner is a `role="status"`
  // region in its own right and carries the outcome, so the same sentence in
  // here was one fact ANNOUNCED twice -- once by the banner and once by this
  // region. Never SHOWN twice: this region is `srOnly`, a one-pixel clipped
  // box, so the duplicate only ever existed for a screen reader. The region
  // stays mounted and empty rather than unmounting, because the next job's
  // first step has to land in a region that was already there.
  //
  // What that costs, recorded here rather than only in a review: the ending
  // is now announced solely by an element that arrives WITH its text in it,
  // which is less reliably read out than a region that was already on the
  // page -- the very reason this one is pre-mounted. So `cancelled` and
  // `lost` may pass in silence where they used to be spoken. It is the shape
  // `RemovalResult` below already has, and it is what "one fact once" buys;
  // a real screen-reader pass is what would settle whether to pay it.
  const bucket = percent === null ? -1 : Math.floor(percent / 10);
  const announcement = useMemo(
    () =>
      ended
        ? ''
        : [
          stepText ?? headline,
          percent === null ? null : `${Math.round(percent)}%`,
        ]
          .filter(Boolean)
          .join(' '),
    // Intentionally narrow: `stepText`, `headline` and `percent` are read at
    // the moment one of these changes, which IS the throttle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [job?.jobId, current?.step.step, bucket, locale, ended],
  );

  if (job === null) {
    // A removal only ever reaches here with no job beside it -- the store
    // dismisses the finished one as it records this, and starting a job
    // clears it -- but the job is read first anyway, so the live thing wins
    // whatever order two changes arrive in.
    if (removal !== null) {
      return <RemovalResult removal={removal} onDismiss={onDismiss} />;
    }
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
          announced, so the next job's first step has to land in one that was
          already on the page. It empties at the end -- the banner below is a
          live region too, and it is the one that says how this job ended. */}
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

/**
 * How a finished uninstall is reported.
 *
 * The pane's third state, beside "nothing is installing" and a job. It wears
 * the job banner's clothes -- same tone, same `role="status"`, same Dismiss --
 * because it answers the same question, and the two are never on screen
 * together.
 *
 * The sentence is the toast's, deliberately: the toast is the notification and
 * this is the record, and one uninstall said two ways would be two facts to
 * reconcile. Under it, only when there is something to say: nothing removes a
 * plugin's pip packages -- not this panel, not the CLI -- so an uninstall that
 * left some names the packages, hands over the line that removes them with the
 * server stopped, and the line that puts the plugin back if that was a
 * mistake. An uninstall that left nothing is the sentence alone.
 */
function RemovalResult({
  removal,
  onDismiss,
}: {
  removal: PluginRemoval;
  onDismiss: () => void;
}) {
  const { t } = useI18n();
  const left = removal.depsLeft.length > 0;

  return (
    <>
      <div className={styles.activityTitle}>{removal.name}</div>

      <div className={styles.resultBanner} role="status" data-tone="success">
        <span>{t('pluginCenter.toast.removed', { plugin: removal.name })}</span>

        {left && (
          <>
            <span className={styles.resultHint}>
              {t('pluginCenter.activity.depsLeft', {
                packages: removal.depsLeft.join(', '),
              })}
            </span>
            {removal.uninstallCommand !== null && (
              <CommandBlock command={removal.uninstallCommand} />
            )}
            {removal.reinstallHint !== '' && (
              <>
                <span className={styles.resultHint}>
                  {t('pluginCenter.activity.reinstall')}
                </span>
                <CommandBlock command={removal.reinstallHint} />
              </>
            )}
          </>
        )}
      </div>

      <div className={styles.cardActions}>
        <button type="button" className={styles.secondaryBtn} onClick={onDismiss}>
          {t('packs.activity.dismiss')}
        </button>
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

  // `role="status"` below, so this element IS the announcement of the ending
  // as well as the sight of it -- which is why the pane's own live region
  // stops here rather than saying the same thing again.
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
 * The one sentence a finished job is reported with, said in one place.
 *
 * The banner is that place: it is what a sighted reader sees AND a
 * `role="status"` region, so the ending is announced by the element carrying
 * it. The pane's own live region -- which exists because a region that mounts
 * with its text already in it is not reliably announced -- keeps the running
 * commentary and stops where this sentence starts. It used to carry this too,
 * which read the ending out twice; being visually hidden, it never showed it.
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
