import { useCallback, useId, useState } from 'react';
import type { GitCommit, GitCommitFile } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import { useI18n } from '../../i18n';
import { ActionMenu } from '../shared/ActionMenu';
import { MoreHorizontalIcon } from '../shared/Icons';
import { displayPath, FileKindChip, FileRowName } from './FileRow';
import { RefSection } from './RefSection';
import { RefError } from './RefRow';
import { relativeTime } from './scm';
import styles from './SourceControl.module.css';

/**
 * The commits on this branch, newest first, a page at a time.
 *
 * Not a reference list, although it wears the same box: branches, remotes and
 * stashes are one short read each and are re-read on every poll, while this
 * one is PAGED -- so re-reading it on a schedule would throw away every page
 * past the first that the reader had loaded. The store keeps it off the poll
 * and refreshes it on the things that actually move HEAD; see
 * `reloadLogIfLive` there.
 *
 * Two rules here are not taste. The file list is read from the row's own
 * CLICK, never from an effect: `loadCommitFiles` caches a sha only once the
 * read answers, and StrictMode double-invokes effects, so an effect would
 * spend two requests on every expand in development and one per row on mount.
 * And ONE commit is expanded at a time -- a page is thirty rows deep, and a
 * second file list under the first pushes both off a 500px panel.
 */
export function HistorySection() {
  const { t } = useI18n();
  const log = useGitStore((s) => s.log);
  const commitFiles = useGitStore((s) => s.commitFiles);
  const open = useGitStore((s) => s.sections.history);
  const historyError = useGitStore((s) => s.historyError);
  const setSectionOpen = useGitStore((s) => s.setSectionOpen);
  const loadMoreLog = useGitStore((s) => s.loadMoreLog);
  const loadCommitFiles = useGitStore((s) => s.loadCommitFiles);
  const announce = useGitStore((s) => s.announce);
  const openGitDiff = useUIStore((s) => s.openGitDiff);
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggleCommit = useCallback(
    (sha: string) => {
      const opening = expanded !== sha;
      setExpanded(opening ? sha : null);
      // From the press, and only when the row is OPENING. Closing and
      // reopening costs nothing either: a commit's tree never changes, so the
      // store answers the second expand out of its cache.
      if (opening) void loadCommitFiles(sha);
    },
    [expanded, loadCommitFiles],
  );

  const copySha = useCallback(
    (sha: string, short: string) => {
      // Wrapped in a promise chain rather than called directly, so an absent
      // `navigator.clipboard` -- which is what a page served over plain http
      // gets -- throws INTO the rejection path instead of past it.
      //
      // Two answers, not one: a toast, which is what the Package Center gives
      // the same event and what a sighted reader sees, and the live message,
      // which is what a screen reader hears. This panel's own keys rather
      // than the Center's -- "Could not copy. Select the text and copy it by
      // hand." is advice about a command block, and there is no text to
      // select on a commit row.
      //
      // The id is IN the sentence, and that is not decoration. The live
      // region holds one string: `announce` writes `liveMessage`, the slots
      // in `SourceControlTab` swap only when an operation releases `busyOp`
      // or `netOp`, and a clipboard write releases neither. So a constant
      // sentence written twice is the same text node twice, which a screen
      // reader announces exactly zero times -- copying commit A and then
      // commit B would say nothing at all for B. Naming the commit makes two
      // consecutive copies two different strings, which is what makes the
      // second one audible.
      const { addToast } = useToastStore.getState();
      void Promise.resolve()
        .then(() => navigator.clipboard.writeText(sha))
        .then(
          () => {
            const said = t('git.history.copied', { sha: short });
            addToast(said, 'success');
            announce(said);
          },
          () => {
            const said = t('git.history.copyFailed');
            addToast(said, 'error');
            announce(said);
          },
        );
    },
    [announce, t],
  );

  const loadMore = useCallback(() => {
    // The guard, not a native `disabled`: the press itself is what sets
    // `loading`, so a natively disabled button would stop being focusable on
    // the very next render and the browser would drop focus to `<body>`.
    if (log.loading) return;
    void loadMoreLog();
  }, [log.loading, loadMoreLog]);

  return (
    <RefSection
      kind="history"
      title={t('git.section.history')}
      // How many rows the section holds, which is the page the reader has
      // loaded rather than the length of the history. Nothing at all until
      // something has been read -- and nothing on an unborn branch either,
      // where zero would be a number about a repository that has no commits
      // to count and the header line already says so.
      count={log.unborn || log.commits.length === 0 ? null : log.commits.length}
      open={open}
      onOpenChange={(next) => setSectionOpen('history', next)}
    >
      {/* The same line the three reference sections draw, for the same
          reason: this read is the panel's own and its failure must not
          replace the refusal the user was reading on the header. The
          store's message rather than `errorSentence`, because the frame
          around it already says which read failed and `git.error.generic`
          ("git failed: ...") inside that frame is the same fact twice; a
          timeout's sentence is already in `message`. */}
      <RefError
        message={historyError === null ? null : historyError.message}
        what={t('git.section.history')}
      />
      {log.commits.map((commit) => (
        <CommitRow
          key={commit.sha}
          commit={commit}
          open={expanded === commit.sha}
          // A sha is forty hexadecimal characters, so it can never name an
          // inherited member of this bare object.
          files={commitFiles[commit.sha]}
          onToggle={() => toggleCommit(commit.sha)}
          onCopy={() => copySha(commit.sha, commit.short)}
          onOpenFile={(file) =>
            openGitDiff({ path: file.path, scope: 'commit', sha: commit.sha })}
        />
      ))}
      {/* Never on an unborn branch: there is no page after a history that
          does not exist, whatever a server answers about one. */}
      {log.hasMore && !log.unborn && (
        <li className={styles.moreRow}>
          <button
            type="button"
            className={styles.linkButton}
            aria-disabled={log.loading}
            onClick={loadMore}
          >
            {t('git.history.loadMore')}
          </button>
        </li>
      )}
    </RefSection>
  );
}

/**
 * One commit, and the files it changed.
 *
 * Two `<li>`s rather than one: the row is a flex line and the file list is a
 * list, so nesting the second inside the first would put a block inside a box
 * whose whole job is to keep one line on one line. The list stays MOUNTED
 * while it is collapsed -- an `aria-controls` pointing at an element that is
 * not in the document names nothing, and a reader offering "go to the
 * controlled element" would have nowhere to go.
 */
function CommitRow({
  commit,
  open,
  files,
  onToggle,
  onCopy,
  onOpenFile,
}: {
  commit: GitCommit;
  open: boolean;
  /** Undefined until the read answers, which is not the same as none. */
  files: GitCommitFile[] | undefined;
  onToggle: () => void;
  onCopy: () => void;
  onOpenFile: (file: GitCommitFile) => void;
}) {
  const { t, locale } = useI18n();
  const filesId = useId();
  const headingId = useId();
  // The dim half of the row: when it was made, and by whom. One string
  // rather than two spans with a separator between them -- the separator
  // would be a punctuation mark that is not the same character in every
  // language, and neither half is a sentence that needs one.
  const when = relativeTime(commit.authoredAt, locale);
  const meta = [when, commit.authorName].filter((part) => part !== '').join(' ');

  return (
    <>
      {/* The subject and the meta live on the ROW as well: below 380px the
          meta is a 1px box off screen, so its own tooltip is unreachable and
          this is the only one a pointer can open. A newline between them,
          because they are two clauses and not one sentence. */}
      <li
        className={styles.row}
        title={meta === '' ? commit.subject : `${commit.subject}\n${meta}`}
      >
        <button
          type="button"
          className={styles.openButton}
          aria-expanded={open}
          aria-controls={filesId}
          onClick={onToggle}
        >
          {/* Seven characters is what the row has room for and what the next
              command line takes; the forty are what a bug report needs, so
              they stay in reach here. */}
          <span className={styles.commitSha} title={commit.sha}>
            {commit.short}
          </span>
          {/* No `title` of its own. This is the elastic part of the row and
              therefore most of its width, and an inner tooltip WINS over an
              ancestor's -- so one here would answer a pointer with the
              subject alone over the whole button, and the row's tooltip
              (subject AND meta), which is the only place the meta can be read
              below 380px, would never open. `FileRowName` and `RefRow`'s name
              span carry none for the same reason. */}
          <span className={`${styles.rowName} ${styles.nameElastic}`}>
            {commit.subject}
          </span>
          {meta !== '' && (
            <span
              className={`${styles.rowDir} ${styles.metaFirm} ${styles.metaOptional}`}
            >
              {meta}
            </span>
          )}
        </button>
        {/* Both shapes, and the container query picks one -- see
            `.rowChoices` in the stylesheet. The verb is about 90px of text
            beside a sha that cannot shrink, which a 180px panel does not
            have. */}
        <div className={styles.rowActions}>
          <div className={styles.rowChoices}>
            <button
              type="button"
              className={styles.rowAction}
              // The verb NAMES the commit it acts on: thirty rows deep, a
              // button called "Copy commit id" is thirty identical buttons.
              aria-label={`${t('git.history.copySha')} ${commit.short}`}
              title={t('git.history.copySha')}
              onClick={onCopy}
            >
              {t('git.history.copySha')}
            </button>
          </div>
          <div className={styles.rowMenu}>
            <ActionMenu
              label={`${t('git.action.more')} ${commit.short}`}
              items={[
                { id: 'copySha', label: t('git.history.copySha'), onSelect: onCopy },
              ]}
              align="end"
              className={styles.iconButton}
            >
              <MoreHorizontalIcon size={13} />
            </ActionMenu>
          </div>
        </div>
      </li>
      <li className={styles.refSublist} id={filesId} hidden={!open}>
        {/* Nothing at all while the read is out: a count drawn from a list
            that has not landed would say "0 file(s)" about a commit that
            changed four. */}
        {open && files !== undefined && (
          <>
            <span className={styles.refSubhead} id={headingId}>
              {t('git.history.files', { count: files.length })}
            </span>
            <ul className={styles.list} role="list" aria-labelledby={headingId}>
              {files.map((file) => (
                <li
                  key={`${file.path}:${file.kind}`}
                  className={styles.row}
                  title={displayPath(file)}
                >
                  <FileKindChip kind={file.kind} />
                  <FileRowName file={file} onOpen={() => onOpenFile(file)} />
                </li>
              ))}
            </ul>
          </>
        )}
      </li>
    </>
  );
}
