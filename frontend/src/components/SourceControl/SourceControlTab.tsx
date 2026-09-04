import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { gitOpKey, useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { ChangeGroup } from './ChangeGroup';
import { CommitBox } from './CommitBox';
import { EmptyStates } from './EmptyStates';
import { IdentityForm } from './IdentityForm';
import { ScmHeader, isLayoutFile } from './ScmHeader';
import shell from '../Sidebar/NodePalette.module.css';
import styles from './SourceControl.module.css';

/**
 * The sidebar's fifth tab: the working tree, and what can be done to it.
 *
 * The panel owns the poll. `attach()` on mount and `detach()` on unmount is
 * the whole lifecycle, because the sidebar mounts only the tab that is open --
 * so opening this one starts the fifteen-second status read, the focus and
 * visibility listeners and the save hook, and switching away or collapsing the
 * sidebar stops all three. Nothing outside the tab needs git, which is why
 * there is no bootstrap hook beside the three in `NodePalette`.
 *
 * `repoState` decides which screen this is, and it is never an error: "there
 * is no project", "git is not installed" and "this is not a repository" all
 * arrive as a 200 with a null status. `loadError` is the separate thing -- a
 * server that cannot be read -- and the header shows it in every state,
 * including after a good first read, so a panel can never sit there quietly
 * showing a status that stopped being true ten minutes ago.
 */
export function SourceControlTab() {
  const { t } = useI18n();
  const repoState = useGitStore((s) => s.repoState);
  const repo = useGitStore((s) => s.repo);
  const status = useGitStore((s) => s.status);
  const loadError = useGitStore((s) => s.loadError);
  const hideLayout = useGitStore((s) => s.hideLayout);
  const identityFormOpen = useGitStore((s) => s.identityFormOpen);
  const liveMessage = useGitStore((s) => s.liveMessage);
  const [announceKey, setAnnounceKey] = useState(0);

  useEffect(() => {
    // Read off the store rather than through a selector, so the effect can
    // hold an empty dependency list: exactly one attach, exactly one detach,
    // whatever re-renders happen in between.
    const { attach, detach } = useGitStore.getState();
    attach();
    return () => detach();
  }, []);

  // The live region is re-keyed rather than merely re-rendered, because two
  // stages in a row can leave the SAME sentence ("Staged Changes 2, Changes
  // 0") and an unchanged text node is announced exactly zero times. A finished
  // operation is the signal: `busyOp` falling back to null with no error is
  // one write that landed. The identity write is the exception -- it moves
  // nothing in the panel and writes no sentence, so re-keying there would
  // re-read whatever the last real operation said.
  useEffect(
    () =>
      useGitStore.subscribe((state, prev) => {
        if (prev.busyOp === null || state.busyOp !== null) return;
        if (prev.busyOp === 'identity') return;
        if (state.lastError !== null) return;
        if (state.liveMessage === '') return;
        setAnnounceKey((n) => n + 1);
      }),
    [],
  );

  // Unstaged and untracked are one list on screen: both are "work that is not
  // in the next commit yet", and git's distinction between them is already
  // carried by the letter on each row.
  const changes = useMemo(() => {
    if (status === null) return [];
    const all = [...status.unstaged, ...status.untracked];
    return hideLayout ? all.filter((file) => !isLayoutFile(file.path)) : all;
  }, [status, hideLayout]);

  let body: ReactNode = null;
  if (repoState === 'unknown') {
    // Nothing has answered yet. A failed first read has already put the
    // reason in the header, so this line is only for the wait itself.
    body = loadError === null
      ? (
        <div className={styles.stateMessage}>
          {t('git.busy', { op: t(gitOpKey('status')) })}
        </div>
      )
      : null;
  } else if (repoState === 'ready' && status !== null) {
    const clean = status.staged.length === 0
      && status.unstaged.length === 0
      && status.untracked.length === 0
      && status.conflicted.length === 0;
    body = (
      <>
        <CommitBox />
        <div className={styles.scroll}>
          {status.merge_in_progress && (
            <div className={styles.banner}>{t('git.merge.banner')}</div>
          )}
          {clean
            ? <div className={styles.stateMessage}>{t('git.empty.clean')}</div>
            : (
              <>
                {status.conflicted.length > 0 && (
                  <ChangeGroup kind="merge" files={status.conflicted} />
                )}
                <ChangeGroup kind="staged" files={status.staged} />
                <ChangeGroup kind="changes" files={changes} />
              </>
            )}
        </div>
      </>
    );
  } else if (repoState !== 'ready') {
    body = (
      <div className={styles.scroll}>
        <EmptyStates
          state={repoState}
          nestedToplevel={repo?.nested_toplevel ?? null}
          gitVersion={repo?.git_version ?? null}
          onInit={() => void useGitStore.getState().init()}
        />
      </div>
    );
  }

  return (
    <>
      {/* Outside `.panelBody`, which is a flex column that nothing may be
          positioned out of, and above the header so its text is never part of
          the reading order of the panel it describes. */}
      <div
        key={announceKey}
        className={styles.live}
        role="status"
        aria-live="polite"
      >
        {liveMessage}
      </div>
      <ScmHeader />
      <div className={shell.panelBody}>
        {identityFormOpen && <IdentityForm />}
        {body}
      </div>
    </>
  );
}
