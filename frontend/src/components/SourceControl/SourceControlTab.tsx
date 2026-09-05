import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { isLayoutFile, useGitStore } from '../../store/gitStore';
import { gitOpKey } from './scm';
import { useI18n } from '../../i18n';
import { BranchesSection } from './BranchesSection';
import { ChangeGroup } from './ChangeGroup';
import { CommitBox } from './CommitBox';
import { EmptyStates } from './EmptyStates';
import { IdentityForm } from './IdentityForm';
import { MergeGroup } from './MergeGroup';
import { RemotesSection } from './RemotesSection';
import { ScmHeader } from './ScmHeader';
import { StashesSection } from './StashesSection';
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
  // Which of the two live regions currently carries the sentence; the other
  // one is empty. See the pair at the bottom of this file.
  const [liveSlot, setLiveSlot] = useState(0);

  useEffect(() => {
    // Read off the store rather than through a selector, so the effect can
    // hold an empty dependency list: exactly one attach, exactly one detach,
    // whatever re-renders happen in between.
    const { attach, detach } = useGitStore.getState();
    attach();
    return () => detach();
  }, []);

  // The sentence changes SIDES on each finished write, because two stages in a
  // row can leave the same words ("Staged Changes 2, Changes 0") and an
  // unchanged text node is announced exactly zero times. A finished operation
  // is the signal -- in EITHER lane: a local write releases `busyOp` and a
  // network one releases `netOp`, and a guard that watched the local lane
  // alone said a first fetch and then nothing for every fetch after it. The
  // identity write is the exception -- it moves nothing in the panel and
  // writes no sentence, so a swap there would re-read whatever the last real
  // operation said, and only the local lane has that operation.
  useEffect(
    () =>
      useGitStore.subscribe((state, prev) => {
        const finishedLocal = prev.busyOp !== null
          && state.busyOp === null
          && prev.busyOp !== 'identity';
        const finishedNet = prev.netOp !== null && state.netOp === null;
        if (!finishedLocal && !finishedNet) return;
        if (state.lastError !== null) return;
        if (state.liveMessage === '') return;
        setLiveSlot((slot) => (slot === 0 ? 1 : 0));
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

  // Nothing to draw a panel from yet. `unknown` is the obvious half; the
  // other is a `ready` repository with no status beside it, which the route
  // never sends -- the two travel together -- and which would otherwise fall
  // past all three branches and leave a header above an empty body. A wait is
  // both truer than that and something the reader can act on.
  const waiting = repoState === 'unknown' || (repoState === 'ready' && status === null);

  let body: ReactNode = null;
  if (waiting) {
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
          {/*
            Outside the clean/dirty branch, because a merge whose every file
            has been settled as "mine" changes no file: the tree is clean, the
            conflict list is empty, and `MERGE_HEAD` is still there with only
            two ways out of it -- the commit box, and this group's Abort. The
            banner is the group's own; the tab drew a second copy of that
            sentence here until the group had a heading to hang it under.
          */}
          {(status.merge_in_progress || status.conflicted.length > 0) && (
            <MergeGroup files={status.conflicted} />
          )}
          {clean
            ? <div className={styles.stateMessage}>{t('git.empty.clean')}</div>
            : (
              <>
                <ChangeGroup kind="staged" files={status.staged} />
                <ChangeGroup kind="changes" files={changes} />
              </>
            )}
          {/*
            Outside the clean/dirty branch above: branches, remotes and stashes
            are properties of the repository, not of the working tree, and a
            clean checkout is exactly when somebody goes looking for another
            branch. Each one reads its own slice of the store, collapsed by
            default and remembered there -- which is also what loads its list
            as the section opens.
          */}
          <BranchesSection />
          <RemotesSection />
          <StashesSection />
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
      {/*
        TWO regions, both mounted from the first render, and the sentence
        alternates between them.

        A `role="status"` element that is INSERTED with its text already
        inside is not reliably announced -- the same lesson
        `PackActivityPane`'s announcer records -- so replacing one region per
        write (which is what a changing `key` does) can be silence. Keeping
        one region and rewriting it has the opposite hole: the second of two
        stages leaves the identical sentence, and an unchanged text node is
        read zero times. Alternating gives both halves: the region a sentence
        lands in has been on the page all along, and a repeated sentence is
        still a change to whichever region is next.

        Outside `.panelBody`, which is a flex column that nothing may be
        positioned out of, and above the header so the text is never part of
        the reading order of the panel it describes.
      */}
      <div className={styles.live} role="status" aria-live="polite">
        {liveSlot === 0 ? liveMessage : ''}
      </div>
      <div className={styles.live} role="status" aria-live="polite">
        {liveSlot === 1 ? liveMessage : ''}
      </div>
      <ScmHeader />
      <div className={shell.panelBody}>
        {identityFormOpen && <IdentityForm />}
        {body}
      </div>
    </>
  );
}
