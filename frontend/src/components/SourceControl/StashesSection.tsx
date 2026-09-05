import { useCallback, useEffect, useRef } from 'react';
import type { StashInfo } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { RefSection } from './RefSection';
import { RefEmpty, RefError, RefRow } from './RefRow';
import { focusRefSection } from './ScmHeader';
import { relativeTime } from './scm';

/** git's own name for a stash, which is what the next command line will say. */
function selectorFor(stash: StashInfo): string {
  return `stash@{${stash.index}}`;
}

/**
 * The stash stack, and the three things that can be done to one.
 *
 * Two rules here are git's and not this panel's. The INDEX is `StashInfo.index`
 * and never the array position: dropping `stash@{0}` renumbers every stash
 * below it, so a row that sent its own position would pop the wrong one the
 * moment two stashes existed and one was dropped. And the MESSAGE is shown
 * exactly as it was written -- it is the one string in this panel a user wrote
 * themselves, and trimming or truncating it would make the list unsearchable
 * by the words that are in it.
 *
 * Only Drop asks first. Pop and Apply put the stash back into the working
 * tree, which is where the user wanted it, and the store already offers to
 * reload any open graph they changed; Drop is the one that throws work away
 * with nothing to undo it.
 *
 * Making a stash is the More menu's, not this section's: it acts on the
 * working tree rather than on the list, and it has to be reachable from a
 * panel whose sections are all collapsed.
 */
export function StashesSection() {
  const { t, locale } = useI18n();
  const stashes = useGitStore((s) => s.stashes);
  // Null while no status has answered, which is not the same as zero.
  const stashCount = useGitStore((s) => s.status?.stash_count ?? null);
  const open = useGitStore((s) => s.sections.stashes);
  const refsError = useGitStore((s) => s.refsError.stashes);
  const setSectionOpen = useGitStore((s) => s.setSectionOpen);
  const refreshRefs = useGitStore((s) => s.refreshRefs);
  const stashPop = useGitStore((s) => s.stashPop);
  const stashApply = useGitStore((s) => s.stashApply);
  const stashDrop = useGitStore((s) => s.stashDrop);

  // The count this list was last re-read for, so one disagreement asks once.
  const askedFor = useRef<number | null>(null);
  // Whether that read is still out. `askedFor` only covers the renders where
  // the count is UNCHANGED, and the window this fires in is exactly where it
  // moves again: a stash write answers with a status first and the list a
  // moment later, and a second stash pushed at the command line inside that
  // moment opened a second concurrent GET for the same list. Same shape as
  // the header's `remotesInFlight`.
  const inFlight = useRef(false);

  /**
   * Read the list again whenever the status says it is the wrong length.
   *
   * The two come from different reads at different times: the status is the
   * fifteen-second poll (and every mutation's answer), while the list is read
   * when the section opens and after a stash write. A `git stash push` at the
   * command line moves only the first, so the rows on screen were a stack
   * that no longer existed -- and the panel had the evidence in its own
   * store. One read per disagreement: the answer either settles it or the
   * count moves again, and a re-read on every render would ask a server that
   * kept answering the old list once a frame.
   *
   * The in-flight guard does not swallow the question: a skipped ask leaves
   * `askedFor` alone, so the next answer that changes `stashes` asks it again
   * -- with the count it disagrees about NOW rather than the one the first
   * read went out for. That next answer is the one after the read being
   * waited on, not the read itself: the store's `set` reaches React before
   * the promise's `finally` does, so the effect the answer runs can still see
   * the flag. For an open section the next answer is at most one poll away
   * (`refreshExpandedRefs` re-reads every open list), and it takes two stash
   * writes landing inside one read to get there at all.
   */
  useEffect(() => {
    if (stashes === null || stashCount === null) return;
    if (!open) {
      // A collapsed section draws no rows at all, and the number beside its
      // heading is the STATUS's own -- so a list that disagrees with it is
      // not on screen to be wrong, and a read here would be a request for
      // something nobody can see. Opening reads the list anyway
      // (`setSectionOpen`), so the disagreement is recorded as asked: without
      // that, the render the open produces would send a second read a moment
      // behind the section's own.
      askedFor.current = stashCount;
      return;
    }
    if (stashes.length === stashCount) {
      askedFor.current = null;
      return;
    }
    if (askedFor.current === stashCount) return;
    // Before the stamp, not after: a read that is skipped here has not been
    // asked, and the answer that is on its way is what runs this again.
    if (inFlight.current) return;
    askedFor.current = stashCount;
    inFlight.current = true;
    void refreshRefs('stashes').finally(() => {
      inFlight.current = false;
    });
  }, [open, refreshRefs, stashCount, stashes]);

  const askThenDrop = useCallback(
    async (stash: StashInfo) => {
      const ok = await confirm({
        title: t('git.stash.dropConfirm', { name: selectorFor(stash) }),
        confirmText: t('git.stash.drop'),
        variant: 'danger',
      });
      if (!ok) return;
      if (await stashDrop(stash.index)) focusRefSection('stashes');
    },
    [stashDrop, t],
  );

  return (
    <RefSection
      kind="stashes"
      title={t('git.section.stashes')}
      // The STATUS wins: it is the half the poll keeps fresh, and every write
      // answers with a new one. The list is a read that happens when the
      // section opens -- so after a `git stash push` at the command line the
      // list said 0 beside a status that said 1, and went on saying it for as
      // long as the tab stayed open. The list is the fallback for the moment
      // before any status has answered.
      count={stashCount ?? stashes?.length ?? 0}
      open={open}
      onOpenChange={(next) => setSectionOpen('stashes', next)}
    >
      <RefError message={refsError} what={t('git.section.stashes')} />
      {refsError === null && stashes !== null && stashes.length === 0 && (
        <RefEmpty text={t('git.stash.empty')} />
      )}
      {(stashes ?? []).map((stash) => {
        const selector = selectorFor(stash);
        const when = relativeTime(stash.created_at, locale);
        // A key rather than a literal ", ": the comma between two clauses is
        // a different character in Chinese, and a row missing one half is
        // the other half alone rather than a dangling separator.
        const meta = stash.branch === null || stash.branch === ''
          ? when
          : t('git.stash.rowMeta', { branch: stash.branch, when });
        return (
          <RefRow
            key={selector}
            name={stash.message}
            // The buttons are named after the selector, not the message: git
            // writes the same "WIP on main: ..." for every stash nobody named.
            identity={selector}
            badge={selector}
            meta={meta}
            // Three things want this row and a 180px panel has room for two.
            // The message is the one a reader is scanning for -- it is the
            // string they wrote -- so below 380px the branch and date come out
            // of the row and stay in its title and its accessible text.
            metaOptional
            actions={[
              {
                id: 'pop',
                label: t('git.stash.pop'),
                onSelect: () => {
                  void stashPop(stash.index).then((ok) => {
                    // The stash is off the stack, so the row it was on is gone.
                    if (ok) focusRefSection('stashes');
                  });
                },
              },
              {
                id: 'apply',
                // No focus move: an apply leaves the stash where it is, so the
                // button that was pressed is still under the pointer.
                label: t('git.stash.apply'),
                onSelect: () => void stashApply(stash.index),
              },
              {
                id: 'drop',
                label: t('git.stash.drop'),
                danger: true,
                onSelect: () => void askThenDrop(stash),
              },
            ]}
          />
        );
      })}
    </RefSection>
  );
}
