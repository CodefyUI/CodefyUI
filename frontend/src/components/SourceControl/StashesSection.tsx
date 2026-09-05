import { useCallback } from 'react';
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
  const stashCount = useGitStore((s) => s.status?.stash_count ?? 0);
  const open = useGitStore((s) => s.sections.stashes);
  const refsError = useGitStore((s) => s.refsError.stashes);
  const setSectionOpen = useGitStore((s) => s.setSectionOpen);
  const stashPop = useGitStore((s) => s.stashPop);
  const stashApply = useGitStore((s) => s.stashApply);
  const stashDrop = useGitStore((s) => s.stashDrop);

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
      // The list wins where it has been read, because those are the rows the
      // section is about to draw; the status carries the count on every poll
      // before that.
      count={stashes?.length ?? stashCount}
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
