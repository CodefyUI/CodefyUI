import { useCallback, useId } from 'react';
import type { BranchInfo, RemoteBranchInfo } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { confirm, prompt } from '../../utils/dialog';
import { PlusIcon } from '../shared/Icons';
import { RefSection } from './RefSection';
import { RefError, RefRow } from './RefRow';
import { focusRefSection } from './ScmHeader';
import { aheadBehindGlyphs, isValidBranchName } from './scm';
import styles from './SourceControl.module.css';

/**
 * Every branch in the repository, and the four things that can be done to one.
 *
 * The NAME is the switch button, rather than a "Switch" beside it: a row whose
 * only purpose is to be checked out reads better as one target than as a label
 * plus a verb, and at 180px there is no room for both. Its accessible name is
 * the whole sentence (`Switch to work`), because twenty rows of a button
 * called "Switch" are twenty buttons a reader cannot tell apart.
 *
 * A local branch with no upstream says nothing about tracking. "Not published"
 * beside every row of a repository that has no remote is noise; the header
 * already says it about the branch the user is on, which is the one they can
 * do something about.
 */
export function BranchesSection() {
  const { t } = useI18n();
  const branches = useGitStore((s) => s.branches);
  const open = useGitStore((s) => s.sections.branches);
  const refsError = useGitStore((s) => s.refsError.branches);
  const setSectionOpen = useGitStore((s) => s.setSectionOpen);
  const createBranch = useGitStore((s) => s.createBranch);
  const checkout = useGitStore((s) => s.checkout);
  const renameBranch = useGitStore((s) => s.renameBranch);
  const deleteBranch = useGitStore((s) => s.deleteBranch);
  const remoteHeadingId = useId();

  const local = branches?.local ?? [];
  const remote = branches?.remote ?? [];

  /** A name the server would refuse, refused while the box is still open. */
  const validateName = useCallback(
    (value: string) => (isValidBranchName(value.trim()) ? null : t('git.branch.invalid')),
    [t],
  );

  const askThenCreate = useCallback(async () => {
    const name = await prompt({ title: t('git.branch.namePrompt'), validate: validateName });
    if (name === null) return;
    // `createBranch` checks the new branch out by default, which is what
    // "New Branch..." means everywhere else this button exists.
    await createBranch(name.trim());
  }, [createBranch, t, validateName]);

  const askThenRename = useCallback(
    async (name: string) => {
      const next = await prompt({
        // No `defaultValue`: the old name is in the question, and a prefilled
        // box is one keystroke away from renaming a branch to itself.
        title: t('git.branch.renamePrompt', { name }),
        validate: validateName,
      });
      if (next === null) return;
      const renamed = next.trim();
      if (renamed === '' || renamed === name) return;
      if (await renameBranch(name, renamed)) focusRefSection('branches');
    },
    [renameBranch, t, validateName],
  );

  /**
   * Delete a branch, and ask again for the one refusal that is a question.
   *
   * git refuses to delete a branch whose commits are on no other branch, and
   * that refusal is not an error -- it is "are you sure", asked by the only
   * thing that can tell. So the answer to `branch_not_merged` is the second
   * confirm rather than a red line, and the line git's own words would have
   * left on screen is taken down before the question is asked.
   */
  const askThenDelete = useCallback(
    async (name: string) => {
      const ok = await confirm({
        title: t('git.branch.deleteConfirm', { name }),
        confirmText: t('git.branch.delete'),
        variant: 'danger',
      });
      if (!ok) return;
      if (await deleteBranch(name, false)) {
        focusRefSection('branches');
        return;
      }
      // Read AFTER the write, which is what put the refusal there.
      const { lastError, dismissError } = useGitStore.getState();
      if (lastError?.code !== 'branch_not_merged') return;
      dismissError();
      const forced = await confirm({
        title: t('git.branch.forceDeleteConfirm', { name }),
        confirmText: t('git.branch.delete'),
        variant: 'danger',
      });
      if (!forced) return;
      if (await deleteBranch(name, true)) focusRefSection('branches');
    },
    [deleteBranch, t],
  );

  return (
    <RefSection
      kind="branches"
      title={t('git.section.branches')}
      // The LOCAL branches: those are the rows the section's actions act on,
      // and a remote-tracking ref is not a branch you have. Null until the
      // list has been read -- the section is closed on a fresh profile, and
      // nothing has counted anything yet.
      count={branches === null ? null : local.length}
      open={open}
      onOpenChange={(next) => setSectionOpen('branches', next)}
      actions={
        <button
          type="button"
          className={styles.iconButton}
          aria-label={t('git.branch.new')}
          title={t('git.branch.new')}
          onClick={() => void askThenCreate()}
        >
          <PlusIcon size={13} />
        </button>
      }
    >
      <RefError message={refsError} what={t('git.section.branches')} />
      {local.map((entry) => (
        <LocalBranchRow
          key={entry.name}
          branch={entry}
          onSwitch={() => void checkout(entry.name, 'local').then((ok) => {
            // The row the button was on is now the current branch, so the
            // button itself has been replaced by the Current marker.
            if (ok) focusRefSection('branches');
          })}
          onRename={() => void askThenRename(entry.name)}
          onDelete={() => void askThenDelete(entry.name)}
        />
      ))}
      {remote.length > 0 && (
        // A list inside the list: these are not branches you have, they are
        // branches somewhere else that you can have. One press makes a local
        // branch that tracks one.
        <li className={styles.refSublist}>
          <span className={styles.refSubhead} id={remoteHeadingId}>
            {t('git.branch.remote')}
          </span>
          <ul className={styles.list} role="list" aria-labelledby={remoteHeadingId}>
            {remote.map((entry) => (
              <RemoteBranchRow
                // The whole ref, because the NAME half is not unique: `main`
                // on two remotes is two rows, and two rows keyed on `main`
                // are one row as far as React is concerned.
                key={remoteRef(entry)}
                branch={entry}
                onSwitch={() => void checkout(remoteRef(entry), 'remote')}
              />
            ))}
          </ul>
        </li>
      )}
    </RefSection>
  );
}

/** One local branch: its name, where it stands, and what can be done to it. */
function LocalBranchRow({
  branch,
  onSwitch,
  onRename,
  onDelete,
}: {
  branch: BranchInfo;
  onSwitch: () => void;
  onRename: () => void;
  onDelete: () => void;
}) {
  const { t } = useI18n();
  const tracking = branch.gone
    ? t('git.upstreamGone')
    : branch.upstream === null
      ? null
      : t('git.aheadBehind', { ahead: branch.ahead ?? 0, behind: branch.behind ?? 0 });
  // The count is drawn as the two numbers and read out as the sentence --
  // the same split the header's branch line makes, and for the same reason:
  // the clause and the branch name were ellipsising each other at 250px.
  // "Upstream deleted" is a state rather than a count and stays as words.
  const counted = !branch.gone && branch.upstream !== null;
  const glyphs = counted
    ? aheadBehindGlyphs(branch.ahead ?? 0, branch.behind ?? 0)
    : null;

  return (
    <RefRow
      name={branch.name}
      // The name is the button, unless this IS the branch you are on -- in
      // which case there is nothing to switch to and the row says so.
      action={branch.current ? null : { label: t('git.branch.switch', { name: branch.name }), onSelect: onSwitch }}
      badge={branch.current ? t('git.branch.current') : null}
      meta={counted ? glyphs : tracking}
      metaLabel={counted && tracking !== null ? tracking : undefined}
      actions={[
        { id: 'rename', label: t('git.branch.rename'), onSelect: onRename },
        // Never on the current branch: git refuses to delete the branch that
        // is checked out, so the button could only offer an error.
        ...(branch.current
          ? []
          : [{ id: 'delete', label: t('git.branch.delete'), danger: true, onSelect: onDelete }]),
      ]}
    />
  );
}

/**
 * The ref a remote-tracking branch IS: `origin/feat/deep`.
 *
 * The server splits one into two fields at the FIRST slash -- `remote` is
 * `origin` and `name` is everything after it -- and re-splits whatever the
 * checkout is sent the same way. So the two halves have to be put back
 * together before either the screen or the request sees them: `feat/deep`
 * alone is refused outright (`invalid_ref`, no slash to split on for a plain
 * branch), and worse when the branch's own name has a slash, where it names a
 * remote called `feat` that does not exist.
 */
function remoteRef(branch: RemoteBranchInfo): string {
  return `${branch.remote}/${branch.name}`;
}

/** One remote-tracking branch, which one press turns into a local one. */
function RemoteBranchRow({
  branch,
  onSwitch,
}: {
  branch: RemoteBranchInfo;
  onSwitch: () => void;
}) {
  const { t } = useI18n();
  // The row says the whole ref, which is the only thing that tells `main` on
  // `origin` from `main` on `upstream`.
  const ref = remoteRef(branch);
  return (
    <RefRow
      name={ref}
      action={{ label: t('git.branch.switch', { name: ref }), onSelect: onSwitch }}
      actions={[]}
    />
  );
}
