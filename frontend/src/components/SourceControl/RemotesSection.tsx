import { useCallback } from 'react';
import { useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { confirm, prompt } from '../../utils/dialog';
import { PlusIcon } from '../shared/Icons';
import { RefSection } from './RefSection';
import { RefEmpty, RefError, RefRow } from './RefRow';
import { focusRefSection } from './ScmHeader';
import { isValidRemoteName, isValidRemoteUrl } from './scm';
import styles from './SourceControl.module.css';

/**
 * The remotes this repository knows, and the three things that can be done to
 * one.
 *
 * The URL on a row is a DISPLAY string and not the remote's real URL: the
 * server masks the credential half of one before it is served, because
 * `GET /remotes` is an open read and people really do paste a token into
 * `git remote add`. Nothing here ever sends one back -- Change URL asks with
 * an EMPTY box, so what reaches the server is what somebody typed rather than
 * a string with `***` in the middle of it.
 *
 * The name is not a button. There is nothing to switch to, and a row that
 * looked pressable and was not would be worse than a plain one.
 */
export function RemotesSection() {
  const { t } = useI18n();
  const remotes = useGitStore((s) => s.remotes);
  const open = useGitStore((s) => s.sections.remotes);
  const refsError = useGitStore((s) => s.refsError.remotes);
  const setSectionOpen = useGitStore((s) => s.setSectionOpen);
  const addRemote = useGitStore((s) => s.addRemote);
  const setRemoteUrl = useGitStore((s) => s.setRemoteUrl);
  const removeRemote = useGitStore((s) => s.removeRemote);

  /**
   * A URL the server would refuse, refused while the box is still open.
   *
   * The message names https and SSH, which is what people paste. `file://` is
   * accepted too -- the server takes it, and a bare repository on a shared
   * drive is a real remote -- but there is no string that says so and this is
   * not the task that adds one.
   */
  const validateUrl = useCallback(
    (value: string) => (isValidRemoteUrl(value.trim()) ? null : t('git.remote.invalidUrl')),
    [t],
  );

  const askThenAdd = useCallback(async () => {
    const name = await prompt({
      title: t('git.remote.namePrompt'),
      // `git.error.invalid` -- the server's own words for this refusal, and
      // there is no key that describes a remote name in particular.
      validate: (value) =>
        isValidRemoteName(value.trim()) ? null : t('git.error.invalid'),
    });
    if (name === null) return;
    const url = await prompt({ title: t('git.remote.urlPrompt'), validate: validateUrl });
    if (url === null) return;
    await addRemote(name.trim(), url.trim());
  }, [addRemote, t, validateUrl]);

  const askThenChangeUrl = useCallback(
    async (name: string) => {
      // No `defaultValue`: see the note at the top of this file.
      const url = await prompt({ title: t('git.remote.urlPrompt'), validate: validateUrl });
      if (url === null) return;
      await setRemoteUrl(name, url.trim());
    },
    [setRemoteUrl, t, validateUrl],
  );

  const askThenRemove = useCallback(
    async (name: string) => {
      const ok = await confirm({
        title: t('git.remote.removeConfirm', { name }),
        confirmText: t('git.remote.remove'),
        variant: 'danger',
      });
      if (!ok) return;
      if (await removeRemote(name)) focusRefSection('remotes');
    },
    [removeRemote, t],
  );

  return (
    <RefSection
      kind="remotes"
      title={t('git.section.remotes')}
      // Null is "not read yet" here too, and a count is as much of a claim as
      // the empty sentence below.
      count={remotes === null ? null : remotes.length}
      open={open}
      onOpenChange={(next) => setSectionOpen('remotes', next)}
      actions={
        <button
          type="button"
          className={styles.iconButton}
          aria-label={t('git.remote.add')}
          title={t('git.remote.add')}
          onClick={() => void askThenAdd()}
        >
          <PlusIcon size={13} />
        </button>
      }
    >
      <RefError message={refsError} what={t('git.section.remotes')} />
      {/* `null` is "not read yet" and never "none", so the empty sentence
          waits for an answer rather than claiming one. */}
      {refsError === null && remotes !== null && remotes.length === 0 && (
        <RefEmpty text={t('git.remote.empty')} />
      )}
      {(remotes ?? []).map((entry) => (
        <RefRow
          key={entry.name}
          name={entry.name}
          meta={entry.fetch_url}
          // The one list whose NAME is the short half. A remote is called
          // `origin` and points at 400px of URL, and a row that shared the
          // shrink between them spent the name's last characters on a URL
          // that had plenty to give: "ori..." beside a readable address.
          firm="name"
          actions={[
            {
              id: 'url',
              label: t('git.remote.changeUrl'),
              onSelect: () => void askThenChangeUrl(entry.name),
            },
            {
              id: 'remove',
              label: t('git.remote.remove'),
              danger: true,
              onSelect: () => void askThenRemove(entry.name),
            },
          ]}
        />
      ))}
    </RefSection>
  );
}
