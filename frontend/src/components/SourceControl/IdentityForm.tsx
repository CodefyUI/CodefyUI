import { useEffect, useId, useRef, useState, type FormEvent } from 'react';
import type { ConfigScope } from '../../api/git';
import { useGitStore } from '../../store/gitStore';
import { useI18n, type TranslationKey } from '../../i18n';
import styles from './SourceControl.module.css';

/**
 * Which config file a value came from.
 *
 * `system` reads as "global": both mean "every repository on this machine",
 * and the difference between them is a fact about the installation, not about
 * the commit that is about to be made.
 */
function scopeKey(scope: ConfigScope | null, value: string | null): TranslationKey {
  if (value === null || value === '') return 'git.identity.missing';
  return scope === 'local' ? 'git.identity.scopeLocal' : 'git.identity.scopeGlobal';
}

/**
 * Name and email, inline above the panel.
 *
 * Inline rather than a modal because it opens BY ITSELF: git refuses a commit
 * with no identity, and the tab answers that refusal by putting the fix where
 * the commit box is instead of throwing a dialog over the work.
 *
 * A field left blank means "leave that one alone" -- the store omits an empty
 * half from the request rather than sending an empty string, which the server
 * refuses. So a user whose name is already set globally can fill in the email
 * and nothing else, which is the common case the scope line exists to explain.
 */
export function IdentityForm() {
  const { t } = useI18n();
  const identity = useGitStore((s) => s.identity);
  const saveIdentity = useGitStore((s) => s.saveIdentity);
  const closeIdentityForm = useGitStore((s) => s.closeIdentityForm);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const seeded = useRef(false);
  const domId = useId();
  const nameId = `${domId}-name`;
  const emailId = `${domId}-email`;

  // The config read is async and starts when the form opens, so the fields are
  // empty for a frame or two and fill in when it lands.
  //
  // ONCE. `identity` is replaced by every config read and by every successful
  // write, and a second one -- a commit refused again for a missing identity,
  // the form reopened from the menu -- would otherwise land in the middle of
  // typing and put the old values back over what was being written.
  useEffect(() => {
    if (identity === null || seeded.current) return;
    seeded.current = true;
    setName(identity.name ?? '');
    setEmail(identity.email ?? '');
  }, [identity]);

  const nothingToSave = name.trim() === '' && email.trim() === '';

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (nothingToSave) return;
    // The store closes the form on success and leaves it open on a refusal,
    // with the reason in the header's error line.
    void saveIdentity({ name, email });
  };

  return (
    <form className={styles.identityForm} onSubmit={onSubmit}>
      <div className={styles.identityTitle}>{t('git.identity.title')}</div>
      <div className={styles.field}>
        <label className={styles.fieldLabel} htmlFor={nameId}>
          {t('git.identity.name')}
        </label>
        <input
          id={nameId}
          className={styles.fieldInput}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <span className={styles.scope}>
          {t(scopeKey(identity?.name_scope ?? null, identity?.name ?? null))}
        </span>
      </div>
      <div className={styles.field}>
        <label className={styles.fieldLabel} htmlFor={emailId}>
          {t('git.identity.email')}
        </label>
        <input
          id={emailId}
          className={styles.fieldInput}
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <span className={styles.scope}>
          {t(scopeKey(identity?.email_scope ?? null, identity?.email ?? null))}
        </span>
      </div>
      {/*
        Save filled and Cancel ghosted, which is the order and the weighting
        `DialogContainer` already uses everywhere else in the app. Save is off
        while both halves are blank because that is the ONE request
        `PUT /config` cannot do anything with -- it answers 400 `invalid_value`
        -- and the store refuses it before the wire anyway.
      */}
      <div className={styles.identityActions}>
        <button
          type="submit"
          className={styles.filledButton}
          aria-disabled={nothingToSave}
          onClick={(e) => {
            if (nothingToSave) e.preventDefault();
          }}
        >
          {t('git.identity.save')}
        </button>
        <button
          type="button"
          className={styles.ghostButton}
          onClick={() => closeIdentityForm()}
        >
          {t('dialog.cancel')}
        </button>
      </div>
    </form>
  );
}
