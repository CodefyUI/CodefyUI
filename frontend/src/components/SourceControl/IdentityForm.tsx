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
 * A half this form does not send means "leave that one alone" -- the store
 * omits an empty half from the request rather than sending an empty string,
 * which the server refuses. TWO halves go unsent: the one left blank, and the
 * one that still holds exactly what it was seeded with. The seed is the
 * identity git answered the config read with, which for most people is the
 * GLOBAL name and email -- so a user who opened this to set a name for one
 * project would otherwise have written the global email into that project's
 * `.git/config` beside it, and pinned the repository to an address nobody
 * chose. Save is off until one of the two really changes, for the same reason.
 */
export function IdentityForm() {
  const { t } = useI18n();
  const identity = useGitStore((s) => s.identity);
  const saveIdentity = useGitStore((s) => s.saveIdentity);
  const closeIdentityForm = useGitStore((s) => s.closeIdentityForm);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const seeded = useRef(false);
  const nameRef = useRef<HTMLInputElement>(null);
  const domId = useId();
  const nameId = `${domId}-name`;
  const emailId = `${domId}-email`;
  const nameScopeId = `${domId}-name-scope`;
  const emailScopeId = `${domId}-email-scope`;

  // Focus the first field as the form appears.
  //
  // This component is mounted only while the form is open, and the case that
  // matters is the one where nobody opened it: a commit refused for a missing
  // identity puts it on screen by itself, above a button the user has just
  // pressed. Landing in the Name field is what makes that a form to fill in
  // rather than a paragraph that appeared somewhere on the page.
  //
  // On mount only. The seeding effect below fills the fields a frame or two
  // later, and a focus that ran again then would take the caret back from
  // whatever the user had already tabbed to.
  useEffect(() => {
    nameRef.current?.focus();
  }, []);

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

  // What the fields were seeded with, and so what a half has to differ from
  // to be worth writing. A half that is blank, or back to the value it was
  // seeded with, is one this repository has no reason to be given.
  const seededName = identity?.name ?? '';
  const seededEmail = identity?.email ?? '';
  const nameChanged = name.trim() !== '' && name.trim() !== seededName;
  const emailChanged = email.trim() !== '' && email.trim() !== seededEmail;
  const nothingToSave = !nameChanged && !emailChanged;

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (nothingToSave) return;
    // An unchanged half goes as the empty string, which is the spelling the
    // store already reads as "leave that one alone" and omits from the
    // request. The store closes the form on success and leaves it open on a
    // refusal, with the reason in the header's error line.
    void saveIdentity({
      name: nameChanged ? name : '',
      email: emailChanged ? email : '',
    });
  };

  return (
    <form className={styles.identityForm} onSubmit={onSubmit}>
      <div className={styles.identityTitle}>{t('git.identity.title')}</div>
      <div className={styles.field}>
        <label className={styles.fieldLabel} htmlFor={nameId}>
          {t('git.identity.name')}
        </label>
        {/*
          The scope line DESCRIBES the field: "for this project" and "from
          global git config" are the difference between a commit signed as you
          and one signed as whoever last used this machine, and a reader who
          hears only the label and the value never learns which they are
          about to write.
        */}
        <input
          id={nameId}
          className={styles.fieldInput}
          aria-describedby={nameScopeId}
          value={name}
          ref={nameRef}
          onChange={(e) => setName(e.target.value)}
        />
        <span className={styles.scope} id={nameScopeId}>
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
          aria-describedby={emailScopeId}
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <span className={styles.scope} id={emailScopeId}>
          {t(scopeKey(identity?.email_scope ?? null, identity?.email ?? null))}
        </span>
      </div>
      {/*
        Save filled and Cancel ghosted, which is the order and the weighting
        `DialogContainer` already uses everywhere else in the app. Save is off
        until one of the halves differs from what was loaded, because a write
        with nothing new in it either says nothing at all -- both halves blank
        is the ONE request `PUT /config` refuses outright, 400 `invalid_value`,
        and the store stops it before the wire anyway -- or copies the global
        identity into this repository for no reason.
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
