import { useId, useState, type FormEvent } from 'react';
import type { InspectionFailure, InspectionState } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import { parseGitHubSource, type Translate } from './pluginStatus';
import packStyles from '../PackCenter/PackCenterModal.module.css';
import styles from './PluginCenterModal.module.css';

/**
 * The box you type a repository into.
 *
 * Installing something the catalog does not list is the one thing the Plugin
 * Center does that the Package Center has no equivalent for, and it is a
 * two-step conversation: this box asks the server to READ a source, and the
 * review card that follows is what accepts what it found. Nothing here
 * installs anything.
 *
 * A pure view of `pluginStore.inspection` apart from what has been typed:
 * the phase decides what the button says, and a refusal that phase carries is
 * printed under the row.
 */

/** *detail*'s *key* when it is a non-empty string, else null. */
function text(detail: Record<string, unknown> | null, key: string): string | null {
  const value = detail?.[key];
  return typeof value === 'string' && value !== '' ? value : null;
}

/** The string members of *detail*'s *key*, or nothing at all. */
function list(detail: Record<string, unknown> | null, key: string): string[] {
  const value = detail?.[key];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

/**
 * What a refused inspection should read as: the complaint, and the offer
 * under it when the refusal carried one.
 *
 * Two of these codes are answered here rather than by `REFUSAL_KEY`, because
 * the useful half of each is in the BODY and not in the code: which id is
 * taken, and which names would have worked. A sentence written server-side
 * could not have said either.
 */
function refusalLines(
  t: Translate, failure: InspectionFailure, source: string,
): { message: string; hint: string | null } {
  if (failure.code === 'reserved_id') {
    return {
      message: t('pluginCenter.review.idConflict', {
        id: text(failure.detail, 'id') ?? source,
      }),
      hint: null,
    };
  }
  if (failure.code === 'unknown_catalog_name') {
    const known = list(failure.detail, 'known');
    return {
      message: t('pluginCenter.source.unknownName', { source }),
      hint: known.length === 0
        ? null
        : t('pluginCenter.source.knownNames', { known: known.join(', ') }),
    };
  }
  // Everything else already has its sentence: the store maps a coded refusal
  // to one before it ever reaches a component, so `message` is prose.
  return {
    message: t('pluginCenter.source.fail', { source, message: failure.message }),
    hint: null,
  };
}

export interface PluginSourceFormProps {
  inspection: InspectionState;
  /** False when the server refuses installs from this browser (remote). */
  canInstall: boolean;
  onReview: (source: string) => void;
}

export function PluginSourceForm({
  inspection, canInstall, onReview,
}: PluginSourceFormProps) {
  const { t } = useI18n();
  const inputId = useId();
  const [source, setSource] = useState('');
  // Typed something that is not a source. Local, because nothing was sent:
  // the store has no state for a request that was never made.
  const [invalid, setInvalid] = useState(false);

  const inspecting = inspection.phase === 'inspecting';
  const typed = source.trim();

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (typed === '') return;
    if (parseGitHubSource(typed) === null) {
      // Refused without a round trip. The server would answer 400
      // `unparseable_source` to the same string, and this is the one refusal
      // a client can be sure of on its own — so the sentence appears as fast
      // as the keystroke that earned it.
      setInvalid(true);
      return;
    }
    setInvalid(false);
    onReview(typed);
  };

  // The newest fact wins: a source that never left this browser is what the
  // user just did, and a stale failure from the last request under it would
  // be two complaints about one box.
  const refusal = invalid
    ? { message: t('pluginCenter.source.invalid'), hint: null }
    : inspection.phase === 'error'
      ? refusalLines(t, inspection.failure, inspection.source)
      : null;

  return (
    <form className={styles.sourceForm} onSubmit={submit}>
      <label htmlFor={inputId}>{t('pluginCenter.source.label')}</label>
      <input
        id={inputId}
        type="text"
        value={source}
        placeholder={t('pluginCenter.source.placeholder')}
        onChange={(event) => setSource(event.target.value)}
        aria-invalid={invalid || undefined}
        autoComplete="off"
        spellCheck={false}
      />
      <button
        type="submit"
        className={packStyles.primaryBtn}
        disabled={inspecting || typed === '' || !canInstall}
        // The footer prints this sentence once; a button that is off because
        // of it says so where the pointer already is.
        title={canInstall ? undefined : t('packs.remoteDisabled')}
      >
        {inspecting ? t('pluginCenter.source.reviewing') : t('pluginCenter.source.review')}
      </button>

      {refusal !== null && (
        <div className={styles.sourceError} role="alert">
          <div>{refusal.message}</div>
          {/* The names that WOULD have worked are an offer, not part of the
              complaint, so they keep the block and drop its colour. */}
          {refusal.hint !== null && (
            <div className={styles.sourceHint}>{refusal.hint}</div>
          )}
        </div>
      )}
    </form>
  );
}
