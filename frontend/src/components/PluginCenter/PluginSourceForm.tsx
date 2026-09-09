import { useId, useState, type FormEvent } from 'react';
import type { InspectionFailure, InspectionState } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import { parseGitHubSource, refusalSentence, type Translate } from './pluginStatus';
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
 * Only the OFFER is written here. The complaint is `refusalSentence`, shared
 * with the store because a refused Install on a ROW reports itself in a
 * toast: two builders for one refusal is how the box's wording and the
 * toast's drift apart. The offer stays because it is the box's alone -- a
 * list of names to type is an answer to somebody typing.
 */
function refusalLines(
  t: Translate, failure: InspectionFailure, source: string,
): { message: string; hint: string | null } {
  const message = refusalSentence(t, failure, source);
  if (failure.code !== 'unknown_catalog_name') return { message, hint: null };

  const known = list(failure.detail, 'known');
  return {
    message,
    hint: known.length === 0
      ? null
      : t('pluginCenter.source.knownNames', { known: known.join(', ') }),
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
  const errorId = useId();
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
      {/* The disclosure this form opens out of carries the name on screen
          (`PluginCenterModal`), so a visible label would print it twice; the
          field keeps it as its accessible name. */}
      <input
        type="text"
        aria-label={t('pluginCenter.source.label')}
        value={source}
        placeholder={t('pluginCenter.source.placeholder')}
        onChange={(event) => {
          setSource(event.target.value);
          // Withdrawn on the keystroke, because it was earned on one: this
          // build judged the STRING, so the moment the string changes the
          // judgement is about something that is no longer in the box.
          setInvalid(false);
        }}
        aria-invalid={invalid || undefined}
        // `role="alert"` announces a refusal when it arrives; this is what
        // says it again to somebody who tabs back into the field.
        aria-describedby={refusal === null ? undefined : errorId}
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
        <div id={errorId} className={styles.sourceError} role="alert">
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
