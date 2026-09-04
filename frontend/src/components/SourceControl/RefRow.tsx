import { useId } from 'react';
import { useGitStore } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { ActionMenu } from '../shared/ActionMenu';
import { MoreHorizontalIcon } from '../shared/Icons';
import styles from './SourceControl.module.css';

/** One hover/focus action at the end of a reference row. */
export interface RefRowAction {
  /** Stable key; also what a test can hang a query off. */
  id: string;
  label: string;
  /** Draws it as the destructive one, which each list has at most one of. */
  danger?: boolean;
  onSelect: () => void;
}

export interface RefRowProps {
  /** What the row IS -- a branch name, a remote name, a stash message. */
  name: string;
  /**
   * What pressing the name does, or null when the name is only a name.
   *
   * `label` is the whole sentence (`Switch to work`), because a list twenty
   * rows deep whose buttons are all called "Switch" is twenty buttons a
   * reader cannot tell apart -- and it is the tooltip too, so a pointer gets
   * the same answer.
   */
  action?: { label: string; onSelect: () => void } | null;
  /**
   * What the row's ACTIONS are named after, when `name` is not unique.
   *
   * A branch name and a remote name are unique by construction; a stash
   * message is not -- git writes "WIP on main: <sha> <subject>" for every
   * stash nobody named, so a list of them would otherwise carry three buttons
   * all called "Drop WIP on main...". This has to be a string that is on
   * screen too, which is why the stash list passes its `badge`.
   */
  identity?: string;
  /** A short chip after the name: "Current", a stash's own selector. */
  badge?: string | null;
  /** The dimmer second half: ahead/behind, a URL, how long ago. */
  meta?: string | null;
  actions: RefRowAction[];
}

/**
 * One row in one reference list.
 *
 * The same box as a file row, and for the same reasons: a `min-width: 0` name
 * that ellipsises, a dimmer half beside it that gives way first, the whole
 * string in a `title`, and the actions collapsed to zero width until a hover
 * or a focus opens them -- so a 180px panel spends its width on the name and
 * a 520px one shows the rest.
 *
 * The actions themselves come in two shapes and CSS picks one, the way a
 * conflict row's do: "Change URL / Remove" and "Pop / Apply / Drop" are both
 * about 125px of text beside a chip that cannot shrink, which a 180px panel
 * does not have -- so below the threshold they are one 24px menu instead.
 */
export function RefRow({ name, action, identity, badge, meta, actions }: RefRowProps) {
  const { t } = useI18n();
  const named = identity ?? name;
  // Addressable, because on a pressable row the meta is INSIDE the button --
  // see the `aria-describedby` below.
  const metaId = useId();
  const hasMeta = meta !== null && meta !== undefined && meta !== '';
  const body = (
    <>
      <span className={styles.rowName}>{name}</span>
      {hasMeta && (
        // Its own `title`, not the row's: a remote URL is one unbroken token
        // in a 180px column and is the FIRST thing here to be ellipsised, so
        // the string in full has to be reachable from the half that was cut.
        <span className={styles.rowDir} id={metaId} title={meta}>{meta}</span>
      )}
    </>
  );

  return (
    // The name lives on the ROW as well: a `title` on the button would not
    // open where the button is not the thing under the pointer.
    <li className={styles.row} title={name}>
      {action === null || action === undefined ? (
        <span className={styles.refNameStatic}>{body}</span>
      ) : (
        <button
          type="button"
          className={styles.openButton}
          aria-label={action.label}
          // The meta is inside this button, and an `aria-label` REPLACES the
          // name a button would otherwise take from its own text -- so
          // "2 to push, 3 to pull" was on screen and announced to nobody. It
          // is the description instead: the name stays the sentence a reader
          // can act on ("Switch to work"), and the tracking half follows it.
          aria-describedby={hasMeta ? metaId : undefined}
          title={action.label}
          onClick={action.onSelect}
        >
          {body}
        </button>
      )}
      {badge !== null && badge !== undefined && (
        <span className={styles.refBadge}>{badge}</span>
      )}
      <div className={styles.rowActions}>
        <div className={styles.rowChoices}>
          {actions.map((one) => (
            <button
              key={one.id}
              type="button"
              className={
                one.danger === true
                  ? `${styles.rowAction} ${styles.dangerAction}`
                  : styles.rowAction
              }
              // The verb NAMES the row it acts on -- see `FileRow`.
              aria-label={`${one.label} ${named}`}
              title={one.label}
              onClick={one.onSelect}
            >
              {one.label}
            </button>
          ))}
        </div>
        {actions.length > 0 && (
          // The same actions in one 24px square, for a panel too narrow for
          // the verbs -- see `.rowChoices` in the stylesheet. The trigger
          // carries the row's identity, so the entries inside it do not have
          // to say it a second time.
          <div className={styles.rowMenu}>
            <ActionMenu
              label={`${t('git.action.more')} ${named}`}
              items={actions.map((one) => ({
                id: one.id,
                label: one.label,
                onSelect: one.onSelect,
              }))}
              align="end"
              className={styles.iconButton}
            >
              <MoreHorizontalIcon size={13} />
            </ActionMenu>
          </div>
        )}
      </div>
    </li>
  );
}

/**
 * Why a list is not on screen.
 *
 * Inside the section rather than on the header's error line: these reads are
 * the panel's own -- the poll refreshes every open section every fifteen
 * seconds -- and a failure nobody asked for must not replace the refusal the
 * user was reading. An open, empty section with no reason given would be
 * worse than both, which is why it is reported at all.
 *
 * Except while the header is saying it already. A backend that stops answering
 * fails the status poll and all three refs reads with one sentence, and the
 * panel printed it in the header and again in every open section -- up to four
 * copies of one fact on a 180px panel. `loadError` is that same server, said
 * once, above.
 */
export function RefError({ message }: { message: string | null }) {
  const { t } = useI18n();
  const loadError = useGitStore((s) => s.loadError);
  if (message === null || loadError !== null) return null;
  return (
    <li className={styles.refError}>{t('git.error.loadFail', { error: message })}</li>
  );
}

/** A list that really is empty, saying so in the section's own words. */
export function RefEmpty({ text }: { text: string }) {
  return <li className={styles.refEmpty}>{text}</li>;
}
