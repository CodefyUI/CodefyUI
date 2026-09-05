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
  /**
   * What the meta MEANS, when what it shows is a shorthand.
   *
   * A branch row draws its tracking count as two arrows and two digits
   * (`↑1 ↓0`), which is a glance's worth of screen and nothing a reader can
   * pronounce -- so the sentence it stands for goes here, and it is what the
   * `title` says and what the row is described by. The visible half is
   * `aria-hidden` when this is given, so the two are never read one after
   * the other.
   *
   * Given ALONE (with an empty `meta`) it still carries: a branch level with
   * its upstream draws no count, and "0 to push, 0 to pull" is still the
   * answer for somebody who cannot see that there is nothing there.
   */
  metaLabel?: string;
  /**
   * Which half of the row never gives up width; the other one ellipsises.
   *
   * `meta` by default, which is a branch row (a name, and a count that is two
   * glyphs wide) and a stash row (a message somebody wrote, and the branch and
   * date it was written on). `name` is the remote list: a short name beside a
   * long URL, where the default proportion spent the name's last characters
   * on a URL that had 200px to give -- "ori..." beside a readable
   * `https://github.com/owner/repo.git`.
   *
   * See the block in the stylesheet: shrinking is weighted by the base size,
   * so a long second half takes pixels off a short first half whatever the
   * shrink factors say.
   */
  firm?: 'name' | 'meta';
  /**
   * Whether a narrow panel DROPS the meta rather than ellipsising it.
   *
   * `firm` divides a row between two halves that both have to be on screen.
   * A stash row has three: a message somebody wrote, the branch and date it
   * was written on, and git's own selector -- and at 180px that came out as
   * `ex... authte... stash@{0}`, a message cut to two characters because the
   * meta had taken its 60% and the chip cannot shrink. Two of the three fit,
   * and the message is the half the reader is looking for.
   *
   * So below the same threshold the row's verbs collapse at, the meta comes
   * out of the row -- off screen rather than deleted, so a reader still hears
   * it, and into the row's `title` as well, because a clipped span is not
   * something a pointer can be over. Above the threshold nothing changes.
   */
  metaOptional?: boolean;
  actions: RefRowAction[];
}

/**
 * One row in one reference list.
 *
 * The same box as a file row, and for the same reasons: a `min-width: 0` name
 * that ellipsises, a dimmer half beside it, the whole string in a `title`, and
 * the actions collapsed to zero width until a hover or a focus opens them --
 * so a 180px panel spends its width on the name and a 520px one shows the rest.
 *
 * Which half gives way is the caller's, through `firm`: a file row's
 * proportion is right for a name beside a directory and wrong for a name
 * beside a URL, where it left `ori...` next to a URL with room to spare.
 *
 * The actions themselves come in two shapes and CSS picks one, the way a
 * conflict row's do: "Change URL / Remove" and "Pop / Apply / Drop" are both
 * about 125px of text beside a chip that cannot shrink, which a 180px panel
 * does not have -- so below the threshold they are one 24px menu instead.
 */
export function RefRow({
  name,
  action,
  identity,
  badge,
  meta,
  metaLabel,
  firm = 'meta',
  metaOptional = false,
  actions,
}: RefRowProps) {
  const { t } = useI18n();
  const named = identity ?? name;
  // Addressable, because on a pressable row the meta is INSIDE the button --
  // see the `aria-describedby` below.
  const metaId = useId();
  // What is drawn, and what is said. They are the same string on every row
  // whose meta is already prose (a URL, "main, 2 hours ago").
  const shownMeta = meta ?? '';
  const spokenMeta = metaLabel ?? '';
  const hasMeta = shownMeta !== '' || spokenMeta !== '';
  // What the meta MEANS, whichever half it was given as.
  const saidMeta = spokenMeta === '' ? shownMeta : spokenMeta;
  const metaClass = [
    styles.rowDir,
    firm === 'name' ? '' : styles.metaFirm,
    metaOptional ? styles.metaOptional : '',
  ]
    .filter((one) => one !== '')
    .join(' ');
  const body = (
    <>
      <span
        className={`${styles.rowName} ${
          firm === 'name' ? styles.nameFirm : styles.nameElastic
        }`}
      >
        {name}
      </span>
      {hasMeta && (
        // Its own `title`, not the row's: a remote URL is one unbroken token
        // in a 180px column and is the FIRST thing here to be ellipsised, so
        // the string in full has to be reachable from the half that was cut.
        <span className={metaClass} id={metaId} title={saidMeta}>
          {spokenMeta === '' ? shownMeta : (
            <>
              {shownMeta !== '' && <span aria-hidden="true">{shownMeta}</span>}
              <span className={styles.srOnly}>{spokenMeta}</span>
            </>
          )}
        </span>
      )}
    </>
  );

  return (
    // The name lives on the ROW as well: a `title` on the button would not
    // open where the button is not the thing under the pointer.
    //
    // A meta the panel may drop joins it, on its own line: below the threshold
    // that span is a 1px box off screen, so its own `title` is unreachable and
    // the row's is the only tooltip a pointer can open. A newline rather than
    // a separator, because the two clauses are already punctuated sentences in
    // their own languages and no third one is needed to join them.
    <li
      className={styles.row}
      title={metaOptional && saidMeta !== '' ? `${name}\n${saidMeta}` : name}
    >
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
                // The destructive one keeps its cue in here too: below 380px
                // this menu is the only shape of a row's actions on screen.
                danger: one.danger,
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
 * It names the LIST it is about. `git.error.loadFail` is the status poll's
 * sentence, and inside the Stashes section it said the repository status
 * could not be read while the status was fine and only `git stash list` had
 * failed -- the wrong thing named, in the one place that knows which.
 *
 * Suppressed while the header is saying the same thing already. A backend
 * that stops answering fails the status poll and all three refs reads at
 * once, and the panel printed it in the header and again in every open
 * section -- up to four copies of one fact on a 180px panel. But only while
 * the header is really drawing THAT sentence: the header shows one line and
 * the operation's refusal wins it, so a `lastError` means `loadError` is not
 * on screen and this is the only place the failed read would be reported.
 */
export function RefError({ message, what }: { message: string | null; what: string }) {
  const { t } = useI18n();
  const loadError = useGitStore((s) => s.loadError);
  const lastError = useGitStore((s) => s.lastError);
  if (message === null) return null;
  if (loadError !== null && lastError === null) return null;
  return (
    <li className={styles.refError}>
      {t('git.error.listFail', { what, error: message })}
    </li>
  );
}

/** A list that really is empty, saying so in the section's own words. */
export function RefEmpty({ text }: { text: string }) {
  return <li className={styles.refEmpty}>{text}</li>;
}
