import styles from './SettingsPopover.module.css';

interface RowProps {
  name: string;
  /**
   * Plain text, or a fragment when a row needs a second line. Optional, and
   * left out by most rows: a description under every one of them meant the
   * popover opened onto ~17 sentences of standing prose. A row carries one
   * only when the setting's consequence cannot be read off its own name --
   * what a run does differently after the toggle moves -- or when the control
   * cannot carry the sentence itself: `title` reaches a hovering mouse and
   * nothing else, and a DISABLED control fires no pointer events, so not even
   * that. Everything else says it on the control, as a `title`.
   */
  desc?: React.ReactNode;
  ctrl: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}

/**
 * One "name / description / control" line of the settings popover.
 *
 * Extracted from SettingsPopover.tsx when the "This Server" section moved into
 * its own component (#193 item 2): both files need this row, and importing it
 * back out of SettingsPopover would have made the two modules import each
 * other.
 *
 * `onClick` widens the click target to the whole row for a pointer; it does
 * NOT make the row a control. Every row that takes one wraps a real `<button>`
 * carrying the setting's name and its `aria-pressed`, so a `role="button"` out
 * here published a second control with the same accessible name and no state
 * -- assistive tech offered two identical "Record node outputs" buttons, and
 * `getByRole('button', { name })` matched both. The tab stop went with it: the
 * row had `tabIndex={0}` and no focus style of its own, so keyboard focus
 * landed on it invisibly one stop before the button that does the same thing.
 */
export function SettingsRow({ name, desc, ctrl, onClick, disabled }: RowProps) {
  const interactive = onClick !== undefined;
  return (
    <div
      className={`${styles.row} ${desc === undefined ? styles.compact : ''} ${interactive ? styles.interactive : ''} ${disabled ? styles.disabled : ''}`}
      onClick={onClick}
    >
      <div>
        <div className={styles.name}>{name}</div>
        {desc !== undefined && <div className={styles.desc}>{desc}</div>}
      </div>
      <div className={styles.ctrl}>{ctrl}</div>
    </div>
  );
}
