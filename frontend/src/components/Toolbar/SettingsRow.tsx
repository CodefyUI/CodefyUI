import styles from './SettingsPopover.module.css';

interface RowProps {
  name: string;
  desc: string;
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
 * other. The markup is unchanged — a row with no `onClick` stays
 * non-interactive (no role, no tab stop), which is what the panel's
 * read-only rows rely on.
 */
export function SettingsRow({ name, desc, ctrl, onClick, disabled }: RowProps) {
  const interactive = onClick !== undefined;
  return (
    <div
      className={`${styles.row} ${interactive ? styles.interactive : ''} ${disabled ? styles.disabled : ''}`}
      onClick={onClick}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      onKeyDown={(e) => {
        if (interactive && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick?.();
        }
      }}
    >
      <div>
        <div className={styles.name}>{name}</div>
        <div className={styles.desc}>{desc}</div>
      </div>
      <div className={styles.ctrl}>{ctrl}</div>
    </div>
  );
}
