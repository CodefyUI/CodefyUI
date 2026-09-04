import type { ReactNode } from 'react';
import styles from './Pill.module.css';

/** Which wash a pill wears. Names a ROLE, not a colour. */
export type PillTone = 'success' | 'warning' | 'info' | 'neutral';

export interface PillProps {
  tone: PillTone;
  /**
   * Prefix the label with a pulsing dot: something is still happening.
   *
   * A dot rather than a spinner, because it says "still going" without
   * claiming to know how far, and it stops dead under reduced motion.
   */
  pulse?: boolean;
  children: ReactNode;
}

/**
 * One status chip for the whole app: a pack's state, a plugin's state, the
 * chip on a plugin row in the sidebar.
 *
 * The tone is an attribute rather than a class so the stylesheet reads as one
 * table of washes, and so a test can assert the ROLE a pill is claiming
 * ("this is a warning") instead of a hashed CSS-module name.
 *
 * Deliberately not a button and not focusable: it reports a state the
 * controls beside it change. A pill that wants a click is a button.
 */
export function Pill({ tone, pulse = false, children }: PillProps) {
  return (
    <span className={styles.pill} data-tone={tone}>
      {pulse && (
        <span className={styles.pillDot} data-role="pulse" aria-hidden="true" />
      )}
      {children}
    </span>
  );
}
