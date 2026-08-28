import { useEffect, useRef } from 'react';
import type { PackLogLine } from '../../store/packStore';
import { useI18n } from '../../i18n';
import styles from './PackCenterModal.module.css';

/** How close to the bottom still counts as "following the log". */
const STICK_PX = 24;

export interface PackLogTailProps {
  lines: PackLogLine[];
  ariaLabel: string;
}

/**
 * The install transcript: pip's own output, the step announcements and any
 * error, verbatim and in the server's English.
 *
 * Deliberately not translated. It is a transcript of what ran — the pip line
 * that failed is the line the user pastes into a search box — and the STEPS
 * around it are what the UI phrases in the reader's language.
 *
 * Sticks to the bottom only while the reader is already there. Scrolling up to
 * read the line that failed and being yanked back down by the next progress
 * message is the behaviour this guards against.
 */
export function PackLogTail({ lines, ariaLabel }: PackLogTailProps) {
  const { t } = useI18n();
  const boxRef = useRef<HTMLDivElement | null>(null);
  const stuckRef = useRef(true);

  useEffect(() => {
    const box = boxRef.current;
    if (box && stuckRef.current) box.scrollTop = box.scrollHeight;
  }, [lines]);

  return (
    <div
      ref={boxRef}
      className={styles.log}
      role="log"
      aria-live="polite"
      aria-relevant="additions"
      aria-label={ariaLabel}
      onScroll={() => {
        const box = boxRef.current;
        if (!box) return;
        stuckRef.current =
          box.scrollHeight - box.scrollTop - box.clientHeight <= STICK_PX;
      }}
    >
      {lines.length === 0 ? (
        <div className={styles.logEmpty}>{t('packs.activity.logEmpty')}</div>
      ) : (
        lines.map((line) => (
          <div key={line.seq} className={styles.logLine} data-kind={line.kind}>
            {line.text}
          </div>
        ))
      )}
    </div>
  );
}
