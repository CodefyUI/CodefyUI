import styles from './ProgressBar.module.css';

export interface ProgressBarProps {
  /** 0..100, or null for indeterminate. Values are clamped. */
  value: number | null;
  /** Accessible name (required: the bar is a standalone widget). */
  label: string;
  /** Fill colour role. Defaults to 'accent'. */
  tone?: 'accent' | 'info' | 'success' | 'warning' | 'error';
  /** Track thickness: 'sm' is 4px (inline, beside a row), 'md' is 8px. */
  size?: 'sm' | 'md';
  /** Render the rounded percentage to the right of the track. */
  showValue?: boolean;
  className?: string;
}

/**
 * One accessible progress bar for the whole app — per-item downloads, an
 * overall job bar, and the indeterminate bar shown while the server restarts.
 *
 * Determinate and indeterminate are one component on purpose: a download that
 * has no total yet has to become a real percentage mid-flight without the
 * caller swapping widgets (which would move focus and re-announce the widget).
 *
 * A null (or NaN) value renders as indeterminate: `aria-valuenow` is omitted,
 * which is how ARIA spells "progress unknown", and the root is `aria-busy`.
 * Sending 0 instead would announce "0 percent", i.e. a wrong number rather
 * than no number.
 */
export function ProgressBar({
  value,
  label,
  tone = 'accent',
  size = 'md',
  showValue = false,
  className,
}: ProgressBarProps) {
  // NaN reaches here from an upstream `downloaded / total` with no total yet;
  // it is "unknown", not zero. Infinity still clamps to a full bar.
  const clamped =
    value === null || Number.isNaN(value) ? null : Math.min(100, Math.max(0, value));
  const indeterminate = clamped === null;

  const rootClass = [styles.root, styles[size], className].filter(Boolean).join(' ');
  const fillClass = [
    styles.fill,
    styles[`tone_${tone}`],
    indeterminate ? styles.indeterminate : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={rootClass} aria-busy={indeterminate || undefined}>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={clamped ?? undefined}
        className={styles.track}
      >
        <div
          className={fillClass}
          style={indeterminate ? undefined : { width: `${clamped}%` }}
        />
      </div>
      {showValue && !indeterminate && (
        /* Hidden from assistive tech: aria-valuenow above already carries this
           number, and reading it twice in a row is noise. */
        <span className={styles.value} aria-hidden="true">
          {Math.round(clamped)}%
        </span>
      )}
    </div>
  );
}
