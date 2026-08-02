import type { ParamDefinition } from '../../types';
import { useI18n } from '../../i18n';
import { useScriptValidation } from '../../hooks/useScriptValidation';
import { CodeEditor } from './CodeEditor';
import styles from './ScriptCodeField.module.css';

/**
 * A `code` param: the editor plus its live policy verdict (core#131).
 *
 * Rendered by `ParamField`, so it appears in every param surface at once —
 * the Node Detail Modal's param column, the side config panel, and the
 * modal's wider Code tab (which passes `variant="tab"` for a taller box).
 * All of them read and write the same store param, so the two editors a
 * user can have open simultaneously stay in lockstep with no extra wiring.
 */
export interface ScriptCodeFieldProps {
  param: ParamDefinition;
  value: unknown;
  onChange: (name: string, value: unknown) => void;
  displayLabel: string;
  variant?: 'panel' | 'tab';
  /** Overrides the variant's default editor height. */
  height?: number;
}

const HEIGHTS: Record<'panel' | 'tab', number> = { panel: 210, tab: 420 };

export function ScriptCodeField({
  param,
  value,
  onChange,
  displayLabel,
  variant = 'panel',
  height,
}: ScriptCodeFieldProps) {
  const { t } = useI18n();
  const code = typeof value === 'string' ? value : String(value ?? param.default ?? '');
  const { status, result } = useScriptValidation(code);

  const rejected = status === 'done' && result !== null && !result.ok;
  const missingRun = status === 'done' && result !== null && result.ok && !result.defines_run;

  return (
    <div className={styles.field}>
      <div className={styles.labelRow}>
        <label className={styles.label}>{displayLabel}</label>
        <span
          className={`${styles.badge} ${
            rejected
              ? styles.badgeError
              : status === 'checking'
                ? styles.badgeChecking
                : status === 'done'
                  ? styles.badgeOk
                  : styles.badgeIdle
          }`}
          data-testid="script-policy-badge"
        >
          {status === 'checking'
            ? t('paramField.code.checking')
            : status === 'unavailable'
              ? t('paramField.code.unavailable')
              : rejected
                ? t('paramField.code.rejected')
                : status === 'done'
                  ? t('paramField.code.ok')
                  : ''}
        </span>
      </div>

      <CodeEditor
        value={code}
        onChange={(next) => onChange(param.name, next)}
        minHeight={height ?? HEIGHTS[variant]}
        ariaLabel={displayLabel}
        errorLine={rejected ? result?.line ?? null : null}
      />

      {rejected && result && (
        <div className={styles.error} role="alert">
          {result.line != null && (
            <span className={styles.errorLine}>
              {t('paramField.code.atLine', { line: result.line })}
            </span>
          )}
          {result.error}
        </div>
      )}

      {missingRun && (
        <div className={styles.warning}>{t('paramField.code.noRun')}</div>
      )}

      {result && result.allowed_modules.length > 0 && (
        <div className={styles.allowed} title={result.allowed_modules.join(', ')}>
          {t('paramField.code.allowed', { modules: result.allowed_modules.join(', ') })}
        </div>
      )}
    </div>
  );
}
