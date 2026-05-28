import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useDialogStore } from '../../store/dialogStore';
import styles from './DialogContainer.module.css';

/**
 * Renders the currently-active confirm / prompt dialog (or nothing when
 * the store is empty). Mounted once at the App root, reads from
 * ``useDialogStore``, calls ``store.close(value)`` to resolve the
 * pending promise.
 *
 * Keyboard contract:
 * - Escape  → cancel  (close with false / null)
 * - Enter   → confirm (close with true / current input value)
 *
 * Visual: matches the HeatmapModal — dark gradient surface, teal
 * accent on the primary action, danger variant uses a warm red. Auto-
 * focuses the input (prompt) or the primary button (confirm).
 */
export function DialogContainer() {
  const active = useDialogStore((s) => s.active);
  const close = useDialogStore((s) => s.close);

  const [inputValue, setInputValue] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const confirmBtnRef = useRef<HTMLButtonElement | null>(null);

  // Reset input + focus the right element each time a new dialog opens.
  useEffect(() => {
    if (!active) return;
    let timerId: ReturnType<typeof setTimeout>;
    if (active.kind === 'prompt') {
      setInputValue(active.defaultValue ?? '');
      setValidationError(null);
      // Focus + select the input on next paint so the default value is
      // overwritable in one keystroke (matches native window.prompt UX).
      timerId = setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 0);
    } else {
      timerId = setTimeout(() => confirmBtnRef.current?.focus(), 0);
    }
    return () => clearTimeout(timerId);
  }, [active]);

  // ESC cancels.
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        close(active.kind === 'prompt' ? null : false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, close]);

  if (!active) return null;

  const variant = active.kind === 'confirm' ? active.variant ?? 'info' : 'info';
  const cancelText = active.cancelText ?? 'Cancel';
  const confirmText =
    active.confirmText ?? (active.kind === 'prompt' ? 'OK' : 'Confirm');

  function handleConfirm() {
    if (!active) return;
    if (active.kind === 'prompt') {
      const validate = active.validate;
      const err = validate ? validate(inputValue) : null;
      if (err) {
        setValidationError(err);
        return;
      }
      close(inputValue);
    } else {
      close(true);
    }
  }

  function handleCancel() {
    if (!active) return;
    close(active.kind === 'prompt' ? null : false);
  }

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    handleConfirm();
  }

  return createPortal(
    <div
      className={styles.backdrop}
      onClick={handleCancel}
      role="dialog"
      aria-modal="true"
      aria-label={active.title}
    >
      <div className={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>{active.title}</span>
        </div>
        <form onSubmit={onSubmit} className={styles.body}>
          {active.message && (
            <div className={styles.message}>{active.message}</div>
          )}
          {active.kind === 'prompt' && (
            <>
              <input
                ref={inputRef}
                type="text"
                className={styles.input}
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value);
                  if (validationError) setValidationError(null);
                }}
                placeholder={active.placeholder}
                aria-label="Dialog input"
              />
              {validationError && (
                <div className={styles.error}>{validationError}</div>
              )}
            </>
          )}
          <div className={styles.footer}>
            <button
              type="button"
              className={styles.cancelBtn}
              onClick={handleCancel}
            >
              {cancelText}
            </button>
            <button
              ref={confirmBtnRef}
              type="submit"
              className={`${styles.confirmBtn} ${variant === 'danger' ? styles.danger : ''}`}
            >
              {confirmText}
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body,
  );
}
