import { useToastStore, type ToastType } from '../../store/toastStore';
import styles from './Toast.module.css';

const TYPE_ICONS: Record<ToastType, string> = {
  success: '\u2713',
  error: '\u2717',
  info: '\u24D8',
  warning: '!',
};

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  const removeToast = useToastStore((s) => s.removeToast);

  if (toasts.length === 0) return null;

  return (
    <div className={styles.container}>
      {toasts.map((toast) => (
        <div key={toast.id} className={`${styles.toast} ${styles[toast.type]}`}>
          <span className={styles.icon}>{TYPE_ICONS[toast.type]}</span>
          <span className={styles.message}>{toast.message}</span>
          {toast.action && (
            <button
              type="button"
              className={styles.action}
              onClick={() => {
                toast.action?.onClick();
                // Dismiss after running, not instead of it. An action toast
                // never times out (it is an error, or sticky); leaving it up
                // would offer to open a panel that is already on screen.
                removeToast(toast.id);
              }}
            >
              {toast.action.label}
            </button>
          )}
          <button type="button" className={styles.close} onClick={() => removeToast(toast.id)}>
            &times;
          </button>
        </div>
      ))}
    </div>
  );
}
