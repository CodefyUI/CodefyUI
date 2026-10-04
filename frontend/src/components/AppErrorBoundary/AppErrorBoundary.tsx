import { Component, useEffect, useRef, useState, type ErrorInfo, type ReactNode } from 'react';
import { useI18n, type TranslationKey } from '../../i18n';
import en from '../../i18n/locales/en';
import { suspendAutosave } from '../../store/tabStore';
import { workspaceFileName } from '../../utils/workspaceFile';
import {
  buildWorkspaceBackup,
  downloadWorkspaceFile,
  readAutosavedTabs,
  startEmptyWorkspace,
  type AutosavedTabs,
} from './workspaceBackup';
import styles from './AppErrorBoundary.module.css';

type Vars = Record<string, string | number>;

/**
 * A UI string, or its English text when the i18n store cannot give one.
 *
 * This screen is up because a render threw, and the store holding the
 * translations is among the things that might have. The English table is a
 * plain object, so reading it cannot fail that way.
 */
function say(key: TranslationKey, vars?: Vars): string {
  try {
    const text = useI18n.getState().t(key, vars);
    if (typeof text === 'string' && text !== '') return text;
  } catch {
    // English below.
  }
  let text: string = en[key];
  for (const [name, value] of Object.entries(vars ?? {})) {
    // A function replacement, as `t` uses, so `$&` in a value stays literal.
    text = text.replace(`{${name}}`, () => String(value));
  }
  return text;
}

interface ErrorText {
  /** One line: the error's name and message. Never empty. */
  message: string;
  /** The JS stack without a first line that only repeats `message`; '' when there is none. */
  stack: string;
}

/** What to show for a thrown value, which need not be an Error at all. */
function errorText(error: unknown): ErrorText {
  try {
    if (error instanceof Error) {
      const name = error.name || 'Error';
      const message = error.message ? `${name}: ${error.message}` : name;
      let stack = typeof error.stack === 'string' ? error.stack : '';
      // V8 opens the stack with the line shown above it; Firefox does not.
      if (stack === message) stack = '';
      else if (stack.startsWith(`${message}\n`)) stack = stack.slice(message.length + 1);
      return { message, stack };
    }
    return { message: String(error) || 'Error', stack: '' };
  } catch {
    // A value String() cannot convert: a throwing toString, or no prototype.
    return { message: 'Error', stack: '' };
  }
}

/** One error the boundary caught, with the component stack React gave it. */
interface Caught {
  error: unknown;
  componentStack: string;
}

/**
 * The Details text: the first error's stacks, then any error that followed
 * it, each with its message, since only the first one's is shown above.
 */
function detailsText(first: Caught, later: Caught[]): string {
  const block = (caught: Caught, withMessage: boolean) => {
    const { message, stack } = errorText(caught.error);
    const head = withMessage ? [message, stack].filter((part) => part !== '').join('\n') : stack;
    return [head, caught.componentStack.trim()].filter((part) => part !== '').join('\n\n');
  };
  const parts = [block(first, false)];
  if (later.length > 0) {
    parts.push(say('appError.details.later'));
    for (const caught of later) parts.push(block(caught, true));
  }
  return parts.filter((part) => part !== '').join('\n\n');
}

type BackupState =
  | { phase: 'idle' }
  | { phase: 'working' }
  | { phase: 'done'; downloaded: boolean; skippedReadOnly: number; unreadable: number }
  | { phase: 'failed'; key: 'appError.backup.readFailed' | 'appError.backup.makeFailed'; reason: string };

type ResetState =
  | { phase: 'idle' }
  | { phase: 'working' }
  | { phase: 'no_copy' }
  | { phase: 'not_changed' }
  | { phase: 'failed'; reason: string };

/** The lines under the buttons, from what the user has tried so far. */
function notesOf(backup: BackupState, reset: ResetState): string[] {
  const notes: string[] = [];
  if (backup.phase === 'failed') notes.push(say(backup.key, { error: backup.reason }));
  if (backup.phase === 'done') {
    if (!backup.downloaded) notes.push(say('appError.backup.empty'));
    if (backup.skippedReadOnly > 0) {
      notes.push(say('workspace.export.skippedReadOnly', { count: backup.skippedReadOnly }));
    }
    if (backup.unreadable > 0) {
      notes.push(say('appError.backup.unreadable', { count: backup.unreadable }));
    }
  }
  if (reset.phase === 'no_copy') {
    // With a complete backup in hand there is no `no_copy`, so a downloaded
    // one here is one that left tabs out: downloading it again cannot help.
    const downloaded = backup.phase === 'done' && backup.downloaded;
    notes.push(say(downloaded ? 'appError.reset.noCopyLossy' : 'appError.reset.noCopy'));
  }
  if (reset.phase === 'not_changed') notes.push(say('appError.reset.notChanged'));
  if (reset.phase === 'failed') notes.push(say('appError.reset.failed', { error: reset.reason }));
  return notes;
}

function RecoveryScreen({ caught }: { caught: Caught[] }) {
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  // One storage job at a time: a backup read while the reset writes could
  // copy half of either. A ref, so a second click lands before any re-render.
  const busy = useRef(false);
  const [backup, setBackup] = useState<BackupState>({ phase: 'idle' });
  const [reset, setReset] = useState<ResetState>({ phase: 'idle' });

  // The page under this screen is gone, and focus went with it.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  // Empty only between React switching to this screen and reporting the
  // error, which happen before the browser paints.
  const [first, ...later] = caught;
  const message = first ? errorText(first.error).message : '';
  const details = first ? detailsText(first, later) : '';

  const downloadBackup = async () => {
    if (busy.current) return;
    busy.current = true;
    setBackup({ phase: 'working' });
    try {
      let saved: AutosavedTabs;
      try {
        saved = await readAutosavedTabs();
      } catch (reason) {
        setBackup({ phase: 'failed', key: 'appError.backup.readFailed', reason: errorText(reason).message });
        return;
      }
      try {
        const now = new Date();
        const { file, skippedReadOnly, unreadable } = buildWorkspaceBackup(saved, now);
        if (file !== null) downloadWorkspaceFile(file, workspaceFileName(now));
        setBackup({ phase: 'done', downloaded: file !== null, skippedReadOnly, unreadable });
      } catch (reason) {
        setBackup({ phase: 'failed', key: 'appError.backup.makeFailed', reason: errorText(reason).message });
      }
    } finally {
      busy.current = false;
    }
  };

  const startEmpty = async () => {
    if (busy.current) return;
    // A file in the user's hands is what makes this safe. Without one, ask.
    const downloaded = backup.phase === 'done' && backup.downloaded;
    if (!downloaded && !window.confirm(say('appError.reset.confirm'))) return;
    busy.current = true;
    setReset({ phase: 'working' });
    try {
      // A file that holds every tab stands in for a copy that does not fit;
      // one that left tabs out does not, or those tabs would be stored nowhere.
      const completeBackup =
        backup.phase === 'done' &&
        backup.downloaded &&
        backup.skippedReadOnly === 0 &&
        backup.unreadable === 0;
      const outcome = await startEmptyWorkspace(new Date(), { completeBackup });
      if (outcome === 'reset') {
        window.location.reload();
        return;
      }
      setReset({ phase: outcome });
    } catch (reason) {
      setReset({ phase: 'failed', reason: errorText(reason).message });
    } finally {
      busy.current = false;
    }
  };

  // `aria-disabled`, not `disabled`: a disabled button drops the focus it
  // holds to <body>, and the click handlers above ignore the press anyway.
  const working = backup.phase === 'working' || reset.phase === 'working';

  return (
    <main className={styles.screen}>
      <div className={styles.card}>
        <h1 ref={headingRef} tabIndex={-1} className={styles.title}>
          {say('appError.title')}
        </h1>
        {message !== '' && (
          <pre className={styles.message}>
            <code>{message}</code>
          </pre>
        )}
        {details !== '' && (
          <details>
            <summary className={styles.summary}>{say('appError.details')}</summary>
            <pre className={styles.stack}>{details}</pre>
          </details>
        )}
        <p className={styles.hint}>
          {/* An empty workspace opens on the welcome screen, which has no
              Graphs panel: a new graph comes first. */}
          {say('appError.hint', {
            newGraph: say('welcome.newGraph'),
            import: say('graphs.import'),
            panel: say('sidebar.tab.graphs'),
          })}
        </p>
        <div className={styles.actions}>
          {/* First: when the crash comes from storage, a reload hits it again. */}
          <button
            type="button"
            className={styles.backupBtn}
            aria-disabled={working}
            onClick={() => void downloadBackup()}
          >
            {say('appError.backup')}
          </button>
          <button
            type="button"
            className={styles.reloadBtn}
            onClick={() => window.location.reload()}
          >
            {say('appError.reload')}
          </button>
          <button
            type="button"
            className={styles.resetBtn}
            aria-disabled={working}
            onClick={() => void startEmpty()}
          >
            {say('appError.reset')}
          </button>
        </div>
        {/* Always mounted, so a screen reader announces what lands in it. */}
        <div role="status" className={styles.status}>
          {notesOf(backup, reset).map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      </div>
    </main>
  );
}

interface Props {
  children: ReactNode;
}

interface State {
  failed: boolean;
  /** In the order React reported them: the first is the cause, the rest followed it. */
  caught: Caught[];
}

/**
 * Catches a render error anywhere in the app (#555) and shows a recovery
 * screen in place of a blank page.
 *
 * Without it React unmounts the whole tree, and Reload is the only way back.
 * When the crash comes from the autosaved workspace itself it happens again
 * after the reload, so the screen offers the backup first -- the workspace as
 * a file, read from storage rather than from the stores that may have thrown
 * -- and then an empty workspace to Import it into.
 *
 * There is no "try again": re-rendering the same state throws the same error.
 * The Node Detail modal keeps its own narrower `TabErrorBoundary`, so a
 * broken detail tab still costs only that tab.
 */
export class AppErrorBoundary extends Component<Props, State> {
  state: State = { failed: false, caught: [] };

  static getDerivedStateFromError(): Partial<State> {
    // Only the switch. Every error also reaches componentDidCatch, in order,
    // which is where the first one can stay first: a cleanup that throws
    // while the broken tree comes down is reported after the error itself.
    return { failed: true };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    try {
      // Before the next autosave: the state that crashed must not be saved
      // over the last good one, or every reload crashes the same way.
      suspendAutosave();
    } catch {
      // The screen comes up regardless.
    }
    // React has already logged the error. The component stack goes into
    // Details: it says which component threw.
    const componentStack = info.componentStack ?? '';
    this.setState((state) => ({ caught: [...state.caught, { error, componentStack }] }));
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return <RecoveryScreen caught={this.state.caught} />;
  }
}
