import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  getGitDiff,
  getGitFile,
  GitApiError,
  GIT_TIMEOUTS_S,
  type GitDiff,
} from '../../api/git';
import { useDialogStore } from '../../store/dialogStore';
import { useGitStore, type GitStoreError } from '../../store/gitStore';
import { useUIStore, type GitDiffTarget } from '../../store/uiStore';
import { useI18n, type TranslationKey } from '../../i18n';
import {
  graphDiffKind,
  summarizeGraphDiff,
  type GraphDiffSummary as GraphDiff,
} from '../../utils/graphDiff';
import { parseUnifiedDiff } from '../../utils/unifiedDiff';
import { DiffView, type DiffViewMode } from './DiffView';
import { GraphDiffSummary } from './GraphDiffSummary';
import { focusScmFallback } from './ScmHeader';
import { errorHint, errorSentence } from './scm';
import styles from './GitDiffModal.module.css';

/**
 * The cap the route puts on a patch, in the unit the sentence names.
 *
 * `MAX_PATCH_BYTES` is 1 MiB (`backend/app/core/git/diff.py`), and what comes
 * back past it is the first megabyte with `truncated` set. The number is here
 * rather than in the locale file so both languages say the same one.
 */
const PATCH_CAP_KB = 1024;

/** What the header calls each pair of sides. */
const SCOPE_KEY: Record<GitDiffTarget['scope'], TranslationKey> = {
  worktree: 'git.diff.scope.worktree',
  index: 'git.diff.scope.index',
  commit: 'git.diff.scope.commit',
};

/** The refs `GET /file` takes by name; everything else has to be a commit id. */
const NAMED_REFS = new Set(['HEAD', 'index', 'worktree']);

/** `validate_sha` on the server: seven to forty hexadecimal characters. */
const SHA = /^[0-9a-f]{7,40}$/;

/**
 * How many characters of a commit id a heading shows.
 *
 * git's own abbreviation is seven, and forty in a title is a heading nobody
 * reads to the end of.
 */
const SHORT_SHA = 7;

/**
 * One file's change, in whichever pair of sides the row that opened it meant.
 *
 * Mounted once at the app root and driven by `uiStore.gitDiff`, like every
 * other modal in `App.tsx`: the row that opens it is inside a scrolling panel
 * and this is a portal at the top of the document, so the state between them
 * cannot be a prop.
 *
 * The body is remounted per TARGET (the key below). A diff is a read about one
 * file at one moment, and reusing the frame for the next one would leave the
 * previous patch on screen while the new read was out.
 */
export function GitDiffModal() {
  const target = useUIStore((s) => s.gitDiff);
  if (target === null) return null;
  return (
    <GitDiffBody
      key={`${target.scope}:${target.sha ?? ''}:${target.path}`}
      target={target}
    />
  );
}

function GitDiffBody({ target }: { target: GitDiffTarget }) {
  const { t } = useI18n();
  const close = useUIStore((s) => s.closeGitDiff);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const stderrId = useId();

  const [diff, setDiff] = useState<GitDiff | null>(null);
  const [summary, setSummary] = useState<GraphDiff | null>(null);
  const [error, setError] = useState<GitStoreError | null>(null);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<DiffViewMode>('unified');
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void (async () => {
      try {
        const answer = await getGitDiff({
          path: target.path,
          scope: target.scope,
          sha: target.sha,
        });
        if (cancelled) return;
        // The graph strip is a SECOND pair of reads, and only for the two
        // documents it can describe. Its failures are its own: a summary
        // nobody can build is a summary that is not drawn, never a diff that
        // is not shown -- which is why both land in one render, and why the
        // strip cannot appear a moment later and push the patch down.
        const kind = graphDiffKind(target.path);
        const sides = kind === null ? null : await readBothSides(target.path, answer);
        if (cancelled) return;
        setDiff(answer);
        setError(null);
        if (kind !== null && sides !== null) {
          setSummary(summarizeGraphDiff(sides.old, sides.new, kind));
        }
      } catch (err) {
        if (cancelled) return;
        setDiff(null);
        setError(readError(err, t));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [t, target]);

  // Focus starts inside the window and goes back where it came from. Not a
  // focus trap -- Tab still walks out into the page underneath, exactly as it
  // does in both Centers and the template gallery.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    surfaceRef.current?.focus();
    return () => {
      if (previouslyFocused && previouslyFocused.isConnected) {
        previouslyFocused.focus();
        return;
      }
      // The row that opened this window can be GONE by the time it closes:
      // the fifteen-second poll re-renders the file groups, and a row whose
      // key changes -- the file was staged or settled from elsewhere -- takes
      // the button that had focus with it. Falling to `<body>` there would
      // start the next Tab at the top of the page, so this lands on the panel
      // instead, the same answer the file rows already give themselves.
      focusScmFallback();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // A confirm dialog can sit on top of this window (10002 is below this
      // rung, but the dialog host owns the key while it is open), and the
      // shortcuts modal renders at 9000 -- BEHIND this one, with no Escape
      // handler of its own, so swallowing the key is what keeps one press
      // from closing this window and leaving a shortcuts window nobody can
      // see.
      if (useDialogStore.getState().active !== null) return;
      if (useUIStore.getState().shortcutsModalOpen) return;
      e.preventDefault();
      close();
      // Both Centers stand down while this window is open, and they read
      // `gitDiff` from the store -- which `close()` has just emptied. Without
      // this, one press would close this window AND the Center underneath it
      // whenever that Center registered its listener first.
      e.stopImmediatePropagation();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close]);

  const file = useMemo(
    () => (diff === null || diff.binary ? null : parseUnifiedDiff(diff.patch)),
    [diff],
  );

  const scopeLabel = t(SCOPE_KEY[target.scope], {
    sha: (target.sha ?? '').slice(0, SHORT_SHA),
  });
  // "No changes" is what GIT said, never what this parser managed to read. The
  // two used to be one test, and a shape the parser did not know -- which is
  // what an unmerged path's combined patch was until it was taught one -- came
  // out as a confident "No changes" over a file full of conflict markers. A
  // patch with bytes in it that yields no hunks is now shown as the text it is.
  const empty = file !== null && diff !== null && diff.patch.trim() === '';
  // Offered only where there is a patch to switch. A conflicted file has no
  // index copy until it is settled -- there is no stage 0 -- so one of the two
  // columns could only be a guess; and while the read is out, over a refusal,
  // over "Binary file; no text diff.", over "No changes" and over a patch that
  // yielded no hunks, the body is one sentence and pressing either radio
  // changes nothing on screen. A choice that does nothing is not a choice.
  const canSplit = target.conflicted !== true
    && error === null
    && !loading
    && !empty
    && file !== null
    && file.hunks.length > 0;
  const stderr = error?.stderr ?? null;

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div
        ref={surfaceRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-label={t('git.diff.title', { path: target.path })}
        tabIndex={-1}
      >
        <div className={styles.header}>
          <div className={styles.titleBlock}>
            <div className={styles.title}>{t('git.diff.title', { path: target.path })}</div>
            <div className={styles.subtitle}>{scopeLabel}</div>
          </div>
          {canSplit && <ViewSwitch mode={mode} onSelect={setMode} />}
          <button
            type="button"
            className={styles.closeBtn}
            onClick={close}
            title={t('git.diff.close')}
            aria-label={t('git.diff.close')}
          >
            &#215;
          </button>
        </div>

        <div className={styles.body}>
          {error !== null ? (
            <div className={styles.stateMessage}>
              {/* One sentence, and git's own tail behind a disclosure -- the
                  shape the header's error line already uses. It stays HERE
                  and never on that line: a read nobody pressed a button for
                  must not replace the refusal the user was reading.

                  `role="alert"` on the SENTENCE and nothing else, for the
                  reason the header's line records: this window opens on a
                  loading line, and a refusal that replaces it inside a
                  `role="dialog"` is announced by nobody -- a dialog body is
                  not a live region. An alert re-announces on any change in
                  its subtree, so the hint, the Details toggle and the `<pre>`
                  stay outside it; opening the stderr must not read the whole
                  refusal out again. */}
              <div className={styles.errorText} role="alert">{errorSentence(error, t)}</div>
              {error.hint !== null && <div className={styles.errorHint}>{error.hint}</div>}
              {stderr !== null && stderr !== '' && (
                <>
                  <button
                    type="button"
                    className={styles.linkBtn}
                    aria-expanded={detailsOpen}
                    aria-controls={stderrId}
                    onClick={() => setDetailsOpen((was) => !was)}
                  >
                    {t('git.error.details')}
                  </button>
                  {/* Mounted whenever there is one and HIDDEN while it is
                      closed: an `aria-controls` pointing at an element that
                      is not in the document names nothing. */}
                  <pre className={styles.stderr} id={stderrId} hidden={!detailsOpen}>
                    {stderr}
                  </pre>
                </>
              )}
            </div>
          ) : loading ? (
            <div className={styles.stateMessage}>{t('git.diff.loading')}</div>
          ) : diff === null ? null : diff.binary ? (
            <div className={styles.stateMessage}>{t('git.diff.binary')}</div>
          ) : empty ? (
            <div className={styles.stateMessage}>{t('git.diff.empty')}</div>
          ) : (
            <>
              {summary !== null && <GraphDiffSummary summary={summary} />}
              {/* In BOTH views: the split one is derived from this same
                  patch, so what was cut off is missing from either. */}
              {diff.truncated && (
                <p className={styles.truncated}>
                  {t('git.diff.truncated', { kb: PATCH_CAP_KB })}
                </p>
              )}
              {file !== null && (file.hunks.length > 0 ? (
                <DiffView file={file} mode={mode} />
              ) : (
                <pre className={styles.rawPatch}>{diff.patch}</pre>
              ))}
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** The two views, in the order they are drawn. */
const VIEW_MODES: readonly DiffViewMode[] = ['unified', 'split'];

/** What each one is called. */
const VIEW_KEY: Record<DiffViewMode, TranslationKey> = {
  unified: 'git.diff.unified',
  split: 'git.diff.split',
};

/**
 * One choice with two answers, as a real radio group.
 *
 * `role="radiogroup"` is a promise about the keyboard as well as about the
 * name: ONE tab stop, and the arrow keys move and choose inside it. It was
 * two plain buttons wearing `role="radio"`, so a reader who heard "radio
 * button, not checked" and reached for the arrow keys got nothing -- a role
 * describing a widget the component was not. The roving `tabIndex` and the
 * four arrow keys below are that contract, written once for the whole group.
 *
 * The name is a key of its own (`git.diff.view`) because a group needs one:
 * the two radios carry the vocabulary of the ANSWERS, and neither says what
 * the question is.
 */
function ViewSwitch({
  mode,
  onSelect,
}: {
  mode: DiffViewMode;
  onSelect: (next: DiffViewMode) => void;
}) {
  const { t } = useI18n();
  const groupRef = useRef<HTMLDivElement | null>(null);

  const move = (delta: number) => {
    const at = VIEW_MODES.indexOf(mode);
    const to = (at + delta + VIEW_MODES.length) % VIEW_MODES.length;
    onSelect(VIEW_MODES[to]);
    // Focus follows the choice, which is what makes the group one tab stop:
    // the radios are in the order of `VIEW_MODES`, so the one to focus is the
    // one at that index. Focusing it does not wait on the re-render.
    groupRef.current
      ?.querySelectorAll<HTMLButtonElement>('[role="radio"]')[to]
      ?.focus();
  };

  return (
    <div
      ref={groupRef}
      className={styles.views}
      role="radiogroup"
      aria-label={t('git.diff.view')}
      onKeyDown={(e) => {
        const forward = e.key === 'ArrowRight' || e.key === 'ArrowDown';
        const back = e.key === 'ArrowLeft' || e.key === 'ArrowUp';
        if (!forward && !back) return;
        // Before anything else: the arrows scroll the body underneath, and a
        // group that both moves and scrolls moves twice as far as it looks.
        e.preventDefault();
        move(forward ? 1 : -1);
      }}
    >
      {VIEW_MODES.map((one) => (
        <button
          key={one}
          type="button"
          role="radio"
          aria-checked={mode === one}
          // The chosen one is the group's single tab stop; Tab leaves the
          // group rather than walking to the answer nobody picked.
          tabIndex={mode === one ? 0 : -1}
          className={styles.viewButton}
          onClick={() => onSelect(one)}
        >
          {t(VIEW_KEY[one])}
        </button>
      ))}
    </div>
  );
}

/** Both whole sides of a graph file, or null when one of them cannot be had. */
async function readBothSides(
  path: string,
  answer: GitDiff,
): Promise<{ old: string | null; new: string | null } | null> {
  // `oldMissing` is the server saying that side does not exist at all: an
  // untracked file has no index copy, and a root commit has no parent.
  const oldSide = fileRefFor(answer.oldMissing ? null : answer.oldRef);
  const newSide = fileRefFor(answer.newMissing ? null : answer.newRef);
  // A side whose ref this build cannot name is not a side to guess at:
  // reading the new one twice would call a real edit "no logic change", and
  // treating it as absent would report every node in the file as added.
  if (oldSide === undefined || newSide === undefined) return null;
  const [before, after] = await Promise.all([
    readSide(path, oldSide),
    readSide(path, newSide),
  ]);
  if (before === undefined || after === undefined) return null;
  return { old: before, new: after };
}

/**
 * The ref `GET /file` can be asked for.
 *
 * `null` in and `null` out is a side that does not exist -- an added file, a
 * root commit -- which the summary takes as an empty side. `undefined` is a
 * ref this build cannot name, which is not the same thing at all.
 *
 * `<sha>^` is the case worth spelling out: the diff response names the old
 * side of a commit in git's own notation, and `/file`'s grammar is HEAD,
 * index, worktree or a bare commit id (`validate_sha`). The parent's own id
 * is in the log the history row was drawn from, so that is where it comes
 * from -- and where the log no longer has that commit (a page dropped by a
 * refresh), the answer is no summary rather than a wrong one.
 */
function fileRefFor(ref: string | null): string | null | undefined {
  if (ref === null) return null;
  if (NAMED_REFS.has(ref)) return ref;
  const lowered = ref.toLowerCase();
  if (SHA.test(lowered)) return lowered;
  if (lowered.endsWith('^')) {
    const child = lowered.slice(0, -1);
    if (SHA.test(child)) {
      const commit = useGitStore.getState().log.commits.find((one) => one.sha === child);
      const parent = commit?.parents[0];
      if (parent !== undefined) return parent;
    }
  }
  return undefined;
}

/**
 * One whole side, `null` where the file is not there, `undefined` where the
 * read is no basis for a summary.
 *
 * A 404 is an ANSWER -- the file was added here, or deleted -- and
 * `summarizeGraphDiff` takes a null side for exactly that. Everything else
 * means the summary would be built on something that is not the file: a
 * refusal, a binary blob, or a side the 2 MiB cap returned unread, which
 * would parse as broken JSON and report "could not parse" about a graph that
 * is perfectly well formed.
 */
async function readSide(path: string, ref: string | null): Promise<string | null | undefined> {
  if (ref === null) return null;
  try {
    const side = await getGitFile({ path, ref });
    if (side.binary || side.truncated) return undefined;
    return side.text;
  } catch (err) {
    if (err instanceof GitApiError && err.code === 'not_found') return null;
    return undefined;
  }
}

/**
 * A refusal this window can draw, from whatever was thrown.
 *
 * The store's own `toStoreError` in the shape a READ needs, rather than the
 * store's copy: that one is private and its whole job is to WRITE the error
 * into the panel, which is exactly what a read refusal must not do (R12).
 * `read` is the bucket every one of these four routes runs under, so it is
 * the number a timeout's sentence carries.
 */
function readError(
  err: unknown,
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string,
): GitStoreError {
  if (!(err instanceof GitApiError)) {
    return {
      code: 'unknown',
      message: err instanceof Error ? err.message : String(err),
      hint: null,
      stderr: null,
      op: null,
    };
  }
  return {
    code: err.code,
    message:
      err.code === 'timeout'
        ? t('git.error.timeout', { seconds: GIT_TIMEOUTS_S.read })
        : err.message,
    hint: errorHint(err.code, err.hint, t),
    stderr: err.stderr,
    op: null,
  };
}
