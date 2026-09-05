import { useMemo } from 'react';
import type { DiffFile, DiffHunk, DiffLine, DiffLineKind } from '../../utils/unifiedDiff';
import { toSplitRows } from '../../utils/unifiedDiff';
import { useI18n } from '../../i18n';
import styles from './GitDiffModal.module.css';

/** Which of the two shapes the patch is drawn in. */
export type DiffViewMode = 'unified' | 'split';

/**
 * The one character that says what happened to a line.
 *
 * Drawn, not implied by the tint: colour alone is not a way to say something
 * (WCAG 1.4.1), and these three are the same three git prints, so a reader who
 * has ever read a patch already knows them. A context line takes a space so
 * the text of every line starts on one edge.
 */
const SIGN: Record<DiffLineKind, string> = { context: ' ', add: '+', del: '-' };

/**
 * How many lines of a patch are DRAWN.
 *
 * The route caps what is fetched at 1 MiB, which is a different question: a
 * megabyte cut exactly where `diff.py` cuts it parses to about twenty thousand
 * lines, and the unified view draws four elements per line. Measured in
 * Chrome 151, building and laying out that grid took 1.3 s of frozen tab
 * -- after `setLoading(false)`, so the loading line was already gone and the
 * window was simply unresponsive -- and toggling to Side by side, which is six
 * elements per line, paid it again. A project directory holding a dataset or a
 * generated JSON reaches that with one click.
 *
 * Two thousand lines is about ten thousand elements, and far more than anybody
 * reads in a side panel. What is cut off is said in words under the last line.
 */
export const MAX_DIFF_LINES = 2000;

/**
 * One file's patch, as hunks of numbered lines.
 *
 * Plain elements with `white-space: pre` rather than a `<pre>` holding the
 * whole thing: a patch has to be a GRID -- two number columns that stay
 * aligned while the text beside them scrolls sideways -- and one `<pre>` per
 * hunk would put block elements inside a tag whose content model is phrasing.
 * The monospace and the preserved whitespace are the stylesheet's, which is
 * all a `<pre>` was ever bringing.
 *
 * Nothing here is virtualised. What keeps that honest is {@link
 * MAX_DIFF_LINES}: the number of lines DRAWN is capped, and the rest is said
 * in one sentence rather than laid out.
 */
export function DiffView({ file, mode }: { file: DiffFile; mode: DiffViewMode }) {
  const { t } = useI18n();
  // Per FILE, not per mode: the two views draw the same lines, so toggling
  // between them must not walk the patch again.
  const drawn = useMemo(() => cutToLimit(file.hunks), [file]);

  return (
    // Its own scroller. A diff is as wide as its longest line, and a modal
    // that grew with it would take the page's horizontal scrollbar with it.
    <div className={styles.diff}>
      {drawn.hunks.map((hunk) => (
        <div className={styles.hunk} key={hunk.header + String(hunk.newStart)}>
          {/* git's own `@@ -a,b +c,d @@`, section heading and all: it is what
              says where in the file the next lines are. */}
          <div className={styles.hunkHeader}>{hunk.header}</div>
          {mode === 'unified' ? <UnifiedHunk hunk={hunk} /> : <SplitHunk hunk={hunk} />}
        </div>
      ))}
      {drawn.cut && (
        // Under the last line that IS drawn, which is where a reader who has
        // scrolled that far finds it.
        <div className={styles.lineCap}>
          {t('git.diff.tooManyLines', { count: MAX_DIFF_LINES })}
        </div>
      )}
    </div>
  );
}

/**
 * The hunks up to the line limit, and whether anything was left out.
 *
 * The last hunk drawn keeps its own header and numbers and loses only the
 * lines past the cap, so the reader is never shown a hunk that claims lines it
 * does not have -- and a patch under the cap is handed back untouched, array
 * and all, so the common case allocates nothing.
 */
function cutToLimit(hunks: DiffHunk[]): { hunks: DiffHunk[]; cut: boolean } {
  let left = MAX_DIFF_LINES;
  const kept: DiffHunk[] = [];
  for (const hunk of hunks) {
    if (left <= 0) return { hunks: kept, cut: true };
    if (hunk.lines.length <= left) {
      kept.push(hunk);
      left -= hunk.lines.length;
      continue;
    }
    kept.push({ ...hunk, lines: hunk.lines.slice(0, left) });
    return { hunks: kept, cut: true };
  }
  return { hunks, cut: false };
}

/** The old and the new interleaved, which is how git writes a patch. */
function UnifiedHunk({ hunk }: { hunk: DiffHunk }) {
  return (
    <>
      {hunk.lines.map((line, index) => (
        <div className={styles.line} data-kind={line.kind} key={index}>
          <span className={styles.lineNo}>{line.oldNo ?? ''}</span>
          <span className={styles.lineNo}>{line.newNo ?? ''}</span>
          <span className={styles.lineSign}>{SIGN[line.kind]}</span>
          <span className={styles.lineText}>
            {line.text}
            {line.noNewline === true && <NoNewline />}
          </span>
        </div>
      ))}
    </>
  );
}

/**
 * The old on the left and the new on the right, paired run by run.
 *
 * Derived from the patch rather than from the two whole files: the route
 * offers both sides, at the cost of two more git reads and a second set of
 * limits, and pairing the `-`/`+` runs of each hunk is the same answer for
 * the part of the file a patch is about.
 *
 * A context line is the SAME object on both sides -- `toSplitRows` copies
 * nothing -- so which column a line is in is the row's answer and never the
 * line's identity.
 */
function SplitHunk({ hunk }: { hunk: DiffHunk }) {
  return (
    <>
      {toSplitRows(hunk).map((row, index) => (
        <div className={styles.splitRow} data-row="" key={index}>
          <Cell line={row.left} side="del" />
          <Cell line={row.right} side="add" />
        </div>
      ))}
    </>
  );
}

/**
 * One half of a split row: a number, a sign and the text, or an empty box.
 *
 * `side` picks the line NUMBER -- old on the left, new on the right -- and
 * nothing else. What the cell is tinted as comes from the line itself, which
 * is the half that knows: `toSplitRows` puts removals on the left and
 * additions on the right today, and a line that says what it is stays right
 * whatever a later pairing rule decides.
 *
 * Three cells rather than one wrapper, because the row is a grid and a box
 * around half of it would not line up with the other half; the tint is on all
 * three, so the whole half is washed.
 */
function Cell({ line, side }: { line: DiffLine | undefined; side: 'add' | 'del' }) {
  if (line === undefined) {
    // A side that has no line here at all -- one run was longer than the
    // other. Empty rather than absent, so the two columns stay in step.
    return (
      <>
        <span className={styles.lineNo} data-kind="empty" />
        <span className={styles.lineSign} data-kind="empty" />
        <span className={styles.lineText} data-kind="empty" />
      </>
    );
  }
  const kind = line.kind;
  return (
    <>
      <span className={styles.lineNo} data-kind={kind}>
        {(side === 'del' ? line.oldNo : line.newNo) ?? ''}
      </span>
      <span className={styles.lineSign} data-kind={kind}>{SIGN[kind]}</span>
      <span className={styles.lineText} data-kind={kind}>
        {line.text}
        {line.noNewline === true && <NoNewline />}
      </span>
    </>
  );
}

/**
 * git's `\ No newline at end of file`, on the line it is about.
 *
 * Without it the commonest form of this change -- an editor or a formatter
 * outside the app adding a final newline to a graph file -- draws `-}` above
 * `+}` with byte-identical text, the same tint pair as any real edit, and
 * nothing anywhere saying what differs. It is a NOTE and not content, so it
 * goes inside the text cell rather than beside it: both views are grids with a
 * fixed column count, and a fifth or seventh child would start a column of its
 * own and push every row out of step.
 */
function NoNewline() {
  const { t } = useI18n();
  return <span className={styles.noNewline}>{t('git.diff.noNewline')}</span>;
}
