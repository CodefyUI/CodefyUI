import type { DiffFile, DiffHunk, DiffLine, DiffLineKind } from '../../utils/unifiedDiff';
import { toSplitRows } from '../../utils/unifiedDiff';
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
 * One file's patch, as hunks of numbered lines.
 *
 * Plain elements with `white-space: pre` rather than a `<pre>` holding the
 * whole thing: a patch has to be a GRID -- two number columns that stay
 * aligned while the text beside them scrolls sideways -- and one `<pre>` per
 * hunk would put block elements inside a tag whose content model is phrasing.
 * The monospace and the preserved whitespace are the stylesheet's, which is
 * all a `<pre>` was ever bringing.
 *
 * Nothing here is virtualised: the route caps a patch at 1 MiB and says so,
 * and a megabyte of lines in one modal is a problem this build does not have.
 */
export function DiffView({ file, mode }: { file: DiffFile; mode: DiffViewMode }) {
  return (
    // Its own scroller. A diff is as wide as its longest line, and a modal
    // that grew with it would take the page's horizontal scrollbar with it.
    <div className={styles.diff}>
      {file.hunks.map((hunk) => (
        <div className={styles.hunk} key={hunk.header + String(hunk.newStart)}>
          {/* git's own `@@ -a,b +c,d @@`, section heading and all: it is what
              says where in the file the next lines are. */}
          <div className={styles.hunkHeader}>{hunk.header}</div>
          {mode === 'unified' ? <UnifiedHunk hunk={hunk} /> : <SplitHunk hunk={hunk} />}
        </div>
      ))}
    </div>
  );
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
          <span className={styles.lineText}>{line.text}</span>
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
 * The kind is the SIDE's -- a left cell is a removal and a right cell an
 * addition, unless the line is context, which both columns show unchanged.
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
  const kind: DiffLineKind = line.kind === 'context' ? 'context' : side;
  return (
    <>
      <span className={styles.lineNo} data-kind={kind}>
        {(side === 'del' ? line.oldNo : line.newNo) ?? ''}
      </span>
      <span className={styles.lineSign} data-kind={kind}>{SIGN[kind]}</span>
      <span className={styles.lineText} data-kind={kind}>{line.text}</span>
    </>
  );
}
