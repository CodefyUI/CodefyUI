/**
 * A unified patch, parsed into the hunks and lines the diff modal renders.
 *
 * `GET /api/git/diff` answers with ONE file's patch, exactly as git printed
 * it (`backend/app/core/git/diff.py:diff`), and this module is the whole of
 * the client's understanding of it: the unified view walks `DiffHunk.lines`,
 * and the side-by-side view walks `toSplitRows(hunk)`. Deriving the split
 * view from the patch here is what lets the tab show two columns without
 * `@codemirror/merge` and without a second `blobs=1` round trip.
 *
 * The patch it is handed is not always well formed. `diff.py` caps the
 * response at one mebibyte and hands over the prefix of the BYTES, so a
 * routine large diff ends mid-hunk, mid-line, or mid-header. Nothing in here
 * throws: a hunk that was cut short keeps the numbers git wrote in its
 * header and the lines that actually arrived, and a header that was cut in
 * half is dropped. The caller shows `git.diff.truncated` from the response's
 * own flag; it never has to infer damage from a parse failure.
 *
 * Line endings: git separates patch lines with LF and leaves the file's own
 * bytes alone, so a CRLF file arrives as lines whose text ends in CR. That
 * CR is kept in `DiffLine.text` -- a change that is ONLY a line ending is a
 * real change, and stripping it here would render both sides identical. The
 * structural lines (`diff --git`, `---`, `+++`, `@@`) are read with a
 * trailing CR tolerated, so a patch that made a round trip through something
 * that rewrote its separators still yields its hunks.
 */

export type DiffLineKind = 'context' | 'add' | 'del';

export interface DiffLine {
  kind: DiffLineKind;
  /** The line without its one-character prefix. A CR of a CRLF file is kept. */
  text: string;
  /** 1-based line number on the old side; absent on an added line. */
  oldNo?: number;
  /** 1-based line number on the new side; absent on a removed line. */
  newNo?: number;
  /**
   * git wrote `\ No newline at end of file` under this line. Set on the line
   * the note is about rather than emitted as a line of its own: counted as a
   * line it would shift every number below it, and rendered as one it would
   * read as content.
   */
  noNewline?: true;
}

export interface DiffHunk {
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  /** The whole `@@ ... @@` line, section heading included, for the separator. */
  header: string;
  lines: DiffLine[];
}

export interface DiffFile {
  /** `/dev/null` for an added file; empty when the patch carried no header. */
  oldPath: string;
  /** `/dev/null` for a deleted file; empty when the patch carried no header. */
  newPath: string;
  hunks: DiffHunk[];
}

/** One row of the side-by-side view. A missing side is an empty cell. */
export interface SplitRow {
  left?: DiffLine;
  right?: DiffLine;
}

const HUNK_HEADER = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$/;
const GIT_HEADER = 'diff --git ';
/** `"quoted path"` or one run of non-space, twice, and nothing else. */
const GIT_HEADER_PAIR = /^("(?:[^"\\]|\\.)*"|\S+) ("(?:[^"\\]|\\.)*"|\S+)$/;
const DEV_NULL = '/dev/null';

/** The single-letter escapes git writes inside a quoted path, as bytes. */
const ESCAPE_BYTES: Record<string, number | undefined> = {
  a: 0x07, b: 0x08, f: 0x0c, n: 0x0a, r: 0x0d, t: 0x09, v: 0x0b,
  '"': 0x22, '\\': 0x5c,
};

/**
 * Parse one file's unified patch.
 *
 * A second `diff --git` ends the parse: the route diffs one path per request,
 * so a second file means the response is not what this was promised, and
 * folding its lines into the first file's hunks would be worse than ignoring
 * them.
 *
 * A row that is the empty string is never content -- git writes a blank
 * context line as a single space -- so the element a trailing newline leaves
 * behind falls through the same door as any other row this does not
 * recognise, and needs no trimming of its own.
 */
export function parseUnifiedDiff(patch: string): DiffFile {
  const rows = patch.split('\n');
  const hunks: DiffHunk[] = [];
  let hunk: DiffHunk | null = null;
  let oldNo = 0;
  let newNo = 0;
  let started = false;
  let seenGitHeader = false;
  let headerOld: string | null = null;
  let headerNew: string | null = null;
  let gitOld: string | null = null;
  let gitNew: string | null = null;

  for (const row of rows) {
    const line = row.endsWith('\r') ? row.slice(0, -1) : row;

    const opened = HUNK_HEADER.exec(line);
    if (opened !== null) {
      hunk = {
        oldStart: Number(opened[1]),
        oldLines: opened[2] === undefined ? 1 : Number(opened[2]),
        newStart: Number(opened[3]),
        newLines: opened[4] === undefined ? 1 : Number(opened[4]),
        header: line,
        lines: [],
      };
      oldNo = hunk.oldStart;
      newNo = hunk.newStart;
      hunks.push(hunk);
      started = true;
      continue;
    }

    if (line.startsWith(GIT_HEADER)) {
      if (seenGitHeader) break;
      seenGitHeader = true;
      const rest = line.slice(GIT_HEADER.length);
      const pair = GIT_HEADER_PAIR.exec(rest);
      if (pair !== null) {
        gitOld = stripSidePrefix(unquotePath(pair[1]), 'a/');
        gitNew = stripSidePrefix(unquotePath(pair[2]), 'b/');
      } else {
        // git quotes a path for a control character, a quote or a backslash
        // -- never for a space -- so `a/my image.png b/my image.png` is four
        // tokens and the pair above cannot split it. Both sides of a
        // non-rename header are the SAME path, so the midpoint is exact.
        // Anything else (a rename of two unequal names) keeps no path at all
        // rather than half of one and half of the other.
        const middle = (rest.length - 1) / 2;
        if (Number.isInteger(middle) && rest.charAt(middle) === ' ') {
          gitOld = stripSidePrefix(rest.slice(0, middle), 'a/');
          gitNew = stripSidePrefix(rest.slice(middle + 1), 'b/');
        }
      }
      continue;
    }

    if (hunk !== null) {
      const prefix = row.charAt(0);
      if (prefix === ' ' || prefix === '+' || prefix === '-') {
        const kind: DiffLineKind = prefix === '+' ? 'add' : prefix === '-' ? 'del' : 'context';
        const entry: DiffLine = { kind, text: row.slice(1) };
        if (kind !== 'add') {
          entry.oldNo = oldNo;
          oldNo += 1;
        }
        if (kind !== 'del') {
          entry.newNo = newNo;
          newNo += 1;
        }
        hunk.lines.push(entry);
        continue;
      }
      if (prefix === '\\') {
        // Matched on the backslash alone, not on the English sentence after
        // it: git translates that sentence in a localized build.
        const previous = hunk.lines[hunk.lines.length - 1];
        if (previous !== undefined) previous.noNewline = true;
        continue;
      }
      // Anything else -- trailing junk, or a hunk header the byte cap cut in
      // half -- closes the hunk. What arrived before it is kept.
      hunk = null;
      continue;
    }

    if (started) continue;
    if (line.startsWith('--- ')) {
      headerOld = headerPath(line.slice(4), 'a/');
      continue;
    }
    if (line.startsWith('+++ ')) {
      headerNew = headerPath(line.slice(4), 'b/');
    }
  }

  return {
    // The `diff --git` line is the fallback for a patch with no `---`/`+++`
    // pair, which is what a binary file's patch is.
    oldPath: headerOld ?? gitOld ?? '',
    newPath: headerNew ?? gitNew ?? '',
    hunks,
  };
}

/**
 * Pair one hunk's removals with the additions that follow them, for the
 * side-by-side view.
 *
 * A context line occupies both sides of one row. Each run of removals is
 * paired index by index with the run of additions immediately after it, and
 * whichever run is longer spills into rows whose other side is empty -- which
 * is what makes a five-line edit read as five rows rather than as a block of
 * removals above a block of additions.
 *
 * The rows reference the hunk's own `DiffLine` objects; nothing is copied.
 */
export function toSplitRows(hunk: DiffHunk): SplitRow[] {
  const rows: SplitRow[] = [];
  const { lines } = hunk;
  let i = 0;
  while (i < lines.length) {
    if (lines[i].kind === 'context') {
      rows.push({ left: lines[i], right: lines[i] });
      i += 1;
      continue;
    }
    const removed: DiffLine[] = [];
    while (i < lines.length && lines[i].kind === 'del') {
      removed.push(lines[i]);
      i += 1;
    }
    const added: DiffLine[] = [];
    while (i < lines.length && lines[i].kind === 'add') {
      added.push(lines[i]);
      i += 1;
    }
    const paired = Math.max(removed.length, added.length);
    for (let k = 0; k < paired; k += 1) rows.push({ left: removed[k], right: added[k] });
  }
  return rows;
}

/** The path out of a `---` / `+++` line: timestamp dropped, quotes undone. */
function headerPath(rest: string, prefix: 'a/' | 'b/'): string {
  // A POSIX `diff` writes the modification time after a tab. git does not,
  // but a patch that reached the tab from elsewhere may.
  const tab = rest.indexOf('\t');
  const value = unquotePath(tab === -1 ? rest : rest.slice(0, tab));
  return stripSidePrefix(value, prefix);
}

function stripSidePrefix(value: string, prefix: 'a/' | 'b/'): string {
  if (value === DEV_NULL) return value;
  return value.startsWith(prefix) ? value.slice(prefix.length) : value;
}

/**
 * Undo the C-style quoting git applies to a path holding non-ASCII bytes.
 *
 * Default `core.quotepath` writes each such byte as `\ooo` octal, so a graph
 * named in Chinese arrives as a row of escapes. The bytes are reassembled
 * through percent-decoding rather than a `TextDecoder` so the module needs no
 * runtime it cannot count on; a sequence that is not valid UTF-8 falls back
 * to the quoted text rather than throwing.
 */
function unquotePath(value: string): string {
  if (value.length < 2 || !value.startsWith('"') || !value.endsWith('"')) return value;
  const body = value.slice(1, -1);
  // The WHOLE reassembly is guarded, not just the decode at the end:
  // `encodeURIComponent` throws on a lone surrogate as readily as
  // `decodeURIComponent` throws on a byte sequence that is not UTF-8, and
  // nothing in this module may throw -- a throw inside a React render blanks
  // the modal instead of showing the patch.
  try {
    let encoded = '';
    for (let i = 0; i < body.length; i += 1) {
      const ch = body.charAt(i);
      if (ch !== '\\') {
        // Taken a code POINT at a time. `core.quotepath=false` -- what a
        // reader with CJK file names sets -- leaves a non-ASCII byte alone,
        // so an astral character can sit inside a body git quoted for some
        // other reason, and half of one is a lone surrogate.
        const point = String.fromCodePoint(body.codePointAt(i) as number);
        encoded += encodeURIComponent(point);
        i += point.length - 1;
        continue;
      }
      const simple = ESCAPE_BYTES[body.charAt(i + 1)];
      if (simple !== undefined) {
        encoded += percentByte(simple);
        i += 1;
        continue;
      }
      const octal = body.slice(i + 1, i + 4);
      if (/^[0-7]{3}$/.test(octal)) {
        encoded += percentByte(parseInt(octal, 8));
        i += 3;
        continue;
      }
      encoded += percentByte(0x5c);
    }
    return decodeURIComponent(encoded);
  } catch {
    return value;
  }
}

function percentByte(byte: number): string {
  return `%${byte.toString(16).padStart(2, '0')}`;
}
