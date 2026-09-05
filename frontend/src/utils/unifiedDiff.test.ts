import { describe, it, expect } from 'vitest';
import { parseUnifiedDiff, toSplitRows } from './unifiedDiff';

// Every patch below except the deliberately damaged ones is real `git diff`
// output, captured from a scratch repository with the exact argv the backend
// uses (`backend/app/core/git/diff.py:_plan`). `\ No newline at end of file`
// is written `\\ ` in a template literal on purpose: an untagged template
// silently drops the backslash in `\ `, which would delete the very character
// under test.

const TWO_HUNKS = `diff --git a/f.txt b/f.txt
index d3a66df..36bd28f 100644
--- a/f.txt
+++ b/f.txt
@@ -1,5 +1,5 @@
 alpha
-bravo
+BRAVO
 charlie
 delta
 echo
@@ -12,8 +12,9 @@ kilo
 lima
 mike
 november
-oscar
-papa
+OSCAR
+PAPA
+extra
 quebec
 romeo
 sierra
`;

const NO_NEWLINE = `diff --git a/n.txt b/n.txt
index 54d55bf..2090089 100644
--- a/n.txt
+++ b/n.txt
@@ -1,3 +1,3 @@
 one
 two
-three
\\ No newline at end of file
+THREE
\\ No newline at end of file
`;

const NO_INDEX = `diff --git a/u.txt b/u.txt
new file mode 100644
index 0000000..92d56ff
--- /dev/null
+++ b/u.txt
@@ -0,0 +1,2 @@
+new one
+new two
`;

const DELETED = `diff --git a/n.txt b/n.txt
deleted file mode 100644
index 2090089..0000000
--- a/n.txt
+++ /dev/null
@@ -1,3 +0,0 @@
-one
-two
-THREE
\\ No newline at end of file
`;

// What `git diff -- <path>` answers for an UNMERGED path -- the combined
// format, not a two-way patch. Captured from a real conflict (git 2.53) with
// the argv `_plan` takes for a conflicted file, which is the tracked worktree
// branch: `diff --no-color --no-ext-diff -M -- conf.txt`. Two parents, so two
// prefix columns and one more `@` at each end of the hunk header.
const CONFLICTED = `diff --cc conf.txt
index f797e30,ffafd7c..0000000
--- a/conf.txt
+++ b/conf.txt
@@@ -1,3 -1,3 +1,7 @@@
  line one
++<<<<<<< HEAD
 +OURS two
++=======
+ THEIRS two
++>>>>>>> feature
  line three
`;

// The same shape where one side lost its final newline: git then writes a
// REMOVAL row inside the combined hunk, which the two-column reading has to
// classify from either column rather than from the first one alone.
const CONFLICTED_REMOVAL = `diff --cc n.txt
index 9f413b8,54a85b9..0000000
--- a/n.txt
+++ b/n.txt
@@@ -1,3 -1,3 +1,7 @@@
  one
  two
- OURS
++<<<<<<< HEAD
++OURS
++=======
+ THEIRS
++>>>>>>> feature
`;

// A conflicted BINARY file: git names the path once, on the `diff --cc` line,
// and writes no `---`/`+++` pair at all.
const CONFLICTED_BINARY = `diff --cc b.bin
index 43c4d52,7edafa3..0000000
Binary files differ
`;

const ONE_LINE = `diff --git a/s.txt b/s.txt
index 6c542ab..bc8c7b4 100644
--- a/s.txt
+++ b/s.txt
@@ -1 +1 @@
-only
+ONLY
`;

describe('parseUnifiedDiff', () => {
  it('reads both paths off the ---/+++ pair with the a/ and b/ prefixes gone', () => {
    const file = parseUnifiedDiff(TWO_HUNKS);
    expect(file.oldPath).toBe('f.txt');
    expect(file.newPath).toBe('f.txt');
  });

  it('splits a two-hunk patch into two hunks with the counts git wrote', () => {
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    expect(hunks).toHaveLength(2);
    expect(hunks[0]).toMatchObject({
      oldStart: 1, oldLines: 5, newStart: 1, newLines: 5,
      header: '@@ -1,5 +1,5 @@',
    });
    expect(hunks[1]).toMatchObject({
      oldStart: 12, oldLines: 8, newStart: 12, newLines: 9,
    });
  });

  it('keeps the whole @@ line in the header, section heading included', () => {
    // The unified view prints this line verbatim as the hunk separator, so
    // the trailing function/section context git appends is part of it.
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    expect(hunks[1].header).toBe('@@ -12,8 +12,9 @@ kilo');
  });

  it('classifies every body line and strips the one-character prefix', () => {
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    expect(hunks[0].lines.map((l) => [l.kind, l.text])).toEqual([
      ['context', 'alpha'],
      ['del', 'bravo'],
      ['add', 'BRAVO'],
      ['context', 'charlie'],
      ['context', 'delta'],
      ['context', 'echo'],
    ]);
  });

  it('numbers context lines on both sides and each change on its own side', () => {
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    expect(hunks[0].lines.map((l) => [l.oldNo, l.newNo])).toEqual([
      [1, 1],
      [2, undefined],
      [undefined, 2],
      [3, 3],
      [4, 4],
      [5, 5],
    ]);
  });

  it('keeps the two sides in step after an unequal run of changes', () => {
    // Hunk 2 removes two lines and adds three, so from `quebec` onwards the
    // old and new numbers differ by one. Getting this wrong is invisible in a
    // unified view and glaring in a split one.
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    expect(hunks[1].lines.find((l) => l.text === 'quebec')).toMatchObject({
      kind: 'context', oldNo: 17, newNo: 18,
    });
    expect(hunks[1].lines.find((l) => l.text === 'sierra')).toMatchObject({
      kind: 'context', oldNo: 19, newNo: 20,
    });
    const extra = hunks[1].lines.find((l) => l.text === 'extra');
    expect(extra).toMatchObject({ kind: 'add', newNo: 17 });
    // Absent, not present-and-undefined: an added line exists on one side only.
    expect(extra).not.toHaveProperty('oldNo');
  });

  it('reads a hunk header that omits the count, which means one line', () => {
    const { hunks } = parseUnifiedDiff(ONE_LINE);
    expect(hunks[0]).toMatchObject({
      oldStart: 1, oldLines: 1, newStart: 1, newLines: 1,
    });
    expect(hunks[0].lines).toHaveLength(2);
  });

  it('marks the line a no-newline note belongs to instead of listing the note', () => {
    // The marker is not a line of the file: counted as one it would shift
    // every number below it, and rendered as one it would look like content.
    const { hunks } = parseUnifiedDiff(NO_NEWLINE);
    expect(hunks[0].lines.map((l) => [l.kind, l.text])).toEqual([
      ['context', 'one'],
      ['context', 'two'],
      ['del', 'three'],
      ['add', 'THREE'],
    ]);
    expect(hunks[0].lines[2].noNewline).toBe(true);
    expect(hunks[0].lines[3].noNewline).toBe(true);
    expect(hunks[0].lines[0].noNewline).toBeUndefined();
  });

  it('reads the --no-index patch of an untracked file, /dev/null side and all', () => {
    const file = parseUnifiedDiff(NO_INDEX);
    expect(file.oldPath).toBe('/dev/null');
    expect(file.newPath).toBe('u.txt');
    expect(file.hunks[0]).toMatchObject({ oldStart: 0, oldLines: 0, newStart: 1, newLines: 2 });
    expect(file.hunks[0].lines.every((l) => l.kind === 'add')).toBe(true);
    expect(file.hunks[0].lines.map((l) => l.newNo)).toEqual([1, 2]);
    expect(file.hunks[0].lines.map((l) => l.oldNo)).toEqual([undefined, undefined]);
  });

  it('reads a deletion, whose new side is /dev/null', () => {
    const file = parseUnifiedDiff(DELETED);
    expect(file.oldPath).toBe('n.txt');
    expect(file.newPath).toBe('/dev/null');
    expect(file.hunks[0].lines.map((l) => l.oldNo)).toEqual([1, 2, 3]);
  });

  it('reads the combined patch git answers for a conflicted path', () => {
    // The one file a diff matters most for. `_plan` sends a conflicted path
    // down the tracked worktree branch, git sees an UNMERGED path and answers
    // in the combined format -- so a parser that only knows `@@ -a,b +c,d @@`
    // opens no hunk at all and the window says "No changes" over a file full
    // of conflict markers.
    const file = parseUnifiedDiff(CONFLICTED);
    expect(file.oldPath).toBe('conf.txt');
    expect(file.newPath).toBe('conf.txt');
    expect(file.hunks).toHaveLength(1);
    expect(file.hunks[0]).toMatchObject({
      // The LAST old range, which is the parent the old column follows.
      oldStart: 1, oldLines: 3, newStart: 1, newLines: 7,
      header: '@@@ -1,3 -1,3 +1,7 @@@',
    });
  });

  it('reads a combined row from either column and drops both of them', () => {
    // Two columns, so a row is an addition when EITHER holds a `+`: the two
    // sides of the conflict are ` +OURS two` and `+ THEIRS two`, and reading
    // only the first column would call one of them context and print it with
    // its neighbour's column still in the text.
    const { hunks } = parseUnifiedDiff(CONFLICTED);
    expect(hunks[0].lines.map((l) => [l.kind, l.text])).toEqual([
      ['context', 'line one'],
      ['add', '<<<<<<< HEAD'],
      ['add', 'OURS two'],
      ['add', '======='],
      ['add', 'THEIRS two'],
      ['add', '>>>>>>> feature'],
      ['context', 'line three'],
    ]);
    expect(hunks[0].lines.map((l) => l.newNo)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it('reads a removal row inside a combined hunk from its second column', () => {
    const { hunks } = parseUnifiedDiff(CONFLICTED_REMOVAL);
    expect(hunks[0].lines.map((l) => [l.kind, l.text])).toEqual([
      ['context', 'one'],
      ['context', 'two'],
      ['del', 'OURS'],
      ['add', '<<<<<<< HEAD'],
      ['add', 'OURS'],
      ['add', '======='],
      ['add', 'THEIRS'],
      ['add', '>>>>>>> feature'],
    ]);
    // The removal is on the old side only, and the rows after it carry on
    // numbering the result.
    expect(hunks[0].lines.map((l) => [l.oldNo, l.newNo])).toEqual([
      [1, 1], [2, 2], [3, undefined],
      [undefined, 3], [undefined, 4], [undefined, 5], [undefined, 6], [undefined, 7],
    ]);
  });

  it('takes a conflicted binary file\'s path off the diff --cc line', () => {
    // No `---`/`+++` pair at all, and the combined header names the path ONCE
    // with no a/ or b/ prefix.
    const file = parseUnifiedDiff(CONFLICTED_BINARY);
    expect(file.oldPath).toBe('b.bin');
    expect(file.newPath).toBe('b.bin');
    expect(file.hunks).toEqual([]);
  });

  it('answers an empty patch with no hunks and no paths rather than throwing', () => {
    // `git diff` says nothing at all about a file that has not changed, and
    // the modal has to render that as "no changes", not as a parse failure.
    for (const patch of ['', '\n', '   ']) {
      expect(parseUnifiedDiff(patch)).toEqual({ oldPath: '', newPath: '', hunks: [] });
    }
  });

  it('keeps a blank context line, which git writes as a bare space', () => {
    // The space is the prefix, so the line is a real empty line of the file.
    // An element that is the empty string is never content, which is also
    // what makes the one a trailing newline leaves behind harmless.
    const patch = [
      '--- a/b.txt',
      '+++ b/b.txt',
      '@@ -1,3 +1,3 @@',
      ' first',
      ' ',
      '-third',
      '+THIRD',
      '',
    ].join('\n');
    const { hunks } = parseUnifiedDiff(patch);
    expect(hunks[0].lines.map((l) => [l.kind, l.text])).toEqual([
      ['context', 'first'],
      ['context', ''],
      ['del', 'third'],
      ['add', 'THIRD'],
    ]);
    expect(hunks[0].lines[1]).toMatchObject({ oldNo: 2, newNo: 2 });
  });

  it('keeps the carriage return of a CRLF file inside the line text', () => {
    // A change that is ONLY a line ending is a real change the reader has to
    // be able to see; swallowing the CR here would render the two sides
    // identical. The separator git writes is always LF, so only content
    // carries a CR.
    const patch = [
      'diff --git a/w.txt b/w.txt',
      'index 1111111..2222222 100644',
      '--- a/w.txt',
      '+++ b/w.txt',
      '@@ -1,2 +1,2 @@',
      ' keep\r',
      '-old\r',
      '+new\r',
    ].join('\n');
    const { hunks } = parseUnifiedDiff(patch);
    expect(hunks).toHaveLength(1);
    expect(hunks[0].lines.map((l) => [l.kind, l.text])).toEqual([
      ['context', 'keep\r'],
      ['del', 'old\r'],
      ['add', 'new\r'],
    ]);
  });

  it('still finds the structure when the patch itself arrives CRLF-separated', () => {
    // Defence in depth: git writes LF, but a patch that made a round trip
    // through something that did not must not lose its hunks to a stray CR
    // stuck on the end of the @@ line.
    const patch = [
      'diff --git a/w.txt b/w.txt',
      '--- a/w.txt',
      '+++ b/w.txt',
      '@@ -1,1 +1,1 @@',
      '-old',
      '+new',
    ].join('\r\n');
    const file = parseUnifiedDiff(patch);
    expect(file.oldPath).toBe('w.txt');
    expect(file.newPath).toBe('w.txt');
    expect(file.hunks[0].header).toBe('@@ -1,1 +1,1 @@');
  });

  it('keeps the partial hunk when the 1 MiB cap cuts the patch mid-hunk', () => {
    // `diff.py` slices the bytes at MAX_PATCH_BYTES and hands the prefix
    // over, so the last hunk routinely holds fewer lines than its own header
    // promises. The header keeps the numbers git wrote; the lines are what
    // actually arrived.
    const cut = TWO_HUNKS.slice(0, TWO_HUNKS.indexOf('-papa'));
    const { hunks } = parseUnifiedDiff(cut);
    expect(hunks).toHaveLength(2);
    expect(hunks[1]).toMatchObject({ oldLines: 8, newLines: 9 });
    expect(hunks[1].lines.map((l) => l.text)).toEqual([
      'lima', 'mike', 'november', 'oscar',
    ]);
  });

  it('keeps a final line the cap chopped in half', () => {
    const cut = `${TWO_HUNKS.slice(0, TWO_HUNKS.indexOf('-papa'))}-pa`;
    const { hunks } = parseUnifiedDiff(cut);
    expect(hunks[1].lines[hunks[1].lines.length - 1]).toMatchObject({
      kind: 'del', text: 'pa', oldNo: 16,
    });
  });

  it('drops a hunk header the cap cut in half instead of throwing', () => {
    const cut = `${TWO_HUNKS.slice(0, TWO_HUNKS.indexOf('@@ -12'))}@@ -12,8 +1`;
    const { hunks } = parseUnifiedDiff(cut);
    expect(hunks).toHaveLength(1);
    expect(hunks[0].lines).toHaveLength(6);
  });

  it('survives a cut that lands inside the file header, before any hunk', () => {
    const cut = TWO_HUNKS.slice(0, 30);
    expect(() => parseUnifiedDiff(cut)).not.toThrow();
    expect(parseUnifiedDiff(cut).hunks).toEqual([]);
  });

  it('falls back to the diff --git line when a binary patch has no ---/+++ pair', () => {
    const patch = [
      'diff --git a/img.png b/img.png',
      'index 1111111..2222222 100644',
      'Binary files a/img.png and b/img.png differ',
      '',
    ].join('\n');
    const file = parseUnifiedDiff(patch);
    expect(file.oldPath).toBe('img.png');
    expect(file.newPath).toBe('img.png');
    expect(file.hunks).toEqual([]);
  });

  it('splits a diff --git header whose file name holds a space', () => {
    // git quotes a path for a control character, a quote or a backslash --
    // never for a space -- so this header is four whitespace-separated
    // tokens. Both sides of a non-rename header are the same path, so the
    // midpoint is exact. Without this a binary or mode-only patch, which has
    // no ---/+++ pair to fall back on, reported no path at all.
    const patch = [
      'diff --git a/my image.png b/my image.png',
      'index 1111111..2222222 100644',
      'Binary files a/my image.png and b/my image.png differ',
      '',
    ].join('\n');
    const file = parseUnifiedDiff(patch);
    expect(file.oldPath).toBe('my image.png');
    expect(file.newPath).toBe('my image.png');
  });

  it('leaves the paths empty rather than inventing a split it cannot verify', () => {
    // A rename of two differently long names holding spaces has no midpoint
    // to find. Reporting nothing beats reporting a path that is half of one
    // name and half of the other.
    const patch = [
      'diff --git a/my old file.png b/new.png',
      'index 1111111..2222222 100644',
      'Binary files differ',
      '',
    ].join('\n');
    expect(parseUnifiedDiff(patch)).toMatchObject({ oldPath: '', newPath: '' });
  });

  it('reads a quoted path holding an astral character, and never throws on a broken one', () => {
    // With `core.quotepath=false` -- the setting a CJK audience reaches for --
    // git leaves a non-ASCII byte alone but still quotes a name holding a
    // double quote, so an emoji can arrive inside the quoted body. Walked by
    // code UNIT that is a lone surrogate, and `encodeURIComponent` throws on
    // one; a throw in here blanks the modal instead of showing the patch.
    // U+1F600, built from its code point rather than typed, because every
    // source file here stays ASCII.
    const grin = String.fromCodePoint(0x1f600);
    const file = parseUnifiedDiff([
      `diff --git "a/x\\"y${grin}.json" "b/x\\"y${grin}.json"`,
      'index 1111111..2222222 100644',
      'Binary files differ',
    ].join('\n'));
    expect(file.oldPath).toBe(`x"y${grin}.json`);
    expect(file.newPath).toBe(`x"y${grin}.json`);

    // A genuinely lone surrogate cannot be repaired, so the quoted text comes
    // back unchanged -- the point is that a `DiffFile` comes back at all.
    const broken = `diff --git "a/x\uD83D.json" "b/x\uD83D.json"\nBinary files differ`;
    expect(() => parseUnifiedDiff(broken)).not.toThrow();
    expect(parseUnifiedDiff(broken).oldPath).toBe('"a/x\uD83D.json"');
  });

  it('unquotes the C-style path git writes for a non-ASCII file name', () => {
    // Default `core.quotepath`. Without this the header of a graph named in
    // Chinese reads as a row of octal escapes.
    const patch = [
      'diff --git "a/\\346\\270\\254.graph.json" "b/\\346\\270\\254.graph.json"',
      '--- "a/\\346\\270\\254.graph.json"',
      '+++ "b/\\346\\270\\254.graph.json"',
      '@@ -1 +1 @@',
      '-a',
      '+b',
    ].join('\n');
    const file = parseUnifiedDiff(patch);
    // U+6E2C, the character those three octal bytes spell in UTF-8. Written
    // as an escape because every source file here stays ASCII.
    expect(file.oldPath).toBe('\u6e2c.graph.json');
    expect(file.newPath).toBe('\u6e2c.graph.json');
  });

  it('drops the timestamp column a POSIX diff appends to the path', () => {
    const patch = [
      '--- a/t.txt\t2026-09-05 10:00:00.000000000 +0800',
      '+++ b/t.txt\t2026-09-05 10:00:01.000000000 +0800',
      '@@ -1 +1 @@',
      '-a',
      '+b',
    ].join('\n');
    const file = parseUnifiedDiff(patch);
    expect(file.oldPath).toBe('t.txt');
    expect(file.newPath).toBe('t.txt');
  });

  it('stops at the second file when a patch somehow carries more than one', () => {
    // The backend diffs one path per request, so a second `diff --git` means
    // the response is not what this parser was promised. Reporting the first
    // file is right; folding the lines of the second one into it is not.
    const patch = `${TWO_HUNKS}diff --git a/other.txt b/other.txt
--- a/other.txt
+++ b/other.txt
@@ -1 +1 @@
-x
+y
`;
    const file = parseUnifiedDiff(patch);
    expect(file.oldPath).toBe('f.txt');
    expect(file.hunks).toHaveLength(2);
  });
});

describe('toSplitRows', () => {
  it('puts a context line on both sides of the same row', () => {
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    const rows = toSplitRows(hunks[0]);
    expect(rows[0].left).toMatchObject({ text: 'alpha', oldNo: 1 });
    expect(rows[0].right).toMatchObject({ text: 'alpha', newNo: 1 });
  });

  it('pairs a run of removals with the run of additions that follows it', () => {
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    const rows = toSplitRows(hunks[0]);
    expect(rows).toHaveLength(5);
    expect(rows[1].left).toMatchObject({ kind: 'del', text: 'bravo', oldNo: 2 });
    expect(rows[1].right).toMatchObject({ kind: 'add', text: 'BRAVO', newNo: 2 });
  });

  it('leaves the other side empty for the remainder of the longer run', () => {
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    const rows = toSplitRows(hunks[1]);
    expect(rows).toHaveLength(9);
    expect(rows.map((r) => [r.left?.text, r.right?.text])).toEqual([
      ['lima', 'lima'],
      ['mike', 'mike'],
      ['november', 'november'],
      ['oscar', 'OSCAR'],
      ['papa', 'PAPA'],
      [undefined, 'extra'],
      ['quebec', 'quebec'],
      ['romeo', 'romeo'],
      ['sierra', 'sierra'],
    ]);
  });

  it('keeps every line number on the side it belongs to', () => {
    const { hunks } = parseUnifiedDiff(TWO_HUNKS);
    const rows = toSplitRows(hunks[1]);
    expect(rows[5].left).toBeUndefined();
    expect(rows[5].right).toMatchObject({ kind: 'add', text: 'extra', newNo: 17 });
    expect(rows[6].left).toMatchObject({ oldNo: 17 });
    expect(rows[6].right).toMatchObject({ newNo: 18 });
  });

  it('puts a run with no removals entirely on the right', () => {
    const { hunks } = parseUnifiedDiff(NO_INDEX);
    const rows = toSplitRows(hunks[0]);
    expect(rows.map((r) => [r.left, r.right?.text])).toEqual([
      [undefined, 'new one'],
      [undefined, 'new two'],
    ]);
  });

  it('puts a run with no additions entirely on the left', () => {
    const { hunks } = parseUnifiedDiff(DELETED);
    const rows = toSplitRows(hunks[0]);
    expect(rows.map((r) => [r.left?.text, r.right])).toEqual([
      ['one', undefined],
      ['two', undefined],
      ['THREE', undefined],
    ]);
  });

  it('starts a new pairing at each run rather than running two together', () => {
    // A del run, an add run, then a second del run: the second run must open
    // its own rows instead of extending the rows of the first.
    const patch = [
      '--- a/x',
      '+++ b/x',
      '@@ -1,4 +1,4 @@',
      '-a',
      '+A',
      '-b',
      '+B',
      ' c',
      ' d',
    ].join('\n');
    const rows = toSplitRows(parseUnifiedDiff(patch).hunks[0]);
    expect(rows.map((r) => [r.left?.text, r.right?.text])).toEqual([
      ['a', 'A'],
      ['b', 'B'],
      ['c', 'c'],
      ['d', 'd'],
    ]);
  });

  it('answers an empty hunk with no rows', () => {
    expect(toSplitRows({
      oldStart: 1, oldLines: 0, newStart: 1, newLines: 0, header: '@@ -1,0 +1,0 @@', lines: [],
    })).toEqual([]);
  });
});
