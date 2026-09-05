import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DiffView } from './DiffView';
import { parseUnifiedDiff } from '../../utils/unifiedDiff';

/*
 * The patch as a grid, in both of its shapes.
 *
 * The numbers are the point. A view that prints the new side's number beside
 * the old side's text is wrong in a way the text itself never gives away --
 * every line is still there, still tinted, still signed, and only the column
 * of digits beside it is a lie. The same goes for the empty half of a row
 * where one run was longer than the other: it is what keeps the two columns
 * in step, and a reader only notices it is gone by finding the wrong lines
 * facing each other.
 *
 * Built through `parseUnifiedDiff` rather than from hand-written `DiffLine`
 * objects: the numbering is exactly what git would have produced, so a test
 * that expects 11 on the left is expecting the line git called 11.
 */
const PATCH = [
  'diff --git a/train.py b/train.py',
  'index 1111111..2222222 100644',
  '--- a/train.py',
  '+++ b/train.py',
  // Starts at 10 on the old side and 20 on the new, so a column that took
  // its number from the other side is off by ten rather than right by luck.
  '@@ -10,4 +20,3 @@ def train():',
  ' start()',
  '-lr = 0.1',
  '-momentum = 0.9',
  '+lr = 0.01',
  ' step()',
  '',
].join('\n');

const file = parseUnifiedDiff(PATCH);

/** The six spans of one split row, in order: number, sign, text, twice. */
const splitRow = (text: string) =>
  [...screen.getAllByText(text)[0].closest('[data-row]')!.children].map(
    (span) => span.textContent,
  );

/** The four spans of one unified line: old number, new number, sign, text. */
const unifiedRow = (text: string) =>
  [...screen.getByText(text).closest('[data-kind]')!.children].map(
    (span) => span.textContent,
  );

describe('DiffView: the unified shape', () => {
  it('numbers each line on the side it exists on', () => {
    render(<DiffView file={file} mode="unified" />);
    expect(unifiedRow('start()')).toEqual(['10', '20', ' ', 'start()']);
    expect(unifiedRow('momentum = 0.9')).toEqual(['12', '', '-', 'momentum = 0.9']);
    expect(unifiedRow('lr = 0.01')).toEqual(['', '21', '+', 'lr = 0.01']);
  });

  it('keeps git\'s own hunk header, section heading and all', () => {
    render(<DiffView file={file} mode="unified" />);
    expect(screen.getByText('@@ -10,4 +20,3 @@ def train():')).toBeTruthy();
  });
});

describe('DiffView: the two columns', () => {
  it('puts the old side and its own numbers on the left, the new on the right', () => {
    render(<DiffView file={file} mode="split" />);
    // The changed pair: 11 belongs to the old file, 21 to the new one.
    expect(splitRow('lr = 0.1')).toEqual(['11', '-', 'lr = 0.1', '21', '+', 'lr = 0.01']);
    // A context line is the SAME object in both cells and still takes a
    // different number in each.
    expect(splitRow('start()')).toEqual(['10', ' ', 'start()', '20', ' ', 'start()']);
  });

  it('leaves an empty half where one run was longer than the other', () => {
    render(<DiffView file={file} mode="split" />);
    // Two lines were removed and one added, so the second removal faces
    // nothing. The cell is drawn and marked empty rather than left out, which
    // is what keeps the row a row.
    expect(splitRow('momentum = 0.9')).toEqual(['12', '-', 'momentum = 0.9', '', '', '']);
    const row = screen.getByText('momentum = 0.9').closest('[data-row]');
    expect(row?.querySelectorAll('[data-kind="empty"]')).toHaveLength(3);
  });

  it('says which half of a row is which without using colour alone', () => {
    render(<DiffView file={file} mode="split" />);
    const cell = screen.getByText('lr = 0.01').closest('[data-kind]');
    expect(cell?.getAttribute('data-kind')).toBe('add');
    expect(screen.getByText('lr = 0.1').closest('[data-kind]')?.getAttribute('data-kind'))
      .toBe('del');
  });
});
