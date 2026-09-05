import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { GraphDiffSummary } from './GraphDiffSummary';
import { useI18n } from '../../i18n';
import type {
  GraphDiffCountKind,
  GraphDiffLine,
  GraphDiffSummary as GraphDiff,
} from '../../utils/graphDiff';

/*
 * The strip above the patch: which sentence each summary line becomes.
 *
 * Every line here is a lookup, and a lookup is the kind of code whose mistakes
 * are invisible -- a summary that says "removed" over an edit that added is
 * still a well-formed sentence in the reader's own language, and the diff
 * underneath it is still correct, so nothing else in the app disagrees. The
 * sentences below are written out in full rather than read from the map the
 * component uses; a table that takes its answer from the thing under test
 * proves only that the thing is consistent with itself.
 */

function summary(over: Partial<GraphDiff> = {}): GraphDiff {
  return { lines: [], more: 0, noLogicChange: false, unparseable: false, ...over };
}

const draw = (over: Partial<GraphDiff> = {}) =>
  render(<GraphDiffSummary summary={summary(over)} />);

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('GraphDiffSummary: the counted lines', () => {
  const COUNTED: [GraphDiffCountKind, number, string][] = [
    ['nodesAdded', 3, '3 node(s) added'],
    ['nodesRemoved', 2, '2 node(s) removed'],
    ['edgesAdded', 1, '1 edge(s) added'],
    ['edgesRemoved', 4, '4 edge(s) removed'],
    ['positionsMoved', 5, '5 node position(s) moved'],
  ];

  it.each(COUNTED)('says "%s" as the sentence it means', (kind, count, sentence) => {
    draw({ lines: [{ kind, count }] });
    expect(screen.getByText(sentence)).toBeTruthy();
  });

  it('is a list, and says so where the stylesheet took that away', () => {
    // `.summary` sets `list-style: none`, which drops the list semantics of a
    // `<ul>` in Safari -- and how many facts the strip asserts, and where
    // "and 3 more" ends it, is the one thing the sentences cannot say.
    draw({ lines: [{ kind: 'nodesAdded', count: 3 }] });
    expect(screen.getByRole('list').getAttribute('role')).toBe('list');
  });

  it('draws five kinds as their own five sentences, in order', () => {
    // Written out, and matched one to one. Asserting only that the five
    // sentences DIFFER cannot see the swap it was there to guard: exchange
    // "added" and "removed" in the map under test and five distinct
    // sentences are still five distinct sentences, each grammatical,
    // translated and about the wrong thing.
    draw({ lines: COUNTED.map(([kind, count]) => ({ kind, count })) });
    expect(screen.getAllByRole('listitem').map((li) => li.textContent)).toEqual(
      COUNTED.map(([, , sentence]) => sentence),
    );
  });
});

describe('GraphDiffSummary: the lines that name a node', () => {
  it('names the node, the parameter and both of its values', () => {
    draw({
      lines: [{ kind: 'param', node: 'linear', param: 'out_features', from: '16', to: '32' }],
    });
    expect(screen.getByText('linear: out_features 16 -> 32')).toBeTruthy();
  });

  it('marks the empty half of a parameter that was only added', () => {
    // `summarizeGraphDiff` gives an empty `from` for a parameter that is on
    // the new side alone. Without a placeholder the line reads
    // "linear: lr  -> 0.01" -- a sentence with a hole where a value goes.
    draw({ lines: [{ kind: 'param', node: 'linear', param: 'lr', from: '', to: '0.01' }] });
    expect(screen.getByText('linear: lr - -> 0.01')).toBeTruthy();
  });

  it('marks the other half for one that was removed', () => {
    draw({ lines: [{ kind: 'param', node: 'linear', param: 'lr', from: '0.01', to: '' }] });
    expect(screen.getByText('linear: lr 0.01 -> -')).toBeTruthy();
  });

  it('says the old and the new type of a node whose type changed', () => {
    draw({ lines: [{ kind: 'typeChanged', node: 'head', from: 'Linear', to: 'Conv2d' }] });
    expect(screen.getByText('head: type Linear -> Conv2d')).toBeTruthy();
  });
});

describe('GraphDiffSummary: the eight-line cap', () => {
  it('draws every line it was given and counts the ones that did not fit', () => {
    const lines: GraphDiffLine[] = Array.from({ length: 8 }, (_, i) => ({
      kind: 'nodesAdded',
      count: i + 1,
    }));
    draw({ lines, more: 2 });
    expect(screen.getByText('and 2 more')).toBeTruthy();
    // Eight sentences and the count under them.
    expect(screen.getAllByRole('listitem')).toHaveLength(9);
  });

  it('draws no such row when everything fit', () => {
    draw({ lines: [{ kind: 'nodesAdded', count: 1 }] });
    expect(screen.getAllByRole('listitem')).toHaveLength(1);
    expect(screen.queryByText(/more/)).toBeNull();
  });
});

describe('GraphDiffSummary: the answers that are not a list', () => {
  it('draws nothing at all for a change it has no sentence for', () => {
    // A segment group, a note resized, a subgraph definition: real changes,
    // none of them summarised in v1, so the summary is empty while
    // `noLogicChange` is false. An empty bordered strip above the patch would
    // be a box that says nothing; the patch below carries the answer.
    const { container } = draw();
    expect(container.firstChild).toBeNull();
    expect(screen.queryByRole('list')).toBeNull();
  });

  it('says when a side could not be read as a graph', () => {
    draw({ unparseable: true });
    expect(screen.getByText('Could not parse as a graph')).toBeTruthy();
  });

  it('says when only the text of the file moved', () => {
    draw({ noLogicChange: true });
    expect(screen.getByText('No logic change')).toBeTruthy();
  });

  it('prefers that refusal to a list built on a side it could not read', () => {
    // The two do not travel together out of `summarizeGraphDiff`; the order
    // is pinned so they cannot start to.
    draw({ unparseable: true, lines: [{ kind: 'nodesAdded', count: 9 }] });
    expect(screen.getByText('Could not parse as a graph')).toBeTruthy();
    expect(screen.queryByText('9 node(s) added')).toBeNull();
  });
});
