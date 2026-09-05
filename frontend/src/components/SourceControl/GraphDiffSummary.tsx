import { useI18n, type TranslationKey } from '../../i18n';
import type {
  GraphDiffCountKind,
  GraphDiffLine,
  GraphDiffSummary as GraphDiff,
} from '../../utils/graphDiff';
import styles from './GitDiffModal.module.css';

/**
 * The placeholder for a parameter that is not on one side at all.
 *
 * `summarizeGraphDiff` gives an empty `from` for a parameter that was ADDED
 * and an empty `to` for one that was removed, and "out_features  -> 32" is a
 * sentence with a hole in it. An ASCII hyphen rather than a word, because the
 * line around it is already the reader's own language and this is a slot
 * marker, not vocabulary.
 */
const ABSENT = '-';

/** The five lines that are a bare number, and the sentence each one gets. */
const COUNT_KEY: Record<GraphDiffCountKind, TranslationKey> = {
  nodesAdded: 'git.gdiff.nodesAdded',
  nodesRemoved: 'git.gdiff.nodesRemoved',
  edgesAdded: 'git.gdiff.edgesAdded',
  edgesRemoved: 'git.gdiff.edgesRemoved',
  positionsMoved: 'git.gdiff.positionsMoved',
};

/**
 * What the change did to the GRAPH, above what it did to the JSON.
 *
 * A saved graph is a document nobody reads as text: two nodes swapping places
 * in the array is a diff of forty lines and no change at all, and one number
 * moving inside `data.params` is a diff of one line that changes what the
 * model is. This strip says the second kind out loud; the patch under it is
 * still the whole truth.
 *
 * Three states, and they are mutually exclusive by construction:
 * `unparseable` (a side that is not a document of this kind -- broken JSON,
 * or JSON that is not a graph), `noLogicChange` (the two sides say the same
 * thing and only their text moved), and a list of lines. A summary with no
 * lines and no flag is a real answer too -- a segment group, a note resized,
 * a subgraph definition, none of which v1 has a sentence for -- and draws
 * nothing at all rather than an empty box.
 */
export function GraphDiffSummary({ summary }: { summary: GraphDiff }) {
  const { t } = useI18n();

  if (summary.unparseable) {
    return <p className={styles.summaryNote}>{t('git.gdiff.unparseable')}</p>;
  }
  if (summary.noLogicChange) {
    return <p className={styles.summaryNote}>{t('git.gdiff.noLogicChange')}</p>;
  }
  if (summary.lines.length === 0) return null;

  return (
    // `role="list"` is spelled out because `list-style: none` takes the list
    // semantics away from a `<ul>` in Safari -- and how many facts the strip
    // is asserting, and where "and 3 more" ends it, is the one thing a reader
    // cannot get from the sentences themselves.
    <ul className={styles.summary} role="list">
      {summary.lines.map((line, index) => (
        <li key={index} className={styles.summaryLine}>{sentence(line, t)}</li>
      ))}
      {summary.more > 0 && (
        <li className={styles.summaryLine}>
          {t('git.gdiff.more', { count: summary.more })}
        </li>
      )}
    </ul>
  );
}

/** One summary line, as the sentence its kind names. */
function sentence(
  line: GraphDiffLine,
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string,
): string {
  if (line.kind === 'typeChanged') {
    return t('git.gdiff.typeChanged', { node: line.node, from: line.from, to: line.to });
  }
  if (line.kind === 'param') {
    return t('git.gdiff.param', {
      node: line.node,
      param: line.param,
      from: line.from === '' ? ABSENT : line.from,
      to: line.to === '' ? ABSENT : line.to,
    });
  }
  return t(COUNT_KEY[line.kind], { count: line.count });
}
