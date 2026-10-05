import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, it, expect, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { ResultsPanel } from './ResultsPanel';
import { useTabStore } from '../../store/tabStore';
import { useRunStore } from '../../store/runStore';
import { useI18n } from '../../i18n';
import styles from './ResultsPanel.module.css';

// The Log tab shows Print's text in its own shape. Print sends what it was
// given as one log entry -- a DataFrame, a padded table, an indented block --
// and the entry reaches the DOM intact; only the stylesheet decides whether
// the browser then folds every run of spaces and every newline into a single
// space, which turned a table into one line.

/**
 * The declarations of each ResultsPanel.module.css rule whose selector is
 * exactly `selector`, read as text.
 *
 * vitest hands a CSS module over as class names and applies none of its
 * rules, so no style in jsdom is ever computed from it (Toolbar.test.tsx reads
 * its stylesheet the same way). What the rule does on screen is the browser's
 * half of the check.
 */
function declarations(selector: string): Record<string, string>[] {
  const css = readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), 'ResultsPanel.module.css'),
    'utf8',
  ).replace(/\/\*[\s\S]*?\*\//g, '');
  return css
    .split('}')
    .map((chunk) => chunk.split('{').slice(-2))
    .filter(([head = '']) => head.trim() === selector)
    .map(([, body = '']) =>
      Object.fromEntries(
        body
          .split(';')
          .map((declaration) => declaration.split(':'))
          .filter((parts) => parts.length > 1)
          .map(([property, ...value]) => [property.trim(), value.join(':').trim()]),
      ),
    );
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('test');
  useRunStore.setState({ runs: [], total: 0, activeCount: 0 });
});

describe('ResultsPanel: the log keeps whitespace', () => {
  it('lays a message out with its own line breaks and spacing, wrapping a line too long for the panel', () => {
    const rules = declarations('.logMessage');
    expect(rules).toHaveLength(1);
    // `pre-wrap`, not `pre`: spaces and newlines are kept, and a long line
    // still wraps inside the panel instead of scrolling it sideways.
    expect(rules[0]['white-space']).toBe('pre-wrap');
    expect(rules[0]['word-break']).toBe('break-word');
  });

  it('puts a Print line into the log exactly as it was sent', () => {
    // What Print logs for pd.DataFrame({'name': ['alice', 'bob'], 'score': [90, 85]}).
    const message = '    name  score\n0  alice     90\n1    bob     85';
    const { activeTabId, addTabLog } = useTabStore.getState();
    // The entry useGraphExecution adds for a node's text output.
    addTabLog(activeTabId, { nodeId: 'print-1', message, kind: 'text', type: 'info' });
    render(<ResultsPanel />);
    const shown = document.querySelector(`.${styles.logMessage}`);
    expect(shown?.textContent).toBe(message);
  });
});
