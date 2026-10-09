import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import type { PackLogLine } from '../../store/packStore';
import { useI18n } from '../../i18n';
import { PackLogTail } from './PackLogTail';

/**
 * The transcript both the Package Center and the Plugin Center draw. Its own
 * file because the gap notice is the one line in it the UI phrases itself,
 * and both panes get it from here.
 */

const gapped: PackLogLine[] = [
  {
    seq: 4, ts: null, kind: 'gap', dropped: 4,
    text: '4 earlier events are no longer available',
  },
  { seq: 5, ts: null, kind: 'log', text: 'Collecting model2vec>=0.8.0' },
  { seq: 7, ts: null, kind: 'step', text: 'done' },
];

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('PackLogTail', () => {
  it('says once, in English, how many earlier lines are gone, then the kept ones verbatim', () => {
    render(<PackLogTail lines={gapped} ariaLabel="Install log" />);
    const log = within(screen.getByRole('log', { name: 'Install log' }));

    const notices = log.getAllByText(/no longer available/);
    expect(notices).toHaveLength(1);
    expect(notices[0]).toHaveTextContent(
      '4 earlier log entries are no longer available. Showing what was kept.',
    );
    expect(notices[0]).toHaveAttribute('data-kind', 'gap');
    expect(log.getByText('Collecting model2vec>=0.8.0')).toBeInTheDocument();
    expect(log.getByText('done')).toBeInTheDocument();
  });

  it('says it in Traditional Chinese for a zh-TW reader', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<PackLogTail lines={gapped} ariaLabel="安裝紀錄" />);

    expect(screen.getByText('較早的 4 筆紀錄已無法取得，以下為保留的部分。'))
      .toHaveAttribute('data-kind', 'gap');
    // The server's own lines stay as they were sent.
    expect(screen.getByText('Collecting model2vec>=0.8.0')).toBeInTheDocument();
  });

  it('counts zero for a gap line that arrived without a count', () => {
    render(
      <PackLogTail
        lines={[{ seq: 1, ts: null, kind: 'gap', text: 'gap' }]}
        ariaLabel="Install log"
      />,
    );
    expect(screen.getByText(/^0 earlier log entries/)).toBeInTheDocument();
  });
});
