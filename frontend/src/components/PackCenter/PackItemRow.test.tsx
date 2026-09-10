import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { PackItem } from '../../api/rest';
import type { PackItemProgress } from '../../store/packStore';
import { useI18n } from '../../i18n';
import { PackItemRow, itemDisplayName } from './PackItemRow';

function item(over: Partial<PackItem> & { id: string } = { id: 'glove-50d' }): PackItem {
  return {
    kind: 'asset',
    repo_id: null,
    url: 'https://example.test/glove.6B.zip',
    size_bytes: 69_000_000,
    license: null,
    status: 'downloading',
    ...over,
  };
}

function renderRow(progress: PackItemProgress | null) {
  render(
    <PackItemRow
      item={item({ id: 'glove-50d' })}
      checked={false}
      onToggle={() => {}}
      progress={progress}
      disabled={false}
    />,
  );
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('itemDisplayName', () => {
  it('prefers the repo id, then the file name, then the item id', () => {
    expect(itemDisplayName(item({ id: 'labse', repo_id: 'sentence-transformers/LaBSE' })))
      .toBe('sentence-transformers/LaBSE');
    expect(itemDisplayName(item({ id: 'glove-50d', url: 'https://x.test/g.zip?sig=abc' })))
      .toBe('g.zip');
    // A URL that ends in its own directory has no file name to show, and a
    // signed one can be nothing but a query — the id is always there.
    expect(itemDisplayName(item({ id: 'glove-50d', url: 'https://x.test/dir/' })))
      .toBe('glove-50d');
    expect(itemDisplayName(item({ id: 'glove-50d', url: null }))).toBe('glove-50d');
  });
});

describe('PackItemRow — the caption beside the bar', () => {
  it('reads as a size while the item is downloading', () => {
    renderRow({ bytesDone: 45 * 1024 * 1024, bytesTotal: 90 * 1024 * 1024, percent: 50 });

    expect(screen.getByText('45 MB / 90 MB')).toBeInTheDocument();
  });

  it('reads as bytes so far when the server never sent a size', () => {
    renderRow({ bytesDone: 1024, bytesTotal: null, percent: null });

    expect(screen.getByText('1.0 KB')).toBeInTheDocument();
  });

  it('says converting instead of a size once the frames carry a caption', () => {
    // The convert step's numbers are WORDS: 10000 of 400000 rendered as
    // "9.8 KB / 391 KB" under a bar at 2.5%, right after a finished 66 MB
    // download. The size the reducer preserves is the download's, and it must
    // not be shown beside a bar that is measuring something else.
    renderRow({
      bytesDone: 69182535,
      bytesTotal: 69182535,
      percent: 2.5,
      text: 'Converting GloVe text to npz (one-time)',
    });

    expect(screen.getByText('Converting (one time)')).toBeInTheDocument();
    expect(screen.queryByText('66 MB / 66 MB')).toBeNull();
    // The server's English caption is the signal, never the copy on screen.
    expect(screen.queryByText(/npz/)).toBeNull();
    expect(screen.getByRole('progressbar', { name: 'glove.6B.zip' }))
      .toHaveAttribute('aria-valuenow', '2.5');
  });

  it('translates that caption with the rest of the panel', () => {
    useI18n.setState({ locale: 'zh-TW' });
    renderRow({ bytesDone: 69182535, bytesTotal: 69182535, percent: 2.5, text: 'Converting' });

    expect(screen.getByText('轉換中（只需一次）')).toBeInTheDocument();
  });
});
