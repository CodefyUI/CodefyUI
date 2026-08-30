import { describe, it, expect, beforeEach } from 'vitest';
import type { PackItem, PackSummary } from '../../api/rest';
import { emptyPackJob, type PackJob } from '../../store/packStore';
import { useI18n } from '../../i18n';
import {
  catalogKey,
  formatBytes,
  jobOverallPercent,
  missingItems,
  statusTone,
  stepLabel,
} from './packStatus';

function item(over: Partial<PackItem> & { id: string }): PackItem {
  return {
    kind: 'hf',
    repo_id: `org/${over.id}`,
    url: null,
    size_bytes: 1024,
    license: 'apache-2.0',
    status: 'missing',
    ...over,
  };
}

function pack(over: Partial<PackSummary> & { id: string }): PackSummary {
  return {
    title: over.id,
    description: '',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: false,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...over,
  };
}

function job(over: Partial<PackJob> = {}): PackJob {
  return { ...emptyPackJob('j1', 'p1'), ...over };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

const t = () => useI18n.getState().t;

describe('statusTone', () => {
  it('maps every pack status onto a wash', () => {
    expect(statusTone('installed')).toBe('success');
    expect(statusTone('installing')).toBe('info');
    expect(statusTone('partial')).toBe('warning');
    expect(statusTone('needs_restart')).toBe('warning');
    expect(statusTone('not_installed')).toBe('neutral');
  });
});

describe('formatBytes', () => {
  it('scales to the largest unit that leaves a readable number', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(10 * 1024)).toBe('10 KB');
    expect(formatBytes(352 * 1024 * 1024)).toBe('352 MB');
    expect(formatBytes(1.5 * 1024 * 1024 * 1024)).toBe('1.5 GB');
  });

  it('says nothing rather than NaN for a size the server never sent', () => {
    expect(formatBytes(Number.NaN)).toBe('0 B');
    expect(formatBytes(-1)).toBe('0 B');
    expect(formatBytes(Number.POSITIVE_INFINITY)).toBe('0 B');
  });
});

describe('missingItems', () => {
  it('is everything that is not already on disk', () => {
    const p = pack({
      id: 'sentence-embeddings',
      items: [
        item({ id: 'a', status: 'present' }),
        item({ id: 'b', status: 'missing' }),
        // A half-written download still has bytes to fetch, so it counts.
        item({ id: 'c', status: 'downloading' }),
      ],
    });
    expect(missingItems(p).map((i) => i.id)).toEqual(['b', 'c']);
    expect(missingItems(pack({ id: 'empty' }))).toEqual([]);
  });
});

describe('jobOverallPercent', () => {
  it('weights each item by its size rather than counting them', () => {
    const p = pack({
      id: 'p1',
      items: [item({ id: 'big', size_bytes: 900 }), item({ id: 'small', size_bytes: 100 })],
    });
    const j = job({
      items: {
        big: { bytesDone: 450, bytesTotal: 900, percent: 50 },
        small: { bytesDone: 100, bytesTotal: 100, percent: 100 },
      },
    });
    // Counting items would say 75%; the bytes say 55%.
    expect(jobOverallPercent(j, p)).toBeCloseTo(55, 5);
  });

  it('uses the size the download reported when the catalog has none', () => {
    const j = job({ items: { x: { bytesDone: 25, bytesTotal: 100, percent: 25 } } });
    expect(jobOverallPercent(j, undefined)).toBeCloseTo(25, 5);
  });

  it('falls back to the step count when nothing has a size', () => {
    const j = job({
      steps: [
        { step: 'pip', label: 'pip', state: 'done' },
        { step: 'verify', label: 'verify', state: 'running' },
      ],
    });
    expect(jobOverallPercent(j, pack({ id: 'p1' }))).toBeCloseTo(50, 5);
  });

  it('is unknown, not zero, before the job has said anything', () => {
    expect(jobOverallPercent(job(), pack({ id: 'p1' }))).toBeNull();
    expect(jobOverallPercent(null, pack({ id: 'p1' }))).toBeNull();
  });

  it('never runs past 100 when a download overshoots its stated size', () => {
    const p = pack({ id: 'p1', items: [item({ id: 'x', size_bytes: 100 })] });
    const j = job({ items: { x: { bytesDone: 400, bytesTotal: 100, percent: 100 } } });
    expect(jobOverallPercent(j, p)).toBe(100);
  });

  it('believes the download over an under-reporting catalog', () => {
    // The bar read 100% while the row underneath still said 449 / 476 MB:
    // the catalog's weight saturated, so every byte past 449 was invisible.
    // The larger of the two promises is the honest denominator.
    const p = pack({ id: 'p1', items: [item({ id: 'x', size_bytes: 449 })] });
    const j = job({ items: { x: { bytesDone: 449, bytesTotal: 476, percent: 94 } } });
    expect(jobOverallPercent(j, p)).toBeCloseTo(94.3, 1);
  });
});

describe('stepLabel', () => {
  it('translates off the step id and carries the item name through', () => {
    expect(stepLabel(t(), 'pip', 'Installing python packages')).toBe(
      'Installing Python packages',
    );
    expect(stepLabel(t(), 'verify', 'verifying')).toBe('Verifying the installation');
    expect(stepLabel(t(), 'download:all-MiniLM-L6-v2', 'Downloading x')).toBe(
      'Downloading all-MiniLM-L6-v2',
    );
    expect(stepLabel(t(), 'convert:glove-6b-50d', 'Converting')).toBe(
      'Preparing glove-6b-50d',
    );
  });

  it('falls back to the server label for a step this build cannot name', () => {
    expect(stepLabel(t(), 'quantise:awq', 'Quantising awq')).toBe('Quantising awq');
    // A known kind with no item is still a step the UI cannot phrase.
    expect(stepLabel(t(), 'download', 'Downloading something')).toBe(
      'Downloading something',
    );
    // Nothing to fall back to: the raw id beats an empty line.
    expect(stepLabel(t(), 'quantise:awq', '')).toBe('quantise:awq');
  });

  it('follows the locale', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(stepLabel(t(), 'verify', 'verifying')).toBe('正在驗證安裝結果');
  });
});

describe('catalogKey', () => {
  it('answers only for a pack this build ships copy for', () => {
    expect(catalogKey('word-vectors', 'title')).toBe('packs.catalog.word-vectors.title');
    expect(catalogKey('word-vectors', 'desc')).toBe('packs.catalog.word-vectors.desc');
    expect(catalogKey('a-pack-from-the-future', 'title')).toBeNull();
    // Nothing inherited off Object.prototype counts as shipped copy.
    expect(catalogKey('constructor', 'title')).toBeNull();
  });
});
