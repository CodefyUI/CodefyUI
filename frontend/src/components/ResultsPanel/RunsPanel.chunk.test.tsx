import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import * as rest from '../../api/rest';
import { useI18n } from '../../i18n';
import { _resetRunStoreForTesting } from '../../store/runStore';
import { _resetSweepStoreForTesting } from '../../store/sweepStore';
import { useToastStore } from '../../store/toastStore';
import { RunsPanel } from './RunsPanel';

// In a file of its own: the lazy chunk's loader is cached per module, and
// here its import fails (a stale `dist/` after an update, a flaky network).
vi.mock('./NewSweepDialog', () => {
  throw new Error('chunk gone');
});

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listRuns: vi.fn(async () => ({ runs: [], total: 0, limit: 50, offset: 0 })),
  };
});

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  _resetRunStoreForTesting();
  _resetSweepStoreForTesting();
});

afterEach(() => {
  act(() => {
    _resetRunStoreForTesting();
    _resetSweepStoreForTesting();
  });
  vi.restoreAllMocks();
});

describe('RunsPanel: the New sweep chunk fails to load', () => {
  it('closes the dialog it was opening and says so', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    render(<RunsPanel panelHeight={400} />);
    const t = useI18n.getState().t;
    fireEvent.click(await screen.findByRole('button', { name: t('sweeps.new.title') }));
    await waitFor(() => expect(useToastStore.getState().toasts).toHaveLength(1));
    expect(useToastStore.getState().toasts[0]).toMatchObject({
      message: t('lazyChunk.failed'),
      type: 'error',
    });
    expect(warn).toHaveBeenCalled();
  });
});
