import { describe, it, expect, beforeEach, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import * as rest from '../../api/rest';
import { useI18n } from '../../i18n';
import { _resetRunStoreForTesting } from '../../store/runStore';
import { _resetSweepStoreForTesting, useSweepStore } from '../../store/sweepStore';
import { useToastStore } from '../../store/toastStore';
import { settle } from '../../test/utils';
import { RunsPanel } from './RunsPanel';

// The New sweep dialog's chunk cannot load (a deploy replaced it, the
// network dropped): the panel closes the dialog instead of leaving the
// canvas shortcuts standing down behind nothing.
vi.mock('./NewSweepDialog', () => {
  throw new Error('chunk failed to load');
});

vi.mock('./LossChart', () => ({ LossChart: () => null }));

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return {
    ...actual,
    listRuns: vi.fn(async () => ({ runs: [], total: 0, limit: 50, offset: 0 })),
    getSweep: vi.fn(),
  };
});

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  _resetRunStoreForTesting();
  _resetSweepStoreForTesting();
});

describe('RunsPanel when the New sweep chunk cannot load', () => {
  it('closes the dialog and keeps the panel usable', async () => {
    render(<RunsPanel panelHeight={400} />);
    await waitFor(() => expect(rest.listRuns).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', { name: 'New sweep' }));
    expect(useSweepStore.getState().newSweepOpen).toBe(true);
    await settle();
    await settle();
    await waitFor(() => expect(useSweepStore.getState().newSweepOpen).toBe(false));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(screen.getByRole('button', { name: 'New sweep' })).toBeInTheDocument();
  });

  it('says so in an error toast', async () => {
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
    warn.mockRestore();
  });
});
