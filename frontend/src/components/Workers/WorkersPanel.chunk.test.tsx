import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { fetchWorkers } from '../../api/workers';
import { useI18n } from '../../i18n';
import { _resetWorkerStoreForTesting, useWorkerStore } from '../../store/workerStore';
import { settle } from '../../test/utils';
import { WorkersPanel } from './WorkersPanel';

// The Add worker dialog's chunk cannot load (a deploy replaced it, the
// network dropped): the panel closes the dialog instead of rendering nothing.
vi.mock('./AddWorkerDialog', () => {
  throw new Error('chunk failed to load');
});

vi.mock('../../api/workers', () => ({
  fetchWorkers: vi.fn(),
  drainWorker: vi.fn(),
  undrainWorker: vi.fn(),
  removeWorker: vi.fn(),
}));

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetWorkerStoreForTesting();
  vi.mocked(fetchWorkers).mockResolvedValue([]);
});

describe('WorkersPanel when the Add worker chunk cannot load', () => {
  it('closes the dialog and keeps the panel usable', async () => {
    render(<WorkersPanel />);
    act(() => useWorkerStore.getState().openPanel());
    await settle();
    const panel = screen.getByRole('dialog', { name: 'Workers' });
    fireEvent.click(within(panel).getByRole('button', { name: 'Add worker' }));
    await settle();
    await settle();
    expect(screen.queryByRole('dialog', { name: 'Add a worker' })).toBeNull();
    expect(screen.getByRole('dialog', { name: 'Workers' })).toBeInTheDocument();
  });
});
