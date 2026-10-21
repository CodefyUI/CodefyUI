import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act } from 'react';
import { render } from '@testing-library/react';
import { Toolbar } from './Toolbar';
import { useUIStore } from '../../store/uiStore';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';
import { _resetDeviceOptionsForTesting } from '../../hooks/useDeviceOptions';
import { settle } from '../../test/utils';

// The Queue on... dialog's chunk cannot load (a deploy replaced it, the
// network dropped): the toolbar closes the dialog instead of rendering
// nothing. A file of its own, so the lazy chunk's module-level cache starts
// empty.
vi.mock('./QueueRunDialog', () => {
  throw new Error('chunk failed to load');
});

vi.mock('../../hooks/useGraphExecution', () => ({
  useGraphExecution: () => ({ execute: vi.fn(), stop: vi.fn() }),
}));

vi.mock('../../api/rest', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/rest')>()),
  getRun: vi.fn(() => Promise.resolve(null)),
  fetchDevices: vi.fn(() =>
    Promise.resolve({
      default: 'cpu',
      devices: [{ value: 'cpu', label: 'CPU', detail: '', available: true }],
    }),
  ),
}));

describe('Toolbar when the Queue on... chunk cannot load', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useToastStore.setState({ toasts: [] });
    useUIStore.setState({ queueRunOpen: false });
    _resetDeviceOptionsForTesting();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    useUIStore.setState({ queueRunOpen: false });
  });

  it('closes the dialog and says the chunk failed', async () => {
    render(<Toolbar />);
    await settle();
    act(() => useUIStore.getState().setQueueRunOpen(true));
    await settle();
    await settle();
    expect(useUIStore.getState().queueRunOpen).toBe(false);
    expect(useToastStore.getState().toasts.some((toast) => toast.type === 'error')).toBe(true);
  });
});
